import pandas as pd
import numpy as np
from mip import *

class Opt_Solver:
	def __init__(self, solver_package):
		self.m = Model(sense=MINIMIZE, solver_name=solver_package)
		self._optimise_timeout = 120 # timeout at 2 minutes, 120 seconds
		self._decision_vars = {}
		self._obj_func_components = None
		self._constr_lhs = None
		self._constr_rhs = None
        
	def add_input_variables(self, input_variables):
		input_variables = input_variables.sort_values('variable_id')
		# Update mip solver variable type column
		typeentries = list(input_variables['type'].unique())
		if 'binary' in typeentries:
			typeentries.remove('binary')
			input_variables['type'] = np.where(input_variables['type']=='binary',BINARY,input_variables['type'])
		if 'continuous' in typeentries:
			typeentries.remove('continuous')
			input_variables['type'] = np.where(input_variables['type']=='continuous',CONTINUOUS,input_variables['type'])
		if not len(typeentries) == 0:
			err = typeentries
			raise Exception(f"Unknown variable type {err} found in df")

		# Add variables to mip solver
		for idx, var_id in enumerate(input_variables['variable_id']):
			self._decision_vars[var_id] = self.m.add_var(lb=input_variables.loc[idx,'lower_bound'],
												ub=input_variables.loc[idx,'upper_bound'],
												var_type=input_variables.loc[idx,'type'],
												name=str(var_id))

	def add_objective_function(self, objective_costs):
		objective_costs = objective_costs.sort_values('variable_id')
		# Minimise violation costs
		obj_fnc = minimize(xsum(objective_costs[objective_costs['variable_id']==i]['cost'].values[0] * self._decision_vars[i] for i in
						  objective_costs['variable_id']))
		self.m.objective = obj_fnc   

	def add_constraints(self, lhs_df, rhs_df):
		rhs_df = rhs_df.sort_values('constraint_id')
		# Iterate through rhs dataframe to add constraint equations
		for idx in rhs_df['constraint_id']:
			element_list = []
			lhs_elements = lhs_df[lhs_df['constraint_id']==idx]
			for variable in lhs_elements['variable_id']:
				multiplier = lhs_elements[lhs_elements['variable_id']==variable]['coefficient'].values[0]
				element = self._decision_vars[variable] * multiplier
				element_list += [element]

			expression = xsum(element_list)
			inequality = rhs_df[rhs_df['constraint_id']==idx]['type'].values[0]
			rhs_value = rhs_df[rhs_df['constraint_id']==idx]['rhs'].values[0]

			if inequality == '<=':
				constraint = expression <= rhs_value
			elif inequality == '==':
				constraint = expression == rhs_value
			elif inequality == '>=':
				constraint = expression >= rhs_value
			else:
				raise Exception("Error with format of inequality type")
			self.m.add_constr(constraint, name=str(idx))
                
	def add_constraints_dynamic(self, lhs_df, rhsD_df):
        # Add dynamic rhs constraints
		if not rhsD_df.empty:
			rhsD_df = rhsD_df.sort_values('constraint_id')
			for idx in rhsD_df['constraint_id']:
				# Find RHS variable
				inequality = rhsD_df[rhsD_df['constraint_id']==idx]['type'].values[0]
				rhs_var_id = rhsD_df.loc[rhsD_df['constraint_id']==idx, 'rhs_variable_id'].values[0]
				rhs_value = self._decision_vars[rhs_var_id]
                
				lhs_elements = lhs_df[lhs_df['constraint_id']==idx]
				# if rhsD_df[rhsD_df['constraint_id']==idx]['interval'].values[0] != None:
				# 	# Muliply LHS variable by coefficient.
				# 	lhs_var_id = lhs_elements['variable_id'].values[0]
				# 	multiplier = lhs_elements['coefficient'].values[0]
				# 	expression = self._decision_vars[lhs_var_id] * multiplier
				# else:
				# The constraint is a summation on lhs to set variable on rhs (used for ppa_lgc_sum_1)
				element_list = []
				for variable in lhs_elements['variable_id']:
					multiplier = lhs_elements[lhs_elements['variable_id']==variable]['coefficient'].values[0]
					element = self._decision_vars[variable] * multiplier
					element_list += [element]
				expression = xsum(element_list)
                    
				# if inequality == '==':
				# 	# Prevent error with optimisation trying to set 0 == variable
				# 	constraint = rhs_value == expression
				# else:
				# 	raise NotImplementedError("Dynamic RHS constraints does not yet allow >= or <= constraints")


				if inequality == '<=':
					constraint = expression <= rhs_value
				elif inequality == '==':
					constraint = rhs_value == expression # Prevent error with optimisation trying to set 0 == variable
				elif inequality == '>=':
					constraint = expression >= rhs_value
				else:
					raise Exception("Error with format of inequality type")

				self.m.add_constr(constraint, name=str(idx))
		
	def add_constraints_bigM(self, lhs_df, bigM_df):
		# Apply Big M constraints
		if not bigM_df.empty:
			bigM_df = bigM_df.sort_values('constraint_id')
			for idx in bigM_df['constraint_id']:
				# Define expression as LHS variable
				lhs_var_idx = bigM_df.loc[bigM_df['constraint_id']==idx, 'lhs_variable_id'].values[0]
				lhs_exp = self._decision_vars[lhs_var_idx]
                
				# Define rhs expression and other fields
				inequality = bigM_df.loc[bigM_df['constraint_id']==idx, 'constr_type'].values[0]
				mode = bigM_df.loc[bigM_df['constraint_id']==idx, 'mode'].values[0]
				Mval = bigM_df.loc[bigM_df['constraint_id']==idx, 'bigM'].values[0]
				rhs_var_idx = bigM_df.loc[bigM_df['constraint_id']==idx, 'rhs_variable_id'].values[0]
				rhs_bvar = self._decision_vars[rhs_var_idx]

				if mode == 'normal':
					rhs_exp = Mval * rhs_bvar
				elif mode == 'inverse':
					rhs_exp = Mval * (1 - rhs_bvar)
				else:
					raise Exception("BigM mode is invalid")
                
				if inequality == '<=':
					constraint = lhs_exp <= rhs_exp
				elif inequality == '==':
					constraint = lhs_exp == rhs_exp
				elif inequality == '>=':
					constraint = lhs_exp >= rhs_exp
				else:
					raise Exception("Error with format of inequality type")
				self.m.add_constr(constraint, name=str(idx))


	def add_sos_type_2(self, sos_name, weights_df, x_samples, y_samples, xvar_df, yvar_df):

		for interval in weights_df['interval'].unique():
			weights_ids = weights_df.loc[weights_df['interval']==interval, 'variable_id']
			weights_ids = weights_ids.sort_values().to_list()
			weights = [self._decision_vars[k] for k in weights_ids]

			xvar_id = xvar_df.loc[xvar_df['interval']==interval, 'variable_id']
			xvar = self._decision_vars[int(xvar_id)]

			yvar_id = yvar_df.loc[yvar_df['interval']==interval, 'variable_id']
			yvar = self._decision_vars[int(yvar_id)]

			# Weights
			self.m.add_sos([(weights[i], x_samples[i]) for i in range(len(x_samples))], 2)
		
			# Constraint for sum of weights 
			sum_of_weights = xsum(weights) == 1
			self.m.add_constr(sum_of_weights, name=sos_name+"_wsum")
			
			# Constraint for linking x_samples
			x_link = xsum([weights[i]*x_samples[i] for i in range(len(x_samples))]) == xvar
			self.m.add_constr(x_link, name=sos_name+"_xlink")

			# Constraint for linking y_samples
			y_link = xsum([weights[i]*y_samples[i] for i in range(len(y_samples))]) == yvar
			self.m.add_constr(y_link, name=sos_name+"_ylink")

		return

	def optimise(self):
		status = self.m.optimize(max_seconds=self._optimise_timeout)
		print(f"OPTIMISATION COMPLETE, Obj Value: {self.m.objective_value}")
		if status != OptimizationStatus.OPTIMAL:
			print("%%%%%%%%%%%%%%%%%%%%% INFEASIBLE MODEL %%%%%%%%%%%%%%%%%%%%%")
			print(f"STATUS: {OptimizationStatus}")
			con_index, newsvr = self.isolate_problem()
			print('Couldn\'t find an optimal solution, but removing con {} fixed INFEASIBLITY'.format(con_index))
			print("INFEASIBLE")#raise ValueError('Linear program infeasible')
			return newsvr, con_index
		else:
			return self.m, None
        

	def isolate_problem(self):
		"""Credit: N.Gorman (nempy)
		"""
		base_prob = self.m
		cons = []
		test_prob = base_prob.copy()
		for con in [con.name for con in base_prob.constrs]:
			[test_prob.remove(c) for c in test_prob.constrs if c.name == con]
			status = test_prob.optimize(max_seconds=self._optimise_timeout)
			cons.append(con)
			if status == OptimizationStatus.OPTIMAL:
				return cons, test_prob
		return []


	def get_decision_variable_vals(self, solver, var_df):
		result = var_df['variable_id'].apply(lambda x: solver.var_by_name(str(x)).x)
		var_df.insert(5,'value',result)
		return var_df
        
	def get_constraint_slack(self, solver, rhs_df):
		result = rhs_df['constraint_id'].apply(lambda x: solver.constr_by_name(str(x)).slack)
		rhs_df.insert(5,'slack',result)
		return rhs_df


