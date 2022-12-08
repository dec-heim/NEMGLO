""" Solver Formatting Functions.
This file provides easier to use functions to interface with in creating variables and constraints for the optimiser.
"""
import pandas as pd
import numpy as np

def create_timeseries_vars(self, var_name, lb, ub, solver_type='continuous'):
	"""
	Creates a series of variables, one for each interval timestamp. Variable is defined by a name, lower bound, upper
	bound and variable type.

	Parameters
	----------
	var_name : str
		Variable name, used to reference and call variable within the optimiser.
	lb : float
		Lower bound parameter for minimum possible value of variable.
	ub : float
		Upper bound parameter for maximum possible value of variable.
	solver_type : ['binary','continuous'], optional
		Variable type. The default is 'continuous'.
	"""
	new_vars = pd.DataFrame({'variable_id': range(self._varid_idx,self._varid_idx+self._n),'interval': range(self._n),
						  'lower_bound': lb, 'upper_bound': ub, 'type': solver_type})
	self._varid_idx = max(new_vars['variable_id']) + 1
	self._var[var_name] = new_vars


def create_period_vars(self, var_name, interval_set, lb, ub, solver_type='continuous'):

	new_vars = pd.DataFrame({'variable_id': range(self._varid_idx, self._varid_idx+len(interval_set)), \
		'interval': interval_set['interval'], 'lower_bound': lb, 'upper_bound': ub, 'type': solver_type})
	self._varid_idx = max(new_vars['variable_id']) + 1
	self._var[var_name] = new_vars


def create_var(self, var_name, lb, ub, solver_type='continuous'):
	"""
	Create a single variable. Variable is defined by a name, lower bound, upper bound and variable type.

	Parameters
	----------
	var_name : str
		Variable name, used to reference and call variable within the optimiser.
	lb : float
		Lower bound parameter for minimum possible value of variable.
	ub : float
		Upper bound parameter for maximum possible value of variable.
	solver_type : ['binary','continuous'], optional
		Variable type. The default is 'continuous'.
	"""
	new_var = pd.DataFrame({'variable_id': range(self._varid_idx,self._varid_idx+1),'interval': None,
						  'lower_bound': lb, 'upper_bound': ub, 'type': solver_type})
	self._varid_idx = max(new_var['variable_id']) + 1
	self._var[var_name] = new_var    

def create_constr_rhs(self, constr_name, constr_type='<=', rhs_value=0.0):
	"""
	Create the right-hand side of a single constrant. This generates a unique constraint ID for the new constraint. 
	Constrant RHS is defined by a name, contraint/inequality type and RHS value.

	Parameters
	----------
	constr_name : str
		Constraint name, used to reference and call constraint within the optimiser.
	constr_type : ['<=','==','>=']
		In/equality type for the constraint. The default is '<='.
	rhs_value : float
		Defined static value for the RHS of the constraint. The default is 0.0.
	"""
	new_constr = pd.DataFrame({'constraint_id': range(self._conid_idx,self._conid_idx+1),'interval': None,
							'type': constr_type, 'rhs': rhs_value})
	self._conid_idx = max(new_constr['constraint_id']) + 1
	self._constr_rhs[constr_name] = new_constr    

def create_constr_rhs_on_interval(self, constr_name, constr_type='<=', rhs_value=0.0, intervals=[]):
	"""
	Create the right-hand side for a set of new constraints corresponding to each interval. This generates unique 
	constraint IDs for consequtive intervals. Constrant RHS is defined by a name, contraint/inequality type and RHS
	value.

	Parameters
	----------
	constr_name : str
		Constraint name, used to reference and call constraint within the optimiser.
	constr_type : ['<=','==','>=']
		In/equality type for the constraint. The default is '<='.
	rhs_value : float
		Defined static value for the RHS of the constraint. The default is 0.0.
	"""
	if intervals == []:
		int_list = range(self._n)
	else:
		int_list = intervals

	if (type(rhs_value) == int) or (type(rhs_value) == float):
		new_constr = pd.DataFrame({'constraint_id': range(self._conid_idx,self._conid_idx+len(int_list)),
							 'interval': int_list, 'type': constr_type, 'rhs': rhs_value})
	elif type(rhs_value) == list:
		new_constr = pd.DataFrame({'constraint_id': range(self._conid_idx,self._conid_idx+len(int_list)),
							 'interval': int_list, 'type': constr_type, 'rhs': rhs_value})
	else:
		raise ValueError("Invalid rhs_value type")
	self._conid_idx = max(new_constr['constraint_id']) + 1
	self._constr_rhs[constr_name] = new_constr

def create_constr_rhs_on_interval_dynamicvar(self, constr_name, constr_type, rhs_var_id_series, intervals=[]):
	"""
	Create the right-hand side for a set of new constraints corresponding to each interval with a dynamic RHS variable.
	The RHS is defined by a variable ID corresponding to each interval. This function generates unique 
	constraint IDs for consequtive intervals. Constrant RHS is defined by a name, contraint/inequality type and series
	of variable IDs for the RHS.

	Parameters
	----------
	constr_name : str
		Constraint name, used to reference and call constraint within the optimiser.
	constr_type : ['<=','==','>=']
		In/equality type for the constraint. The default is '<='.
	rhs_var_id_series : pd.DataFrame()
		Dataframe containing interval column and variable_id column.
	"""
	if intervals == []:
		int_list = range(self._n)
	else:
		int_list = intervals

	new_constr = pd.DataFrame({'constraint_id': range(self._conid_idx,self._conid_idx+len(int_list)),
							   'interval': int_list, 'type': constr_type})
	new_constr = new_constr.merge(rhs_var_id_series, on=['interval'])
	new_constr.rename(columns={'variable_id':'rhs_variable_id'},inplace=True)
	self._conid_idx = max(new_constr['constraint_id']) + 1
	self._constr_rhs_dynamic[constr_name] = new_constr

def create_constr_rhs_dynamicvar(self, constr_name, constr_type, rhs_var_id):
	"""
	Create the right-hand side as a dynamic variable ID for a single new constraint. This function generates a unique 
	constraint ID. Constrant RHS is defined by a name, contraint/inequality type and variable ID integer for the RHS.

	Parameters
	----------
	constr_name : str
		Constraint name, used to reference and call constraint within the optimiser.
	constr_type : ['<=','==','>=']
		In/equality type for the constraint. The default is '<='.
	rhs_var_id : int
		Variable ID for parameter to sit on the RHS.
	"""
	new_constr = pd.DataFrame({'constraint_id': range(self._conid_idx,self._conid_idx+1),'interval': None,
							'type': constr_type, 'rhs_variable_id': rhs_var_id})
	self._conid_idx = max(new_constr['constraint_id']) + 1
	self._constr_rhs_dynamic[constr_name] = new_constr

def create_constr_lhs_sum_intervals(self, constr_name, constr_rhs, coefficient_map=1):
	"""
	** IN-DEVELOPMENT **
	Create definitions for the LHS of a defined single constraint (by a previous RHS formulation), mapping a set of 
	variables for all intervals to be summated on the LHS with coefficients of 1. The RHS arugment must already be
	defined.

	Parameters
	----------
	constr_name : str
		Constraint name, used to reference and call constraint within the optimiser.
	constr_rhs : pd.DataFrame()
		The dataframe entry for the corresponding RHS constraint. In the form of, self._constr_rhs[{var_name}]
	coefficient_map : int
		The LHS coefficient mapped to each LHS variable. The default is 1.
	"""
	new_lhs = pd.DataFrame()
	for var in coefficient_map:
		coeff_df = pd.DataFrame({'constraint_id': constr_rhs['constraint_id'].values[0],
							   'variable_id': self._var[var]['variable_id'],
							   'coefficient': coefficient_map[var]})
		new_lhs = pd.concat([new_lhs, coeff_df])
	self._constr_lhs[constr_name] = new_lhs


def create_constr_lhs_sum_up_to_intervals(self, constr_name, constr_rhs, coefficient_map):
	
	new_lhs = pd.DataFrame()
	prev_itval_id = -1
	for idx, itval_id in enumerate(constr_rhs['interval']):
		for var in coefficient_map:
			sel_var_ids = self._var[var][(self._var[var]['interval'] > prev_itval_id) & \
				(self._var[var]['interval'] <= itval_id)]
			coeff_df = pd.DataFrame({'constraint_id': constr_rhs['constraint_id'][idx],
								'variable_id': sel_var_ids['variable_id'],
								'coefficient': coefficient_map[var]})
			new_lhs = pd.concat([new_lhs, coeff_df])
		prev_itval_id = itval_id
	self._constr_lhs[constr_name] = new_lhs


def create_constr_lhs_on_interval(self, constr_name, constr_rhs, coefficient_map):
	"""
	Create definitions for the LHS of a defined set of RHS constraints corresponding to each interval (by previous RHS 
	formulation). The mapping of LHS variables and coefficients may either be a single variable applied to all intervals
	of the RHS constraint set, or a different (interval-specific) variable corresponding to interval-specific RHS
	constraints. The coefficient map parameter allows multiple LHS variables to be considered in a single function call.

	Parameters
	----------
	constr_name : str
		Constraint name, used to reference and call constraint within the optimiser.
	constr_rhs : pd.DataFrame()
		The dataframe entry for the corresponding RHS constraint. In the form of, self._constr_rhs[{constr_name}]
	coefficient_map : dict
		Defines the matching coefficients to each variable ID set. The same coefficient must be applied to all LHS
		variables of the same name for all intervals. In the form of, {'{var_name_1}': 1, '{var_name_2}': -1}
	"""
	new_lhs = pd.DataFrame()
	for var in coefficient_map:
		# Collect rhs terms
		rhs_df = constr_rhs[['constraint_id','interval']] 
		# Collect lhs terms with coefficient
		lhs_coef = self._var[var][['variable_id','interval']]
		lhs_coef.insert(2,'coefficient',coefficient_map[var])
		# Merge lhs terms to rhs terms
		if (len(lhs_coef) == 1) and (lhs_coef['interval'].isnull().any()): # apply single var to all intervals
			rhs_df.insert(2,'variable_id',lhs_coef.loc[0,'variable_id'])
			rhs_df.insert(3,'coefficient',lhs_coef.loc[0,'coefficient'])
			new_lhs = pd.concat([new_lhs, rhs_df])
		else:
			new_lhs = pd.concat([new_lhs, rhs_df.merge(lhs_coef, on=['interval'])])
	
	if constr_name in self._constr_lhs:
		self._constr_lhs[constr_name] = pd.concat([self._constr_lhs[constr_name], new_lhs])
	else:
		self._constr_lhs[constr_name] = new_lhs

def create_constr_lhs_on_interval_dynamic(self, constr_name, constr_rhs, decision_vars, coefficient_list):
	"""
	Create definitions for LHS variable with a dynamic coefficient across intervals. For example used to define VRE
	availability where LHS arguments are dynamic coefficient (renewable trace) * single-variable across intervals (VRE 
	capacity) == RHS dynamic variable (VRE availability)

	Parameters
	----------
	constr_name : str
		Constraint name, used to reference and call constraint within the optimiser.
	constr_rhs : pd.DataFrame
		The dataframe entry for the corresponding RHS constraint. In the form of, self._constr_rhs[{var_name}]
	decision_vars : pd.DataFrame
		The dataframe of variables to be used as LHS arguments. In the form of, self._var[name_cap].
	coefficient_list : pd.Series
		Dynamic values to be applied as coefficients, ordered by consequtive intervals.

	Raises
	------
	NotImplementedError
		If the decision variable dataframe passed has variables for distinct intervals this error will raise. Only a 
		single variable with non-descript interval data can be passed as decision_vars.
	"""
	new_lhs = constr_rhs[['constraint_id','interval']]

	if not (len(decision_vars['interval']==1) & (decision_vars['interval'].values[0] == None)):
		new_lhs = new_lhs.merge(decision_vars[['variable_id','interval']], on='interval')
	else:
		lhs_var = decision_vars['variable_id'].values[0]
		new_lhs.insert(2,'variable_id',lhs_var)

	new_lhs.insert(3,'coefficient',coefficient_list)
	self._constr_lhs[constr_name] = new_lhs

def create_constr_lhs_on_interval_fixed_rhs(self, constr_name, constr_rhs, decision_vars, coefficient_map):
	"""
	** IN DEVELOPMENT ** 
	```{note}
	Redundancy with <create_constr_lhs_on_interval()>. Remove expression in future bug fixes.
	```
	"""
	new_lhs = pd.DataFrame()
	for var in coefficient_map:
		lhs_coef = decision_vars[var][['variable_id','interval']]
		lhs_coef.insert(2,'coefficient',coefficient_map[var])
		lhs_coef.insert(0,'constraint_id',constr_rhs['constraint_id'].values[0])
		new_lhs = pd.concat([new_lhs, lhs_coef])
	self._constr_lhs[constr_name] = new_lhs
	#return new_lhs


def create_constr_lhs_ramp(self, constr_name, constr_rhs, coeff_map_curr, coeff_map_next):

	new_lhs = pd.DataFrame()
	# Mapping coefficients for current intervals
	for var in coeff_map_curr:
		rhs = constr_rhs.copy()
		rhs = rhs[['constraint_id','interval']]
		lhs_coef = self._var[var][['variable_id','interval']]
		lhs_coef.insert(2,'coefficient', coeff_map_curr[var])
		lhs_coef = rhs.merge(lhs_coef, on='interval')
		new_lhs = pd.concat([new_lhs, lhs_coef])

	# Mapping coefficients for next intervals
	for var in coeff_map_next:
		rhs = constr_rhs.copy()
		rhs = rhs[['constraint_id','interval']]
		rhs.insert(2,'next',rhs['interval'] + 1) # NEW

		lhs_coef = self._var[var][['variable_id','interval']]
		lhs_coef.insert(2,'coefficient', coeff_map_next[var])
		lhs_coef.rename(columns={'interval': 'next'},inplace=True) # NEW
		lhs_coef = rhs.merge(lhs_coef, on='next') # DIFFERENT
		new_lhs = pd.concat([new_lhs, lhs_coef[lhs_coef.columns[lhs_coef.columns != 'next']]])

	new_lhs.to_csv(rf'E:\PROJECTS\NEMGLO\localonly\lhs_{constr_name}.csv')
	self._constr_lhs[constr_name] = new_lhs

	return


def create_contr_lhs_on_interval_matrix(self, constr_name, constr_rhs, coeff_matrix):
	"""
	
	|		| h2e_load	| h2_production	|
	|	-2	|	NaN		|	1.0			|
	|	-1	|	-1.0	|	NaN			|

	The above matrix index indicates the interval offset to apply for that row of variables, the values within the coefficents.
	"""
	new_lhs = pd.DataFrame()
	coeff_matrix = coeff_matrix.replace({np.nan: None})

	# Mapping coefficients based on matrix index
	for row, col in [(x, y) for x in coeff_matrix.index for y in coeff_matrix.columns]:
		if coeff_matrix.loc[row, col] is not None:
			rhs = constr_rhs.copy()
			rhs = rhs[['constraint_id','interval']]
			rhs.insert(2,'match_offset',rhs['interval'] + row) # Row is the offset to shift interval series

			lhs_coef = self._var[col][['variable_id','interval']]
			lhs_coef.insert(2,'coefficient', coeff_matrix.loc[row, col])
			lhs_coef = lhs_coef.rename(columns={'interval': 'match_offset'})
			lhs_coef = rhs.merge(lhs_coef, on='match_offset')
			new_lhs = pd.concat([new_lhs, lhs_coef[lhs_coef.columns[lhs_coef.columns != 'match_offset']]])

	# new_lhs.to_csv(rf'E:\PROJECTS\NEMGLO\localonly\lhs_{constr_name}.csv')
	self._constr_lhs[constr_name] = new_lhs




def create_constr_bigM_on_interval(self, constr_name, lhs_vars, rhs_bin_vars, constr_type='<=', mode='inverse', 
								   Mvalue=10000.0):
	"""
	Creates both the LHS and RHS elements in a bigM constraint formulation in one function.

	Parameters
	----------
	constr_name : str
		Constraint name, used to reference and call constraint within the optimiser.
	lhs_vars : pd.DataFrame
		The dataframe of LHS variables (interval-specific) to form the constraint. In the form of,
		self._var['{var_name}'].
	rhs_bin_vars : pd.DataFrame
		The dataframe of RHS variables (interval-specific) to form the constraint. In the form of,
		self._var['{var_name}'].
	constr_type : ['<=','==','>=']
		In/equality type for the constraint. The default is '<='.
	mode : ['inverse','normal']
		Determines the structure of the bigM constraint formulation. 'inverse' structures the formulation as:
		LHS <= M(1-RHS). 'normal' structures the formulation as: LHS <= M(RHS). The default is 'inverse'.
	Mvalue : float, optional
		The magnitude of the bigM 'swamping' parameter. The default is 10000.0.

	Raises
	------
	ValueError
		Raised if mode parameter is passed as something other than 'inverse' or 'normal'.
	"""
	if not ((mode == 'inverse') or (mode == 'normal')):
		raise ValueError('Unrecognised mode variable. Must be: normal, inverse')

	new_constr = pd.DataFrame({'constraint_id': range(self._conid_idx,self._conid_idx+self._n),
							'interval': range(self._n), 
							'constr_type': constr_type,
							'mode': mode,
							'bigM': Mvalue})

	lhs = lhs_vars[['variable_id','interval']]
	new_constr = new_constr.merge(lhs, on=['interval'])
	new_constr.rename(columns={'variable_id':'lhs_variable_id'},inplace=True)
	rhs = rhs_bin_vars[['variable_id','interval']]
	new_constr = new_constr.merge(rhs, on=['interval'])
	new_constr.rename(columns={'variable_id':'rhs_variable_id'},inplace=True)
	
	self._conid_idx = max(new_constr['constraint_id']) + 1
	self._constr_bigM[constr_name] = new_constr

def create_objective_cost(self, var_name, decision_var_series, cost=1000.0):
	"""
	Creates an objective cost parameter corresponding to a decision variable or set of decision variables. If the cost
	is passed as a float, the single value is mapped to all decision variables. However, if passed as a pd.DataFrame
	with a defined interval column, the unique interval-specific costs are mapped corresponding to interval-specific
	variables.

	Parameters
	----------
	var_name : str
		Variable name, used to reference and call variable within the optimiser.
	decision_var_series : pd.DataFrame
		The dataframe containing the decision variables to be costed in the objective function. In the form of, 
		self._var['{var_name}'].
	cost : float
		The corresponding cost mapped to the decision variables passed. The default is 1000.0.
	"""
	new_costs = decision_var_series[['variable_id','interval']]
	if (type(cost) == int) or (type(cost) == float):
		new_costs.insert(2,'cost', cost)
	elif type(cost) == pd.DataFrame:
		new_costs = new_costs.merge(cost, on=['interval'])
	self._objective_cost[var_name] = new_costs

def create_sos_type_2(self, x_samples, y_samples, sos_name, weight_var_name, x_var_name, y_var_name):

	# Arrange weight variable df
	weights_df = pd.concat([self._var[k] for k in self._var.keys() if k.__contains__(weight_var_name)], \
		ignore_index=True)
	weights_df = weights_df[['variable_id','interval']]

	x_var_df = self._var[x_var_name][['variable_id','interval']]
	y_var_df = self._var[y_var_name][['variable_id','interval']]

	self._sos_2[sos_name] = {
		'weights': weights_df,
		'x_samples': x_samples,
		'y_samples': y_samples,
		'x_vars': x_var_df,
		'y_vars': y_var_df
	}


def remove_element(plan, name, all_assoc=True, as_var=False, as_objective_cost=False, as_constr_lhs=False, \
	as_constr_rhs=False, as_constr_rhs_dynamic=False, as_constr_bigM=False, as_sos_2=False):
	"""Remove variable and/or constraint elements from the Plan object parsed.

	Parameters
	----------
	plan : nemglo.planner.Plan
		Plan object to remove variable/constraints from.
	name : str
		Variable or constraint name to remove. The identifier of a component object may also be used to remove that
		component entirely using all_assoc.
	all_assoc : bool, optional
		Set as true to remove all associations of the name from the Plan object, in all variable and constraint
		attributes.  
	as_var : bool, optional
		Set as true to remove variable name from the variable attributes of Plan object, by default False.
	as_objective_cost : bool, optional
		Set as true to remove variable name from the objective cost attributes of Plan object, by default False.
	as_constr_lhs : bool, optional
		Set as true to remove constraint name from the LHS constraint attributes of Plan object, by default False.
	as_constr_rhs : bool, optional
		Set as true to remove constraint name from the RHS constraint attributes of Plan object, by default False
	as_constr_rhs_dynamic : bool, optional
		Set as true to remove constraint name from the Dynamic RHS constraint attributes of Plan object, by default
		False.
	as_constr_bigM : bool, optional
		Set as true to remove constraint name from the Big M constraint attributes of Plan object, by default False.
	as_sos_2 : bool, optional
		Set as true to remove constraint name from the SOS Type 2 constraint attributes of Plan object, by default
		False.
	"""
	if as_var or all_assoc:
		for item in [k for k in plan._var.keys() if k.__contains__(name)]:
			plan._var.pop(item)
			print("Plan '{}' removed variable '{}'".format(plan._id, item))

	if as_objective_cost or all_assoc:
		for item in [k for k in plan._objective_cost.keys() if k.__contains__(name)]:
			plan._objective_cost.pop(item)
			print("Plan '{}' removed objective cost '{}'".format(plan._id, item))

	if as_constr_lhs or all_assoc:
		for item in [k for k in plan._constr_lhs.keys() if k.__contains__(name)]:
			plan._constr_lhs.pop(item)
			print("Plan '{}' removed constraint LHS '{}'".format(plan._id, item))

	if as_constr_rhs or all_assoc:
		for item in [k for k in plan._constr_rhs.keys() if k.__contains__(name)]:
			plan._constr_rhs.pop(item)
			print("Plan '{}' removed constraint RHS '{}'".format(plan._id, item))

	if as_constr_rhs_dynamic or all_assoc:
		for item in [k for k in plan._constr_rhs_dynamic.keys() if k.__contains__(name)]:
			plan._constr_rhs_dynamic.pop(item)
			print("Plan '{}' removed constraint RHS dynamic '{}'".format(plan._id, item))

	if as_constr_bigM or all_assoc:
		for item in [k for k in plan._constr_bigM.keys() if k.__contains__(name)]:
			plan._constr_bigM.pop(item)
			print("Plan '{}' removed objective cost '{}'".format(plan._id, item))

	if as_sos_2 or all_assoc:
		for item in [k for k in plan._sos_2.keys() if k.__contains__(name)]:
			plan._sos_2.pop(item)
			print("Plan '{}' removed objective cost '{}'".format(plan._id, item))

	if all_assoc:
		if len([k for k in plan._components if k._id.__contains__(name)]) > 0:
			plan._components.remove([k for k in plan._components if k._id.__contains__(name)][0])
			print("Plan '{}' unlinked the component '{}'".format(plan._id, name))
		else:
			print("Plan '{}' does not have any component '{}' to remove".format(plan._id, name))






