import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from nemglo.backend.optimiser_formatters import *
from nemglo.backend.solver_operator import Opt_Solver as solver
from nemglo.planning.data_fetch import *
from nemglo.defaults import *

from nemglo.backend import input_validation as inv

class Plan():
    """ Main Planner object to create and run the load optimisation.

    """
    def __init__(self, identifier):

        # Error check on parsed arguments
        assert isinstance(identifier, str), "Plan Argument: 'identifier' must be a str"
        
        # Object Parameters
        self._id = identifier
        self._components = {}

        # Market Data
        self._prices = []
        self._lgc_price = None
        self._timeseries = []
        self._n = None # KEEP AS IS
        self._intlength = None # KEEP AS IS
        self._rec_price = None

        # # Market => VRE Data
        # self._vre_det = None
        # self._vre_cf_1 = []
        # self._vre_cf_2 = []

        # Optimiser Data
        self._solver = 'GRB'
        self._var = {}
        self._objective_cost = {}
        self._constr_lhs = {}
        self._constr_rhs = {}
        self._constr_rhs_dynamic = {}
        self._constr_bigM = {}
        self._sos_2 = {}
        self._varid_idx = 0
        self._conid_idx = 0

        # Result Data
        self._out_vars = None
        self._out_constr = None



    def load_market_prices(self, market_prices):
        """
        Load market inputs to planner module.

        Parameters
        ----------
        timeseries : bool
                List of timestamps defining the intervals in the simulation period. The default is True.

        market_prices : pandas.DataFrame
            Timeseries data of energy prices to consider in optimisation.

            ========  ================================================
            Columns:  Description:
            Time      Dispatch interval timestamps (as `datetime.datetime`)
            Prices	  Market energy prices, $/MWh (as `np.float64`)
            ========  ================================================
        
        """

        self._validate_market_prices(market_prices)

        if any([self._timeseries, self._prices]):
            print("WARNING: Overwriting existing market_prices")

        self._timeseries = market_prices['Time'].to_list() #timeseries
        self._prices = market_prices['Prices'].to_list() #prices
        self._n = len(self._timeseries)
        self._intlength = np.diff(self._timeseries).min().seconds / 60


    def _validate_market_prices(self, market_prices):
        schema = inv.DataFrameSchema(name="market_prices", primary_keys=['Time', 'Prices'])
        schema.add_column(inv.SeriesSchema(name='Time', data_type=np.dtype('datetime64[ns]'), no_duplicates=True, ascending_order=True))
        schema.add_column(inv.SeriesSchema(name='Prices', data_type=np.float64, must_be_real_number=True))
        schema.validate(market_prices)


    def _load_rec_price(self, rec_price):
        assert isinstance(rec_price, (float, pd.DataFrame)), "load_rec_price Argument: 'rec_price' must be a float or pd.DataFrame"

        if type(rec_price) == pd.DataFrame:
            self._validate_certificate_prices(rec_price)

        self._rec_price = rec_price


    def _validate_certificate_prices(self, certificate_prices):
        schema = inv.DataFrameSchema(name="certificate_prices", primary_keys=['Time', 'Prices'])
        schema.add_column(inv.SeriesSchema(name='Time', data_type=np.dtype('datetime64[ns]'), no_duplicates=True, \
            ascending_order=True, minimum=self._timeseries[0], maximum=self._timeseries[-1]))
        schema.add_column(inv.SeriesSchema(name='REC_price', data_type=np.float64, must_be_real_number=True))
        schema.validate(certificate_prices)


    def _price_mkt_rec(self, allow_buying=True, allow_selling=False):
        name_rec = self._id + '-mkt_rec_buy'
        name_rec_sell = self._id + '-mkt_rec_sell'
       
        # Price series depending on float or df
        cost = self._rec_price
        if type(cost) == pd.DataFrame:
            cost['interval'] = range(self._n)
            cost.rename(columns={'REC_price': 'cost'}, inplace=True)
            cost['cost'] = cost['cost'].mul(self._intlength/60)
        else:
            cost = cost * (self._intlength/60)

        if allow_buying:
            create_timeseries_vars(self, var_name=name_rec, lb=0.0, ub=np.inf)
            create_objective_cost(self, var_name=name_rec, decision_var_series=self._var[name_rec],
                                    cost=cost)

        if allow_selling:
            create_timeseries_vars(self, var_name=name_rec_sell, lb=0.0, ub=np.inf)
            create_objective_cost(self, var_name=name_rec_sell, decision_var_series=self._var[name_rec_sell],
                                    cost=(-1 * cost))

    
    def _set_mkt_rec_sum(self, allow_buying=True, allow_selling=False):
        name_rec = self._id + '-mkt_rec_buy'
        name_rec_sum = self._id + '-mkt_rec_buy_sum'
        name_rec_sell = self._id + '-mkt_rec_sell'
        name_rec_sell_sum = self._id + '-mkt_rec_sell_sum'

        if allow_buying:
            # New variable for summation of vre_avail series
            create_var(self, var_name=name_rec_sum, lb=0.0, ub=np.inf)

            # New constr with variable as RHS
            rhs_var_id = self._var[name_rec_sum]['variable_id'].values[0]
            create_constr_rhs_dynamicvar(self, constr_name=name_rec_sum, constr_type='==', rhs_var_id=rhs_var_id)

            # New lhs summation of constraint
            create_constr_lhs_on_interval_fixed_rhs(self, constr_name=name_rec_sum,
                                                    constr_rhs=self._constr_rhs_dynamic[name_rec_sum],
                                                    decision_vars=self._var,
                                                    coefficient_map={name_rec: 1})

        if allow_selling:
            # New variable for summation of vre_avail series
            create_var(self, var_name=name_rec_sell_sum, lb=0.0, ub=np.inf)

            # New constr with variable as RHS
            rhs_var_id = self._var[name_rec_sell_sum]['variable_id'].values[0]
            create_constr_rhs_dynamicvar(self, constr_name=name_rec_sell_sum, constr_type='==', rhs_var_id=rhs_var_id)

            # New lhs summation of constraint
            create_constr_lhs_on_interval_fixed_rhs(self, constr_name=name_rec_sell_sum,
                                                    constr_rhs=self._constr_rhs_dynamic[name_rec_sell_sum],
                                                    decision_vars=self._var,
                                                    coefficient_map={name_rec_sell: 1})


    def _price_mkt_rec_sum(self, allow_buying=True, allow_selling=False):
        name_rec_sum = self._id + '-mkt_rec_buy_sum'
        name_rec_sell_sum = self._id + '-mkt_rec_sell_sum'

        cost = self._rec_price * (self._intlength/60)

        if allow_buying:
            create_var(self, var_name=name_rec_sum, lb=0.0, ub=np.inf)
            create_objective_cost(self, var_name=name_rec_sum, decision_var_series=self._var[name_rec_sum],
                                    cost=cost)

        if allow_selling:
            create_var(self, var_name=name_rec_sell_sum, lb=0.0, ub=np.inf)
            create_objective_cost(self, var_name=name_rec_sell_sum, decision_var_series=self._var[name_rec_sell_sum],
                                    cost=(-1 * cost))


    def _set_acquire_rec_on_interval(self, allow_buying=True, allow_selling=False):
        name_rec = self._id + '-mkt_rec_buy'
        name_rec_sell = self._id + '-mkt_rec_sell'
        name_load = self._components['Electrolyser'][0]._id + '-mw_load'
        name_constr = self._id+'-rec_acquire'

        coeffs = {}
        if allow_buying:
            coeffs.update({name_rec: 1})

        if allow_selling:
            coeffs.update({name_rec_sell: -1})

        if 'Generator' in self._components:
            for gen in self._components['Generator']:
                name_ppa_rec = gen._id + '-vre_avail'
                coeffs.update({name_ppa_rec: 1})

        rhs_series = self._var[name_load][['interval', 'variable_id']]
        create_constr_rhs_on_interval_dynamicvar(self, constr_name=name_constr,
                                    constr_type='>=', rhs_var_id_series=rhs_series)

        create_constr_lhs_on_interval(self, constr_name=name_constr, constr_rhs=self._constr_rhs_dynamic[name_constr],
                                        coefficient_map=coeffs)


    def autoadd_rec_price_on_interval(self, rec_price, allow_buying=True, allow_selling=True):
        assert isinstance(allow_buying, bool), "`autoadd_rec_price_on_interval` Argument: 'allow_buying' must be a bool"
        assert isinstance(allow_selling, bool), "`autoadd_rec_price_on_interval` Argument: 'allow_buying' must be a bool"

        self._load_rec_price(rec_price)
        self._price_mkt_rec(allow_buying, allow_selling)
        self._set_mkt_rec_sum(allow_buying, allow_selling)
        self._set_acquire_rec_on_interval(allow_buying, allow_selling)


    def _set_acquire_rec_on_sum(self, allow_buying=True, allow_selling=False):
        name_rec_sum = self._id + '-mkt_rec_buy_sum'
        name_rec_sell_sum = self._id + '-mkt_rec_sell_sum'
        name_load = self._components['Electrolyser'][0]._id + '-mw_load_sum'
        name_constr = self._id+'-rec_acquire'

        coeffs = {}
        if allow_buying:
            coeffs.update({name_rec_sum: 1})

        if allow_selling:
            coeffs.update({name_rec_sell_sum: -1})

        if 'Generator' in self._components:
            for gen in self._components['Generator']:
                name_ppa_rec = gen._id + '-ppa_rec_sum'
                coeffs.update({name_ppa_rec: 1})

        rhs_series = self._var[name_load]['variable_id'].values[0]
        create_constr_rhs_dynamicvar(self, constr_name=name_constr,
                                    constr_type='>=', rhs_var_id=rhs_series)

        create_constr_lhs_on_interval(self, constr_name=name_constr, constr_rhs=self._constr_rhs_dynamic[name_constr],
                                        coefficient_map=coeffs)


    def autoadd_rec_price_on_total(self, rec_price, allow_buying=True, allow_selling=True):
        assert isinstance(allow_buying, bool), "`autoadd_rec_price_on_total` Argument: 'allow_buying' must be a bool"
        assert isinstance(allow_selling, bool), "`autoadd_rec_price_on_total` Argument: 'allow_buying' must be a bool"

        self._load_rec_price(rec_price)
        assert isinstance(self._rec_price, float), "`autoadd_rec_price_on_total` requires a static REC price input as a float type. \
            Instead use `_on_interval` if you wish to define a dynamic REC price"
        self._price_mkt_rec_sum(allow_buying, allow_selling)
        self._set_acquire_rec_on_sum(allow_buying, allow_selling)
       

    def view_variable(self, name):
        """View optimiser variables which are in the planner object.

        Parameters
        ----------
        name : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        view_items = dict((k, self._var[k]) for k in [var for var in self._var if var.__contains__(name)])
        if len(view_items) > 0:
            view_items = pd.concat(view_items)
            view_items = view_items.droplevel(1, axis=0)
            view_items.index.name = 'variable_name'
            view_items.reset_index(inplace=True)
        else:
            print("Variable name not found in Plan: '{}'".format(self._id))
        return view_items
    
    # def view_constraint(self, constraint_name):
    # 	"""_summary_

    # 	Parameters
    # 	----------
    # 	constraint_name : _type_
    # 		_description_

    # 	Returns
    # 	-------
    # 	[name] : pandas.DataFrame
    # 		Timeseries data of generator trace.

    # 		========  ================================================
    # 		Columns:  Description:
    # 		Time      Dispatch interval timestamps (as `datetime.datetime`)
    # 		{duid}	  Capacity factor trace values between 0 and 1, % (as `np.float64`)
    # 		========  ================================================
        
    # 	"""

    # 	# if the interval is not given use (zero or none) 

    # 	# For the normal rhs one:
    # 	# columns: constraint_name, constraint_id, interval, coeff_var_1, var_1_name[var_id_1], .... , type, rhs



    # 	return var

    def add_emissions(self, emissions_obj):

        self._emissions += [emissions_obj]


    def relax_and_price_constr_violation(self, constr_name, sense, cost):
        """
        Need to consider the sense + or - of variable in LHS as pos or neg cost
        """
        # Not advisable to use relief var on equal bound production limit constraints
        if constr_name.__contains__('equal'):
            print("WARNING: It is not advisable to relief and price violation for `equal` bound constraints.\
                The cost implementation is not valid.")

        # Get constr_rhs
        if constr_name in self._constr_rhs_dynamic:
            print("in dynamic")
            sel_df = self._constr_rhs_dynamic[constr_name]
        elif constr_name in self._constr_rhs:
            sel_df = self._constr_rhs[constr_name]
            print("in normal RHS")
        else:
            raise Exception("`price_constraint_violation` could not find constr_name in planner object.\
                Only RHS and RHS_dynamic objects are considered")

        # Create violation variables
        if sense == '+':
            create_period_vars(self, var_name=f"relief_{constr_name}", interval_set=sel_df, lb=0, ub=np.inf)
        elif sense == '-':
            create_period_vars(self, var_name=f"relief_{constr_name}", interval_set=sel_df, lb=-np.inf, ub=0)
        else:
            raise Exception('Invalid `sense` argument. Must be one of '-' or '+'')

        # Insert relief vars into the LHS of constraint
        if constr_name in self._constr_rhs:
            create_constr_lhs_on_interval(self, constr_name=constr_name, constr_rhs=self._constr_rhs[constr_name],\
                coefficient_map={f"relief_{constr_name}": 1})

        if constr_name in self._constr_rhs_dynamic:
            create_constr_lhs_on_interval_dynamic(self, constr_name=constr_name, \
                constr_rhs=self._constr_rhs_dynamic[constr_name], coefficient_map={f"relief_{constr_name}": 1})

        # Price relief vars in objective cost function
        create_objective_cost(self, var_name=f"relief_{constr_name}", \
            decision_var_series=self._var[f"relief_{constr_name}"], cost=cost)

        return


    def optimise(self, solver_name="CBC", save_debug=False, save_results=False, results_dir=None):
        """
        Function called to run the optimisation solver. Files for decision variables and constraints are generated if
        save_debug_files is set as True. The generated optimisation results are saved in the planner module object.

        Parameters
        ----------
        save_debug_files : bool
                Save the decision variable and constraint files to local directory. The default is True.
        """
        svr = solver(solver_package=solver_name)

        # Save format
        if save_debug | save_results:
            # Check if results_dir is not None
            if (results_dir is None) or (not os.path.isdir(results_dir)):
                print("create results dir")
                if not os.path.isdir('results'):
                    os.mkdir('results')
                results_dir = os.path.join(os.getcwd(),'results')

            # Get timestamp for saving files
            timestamp = dt.strftime(dt.now(), "%Y_%m_%d_%H_%M")

            if save_debug:
                debug_fld = "NEMGLO_debug_" + timestamp
                sim_debug_dir = os.path.join(results_dir, debug_fld)
                os.mkdir(sim_debug_dir)

            if save_results:
                results_fld = "NEMGLO_results_" + timestamp
                sim_results_dir = os.path.join(results_dir, results_fld)
                os.mkdir(sim_results_dir)

    
        # Decision Variables
        decision_vars = pd.concat(self._var)
        decision_vars = decision_vars.droplevel(1, axis=0)
        decision_vars.index.name = 'variable_name'
        decision_vars.reset_index(inplace=True)
        svr.add_input_variables(input_variables=decision_vars)
        if save_debug:
            decision_vars.to_csv(os.path.join(sim_debug_dir, timestamp+"-DecisionVariables.csv"))

    # Objective Function Costs
        if self._objective_cost:
            obj_costs = pd.concat(self._objective_cost)
            obj_costs = obj_costs.droplevel(1, axis=0)
            obj_costs.index.name = 'variable_name'
            obj_costs.reset_index(inplace=True)
            svr.add_objective_function(objective_costs=obj_costs)
            if save_debug:
                obj_costs.to_csv(os.path.join(sim_debug_dir, timestamp+"-ObjectiveCosts.csv"))
        else:
            obj_costs = None

    # Constraint LHS
        if self._constr_lhs:
            constr_lhs = pd.concat(self._constr_lhs)
            constr_lhs = constr_lhs.droplevel(1, axis=0)
            constr_lhs.index.name = 'constraint_name'
            constr_lhs.reset_index(inplace=True)
            if save_debug:
                constr_lhs.to_csv(os.path.join(sim_debug_dir, timestamp+"-LHS.csv"))
        else:
            constr_lhs = None

    # Constraint RHS
        if self._constr_rhs:
            constr_rhs = pd.concat(self._constr_rhs)
            constr_rhs = constr_rhs.droplevel(1, axis=0)
            constr_rhs.index.name = 'constraint_name'
            constr_rhs.reset_index(inplace=True)
            svr.add_constraints(constr_lhs, constr_rhs)

            if save_debug:
                constr_rhs.to_csv(os.path.join(sim_debug_dir, timestamp+"-RHS.csv"))
        else:
            constr_rhs = None

    # Constraint RHS Dynamic
        if self._constr_rhs_dynamic:
            constr_rhsD = pd.concat(self._constr_rhs_dynamic)
            constr_rhsD = constr_rhsD.droplevel(1, axis=0)
            constr_rhsD.index.name = 'constraint_name'
            constr_rhsD.reset_index(inplace=True)
            svr.add_constraints_dynamic(constr_lhs, constr_rhsD)

            if save_debug:
                constr_rhsD.to_csv(os.path.join(sim_debug_dir, timestamp+"-RHS_dynamic.csv"))
        else:
            constr_rhsD = None

    # Big M constraint sets
        if self._constr_bigM:
            constr_bigM = pd.concat(self._constr_bigM)
            constr_bigM = constr_bigM.droplevel(1, axis=0)
            constr_bigM.index.name = 'constraint_name'
            constr_bigM.reset_index(inplace=True)
            svr.add_constraints_bigM(constr_lhs, constr_bigM)

            if save_debug:
                constr_bigM.to_csv(os.path.join(sim_debug_dir, timestamp+"-BigM.csv"))
        else:
            constr_bigM = None

    # SOS Type 2 set
        if self._sos_2:
            for sos_name in self._sos_2.keys():
                data = self._sos_2[sos_name]
                svr.add_sos_type_2(sos_name=sos_name,
                                    weights_df=data['weights'],
                                    x_samples=data['x_samples'],
                                    y_samples=data['y_samples'],
                                    xvar_df=data['x_vars'],
                                    yvar_df=data['y_vars'])

        # Concat RHS IDs
        all_rhs = pd.concat(
            [constr_rhs, constr_rhsD, constr_bigM], ignore_index=True)
        # Execute Optimiser
        # result_solver, removed_constrs = svr.optimise()
        result_solver, removed_constrs = svr.optimise()

        if removed_constrs:
            print("!Could not resolve constraints: {}".format(removed_constrs))
            rhs_problem_constrs = all_rhs['constraint_id'].isin([int(item) for item in removed_constrs])
            print(all_rhs[rhs_problem_constrs])
            all_rhs = all_rhs[~rhs_problem_constrs]

        self._out_vars = svr.get_decision_variable_vals(result_solver, decision_vars)
        self._out_constr = svr.get_constraint_slack(result_solver, all_rhs)
        """ need to remove searching for missing constraint"""

        # Optimisation outputs
        json_data, json_data['vars'], json_data['vars']['timeseries'], json_data['vars']['static'], json_data['inputtraces'] = {}, {}, {}, {}, {}
        #json_data['vars']['timeseries']['time'] = pd.Series(self._timeseries).to_json(orient='values')
        for variable in self._out_vars['variable_name'].unique():
            output = self._format_out_vars_timeseries(variable)

            if variable in ['h2_production','h2e_load']:
                json_data['vars']['timeseries'][variable] = output['value'].to_json(orient='values')
            else:
                json_data['vars']['static'][variable] = output['value'].to_json(orient='values')

            if save_results:
                output.to_csv(os.path.join(sim_results_dir,timestamp+"-"+variable+".csv"))

        # Input traces
        prc_output = pd.DataFrame({'time': self._timeseries,'interval':range(0,len(self._timeseries)),'value':self._prices})

        # if self._emissions:
        # 	em_output = pd.DataFrame({'time': self._timeseries,'interval':range(0,len(self._timeseries)),'value':self._emissions[0]._trace})

        json_data['inputtraces']['prices'] = prc_output.to_json()
        if save_results:
            prc_output.to_csv(os.path.join(sim_results_dir,timestamp+"-energy_price.csv"))
            if self._emissions:
                em_output.to_csv(os.path.join(sim_results_dir,timestamp+"-grid_emissions.csv"))

        # Write to file
        if save_results:
            with open(os.path.join(sim_results_dir,'outputs.json'), 'w') as json_file:
                json.dump(json_data, json_file)


        # Save costs to file
        if save_results:
            real_costs = self._sum_cost_components(cost_components='real')
            real_costs.to_csv(os.path.join(sim_results_dir,timestamp+"-cost_sum_REAL.csv"))

            shadow_costs = self._sum_cost_components(cost_components='shadow')
            shadow_costs.to_csv(os.path.join(sim_results_dir,timestamp+"-cost_sum_SHADOW.csv"))

            all_costs = self._sum_cost_components(cost_components='all')
            all_costs.to_csv(os.path.join(sim_results_dir,timestamp+"-cost_sum_ALL.csv"))

            for variable in self._objective_cost:
                output = self._compute_cost_component(variable)
                output.to_csv(os.path.join(sim_results_dir,timestamp+"-cost_"+variable+".csv"))

        # # Save all results to file
        # if save_results:
        #     # Optimiser Variables
        #     for variable in self._out_vars['variable_name'].unique():
        #         output = self._format_out_vars_timeseries(variable)
        #         output.to_csv(os.path.join(sim_results_dir,timestamp+"_"+variable+".csv"))

        #     # Input Variables
        #     prc_output = pd.DataFrame({'time': self._timeseries,'interval':range(0,len(self._timeseries)),'value':self._prices})
        #     prc_output.to_csv(os.path.join(sim_results_dir,timestamp+"_energy_price.csv"))

        return result_solver


    def get_load(self):
        """
        Retrieve the optimisation result for each interval load variable.

        Returns
        -------
        pd.DataFrame
                Returns dataframe containing timestamped and interval numbered load results.
        """
        elec = self._components['Electrolyser']
        prefix = elec[0]._id
        return self._format_out_vars_timeseries(prefix+'-mw_load')

    def get_production(self):
        """
        Retrieve the optimisation result for each interval h2 production variable.

        Returns
        -------
        pd.DataFrame
                Returns dataframe containing timestamped and interval numbered production results.
        """
        elec = self._components['Electrolyser']
        prefix = elec[0]._id
        return self._format_out_vars_timeseries(prefix+'-h2_produced')

    def get_storage_fill(self):
        """
        Retrieve the optimisation result for each interval h2 stored variable.

        Returns
        -------
        pd.DataFrame
                Returns dataframe containing timestamped and interval numbered stored results.
        """
        elec = self._components['Electrolyser']
        prefix = elec[0]._id
        return self._format_out_vars_timeseries(prefix+'-h2_stored')

    def get_impact_emissions(self):
        return self._format_out_vars_timeseries('impact_emissions')

    def get_vre_availability(self, identifier):
        """
        Retrieve the optimisation result for each interval vre availability for the specified DUID.

        Parameters
        ----------
        duid : str
                The DUID for which the vre availability should be returned for.

        Returns
        -------
        pd.DataFrame
                Returns dataframe containing timestamped and interval numbered vre availability results.
        """
        gens = self._components['Generator']
        specific_gen = [i for i in gens if i._id == identifier]
        prefix = specific_gen[0]._id
        return self._format_out_vars_timeseries(prefix+'-vre_avail')

    def get_vre_capacity_factor(self, identifier):
        """
        Retrieve the optimisation result for the average capacity factor over the simulation period for the specified
        DUID.

        Parameters
        ----------
        duid : str
                The DUID for which the capacity factor should be returned for.

        Returns
        -------
        float
                The capacity factor value corresponding to the specified VRE generator.
        """
        gens = self._components['Generator']
        specific_gen = [i for i in gens if i._id == identifier]
        return round(specific_gen[0]._trace[specific_gen[0]._duid].mean(), 3) * 100


    def get_total_consumption(self):
        """
        Retrieve the optimisation result for the total energy consumed by the load over the simulation period.

        Returns
        -------
        float
                The total energy consumption in MWh of the load.
        """
        elec = self._components['Electrolyser']
        prefix = elec[0]._id
        return float(self._out_vars[self._out_vars['variable_name'] == prefix+'-mw_load_sum'].value) * (self._intlength/60)


    def get_total_h2_production(self):
        """
        Retrieve the optimisation result for the total h2 produced by the load over the simulation period.

        Returns
        -------
        float
                The total energy consumption in MWh of the load.
        """
        elec = self._components['Electrolyser']
        prefix = elec[0]._id
        return float(self._out_vars[self._out_vars['variable_name'] == prefix+'-h2_produced_sum'].value)


    def get_load_capacity_factor(self):
        """
        Retrieve the optimisation result for the average capacity factor of the load over the simulation period.

        Returns
        -------
        load_capfac : float
                The capacity factor value of the electrolyser load.
        """
        elec = self._components['Electrolyser']
        load_mwh = self.get_total_consumption()
        load_capfac = load_mwh / (elec[0]._capacity * self._n * (self._intlength/60))
        return round(load_capfac,3) * 100


    def get_rec_on_interval(self, as_mwh=True):
        # Get REC series
        rec_df = self.get_load()[['interval','time','value']]
        rec_df.rename(columns={'value': str(self._components['Electrolyser'][0]._id) + ' RECs surrendered'}, \
            inplace=True)
        for series in [item for item in self._var.keys() if (item.__contains__('rec') and not item.__contains__('sum'))\
            or item.__contains__('vre_avail')]:
            res_set = self._format_out_vars_timeseries(series)
            res_set.loc[:,'value'] = res_set['value'].round(4)
            rec_df = rec_df.merge(res_set, on=['interval','time'], how='left')
            rec_df = rec_df.rename(columns={'value': series})

        if as_mwh:
            # Convert to MWh
            index_filt = rec_df.columns[~rec_df.columns.isin(['interval','time'])]
            rec_df.loc[:,index_filt] = np.ceil(rec_df[index_filt].mul((self._intlength/60)))

        # Rename REC columns
        for col in rec_df.columns:
            if col.__contains__(self._id) & col.__contains__('buy'):
                rec_df.rename(columns={col: 'Market RECs bought'}, inplace=True)
            elif col.__contains__(self._id) & col.__contains__('sell'):
                rec_df.rename(columns={col: 'Market RECs sold'}, inplace=True)
            elif 'Generator' in self._components:
                for gen in self._components['Generator']:
                    if col.__contains__(gen._id):
                        rec_df.rename(columns={col: '{} RECs acquired'.format(gen._id)}, inplace=True)

        return rec_df

    def get_rec_summary(self):
        rec_df = self._out_vars[(self._out_vars['variable_name'].str.contains('rec')) & \
                                (self._out_vars['interval'].isnull())]
        rec_df = rec_df.pivot_table(values='value',columns='variable_name')
        rec_df.columns.name = None
        rec_df = rec_df.round(4)
        rec_df = rec_df.mul((self._intlength/60)) # MWh
        rec_df = np.ceil(rec_df)

        # Add Electrolyser RECs surrendered
        elec_data = np.ceil(round(self.get_total_consumption(),4))
        rec_df.insert(0, str(self._components['Electrolyser'][0]._id) + ' RECs surrendered', \
            elec_data)

        # Rename REC columns
        for col in rec_df.columns:
            if col.__contains__(self._id) & col.__contains__('buy'):
                rec_df.rename(columns={col: 'Market RECs bought'}, inplace=True)
            elif col.__contains__(self._id) & col.__contains__('sell'):
                rec_df.rename(columns={col: 'Market RECs sold'}, inplace=True)
            elif 'Generator' in self._components:
                for gen in self._components['Generator']:
                    if col.__contains__(gen._id):
                        rec_df.rename(columns={col: '{} RECs acquired'.format(gen._id)}, inplace=True)
        return rec_df


    def get_lgc_summary(self):
        """
        Retrieve the optimisation result for LGCs received and surrendered by source.

        Returns
        -------
        lgc_summary : pd.DataFrame
                Returns a dataframe summarising the amount of LGCs received (and surrendered) in MWh terms, mapping to
                each source.
        """
        lgc_summary = pd.DataFrame(
            columns=['LGC_source', 'MWh_volume', 'percentage'])
        if self._out_vars['variable_name'].isin(['ppa_lgc_sum_1']).any():
            lgc_summary = pd.concat(
                [lgc_summary, self._format_out_lgc_stats('ppa_lgc_sum_1')])
        if self._out_vars['variable_name'].isin(['ppa_lgc_sum_2']).any():
            lgc_summary = pd.concat(
                [lgc_summary, self._format_out_lgc_stats('ppa_lgc_sum_2')])
        if self._out_vars['variable_name'].isin(['lgc_from_market']).any():
            lgc_summary = pd.concat(
                [lgc_summary, self._format_out_lgc_stats('lgc_from_market')])
        return lgc_summary


    def get_costs(self, exclude_shadow=True):
        output = pd.DataFrame()
        for variable in self._objective_cost:
            result = self._compute_cost_component(variable)
            result.insert(0,'variable_name',variable)
            output = pd.concat([output, result],ignore_index=True)

        table = output.pivot_table(index=['time','interval'], columns='variable_name', \
            values='value')
        table.insert(0,'total_cost',table.sum(axis=1))
        table = table.reset_index()
        table.columns.name = None

        if exclude_shadow:
            for col in table.columns:
                if any([col.__contains__(shadow_cost) for shadow_cost in cost_variables_shadow]):
                    table.drop(col, axis=1, inplace=True)
        return table

    # def get_costs_real(self):

    # 				real_costs = self._sum_cost_components(cost_components='real')
    # 		real_costs.to_csv(os.path.join(sim_results_dir,timestamp+"-cost_sum_REAL.csv"))

    # 		shadow_costs = self._sum_cost_components(cost_components='shadow')
    # 		shadow_costs.to_csv(os.path.join(sim_results_dir,timestamp+"-cost_sum_SHADOW.csv"))

    # 		all_costs = self._sum_cost_components(cost_components='all')
    # 		all_costs.to_csv(os.path.join(sim_results_dir,timestamp+"-cost_sum_ALL.csv"))

            

    def _compute_cost_component(self, var_name):
        cost_df = self._objective_cost[var_name]
        cost_df = cost_df.merge(self._out_vars[['variable_id','interval','value']], on=['variable_id','interval'])
        cost_df.rename(columns={'value':'variable_value'},inplace=True)
        cost_df.insert(3,'value', cost_df['cost'] * cost_df['variable_value'])

        if len(cost_df['interval']) == self._n:
            cost_df.insert(0,'time', self._timeseries)
        else:
            cost_df.insert(0,'time', np.nan)

        return cost_df[['time','interval','variable_value','cost','value']].reset_index(drop=True)


    def _sum_cost_components(self, cost_components):
        df = pd.DataFrame({'interval':range(0,len(self._timeseries))})

        if cost_components == 'real':
            filt_list = cost_variables_realized
        elif cost_components == 'shadow':
            filt_list = cost_variables_shadow
        elif cost_components == 'all':
            filt_list = cost_variables_realized + cost_variables_shadow
        else:
            raise Exception("Invalid cost_components argument. Must be one of `real`, `shadow`, `all`")

        for var_name in self._objective_cost:
            if any([var_name.__contains__(filt_list[i]) for i in range(len(filt_list))]):
                cost_df = self._compute_cost_component(var_name)
                df = df.merge(cost_df[['interval','value']], on=['interval'], how='left')
                df.rename(columns={'value': var_name}, inplace=True)

        df.set_index('interval', inplace=True)
        df['value'] = df.sum(axis=1)
        df.reset_index(inplace=True)
        df.insert(0,'time', self._timeseries)

        return df[['time','interval','value']]


    def _format_out_vars_timeseries(self, var_name):
        sim_times = pd.DataFrame({'interval': range(self._n), 'time': self._timeseries})

        result = self._out_vars[self._out_vars['variable_name'] == var_name][[
            'interval', 'value']]
        if (len(result) > 1) & (result['interval'].values[0] != None):
            result = result.sort_values('interval')
            result = result.merge(sim_times, on='interval', how='right')
        return result.reset_index(drop=True)

    def _format_out_lgc_stats(self, component_name):
        name_map = {'ppa_lgc_sum_1': self._vre_det['DUID'][1], 'ppa_lgc_sum_2': self._vre_det['DUID'][2],
                    'lgc_from_market': 'Market'}
        total_mwh = self.get_total_consumption()
        value = float(self._out_vars[self._out_vars['variable_name']
                      == component_name].value) * (self._intlength/60)
        percent = (value / total_mwh) * 100
        lgc_component = pd.DataFrame({'LGC_source': [name_map[component_name]], 'MWh_volume': [int(np.floor(value))],
                                      'percentage': [percent]})
        return lgc_component


    def _find_component_class(self, component_class):
        return [c for c in self._components if c.__class__.__name__ == component_class]