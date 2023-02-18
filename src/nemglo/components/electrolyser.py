import pandas as pd
import numpy as np
from datetime import timedelta
from nemglo.planning.planner import Plan
from nemglo.backend import input_validation as inv
from nemglo.backend.optimiser_formatters import *
from nemglo.defaults import *

class Electrolyser:
    """Object to store the user input parameters of electrolyser, check and validate inputs, then perform loading
    actions to planner object.

    Parameters
    ----------
    system_plan : nemglo.planner.Plan
        The system plan object for which the Electrolyser should be linked to.
        
    identifier : str
        A unique identifier to refer to the object as. For example, 'H2E'.

    """
    def __init__(self, system_plan, identifier):
        # Error check on parsed arguments
        assert isinstance(system_plan, Plan), "Electrolyser Argument: 'system_plan' must be nemglo.planner.Plan object"
        assert isinstance(identifier, str), "Electrolyser Argument: 'identifier' must be a str"
        inv.validate_unique_id(self.__class__, system_plan, identifier)

        # Link object to Plan
        self._system_plan = system_plan
        cname = self.__class__.__name__
        self._system_plan._components.update({cname: [self]})
        self._id = identifier

        # Generic Load Characteristics
        self._capacity = None
        self._maxload = None
        self._minload = None
        self._offload = None

        # (Electricity -> Hydrogen Production) Conversion - Specific Energy Consumption
        self._type = None
        self._profile = 'fixed'
        self._sec_nominal = None  # kWh/kg
        self._sec_conversion = None # set default at 100%
        self._sec_system = None  # Multiplication of _nominal and _conversion factor

        # Internally defined parameters
        self._sec_variable_points = None # pd.Dataframe mapping x-y points

        # Hydrogen only model - production benefit price
        self._h2_price_kg = None # $/kg

        # Hydrogen storage
        self._storage_max = None # Max storage in kg H2
        self._storage_initial = None
        self._storage_final = None
        self._storage_drain = None


    # ========================= Input & User Functions =========================
    def load_h2_parameters_preset(self, capacity, maxload, minload, offload, electrolyser_type, sec_profile, 
        h2_price_kg=None):
        """Load Electrolyser parameters to define the object.

        Parameters
        ----------
        capacity : float
            The desired rated capacity of the Electrolyser, in MW.
        maxload : float
            The maximum load operation of the Electrolyser, in MW. At present, NEMGLO assumes capacity parameter.
            `maxload` has null effect on the optimiser.
        minload : float
            The minimum stable loading (MSL) of the Electrolyser, in MW.
        offload : float
            The off state of the Electrolyser, in MW. In other words, absolute minimum value.
        electrolyser_type : str
            Either set as 'PEM' or 'AE'. This determines which set of predefined electrolyser characteristics are used.
        sec_profile : str
            Either set as 'fixed' or 'variable'. This determines the behaviour of the conversion function from
            electricity, as MW, to hydrogen, as kg.
        h2_price_kg : float, optional
            The sale price or production benefit received for the volume of hydrogen produced, by default None.
        """
        # Validate Inputs
        inv.validate_positive_float(capacity, 'capacity', type(self).__name__)
        inv.validate_positive_float(maxload, 'maxload', type(self).__name__)
        inv.validate_positive_float(minload, 'minload', type(self).__name__)
        inv.validate_positive_float(offload, 'offload', type(self).__name__)
        assert isinstance(electrolyser_type, str), "Electrolyser Argument: 'electrolyser_type' must be a str"
        assert isinstance(sec_profile, str), "Electrolyser Argument: 'sec_profile' must be a str"
        assert isinstance(h2_price_kg, (float, type(None))), "Electrolyser Argument: 'h2_price_kg' must be a \
            (float, or None)"

        assert electrolyser_type in ['PEM','AE'], \
            "Electrolyser Argument: 'electrolyser_type' must be one of ['PEM','AE']"
        assert sec_profile in ['variable','fixed'], \
            "Electrolyser Argument: 'sec_profile' must be one of ['variable','fixed']"

        # Store Values
        self._capacity = capacity
        self._maxload = maxload
        self._minload = minload
        self._offload = offload
        self._type = electrolyser_type
        self._profile = sec_profile
        self._h2_price_kg = h2_price_kg

        if self._type == "PEM":
            self._sec_nominal = SEC_NOMINAL_PEM
            self._sec_conversion = SEC_CONVERSION_PEM
            self._sec_variable_points = pd.DataFrame(SEC_PROFILE_PEM)
        elif self._type == "AE":
            self._sec_nominal = SEC_NOMINAL_AE
            self._sec_conversion = SEC_CONVERSION_AE
            self._sec_variable_points = pd.DataFrame(SEC_PROFILE_AE)
        
        self._sec_system = self._sec_nominal * self._sec_conversion


    def load_h2_parameters_custom(self, capacity, maxload, minload, offload, sec_profile, sec_nominal, sec_conversion,\
        sec_datapoints, h2_price_kg=None):
        raise NotImplementedError("Custom parameters are not permitted in this version of NEMGLO")


    def load_h2_storage_parameters(self):
        raise NotImplementedError("Storage parameters are not permitted in this version of NEMGLO")


    def add_electrolyser_operation(self):

        # Verify input parameters exist
        assert (self._maxload is not None) and (self._offload is not None), \
            "Missing Electrolyser Inputs. Be sure to call `load_h2_parameters` before `add_`"

        # Define MW Load variable
        inv.validate_existing(self._system_plan, self._id+'-mw_load', as_var=True, as_objective_cost=True)
        self._set_mw_load()

        # Define MW Load Sum
        inv.validate_existing(self._system_plan, self._id+'-mw_load_sum', as_var=True, as_constr_rhs_dynamic=True, \
            as_constr_lhs=True)
        self._set_mw_load_sum()

        # Fixed or Variable SEC
        if self._profile == 'fixed':
            inv.validate_existing(self._system_plan, self._id+'-h2_produced', as_var=True, as_constr_rhs_dynamic=True, \
            as_constr_lhs=True)
            self._set_h2_production_fixed()
        elif self._profile == 'variable':
            inv.validate_existing(self._system_plan, self._id+'-h2_produced', as_var=True)
            inv.validate_existing(self._system_plan, self._id+'-sec', as_sos_2=True)
            self._set_h2_production_variable()
        
        # Define H2 Production Sum
        inv.validate_existing(self._system_plan, self._id+'h2_produced_sum', as_var=True, as_constr_rhs_dynamic=True, \
            as_constr_lhs=True)
        self._set_h2_production_sum()

        # Production benefit price
        if self._h2_price_kg:
            inv.validate_existing(self._system_plan, self._id+'-h2_produced', as_objective_cost=True)
            self._price_h2_production()

        if self._minload > 0:
            self._set_h2_msl_limit()


    def remove_electrolyser_operation(self):
        """Removes all assosciations of the Electrolyser object from the Plan object.
        """
        remove_element(self._system_plan, self._id, all_assoc=True)


    # ==================== Electrolyser Operation Functions ====================
    def _set_mw_load(self):
        """Defines Electrolyser MW load variables and prices variables according to input market prices in Plan object.
        """
        planner = self._system_plan
        name_load = self._id + '-mw_load'

        # Create Load variables
        create_timeseries_vars(planner, var_name=name_load, lb=self._offload, ub=self._maxload)

        # Extract planner module prices
        price_series = pd.DataFrame({'interval': range(planner._n), 'cost': [planner._prices[i] * \
            (planner._intlength/60) for i in range(0, len(planner._prices))]})
        
        # Price the variables in optimisation cost function
        create_objective_cost(planner, var_name=name_load, decision_var_series=planner._var[name_load], \
            cost=price_series)


    def _set_mw_load_sum(self):
        """Add a variable summating all interval data for the Electrolyser mw_load variables.
        """
        planner = self._system_plan
        name_load_sum = self._id + '-mw_load_sum'
        name_load = self._id + '-mw_load'

        # New variable for summation of load series
        create_var(planner, var_name=name_load_sum, lb=0.0, ub=np.inf)
        
        # New constr with variable as RHS
        rhs_var_id = planner._var[name_load_sum]['variable_id'].values[0]
        create_constr_rhs_dynamicvar(planner, constr_name=name_load_sum, constr_type='==', rhs_var_id=rhs_var_id)

        # New lhs summation of constr
        create_constr_lhs_on_interval_fixed_rhs(planner, constr_name=name_load_sum,
                                                    constr_rhs=planner._constr_rhs_dynamic[name_load_sum],
                                                    decision_vars=planner._var, coefficient_map={name_load: 1})


    def _set_h2_production_fixed(self):
        """
        Creates a timeseries of Hydrogen Production variables for the optimiser determined by a fixed (linear) specific
        energy consumption (kWh/kg). Each variable is set by the equation of:
        h2_production [kg] = (1 / sec_system) [kg/kWh] * 10^3 [kWh/MWh] * h2e_load [MW] * \
            (Interval Length [min] / 60) [h]
        """
        planner = self._system_plan
        product_name = self._id + '-h2_produced'
        load_name = self._id + '-mw_load'

        # H2 production variable in kg
        create_timeseries_vars(planner, var_name=product_name, lb=0.0, ub=np.inf)
        
        # H2 production variable as determined by equation prior
        rhs_series = planner._var[product_name][['interval', 'variable_id']]
        create_constr_rhs_on_interval_dynamicvar(planner, constr_name=product_name,
                            constr_type='==', rhs_var_id_series=rhs_series)

        coeff_val = (planner._intlength / 60) * 10**3 * (1 / (self._sec_system))

        create_constr_lhs_on_interval(planner, constr_name=product_name,
                                        constr_rhs=planner._constr_rhs_dynamic[product_name],
                                        coefficient_map={load_name: coeff_val})


    def _set_h2_production_variable(self):
        """
        Creates a timeseries of Hydrogen Production variables for the optimiser determined by a variable specific
        energy consumption (kWh/kg) relationship. The determined y-variable (Hydrogen Production [kg]) is resolved by
        a Special Ordered Set (SOS) Type 2 in the optimisation solver. In essence, a piecewise linear approximation of
        the defined `_sec_variable_points`. 
        
        The fundamental equation converting SEC [kWh/kg] to [kg] is defined by:
        h2_production [kg] = (1 / sec_system) [kg/kWh] * 10^3 [kWh/MWh] * h2e_load [MW] * \
            (Interval Length [min] / 60) [h]
        """
        planner = self._system_plan
        product_name = self._id + '-h2_produced'
        load_name = self._id + '-mw_load'

        # Calculate sec_variable_points in terms of x: h2e_load [MW] and y: h2_volume [kg]
        self._sec_variable_points['h2e_load'] = self._sec_variable_points['h2e_load_pct'] * self._capacity
        
        self._sec_variable_points['sec'] = self._sec_variable_points['nominal_sec_pct'] * self._sec_system

        self._sec_variable_points['h2_volume'] = self._sec_variable_points['h2e_load'] * (planner._intlength / 60) \
            * 10**3 * (1 / self._sec_variable_points['sec'])
        
        # Extract x, y sample points from user defined _sec_variable_points
        x_samples = self._sec_variable_points['h2e_load'].to_list()
        y_samples = self._sec_variable_points['h2_volume'].to_list()

        # Declare Variables
        ## x_var: h2e_load already declared
        ## y_var:
        create_timeseries_vars(planner, var_name=product_name, lb=0.0, ub=np.inf)

        ## weight vars:
        for idx in range(len(x_samples)):
            create_timeseries_vars(planner, var_name=self._id+'-sec_w'+str(idx), lb=0, ub=1)

        # Define weights and link constraints in solver
        create_sos_type_2(planner, x_samples, y_samples, sos_name=self._id+'-sec', weight_var_name=self._id+'-sec_w',
            x_var_name=load_name, y_var_name=product_name)


    def _price_h2_production(self):
        """
        Defines a price for Hydrogen production [$/kg] as a benefit (not cost) in the optimisation solver.
        """
        planner = self._system_plan
        product_name = self._id + '-h2_produced'
        create_objective_cost(planner, var_name=product_name, decision_var_series=planner._var[product_name], \
            cost=-1*self._h2_price_kg)


    def _set_h2_production_sum(self):
        """Add a variable summating all interval data for the Electrolyser h2 production variables.
        """
        planner = self._system_plan
        name_h2_sum = self._id + '-h2_produced_sum'
        name_h2 = self._id + '-h2_produced'

        # New variable for summation of load series
        create_var(planner, var_name=name_h2_sum, lb=0.0, ub=np.inf)
        
        # New constr with variable as RHS
        rhs_var_id = planner._var[name_h2_sum]['variable_id'].values[0]
        create_constr_rhs_dynamicvar(planner, constr_name=name_h2_sum, constr_type='==', rhs_var_id=rhs_var_id)

        # New lhs summation of constr
        create_constr_lhs_on_interval_fixed_rhs(planner, constr_name=name_h2_sum,
                                                    constr_rhs=planner._constr_rhs_dynamic[name_h2_sum],
                                                    decision_vars=planner._var, coefficient_map={name_h2: 1})


    # =================== Electrolyser MSL & Ramp Functions ====================
    def _set_h2_msl_limit(self, viol_cost=1000.0, Mvalue=1000.0):
        planner = self._system_plan
        par_name = self._id + '-msl_'
        load_name = self._id + '-mw_load'

        # Limit H2 MW load operation to above self._minload
        # Create continuous violation variable and binary indicators
        create_timeseries_vars(planner, var_name=par_name+'violate', lb=0, ub=np.inf)
        create_timeseries_vars(planner, var_name=par_name+'penalise', lb=0, ub=1, solver_type='binary')
        create_timeseries_vars(planner, var_name=par_name+'relieve', lb=0, ub=1, solver_type='binary')

        # Create penalty and relief costs
        create_objective_cost(planner, var_name=par_name+'penalise',
                              decision_var_series=planner._var[par_name+'penalise'],
                              cost=viol_cost)
        create_objective_cost(planner, var_name=par_name+'relieve',
                              decision_var_series=planner._var[par_name+'relieve'],
                              cost=-1*viol_cost)

        # 0 to MSL violation region
        create_constr_rhs_on_interval(planner, constr_name=par_name+'violate', constr_type='>=',
                                      rhs_value=self._minload)
        create_constr_lhs_on_interval(planner, constr_name=par_name+'violate',
                                          constr_rhs=planner._constr_rhs[par_name+'violate'],
                                          coefficient_map={load_name: 1, par_name+'violate': 1})

        # Relax condition for electrolyser off state (P=0)
        create_constr_bigM_on_interval(planner, constr_name=par_name+'penalise',
                                       lhs_vars=planner._var[par_name+'violate'],
                                       rhs_bin_vars=planner._var[par_name+'penalise'],
                                       constr_type='<=', mode='normal', Mvalue=Mvalue)
        create_constr_bigM_on_interval(planner, constr_name=par_name+'relieve',
                                       lhs_vars=planner._var[load_name],
                                       rhs_bin_vars=planner._var[par_name+'relieve'],
                                       constr_type='<=', mode='inverse', Mvalue=Mvalue)


    def _set_ramp_variable(self):
        """Set the ramp variable

        .. note:: The units of the ramp variable are in MW. The user should consider this and dispatch interval length
                    for conventional reporting in MW/min. 
        """
        planner = self._system_plan
        load_name = self._id + '-mw_load'
        ramp_name = self._id + '-mw_ramp'
        down_name = self._id + '-mw_ramp_down'
        up_name = self._id + '-mw_ramp_up'

        timeless1 = pd.DataFrame(index=planner._timeseries[:-1], data={'interval':range(planner._n-1)})
        create_period_vars(planner, ramp_name, timeless1, lb=0, ub=np.inf)

        # Set the variable equal to the abs value difference between intervals 
        rhs_series = planner._var[ramp_name][['interval', 'variable_id']]

        # Ramp down
        create_constr_rhs_on_interval_dynamicvar(planner, down_name, constr_type="<=", \
            rhs_var_id_series=rhs_series, intervals=timeless1['interval'].to_list())
        offset_matrix = pd.DataFrame(data={load_name: [1,-1]},index=[0,1])
        create_contr_lhs_on_interval_matrix(planner, constr_name=down_name, constr_rhs=planner._constr_rhs_dynamic[down_name], \
            coeff_matrix=offset_matrix)

        # Ramp up
        create_constr_rhs_on_interval_dynamicvar(planner, up_name, constr_type="<=", \
            rhs_var_id_series=rhs_series, intervals=timeless1['interval'].to_list())
        offset_matrix = pd.DataFrame(data={load_name: [-1,1]},index=[0,1])
        create_contr_lhs_on_interval_matrix(planner, constr_name=up_name, constr_rhs=planner._constr_rhs_dynamic[up_name], \
            coeff_matrix=offset_matrix)


    def _set_ramp_cost(self, cost=10.0):
        planner = self._system_plan
        ramp_name = self._id+'-mw_ramp'

        create_objective_cost(planner, ramp_name, decision_var_series=planner._var[ramp_name], cost=cost)


    # ================ Electrolyser Production Target Functions ================
    def _set_force_h2_load(self, mw_value):
        planner = self._system_plan
        const_name = self._id + '-mw_force'
        load_name = self._id + '-mw_load'
        create_constr_rhs_on_interval(planner, constr_name=const_name, constr_type='==', rhs_value=mw_value)
        create_constr_lhs_on_interval(planner, constr_name=const_name, constr_rhs=planner._constr_rhs[const_name],
                                      coefficient_map={load_name: 1}) 


    def _set_production_target(self, target_value=100, bound="max", period="hour"):
        """Set production target

        .. warning:: In development (from old version)
        """
        planner = self._system_plan

        original = pd.DataFrame(index=planner._timeseries, data={'interval':range(planner._n)})
        original['TimeBegin'] = original.index - timedelta(minutes=planner._intlength)

        if period == "interval":
            newseries = original.copy()
            create_timeseries_vars(planner, f'production_target_{bound}_{period}', lb=0, ub=np.inf)
        elif period == "hour":
            newseries = original.copy().resample('H',closed='right',label='right').max()
            create_period_vars(planner, f'production_target_{bound}_{period}', newseries, lb=0, ub=np.inf)
        elif period == "day":
            newseries = original.copy().resample('D',closed='right').max()
            create_period_vars(planner, f'production_target_{bound}_{period}', newseries, lb=0, ub=np.inf)
        elif period == "week":
            newseries = original.set_index('TimeBegin').resample('W',closed='right').max()
            create_period_vars(planner, f'production_target_{bound}_{period}', newseries, lb=0, ub=np.inf)
        elif period == "month":
            newseries = original.set_index('TimeBegin').resample('M',closed='right').max()
            create_period_vars(planner, f'production_target_{bound}_{period}', newseries, lb=0, ub=np.inf)
        elif period == "year":
            newseries = original.set_index('TimeBegin').resample('A',closed='right').max()
            create_period_vars(planner, f'production_target_{bound}_{period}', newseries, lb=0, ub=np.inf)
        else:
            raise Exception("Invalid period passed. Valid periods: ['interval','hour','day','week','month','year']")

        # Check Bound Equality
        if bound == "max":
            equality = "<="
        elif bound == "min":
            equality = ">="
        elif bound == "equal":
            equality = "=="
        else:
            raise Exception("bound variable must be either: max, min, equal")

        # Set the production target variables to user defined value
        # e.g. production target >= 100
        # TODO: MAKE ELASTIC
        create_constr_rhs_on_interval(planner, f'prd_tgt_{bound}_{period}', constr_type=equality, \
            rhs_value=target_value, intervals=newseries['interval'].to_list())
        
        create_constr_lhs_on_interval(planner, f'prd_tgt_{bound}_{period}', \
            constr_rhs=planner._constr_rhs[f'prd_tgt_{bound}_{period}'], \
            coefficient_map={f'production_target_{bound}_{period}': 1})


        # Set the production target variables to be the summation of DI production outputs.
        rhs_series = planner._var[f'production_target_{bound}_{period}'][['interval', 'variable_id']]
        create_constr_rhs_on_interval_dynamicvar(planner, f'sum_prd_tgt_{bound}_{period}', constr_type="==", \
            rhs_var_id_series=rhs_series, intervals=newseries['interval'].to_list())

        create_constr_lhs_sum_up_to_intervals(planner, constr_name=f'sum_prd_tgt_{bound}_{period}', \
            constr_rhs=planner._constr_rhs_dynamic[f'sum_prd_tgt_{bound}_{period}'], \
            coefficient_map={self._id+'-h2_produced': 1})


        # Configuration is..
        # var: production_target_{bound}_{period} >= defined_value
        #  sum(h2_production) = RHS production_target...

        # To make this elastic add param in prd_tgt equality constraint

      
    # ======================= Hydrogen Storage Functions =======================
    def _set_h2_production_tracking(self):
        planner = self._system_plan
        stor_name = self._id + '-h2_stored'
        prod_name = self._id+'-h2_produced'

        # Var to reflect tracking sum of h2 produced.
        create_timeseries_vars(planner, var_name=stor_name, lb=0, ub=np.inf)

        # Constr: h2_stored = h2_stored[t-1] + h2_production[t]
        rhs_series = planner._var[stor_name][['interval', 'variable_id']]
        create_constr_rhs_on_interval_dynamicvar(planner, constr_name=stor_name, constr_type='==', \
            rhs_var_id_series=rhs_series)

        # Create LHS on interval-shifting matrix
        coeff_matrix = pd.DataFrame(data={stor_name: [1,np.nan], prod_name: [np.nan,1]}, index=[-1,0])
        create_contr_lhs_on_interval_matrix(planner, constr_name=stor_name, \
            constr_rhs=planner._constr_rhs_dynamic[stor_name], coeff_matrix=coeff_matrix)


    def _set_h2_storage_initial(self):
        planner = self._system_plan
        stor_name = self._id + '-h2_stored'
        init_name = self._id + '-h2_stored_initial'

        first_interval = pd.DataFrame(data={'interval':[0]})
        create_period_vars(planner, var_name=init_name, interval_set=first_interval, lb=0, ub=np.inf)
        create_constr_rhs_on_interval(planner, constr_name=init_name, constr_type='==', \
            rhs_value=self._storage_initial, intervals=[0])
        create_constr_lhs_on_interval(planner, constr_name=init_name, \
            constr_rhs=planner._constr_rhs[init_name], coefficient_map={init_name: 1})

        # Add initial value as parameter to LHS of storage amount
        create_constr_lhs_on_interval(planner, constr_name=stor_name, \
            constr_rhs=planner._constr_rhs_dynamic[stor_name], coefficient_map={init_name: 1})


    def _set_h2_storage_final(self):
        planner = self._system_plan
        stor_name = self._id + '-h2_stored'
        final_name = self._id + '-h2_stored_final'
        force_final = self._id + '-h2_stored_final_constraint'

        final_interval = pd.DataFrame(data={'interval':[planner._n - 1]})
        create_period_vars(planner, var_name=final_name, interval_set=final_interval, lb=0, ub=np.inf)
        create_constr_rhs_on_interval(planner, constr_name=final_name, constr_type='==', \
            rhs_value=self._storage_final, intervals=[planner._n - 1])
        create_constr_lhs_on_interval(planner, constr_name=final_name, \
            constr_rhs=planner._constr_rhs[final_name], coefficient_map={final_name: 1})

        # New constraint to force 'h2_stored_final' to equal 'h2_storage_amount' in last interval
        rhs_series = planner._var[stor_name][['interval', 'variable_id']]
        create_constr_rhs_on_interval_dynamicvar(planner, constr_name=force_final, constr_type="==", \
            rhs_var_id_series=rhs_series, intervals=[planner._n - 1])

        create_constr_lhs_on_interval(planner, constr_name=force_final, \
            constr_rhs=planner._constr_rhs_dynamic[force_final], coefficient_map={final_name: 1})


    def _set_h2_storage_max(self):
        planner = self._system_plan
        stor_name = self._id + '-h2_stored'
        max_name = self._id + '-h2_stored_limit'
        
        create_constr_rhs_on_interval(planner, constr_name=max_name, constr_type='<=', \
            rhs_value=self._storage_max)

        create_constr_lhs_on_interval(planner, constr_name=max_name, \
            constr_rhs=planner._constr_rhs[max_name], coefficient_map={stor_name: 1})


    def _set_storage_external_flow(self, external_flow=-500):
        """Set storage external flow

        .. todo:: Add dynamic storage external_flow that varies between intervals
        
        """
        planner = self._system_plan
        stor_name = self._id + '-h2_stored'
        ext_name = self._id + '-h2_stored_extflow'
        self._storage_ext_flow = [external_flow] * planner._n

        # Create external in/out-flow variable and set to defined mass-flow value.
        create_timeseries_vars(planner, var_name=ext_name, lb=-np.inf, ub=np.inf)
        create_constr_rhs_on_interval(planner, constr_name=ext_name, constr_type="==", \
            rhs_value=self._storage_ext_flow)
        create_constr_lhs_on_interval(planner, constr_name=ext_name, \
            constr_rhs=planner._constr_rhs[ext_name], coefficient_map={ext_name: 1})

        # Add external in/out-flow variable as LHS argument to storage system
        create_constr_lhs_on_interval(planner, constr_name=stor_name, \
            constr_rhs=planner._constr_rhs_dynamic[stor_name], coefficient_map={ext_name: 1})

