import pandas as pd
import numpy as np
from nemglo.backend.optimiser_formatters import *
from nemglo.backend import input_validation as inv
from nemglo.planning.planner import Plan

class Emissions:
    """Object to store the user input parameters of Emissions, check and validate inputs,
    then perform loading actions to planner object.
    """
    def __init__(self, system_plan, identifier):
        # Error check on parsed arguments
        assert isinstance(system_plan, Plan), "Generator Argument: 'system_plan' must be nemglo.planner.Plan object"
        assert isinstance(identifier, str), "Generator Argument: 'identifier' must be a str"
        inv.validate_unique_id(self.__class__, system_plan, identifier)

        # Link object to Plan
        self._system_plan = system_plan
        cname = self.__class__.__name__
        self._system_plan._components.update({cname: [self]})
        self._id = identifier

        # Emissions Characteristics
        self._trace = None # tCO2-e
        self._shadow_price = None # $/tCO2-e

    
    def load_emissions(self, trace, shadow_price):
        # Validate Inputs
        assert isinstance(shadow_price, float), "Emissions Argument: 'shadow_price' must be a float"
        self._validate_trace(trace)

        # Store Values
        self._trace = trace
        self._shadow_price = shadow_price


    def _validate_trace(self, trace):
        schema = inv.DataFrameSchema(name="trace", primary_keys=['Time', 'Intensity_Index'])
        schema.add_column(inv.SeriesSchema(name='Time', data_type=np.dtype('datetime64[ns]'), no_duplicates=True, \
            ascending_order=True, minimum=self._system_plan._timeseries[0], maximum=self._system_plan._timeseries[-1]))
        schema.add_column(inv.SeriesSchema(name='Intensity_Index', data_type=np.float64, must_be_real_number=True, \
            not_negative=True))
        schema.validate(trace)


    def add_emissions(self):
        planner = self._system_plan
        vre_prop_name = self._id + '-vre_proportion'
        co2_name = self._id + '-impact_emissions'
        grid_name = self._id + '-grid_load'
        load_name = planner._components['Electrolyser'][0]._id + '-mw_load'
        load_cap = planner._components['Electrolyser'][0]._capacity

        # > If there's at least one vre generator...
        if 'Generator' in planner._components:
            # Create `vre_proportion` variable equal to vre_avail but capped at load capacity. Priced in opt to minimise var.
            create_timeseries_vars(planner, var_name=vre_prop_name, lb=0, ub=np.inf)
            # vre_prop will not exceed max load
            create_constr_rhs_on_interval(planner, constr_name=vre_prop_name+"_maxload", constr_type="<=", rhs_value=load_cap) 
            create_constr_lhs_on_interval(planner, constr_name=vre_prop_name+"_maxload",
                constr_rhs=planner._constr_rhs[vre_prop_name+"_maxload"], coefficient_map={vre_prop_name: 1})
            # vre_prop will not exceed combined vre avails
            var_ids = planner._var[vre_prop_name][['interval', 'variable_id']]
            create_constr_rhs_on_interval_dynamicvar(planner, constr_name=vre_prop_name+"_avails",
                constr_type='>=', rhs_var_id_series=var_ids)
            coeffs = {}
            for gen in planner._components['Generator']:
                name_ppa_rec = gen._id + '-vre_avail'
                coeffs.update({name_ppa_rec: 1})
            create_constr_lhs_on_interval(planner, constr_name=vre_prop_name+"_avails",
                constr_rhs=planner._constr_rhs_dynamic[vre_prop_name+"_avails"], coefficient_map=coeffs)
            # Incentivise var to max out to upper limit
            create_objective_cost(planner, var_name=vre_prop_name, decision_var_series=planner._var[vre_prop_name], cost=-1)
        else:
            create_timeseries_vars(planner, var_name=vre_prop_name, lb=0, ub=0)

        # Create `grid_load` variable which is mw_load of electrolyser less RE propotion
        create_timeseries_vars(planner, var_name=grid_name, lb=0, ub=np.inf)
        
        # Set `impact emissions` variable
        var_ids = planner._var[grid_name][['interval', 'variable_id']]
        create_constr_rhs_on_interval_dynamicvar(planner, constr_name=grid_name,
            constr_type='==', rhs_var_id_series=var_ids)

        coeffs = {load_name: 1, vre_prop_name: -1}

        create_constr_lhs_on_interval(planner, constr_name=grid_name, constr_rhs=planner._constr_rhs_dynamic[grid_name],
                                        coefficient_map=coeffs)


        # Create `impact emissions` variable
        create_timeseries_vars(planner, var_name=co2_name, lb=0, ub=np.inf)
        # Set `impact emissions` variable
        var_ids = planner._var[co2_name][['interval', 'variable_id']]
        create_constr_rhs_on_interval_dynamicvar(planner, constr_name=co2_name,
            constr_type='==', rhs_var_id_series=var_ids)
        # Set `impact emissions` volume
        coeffs = self._trace['Intensity_Index'].mul(planner._intlength / 60).to_list()
        create_constr_lhs_on_interval_dynamic(planner, constr_name=co2_name,
            constr_rhs=planner._constr_rhs_dynamic[co2_name],
            decision_vars=planner._var[grid_name],
            coefficient_list=coeffs)


    def _price_emissions(self):
        planner = self._system_plan
        co2_name = self._id + '-impact_emissions'

        # Price `impact emissions` in optimiser
        series = planner._var[co2_name]
        create_objective_cost(planner, var_name=co2_name,
                    decision_var_series=series, cost=self._shadow_price)


    def _set_emissions_limit(self, limit_value): #limit_value in tCo2/tH2
        planner = self._system_plan
        co2_name = self._id + '-impact_emissions'
        co2tot_name = self._id + '-total_emissions'
        const_nm = self._id + '-em_limit'
        prodsum_name = planner._components['Electrolyser'][0]._id + '-h2_produced_sum'

        # Create variable for total emissions (kg) over simulation
        create_var(planner, var_name=co2tot_name, lb=0, ub=np.inf)
        rhs_var_id = planner._var[co2tot_name]['variable_id'].values[0]
        create_constr_rhs_dynamicvar(planner, constr_name=co2tot_name, constr_type="==", rhs_var_id=rhs_var_id)
        create_constr_lhs_on_interval_fixed_rhs(planner, constr_name=co2tot_name,
                                                    constr_rhs=planner._constr_rhs_dynamic[co2tot_name],
                                                    decision_vars=planner._var, coefficient_map={co2_name: 1000})

        # Create const for sum of emissions <= limit value
        # limit_value (tCo2/tH2) * h2 production >= total_emissions
        rhs_var_id = planner._var[co2tot_name]['variable_id'].values[0]
        create_constr_rhs_dynamicvar(planner, constr_name=const_nm, constr_type=">=", rhs_var_id=rhs_var_id)
        create_constr_lhs_on_interval(planner, constr_name=const_nm, constr_rhs=planner._constr_rhs_dynamic[const_nm],\
            coefficient_map={prodsum_name: limit_value})
        