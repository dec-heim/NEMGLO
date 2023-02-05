import pandas as pd
import numpy as np
from nemglo.planning.planner import Plan
from nemglo.backend import input_validation as inv
from nemglo.backend.optimiser_formatters import *

class Generator:
    """Generator class is used to create an object to store parameters defining an Variable Renewable Energy (VRE)
    source for the load to contract a Power Purchase Agreements (PPA) with.

    Parameters
    ----------
    system_plan : nemglo.planner.Plan
        The system plan object for which the Generator should be linked to.
    
    identifier : str
        A unique identifier to refer to the object as. For example, 'VRE1' for the first PPA, 'VRE2' for the second.
    
    .. todo:: Inherit data directly via data_fetch and flag an attribute to signify this vs custom inputs.
        From using python, writing a yaml file to show inputs saved in simulation.

    """
    def __init__(self, system_plan, identifier):
        # Error check on parsed arguments
        assert isinstance(system_plan, Plan), "Generator Argument: 'system_plan' must be nemglo.planner.Plan object"
        assert isinstance(identifier, str), "Generator Argument: 'identifier' must be a str"
        inv.validate_unique_id(self.__class__, system_plan, identifier)

        # Link object to Plan
        self._system_plan = system_plan
        cname = self.__class__.__name__
        if cname in self._system_plan._components:
            self._system_plan._components.update({cname: self._system_plan._components[cname] + [self]})
        else:
            self._system_plan._components.update({cname: [self]})
        self._id = identifier

        # Generic Plant Characteristics
        self._duid = None
        self._capacity = None
        self._trace = None

        # PPA Structure
        self._ppa_strike = None
        self._ppa_floor = None

    # Functions to validate inputs
    def load_vre_parameters(self, duid, capacity, trace, ppa_strike, ppa_floor=None):
        """Load VRE parameters defining the Generator object.

        Parameters
        ----------
        duid : str
            The unique string of the VRE Generator. If using a custom input, create a custom name for reference.
        capacity : float
            The desired rated capacity of the VRE Generator, in MW. Note the input trace will be scaled accordingly.
        trace : pandas.DataFrame
            Timeseries data of generator trace.

            ========  ================================================
            Columns:  Description:
            Time      Dispatch interval timestamps (as `datetime.datetime`)
            {duid}	  Capacity factor trace values between 0 and 1, % (as `np.float64`)
            ========  ================================================
        
        ppa_strike : float
            The strike price for the Power Purchase Agreement (PPA) with the VRE Generator, in $/MWh.
        ppa_floor : float, optional
            A floor price for the PPA contract if desired, in $/MWh, by default None.
        
        """
        # Validate Inputs
        assert isinstance(duid, str), "Generator Argument: 'duid' must be a str"
        assert isinstance(capacity, float), "Generator Argument: 'capacity' must be a float"
        assert isinstance(trace, pd.DataFrame), "Generator Argument: 'trace' must be a pd.DataFrame"
        assert isinstance(ppa_strike, float), "Generator Argument: 'ppa_strike' must be a float"
        assert isinstance(ppa_floor, (float, type(None))), "Generator Argument: 'ppa_floor' must be a \
            (float, or None)"

        self._validate_trace(trace, duid)
    
        # Store Values
        self._duid = duid
        self._capacity = capacity
        self._trace = trace
        self._ppa_strike = ppa_strike
        self._ppa_floor = ppa_floor


    def _validate_trace(self, trace, duid):
        schema = inv.DataFrameSchema(name="trace", primary_keys=['Time', duid])
        schema.add_column(inv.SeriesSchema(name='Time', data_type=np.dtype('datetime64[ns]'), no_duplicates=True, \
            ascending_order=True, minimum=self._system_plan._timeseries[0], maximum=self._system_plan._timeseries[-1]))
        schema.add_column(inv.SeriesSchema(name=duid, data_type=np.float64, must_be_real_number=True, \
            not_negative=True))
        schema.validate(trace)


    # Functions to call from within the planner
    def add_ppa_contract(self):
        """Constructs a PPA contract between the load and VRE Generator, given the defined parameters of the generator.

        .. tip:: This function is for standard python use. For more advanced control / direct interfacing, instead use
            the `_set` functions.
        """
        # Verify input parameters exist
        assert (self._duid and self._capacity and any(self._trace) and self._ppa_strike), \
            "Missing Generator Inputs. Be sure to call `load_parameters` before `add_`"

        # Define capacity variable
        inv.validate_existing(self._system_plan, self._id+'-vre_cap', as_var=True, as_constr_lhs=True, \
            as_constr_rhs=True)
        self._set_capacity_var()
        self._set_fixed_capacity()

        # Define VRE availability
        inv.validate_existing(self._system_plan, self._id+'-vre_avail', as_var=True, as_constr_lhs=True, \
            as_constr_rhs_dynamic=True)
        self._set_availability()

        # Define PPA cost function
        inv.validate_existing(self._system_plan, self._id+'-ppa_cfd', as_objective_cost=True)
        self._set_ppa_cost_function()

        # Define PPA volume sum for RECs
        inv.validate_existing(self._system_plan, self._id+'-ppa_rec_sum', as_var=True, as_constr_lhs=True, \
            as_constr_rhs_dynamic=True)
        self._set_ppa_rec_sum()


    def remove_ppa_contract(self):
        """Removes all assosciations of the Generator object from the Plan object.
        """
        remove_element(self._system_plan, self._id, all_assoc=True)


    def _set_capacity_var(self, ub=np.inf):
        """Defines Generator capacity variable in the optimiser as -> Capacity = 100.0
        """
        name_cap = self._id + '-vre_cap'
        planner = self._system_plan

        create_var(planner, var_name=name_cap, lb=0, ub=ub)


    def _set_fixed_capacity(self):
        name_cap = self._id + '-vre_cap'
        planner = self._system_plan

        create_constr_rhs(planner, constr_name=name_cap, constr_type='==', rhs_value=self._capacity)
        
        create_constr_lhs_on_interval(planner, constr_name=name_cap, \
            constr_rhs=planner._constr_rhs[name_cap], coefficient_map={name_cap: 1})
        

    def _set_availability(self):
        """Defines the VRE generator's availability in the optimiser.
        """
        name_avail = self._id + '-vre_avail'
        name_cap = self._id + '-vre_cap'
        planner = self._system_plan

        create_timeseries_vars(planner, var_name=name_avail, lb=0.0, ub=np.inf)
        
        var_ids = planner._var[name_avail][['interval', 'variable_id']]
        create_constr_rhs_on_interval_dynamicvar(planner, constr_name=name_avail,
            constr_type='==', rhs_var_id_series=var_ids)

        create_constr_lhs_on_interval_dynamic(planner, constr_name=name_avail,
            constr_rhs=planner._constr_rhs_dynamic[name_avail],
            decision_vars=planner._var[name_cap],
            coefficient_list=self._trace[self._duid])


    def _set_ppa_cost_function(self):
        """Add the PPA cost function of the VRE generator as a CfD traded on the variable volume of available
        generation.

        .. note:: The cost component in the objective function is expressed in $/MW.
        """
        planner = self._system_plan
        strike = self._ppa_strike
        spotprices = planner._prices
        intlength = planner._intlength
        n = planner._n
        floor = self._ppa_floor
        name_avail = self._id + '-vre_avail'

        if floor != None:
            # PPA takes the difference between strike and floor if spot < floor.
             cfd_val = [(strike - max(floor, spotprices[i])) * (intlength/60) for i in range(0, len(spotprices))]
        else:
            # PPA takes the difference between strike and spot.
            cfd_val = [(strike - spotprices[i]) * (intlength/60) for i in range(0, len(spotprices))]

        series = planner._var[name_avail]
        price_series = pd.DataFrame({'interval': range(n), 'cost': cfd_val})

        create_objective_cost(planner, var_name=self._id+'-ppa_cfd', decision_var_series=series, cost=price_series)


    def _set_ppa_rec_sum(self):
        """Add a summation of the traded volume of the PPA CfD which can used to consider the volume of Renewable Energy
        Certificates received by the load on the assumption of a bundled PPA.

        .. note:: The `ppa_rec_sum` variable is expressed in MW.
        """
        planner = self._system_plan
        name_rec_sum = self._id + '-ppa_rec_sum'
        name_avail = self._id + '-vre_avail'

        # New variable for summation of vre_avail series
        create_var(planner, var_name=name_rec_sum, lb=0.0, ub=np.inf)

        # New constr with variable as RHS
        rhs_var_id = planner._var[name_rec_sum]['variable_id'].values[0]
        create_constr_rhs_dynamicvar(planner, constr_name=name_rec_sum, constr_type='==', rhs_var_id=rhs_var_id)

        # New lhs summation of constraint
        create_constr_lhs_on_interval_fixed_rhs(planner, constr_name=name_rec_sum,
                                                constr_rhs=planner._constr_rhs_dynamic[name_rec_sum],
                                                decision_vars=planner._var,
                                                coefficient_map={name_avail: 1})
