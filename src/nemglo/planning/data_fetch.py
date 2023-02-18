import pandas as pd
import numpy as np
import logging
import os
from nemosis import dynamic_data_compiler, static_table
from nemed import *
from nemed.downloader import download_unit_dispatch
from datetime import datetime as dt, timedelta
from nemglo.backend import input_validation as inv

logger = logging.getLogger(__name__)

class Market:
    def __init__(self, local_cache=r'.\cache', intlength=5, region=None, start_date=None, end_date=None, duid_1=None, \
        duid_2=None, emissions_type=None):
        logger.info("Establish nemosis_data logger")
        self._check_data_inputs(local_cache, intlength, region, start_date, end_date, duid_1, duid_2, emissions_type)
        local_cache = _check_cache(local_cache)
        # Inputs
        self._cache = local_cache
        self._intlength = intlength
        self._region = region
        self._start = start_date + ":00"
        self._end = end_date + ":00"
        self._duid_1 = duid_1
        self._duid_2 = duid_2
        self._emissions_type = emissions_type
        # Outputs
        self._info = None
        self._prices = None
        self._generation = None

    def _check_data_inputs(self, local_cache, intlength, region, start_date, end_date, duid_1, duid_2, \
        emissions_type):
        """Validate correct argument types have been used.
        """
        # Required arguments
        inv.validate_variable_type(intlength, int, "intlength", "market_data")
        inv.validate_variable_type(local_cache, str, "local_cache", "market_data")
        inv.validate_variable_type(region, str, "region", "market_data")
        inv.validate_variable_type(start_date, str, "start_date", "market_data")
        inv.validate_variable_type(end_date, str, "end_date", "market_data")
        inv.validate_and_convert_date(start_date, "start_date")
        inv.validate_and_convert_date(end_date, "start_date")

        # Allowed arguments
        if duid_1 != None:
            inv.validate_variable_type(duid_1, str, "duid_1", "market_data")
        if duid_2 != None:
            inv.validate_variable_type(duid_2, str, "duid_2", "market_data")
        if emissions_type != None:
            inv.validate_variable_type(emissions_type, str, "emissions_type", "market_data")
            _check_statement(((emissions_type == 'Total') | (emissions_type == 'Marginal')), "Invalid `emissions_type` \
                input. Must be either of ['Total','Marginal']")
    
    def get_prices(self):
        """
        Retrieve the NEM historical regional reference node price for the 
        defined parameters of the nemosis data loader object.

        Returns
        -------
        pd.DataFrame()
            Returns pd.DataFrame with columns = ['Time','Prices'].
        """
        prices = dynamic_data_compiler(start_time=self._start, 
                                       end_time=self._end, 
                                       table_name='DISPATCHPRICE', 
                                       raw_data_location=self._cache,
                                       fformat='feather',
                                       filter_cols=['REGIONID'],
                                       filter_values=([[self._region]]),
                                       select_columns=['SETTLEMENTDATE','REGIONID','RRP'],
                                       keep_csv=False)
        prices.set_index('SETTLEMENTDATE',inplace=True)

        # Resample data if requested interval length is not 5 minutes
        if self._intlength != 5:
            self._prices = pd.DataFrame(prices.resample(str(self._intlength)+'min', 
                                  label='right', origin='end')['RRP'].mean())
        else:
            self._prices = pd.DataFrame(prices[['RRP']])

        self._prices.reset_index(inplace=True)
        self._prices.columns = ['Time', 'Prices']
        return self._prices


    def get_generation(self, cf_trace=True):
        df = self._download_all_unitdispatch()
        if not cf_trace:
            return df
        else:
            traces = self._calculate_capacity_factor_traces(df, region_filter=True)
            traces[traces < 0] = 0
            self._generation = traces.reset_index()
            if (self._duid_1 != None) & (self._duid_2 != None):
                return self._generation[['Time', self._duid_1, self._duid_2]]
            elif (self._duid_1 != None):
                return self._generation[['Time', self._duid_1]]
            elif (self._duid_2 != None):
                return self._generation[['Time', self._duid_2]]
            else:
                return None


    def _download_geninfo(self, filter_regions=['NSW1','QLD1','VIC1','SA1','TAS1'], filter_tech=None):
        """
        Downloads the Generator Registration and Exemption List data from AEMO.
        """

        gen_info = static_table(table_name="Generators and Scheduled Loads",
                          raw_data_location=self._cache,
                          select_columns=['Station Name','DUID','Region',
                          'Fuel Source - Descriptor','Reg Cap (MW)'],
                          filter_cols=['Region'],
                          filter_values=[filter_regions])
        print(gen_info)
        gen_info['Reg Cap (MW)'] = np.where(gen_info['Reg Cap (MW)'] == '-', np.nan, \
            gen_info['Reg Cap (MW)'])
        gen_info['Reg Cap (MW)'] = gen_info['Reg Cap (MW)'].astype(float)

        if filter_tech:
            self._info = gen_info[(gen_info['Reg Cap (MW)'].astype(float) >= 30) & \
            (gen_info['Fuel Source - Descriptor'].isin(filter_tech))]
        else:
            self._info = gen_info[(gen_info['Reg Cap (MW)'].astype(float) >= 30) & \
                (gen_info['Fuel Source - Descriptor'].isin(['Wind','Solar','solar']))]


    def _calculate_capacity_factor_traces(self, dataset, region_filter):
        """
        Divides the DUID generation dispatch values by the registered capacity
        for the DUID as found in AEMO's Registration and Exemption List
        """
        if region_filter:
            self._download_geninfo(filter_regions=[self._region])
        else:
            self._download_geninfo()
        info = self._info
        result = dataset.copy()

        for unit in result.columns:
            regcap = info[info['DUID']==unit]['Reg Cap (MW)']
            if not regcap.empty:
                if isinstance(regcap.values[0],float):
                    result[unit] = result[unit].div(float(regcap))
                else:
                    result.drop(unit, axis=1, inplace=True)
            else:
                result.drop(unit, axis=1, inplace=True)
        result.columns.name = None
        return result

    def _download_all_unitdispatch(self):

        # Download via nemed function
        dispatch_table = self._download_unit_dispatch(start_time=self._start,
                                                      end_time=self._end,
                                                      cache=self._cache,
                                                      return_all=False,
                                                      check=True)
        # Set DUIDs as columns
        # (aggfunc purpose is to resolve mismatching Scada and InitialMW)
        vrep = dispatch_table.pivot(index='Time', values='Dispatch', columns='DUID')

        # Resample data if requested interval length is not 5 minutes
        if self._intlength != 5:
            result = vrep.resample(str(self._intlength)+'min', 
                                label='right', origin='end').mean()
        else:
            result = vrep
        return result


    def _download_unit_dispatch(self, start_time, end_time, cache, return_all=True, check=True):
        """Downloads historical SCADA dispatch data via NEMOSIS. Inherited from NEMED

        Parameters
        ----------
        start_time : str
            Start Time Period in format 'yyyy/mm/dd HH:MM:SS'
        end_time : str
            End Time Period in format 'yyyy/mm/dd HH:MM:SS'
        cache : str
            Raw data location in local directory
        source_initialmw : bool
            Whether to download initialmw column from DISPATCHLOAD table, by default False
        source_scada : bool
            Whether to download scada column from DISPATCH_UNIT_SCADA table, by default True
        overwrite : str
            The data value to overwrite in the returned 'Dispatch' column if there is a discrepency in initialmw and scada.
            Must be one of ['initialmw','scada','average']. If one of source_initialmw or source_scada is False, `overwrite`
            has null effect. By default 'scada'.
        return_all : bool
            Whether to return all columns or only ['Time','DUID','Dispatch'], by default False.
        check : bool
            Whether to check for, and remove duplicates after function is complete, by default True.

        Returns
        -------
        pd.DataFrame
            Returns generation data as per NEMOSIS

        """
        DISPATCH_INT_MIN = 5
        # Check inputs
        cache = hp._check_cache(cache)
        assert(isinstance(start_time, str)), "`start_time` must be a string in format yyyy/mm/dd HH:MM:SS"
        assert(isinstance(end_time, str)), "`end_time` must be a string in format yyyy/mm/dd HH:MM:SS"

        # Adjust timestamps for Scada interval-beginning
        shift_stime = dt.strptime(start_time, "%Y/%m/%d %H:%M:%S")
        shift_stime = shift_stime + timedelta(minutes=DISPATCH_INT_MIN)
        shift_etime = dt.strptime(end_time, "%Y/%m/%d %H:%M:%S")
        shift_etime = shift_etime + timedelta(minutes=DISPATCH_INT_MIN)
        get_start_time = dt.strftime(shift_stime, "%Y/%m/%d %H:%M:%S")
        get_end_time = dt.strftime(shift_etime, "%Y/%m/%d %H:%M:%S")

        # Download Dispatch Unit Scada table via NEMOSIS (this includes Non-Scheduled generators)
        disp_scada = dynamic_data_compiler(
            start_time=get_start_time,
            end_time=get_end_time,
            table_name="DISPATCH_UNIT_SCADA",
            raw_data_location=cache,
            select_columns=["SETTLEMENTDATE", "DUID", "SCADAVALUE"],
            fformat="feather",
        )
        disp_scada["Time"] = disp_scada["SETTLEMENTDATE"] - timedelta(minutes=DISPATCH_INT_MIN)
        master = disp_scada[['Time', 'DUID', 'SCADAVALUE']]
        master['Dispatch'] = master['SCADAVALUE']

        # Final check for intervention periods and duplicates entries
        if check:
            final = self._clean_duplicates(master)
        else:
            final = master

        # Return dataset
        if return_all:
            return final
        else:
            return final[['Time', 'DUID', 'Dispatch']]


    def _clean_duplicates(self, table, value_col="Dispatch"):
        """Filtering inherited from NEMED process
        """
        if any(table.duplicated(subset=['Time', 'DUID'])):
            print("Duplicate Timestamped DUIDs found. Updating dataset for duplicates.")
            # Take average values where duplicates differ
            table_clean = table.pivot_table(index=["Time", "DUID"], values=value_col, aggfunc=np.mean)
            table_clean = table_clean.reset_index()

            # Remove duplicates where Time and DUID match
            table_clean = table_clean.drop_duplicates(subset=["Time", "DUID"])
            return table_clean
        else:
            return table


    def estimate_ppa_costs(self, vre_capacity, strike_price, floor_price):
        ## NEED TO ADD INPUT and PREPROCESSES COMPLETE CHECKS

        # always using duid_1

        cfd_volume = self._generation[self._duid_1] * vre_capacity

        if floor_price != None:
            cfd_value = [(strike_price - max(floor_price, self._prices['Prices'][i])) * (self._intlength/60) \
                for i in range(0, len(self._prices['Prices']))]
        else:
             cfd_value = [(strike_price - self._prices['Prices'][i]) * (self._intlength/60) \
                for i in range(0, len(self._prices['Prices']))]           

        cfd = pd.DataFrame({'Time': [self._prices['Time'][i] for i in range(0, min(len(cfd_value), len(cfd_volume)))],
                            'Cost': [cfd_volume[i] *cfd_value[i] for i in range(0, min(len(cfd_value), len(cfd_volume)))],
                            'PriceDiff': [cfd_value[i] for i in range(0, min(len(cfd_value), len(cfd_volume)))],
                            'Volume': [cfd_volume[i] for i in range(0, min(len(cfd_value), len(cfd_volume)))]})
        return cfd


    def get_emissions_intensity(self):
        """_summary_

        USING MARGINAL EMISSIONS: Error may occur with SSL and requests module. See fix: https://www.youtube.com/watch?v=mN8SLBsvSCU

        Returns
        -------
        _type_
            _description_
        """
        emissions_type = self._emissions_type
        if emissions_type == "Total":

            nemed_df = get_total_emissions(start_time=self._start[:-3],
                                           end_time=self._end[:-3],
                                           cache=self._cache,
                                           filter_regions=[self._region],
                                           by=None,
                                           generation_sent_out=True,
                                           assume_energy_ramp=True,
                                           return_pivot=False)
            nemed_df.rename(columns={'TimeEnding': 'Time'},inplace=True)

            # Adjust time resolution to set interval length
            if self._intlength != 5:
                emissions = nemed_df.set_index('Time').resample(str(self._intlength)+'min', label='right', origin='end').sum()
                emissions['Intensity_Index'] = emissions['Total_Emissions'] / emissions['Energy']
                emissions.reset_index(inplace=True)
            else:
                emissions = nemed_df
            return emissions[['Time', 'Intensity_Index']]

        elif emissions_type == "Marginal":
            
            nemed_df = get_marginal_emissions(start_time=self._start[:-3],
                                              end_time=self._end[:-3],
                                              cache=self._cache+"_2")

            nemed_df = nemed_df[nemed_df['Region'] == self._region]

            # Adjust time resolution to set interval length
            if self._intlength != 5:
                emissions = nemed_df.set_index('Time').resample(str(self._intlength)+'min', label='right', origin='end').mean()
                emissions.reset_index(inplace=True)
            else:
                emissions = nemed_df
            return emissions[['Time', 'Intensity_Index']]           


def _check_statement(statement, err_message, log_type=logger.exception):
    try:
        assert statement, err_message
    except AssertionError as e:
        log_type(e)
        raise e


def convert_dt_order(string):
    """
    Convert defined input datetime string into format required by nemosis

    Parameters
    ----------
    string : str
        Input datetime string as ('%d/%m/%Y %H:%M').

    Returns
    -------
    ret_str : str
        Output datetime string as ('%Y/%m/%d %H:%M:%S').
    """
    ret_str = dt.strptime(string,'%d/%m/%Y %H:%M')
    ret_str = dt.strftime(ret_str, '%Y/%m/%d %H:%M:%S')
    return ret_str

def _check_cache(cache):
    """Check the cache folder directory exists
    Parameters
    ----------
    cache : str
        Folder path location to be used as local cache.
    Returns
    -------
    str
        Existing (or updated if error) cache directory path.
    """
    try:
        assert(isinstance(cache, str))
    except AssertionError as e:
        logger.exception("Cache input must be a string")
        raise e

    if not os.path.isdir(cache):
        cache = os.path.join(os.getcwd(), "cache_nemglo")
        if os.path.isdir(cache):
            logger.warning(f"Input cache path is invalid or not found. Using existing cache directory at {cache}")
        else:
            os.mkdir(cache)
            logger.warning(f"Input cache path is invalid or not found. Creating new cache directory at {cache}")
    return cache