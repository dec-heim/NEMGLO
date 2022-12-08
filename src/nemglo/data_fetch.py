import pandas as pd
import numpy as np
import os
from nemosis import dynamic_data_compiler, static_table
from nemed import *
from nemed.downloader import download_unit_dispatch
from datetime import datetime as dt, timedelta

class nemosis_data:
	""" Using as a wrapper for nemosis to grab data that can then be input to 
	the optimiser. 
	Alternatively, optimiser may be used with custom data in place of 
	nemosis_data, however this is not yet integrated/tested in this version.
	"""
	
	def __init__(self, intlength=5, local_cache=r'.\cache'):
		if not os.path.isdir(local_cache):
			print("Cache location does not exist. Creating cache location at \
		 current directory")
			local_cache = '.\cache'
			os.mkdir(local_cache)

		self._cache = local_cache
		self._start = None
		self._end = None
		self._region = None
		self._intlength = intlength
		self._duid_1 = None
		self._duid_2 = None
		self._prices = None
		self._vre_hist = None
		self._vre_cf = None
		self._alldata = None
		self._info = None
		
	def set_unit(self, duid_1, duid_2):
		"""
		Sets the unit parameters for the nemosis data loader object.

		Parameters
		----------
		duid_1 : str
			Dispatchable Unit Identifer to collect generation data for.
		duid_2 : str
			Dispatchable Unit Identifer to collect generation data for.
		"""
		self._duid_1 = duid_1
		if duid_2:
			self._duid_2 = duid_2
		
	def set_region(self, region):
		"""
		Sets the NEM market region parameter for the nemosis data loader object.

		Parameters
		----------
		region : str
			Defined region to collect price data for.
		"""
		self._region = region
		
	def set_dates(self, start, end):
		"""
		Sets the date range for the nemosis data loader object.

		Parameters
		----------
		start : str
			Start timestamp entered in the format ('%d/%m/%Y %H:%M').
		end : str
			End timestamp entered in the format ('%d/%m/%Y %H:%M').
		"""
		self._start = convert_dt_order(start)
		self._end = convert_dt_order(end)
		
	def set_intlen(self,newintlen):
		"""
		Sets the dispatch interval length for sampling historical data. Raw 
		data is formatted in 5 minute dispatch intervals. Setting this 
		parameter to 30min or otherwise will aggregate the data by averaging.

		Parameters
		----------
		newintlen : int, np.float64
			Defined interval length to resample historical dispatch intervals.
		"""
		self._intlength = newintlen
	
	def get_prices(self, valueonly=False):
		"""
		Retrieve the NEM historical regional reference node price for the 
		defined parameters of the nemosis data loader object.

		Parameters
		----------
		valueonly : bool, optional
			Set true to return a the raw list of values. The default is False.

		Returns
		-------
		list or pd.DataFrame()
			Returns the RRP as list if valueonly, else as pd.DataFrame with 
			dispatch interval timestamps.
		"""
		self._download_prices()
		if valueonly:
			return list(self._prices['RRP'].values)
		else:
			self._prices.reset_index(inplace=True)
			self._prices.columns = ['Time', 'Prices']
			return self._prices

	def get_timestamp(self):
		"""
		Retrieve the timestamps for the defined period. 

		Returns
		-------
		list
			Strictly returns the timestamps corresponding to price data.
		"""
		self._download_prices()
		return list(self._prices.index)
	
	def get_vre_dispatch(self, measure='INITIALMW'):
		"""
		Retrieve the vre MW dispatch for the defined DUIDs.
		
		Returns
		-------
		pd.DataFrame()
			Contains SettlementDate as index, and columns corresponding to 
			defined duids.
		"""
		if measure not in ["INITIALMW", "TOTALCLEARED"]:
			raise ValueError("measure arguemnt must be either 'INITIALMW' or 'TOTALCLEARED'")

		df = self._download_vre_vianemed(measure=measure, filt_off=False)
		#self._download_vre(measure=measure)
		return df
	
	def get_vre_traces(self, measure='INITIALMW'):
		"""
		Retrieve the vre traces for the defined DUIDs.

		Returns
		-------
		pd.DataFrame()
			Contains SettlementDate as index, and columns corresponding to 
			defined duids.
		"""
		if measure not in ["INITIALMW", "TOTALCLEARED"]:
			raise ValueError("measure arguemnt must be either 'INITIALMW' or 'TOTALCLEARED'")

		df = self._download_vre_vianemed(measure=measure, filt_off=False)
		#self._download_vre(measure=measure)
		traces = self._calculate_capacity_factor_traces(df)
		traces[traces < 0] = 0
		self._vre_cf = traces.reset_index()
		return traces.reset_index()
	
	def _calculate_capacity_factor_traces(self, dataset):
		"""
		Divides the DUID generation dispatch values by the registered capacity
		for the DUID as found in AEMO's Registration and Exemption List
		"""
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
		
	def _download_prices(self):
		"""
		Downloads the price corresponding to defined region. Resamples the 
		dataseries if the dispatch interval length is not 5 minutes.
		"""
		prices = dynamic_data_compiler(start_time=self._start, 
								 end_time=self._end, 
								 table_name='DISPATCHPRICE', 
								 raw_data_location=self._cache,
								 fformat='feather',
								 filter_cols=['REGIONID'],
								 filter_values=([[self._region]]),
								 select_columns=['SETTLEMENTDATE','REGIONID',
												 'RRP'],
								 keep_csv=False)
		prices.set_index('SETTLEMENTDATE',inplace=True)
		# Resample data if requested interval length is not 5 minutes
		if self._intlength != 5:
			self._prices = pd.DataFrame(prices.resample(str(self._intlength)+'min', 
								  label='right', origin='end')[['RRP']].mean())
		else:
			self._prices = pd.DataFrame(prices[['RRP']])

	def _download_geninfo(self, filter_regions=['NSW1','QLD1','VIC1','SA1','TAS1'], filter_tech=None):
		"""
		Downloads the Generator Registration and Exemption List data from AEMO.
		"""

		gen_info = static_table(table_name="Generators and Scheduled Loads",
						  raw_data_location=self._cache,
						  select_columns=['Station Name','DUID','Region',
						  'Fuel Source - Descriptor','Reg Cap (MW)'],
						  filter_cols=['Region'],
						  filter_values=filter_regions)

		gen_info['Reg Cap (MW)'] = np.where(gen_info['Reg Cap (MW)'] == '-', np.nan, \
			gen_info['Reg Cap (MW)'])
		gen_info['Reg Cap (MW)'] = gen_info['Reg Cap (MW)'].astype(float)

		if filter_tech:
			self._info = gen_info[(gen_info['Reg Cap (MW)'].astype(float) >= 30) & \
			(gen_info['Fuel Source - Descriptor'].isin(filter_tech))]
		else:
			self._info = gen_info[(gen_info['Reg Cap (MW)'].astype(float) >= 30) & \
				(gen_info['Fuel Source - Descriptor'].isin(['Wind','Solar','solar']))]


	def _download_dudetail(self):
		capacity = dynamic_data_compiler(start_time=self._start, 
									end_time=self._end, 
									table_name='DUDETAIL', 
									raw_data_location=self._cache,
									fformat='feather')
		return capacity.loc[:,['EFFECTIVEDATE','DUID','MAXCAPACITY','REGISTEREDCAPACITY']]


	def _download_vre_vianemed(self, measure, filt_off=False):
		# Filter unit values
		if self._duid_2 != None:
			filvals = [self._duid_1, self._duid_2]
		else:
			filvals = [self._duid_1]

		if filt_off:
			filvals = None

		# Download via nemed function
		dispatch_table = download_unit_dispatch(start_time=self._start,
												end_time=self._end,
												cache=self._cache,
												source_initialmw=False,
												source_scada=True,
												overwrite='scada',
												return_all=False,
												check=True)
		# Set DUIDs as columns
		# (aggfunc purpose is to resolve mismatching Scada and InitialMW)
		vrep = dispatch_table.pivot_table(index='Time',values='Dispatch', columns='DUID',\
			aggfunc='max')

		# Resample data if requested interval length is not 5 minutes
		if self._intlength != 5:
			result = vrep.resample(str(self._intlength)+'min', 
								label='right', origin='end').mean()
		else:
			result = vrep
		return result

class nemed_data:
	""" Using as a wrapper for nemed to retrieve emissions data that can then be
	input to the optimiser. 
	Alternatively, optimiser may be used with custom data in place of 
	nemed_data, however this is not yet integrated/tested in this version.
	"""
	def __init__(self, intlength=5, local_cache=r'.\cache'):
		# Check cache folder exists, else create one.
		if not os.path.isdir(local_cache):
			print("Cache location does not exist. Creating cache location at \
			current directory")
			local_cache = '.\cache'
			os.mkdir(local_cache)

		self._cache = local_cache
		self._start = None
		self._end = None
		self._region = None
		self._intlength = intlength
		self._emissions = None
		self._intensity = None

	# TO BE CONTINUED - FUTURE DEV

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