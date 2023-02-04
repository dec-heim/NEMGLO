# class DataFrameSchema and class SeriesSchema creditted to N.Gorman (nempy)
import pandas as pd
import numpy as np
from datetime import datetime as dt
import logging

logger = logging.getLogger(__name__)

class DataFrameSchema:
    def __init__(self, name, primary_keys=None):
        self.name = name
        self.primary_keys = primary_keys
        self.columns = {}
        self.required_columns = []

    def add_column(self, column, optional=False):
        self.columns[column.name] = column
        if not optional:
            self.required_columns.append(column.name)

    def validate(self, df):
        for col in df:
            if col not in self.columns:
                raise UnexpectedColumn("Column '{}' is not allowed in DataFrame '{}'. Permitted columns are {}"\
                    .format(col, self.name, list(self.columns.keys())))

        for col in self.required_columns:
            if col not in df.columns:
                raise MissingColumnError("Column '{}' not in DataFrame '{}'. Required columns are {}"\
                    .format(col, self.name, self.required_columns))

        for col in self.columns:
            if col in df.columns:
                self.columns[col].validate(df[col])

        if self.primary_keys is not None:
            self._check_for_repeated_rows(df)

    def _check_for_repeated_rows(self, df):
        cols_in_df = [col for col in self.primary_keys if col in df.columns]
        if len(df.index) != len(df.drop_duplicates(cols_in_df)):
            raise RepeatedRowError('{} should only have one row for each {}.'.format(self.name, ' '.join(cols_in_df)))


class SeriesSchema:
    def __init__(self, name, data_type, allowed_values=None, must_be_real_number=False, not_negative=False,
                 no_duplicates=False, ascending_order=False, minimum=None, maximum=None):
        self.name = name
        self.data_type = data_type
        self.allowed_values = allowed_values
        self.must_be_real_number = must_be_real_number
        self.not_negative = not_negative
        self.no_duplicates = no_duplicates
        self.ascending_order = ascending_order
        self.min = minimum
        self.max = maximum

    def validate(self, series):
        self._check_data_type(series)
        self._check_allowed_values(series)
        self._check_is_real_number(series)
        self._check_is_not_negtaive(series)
        self._check_no_duplicates(series)
        self._check_is_ordered(series)
        self._check_min_max(series)

    def _check_data_type(self, series):
        if self.data_type == str:
            if not all(series.apply(lambda x: type(x) == str)):
                raise ColumnDataTypeError("All elements of column '{}' should have type str".format(self.name))
        elif self.data_type == callable:
            if not all(series.apply(lambda x: callable(x))):
                raise ColumnDataTypeError("All elements of column '{}' should have type callable".format(self.name))
        elif self.data_type != series.dtype:
            raise ColumnDataTypeError("Column '{}' should have type '{}'".format(self.name, self.data_type))

    def _check_allowed_values(self, series):
        if self.allowed_values is not None:
            if not series.isin(self.allowed_values).all():
                raise ColumnValues("The column '{}' can only contain the values '{}'.".format(self.name, \
                    self.allowed_values))

    def _check_is_real_number(self, series):
        if self.must_be_real_number:
            if np.inf in series.values:
                raise ColumnValues("Value inf not allowed in column '{}'.".format(self.name))
            if np.NINF in series.values:
                raise ColumnValues("Value -inf not allowed in column '{}'.".format(self.name))
            if series.isnull().any():
                raise ColumnValues("Null values not allowed in column '{}'.".format(self.name))

    def _check_is_not_negtaive(self, series):
        if self.not_negative:
            if series.min() < 0.0:
                raise ColumnValues("Negative values not allowed in column '{}'.".format(self.name))

    def _check_no_duplicates(self, series):
        if self.no_duplicates:
            if series.duplicated().any():
                raise ColumnValues("Duplicate values not allowed in column '{}'.".format(self.name))

    def _check_is_ordered(self, series):
        if self.ascending_order:
            if not all(series.reset_index(drop=True) == series.sort_values(ascending=True).reset_index(drop=True)):
                raise ColumnValues("Values must be sequenced in ascending order in column '{}'.".format(self.name))

    def _check_min_max(self, series):
        if self.min:
            if series.min() < self.min:
                raise ColumnValues("Minimum allowed value in column '{}' is {}.".format(self.name, self.min))
        if self.max:
            if series.max() > self.max:
                raise ColumnValues("Maximum allowed value in column '{}' is {}.".format(self.name, self.max))


def validate_unique_id(c_definition, system_plan, identifier):
    for other_gen in [c for c in system_plan._components if c.__class__ is c_definition]:
        if other_gen._id == identifier:
            raise ComponentError("Cannot create a {} object of the same name as an existing instance in Plan '{}'."\
                .format(c_definition, system_plan._id))


def validate_existing(system_plan, name, as_var=False, as_objective_cost=False, as_constr_lhs=False, \
    as_constr_rhs=False, as_constr_rhs_dynamic=False, as_constr_bigM=False, as_sos_2=False):
    if as_var:
        if name in system_plan._var:
            raise ComponentError("Cannot create a new variable {} as it already exists in Plan '{}'."\
                .format(name, system_plan._id))
    if as_objective_cost:
        if name in system_plan._objective_cost:
            raise ComponentError("Cannot create a new objective cost {} as it already exists in Plan '{}'."\
                .format(name, system_plan._id))
    if as_constr_lhs:
        if name in system_plan._constr_lhs:
            raise ComponentError("Cannot create a new constraint LHS {} as it already exists in Plan '{}'."\
                .format(name, system_plan._id))
    if as_constr_rhs:
        if name in system_plan._constr_rhs:
            raise ComponentError("Cannot create a new constraint RHS {} as it already exists in Plan '{}'."\
                .format(name, system_plan._id))
    if as_constr_rhs_dynamic:
        if name in system_plan._constr_rhs_dynamic:
            raise ComponentError("Cannot create a new constraint RHS dynamic {} as it already exists in Plan '{}'."\
                .format(name, system_plan._id))
    if as_constr_bigM:
        if name in system_plan._constr_bigM:
            raise ComponentError("Cannot create a new constraint bigM {} as it already exists in Plan '{}'."\
                .format(name, system_plan._id))
    if as_sos_2:
        if name in system_plan._sos_2:
            raise ComponentError("Cannot create a new SOS-2 {} as it already exists in Plan '{}'."\
                .format(name, system_plan._id))
  

def validate_positive_float(input, in_name, c_name):
    assert isinstance(input, float), "{} Argument: '{}' must be a float.".format(c_name, in_name)
    assert input >= 0, "{} Argument: '{}' must be a positive number.".format(c_name, in_name)

def validate_variable_type(var, vartype, inputname, functionname):
    try:
        assert isinstance(var, vartype), f"`{inputname}` of `{functionname}` cannot be {var}. Accepted types: {vartype}"
    except AssertionError as e:
        logger.exception(e)
        raise e

def validate_and_convert_date(date_string, inputname, date_format="%Y/%m/%d %H:%M"):
    try:
        date_obj = dt.strptime(date_string, date_format)
        return date_obj
    except ValueError:
        e = (f"Invalid date string was passed {date_string} as {inputname}. Required date format: {date_format}")
        logger.exception(e)


def validate_get_market_data(conf):
    """Validate required and allowed fields for '/api/get-market-data' call.
    """
    _check_fields(d=conf,
                  name="conf",
                  required={"market_data"},
                  allowed={"emissions_data"})

    _check_fields(d=conf['market_data'],
                  name="conf['market_data']",
                  required={"start_date", "end_date", "region", "dispatch_interval_length"},
                  allowed={"start_time","end_time"}
                  )

    if 'emissions_data' in conf:
        _check_fields(d=conf['emissions_data'],
                      name="conf['emissions_data']",
                      required={"emissions_type"})


def validate_get_generator_data(conf):
    """Validate required and allowed fields for '/api/get-generator-data' call.    
    """
    _check_fields(d=conf,
                  name="conf",
                  required={"market_data","ppa"},
                  allowed={""})

    _check_fields(d=conf['market_data'],
                  name="conf['market_data']",
                  required={"start_date", "end_date", "region", "dispatch_interval_length"},
                  allowed={"start_time","end_time"}
                  )

    _check_fields(d=conf['ppa'],
                  name="conf['ppa']",
                  required={"duid", "capacity", "strike_price"},
                  allowed={"floor_price"})


def validate_get_data(conf):
    """Validate required and allowed fields for '/api/get-data' call.    
    """
    _check_fields(d=conf,
                  name="conf",
                  required={"market_data", "electrolyser_load", "ppa_1", "ppa_2", "rec"},
                  allowed={""})
    logger.warning("No subfield checks on '/api/get-data'")
    # _check_fields(d=conf['market_data'],
    #               name="conf['market_data']",
    #               required={"start_date", "end_date", "region", "dispatch_interval_length"},
    #               allowed={""}
    #               )

    # _check_fields(d=conf['ppa'],
    #               name="conf['ppa']",
    #               required={"duid", "capacity", "strike_price"},
    #               allowed={"floor_price"})


def _check_fields(d, name="", required={""}, allowed={""}, log_type=logger.exception):
    err_message = f"API param: {name} has invalid fields. Required fields are: {required}. Allowed fields are: {allowed}"
    try:
        assert required <= d.keys() <= (required | allowed), err_message
    except AssertionError as e:
        log_type(e)
        raise e

class RepeatedRowError(Exception):
    """Raise for repeated rows."""


class ColumnDataTypeError(Exception):
    """Raise for columns with incorrect data types."""


class MissingColumnError(Exception):
    """Raise for required column missing."""


class UnexpectedColumn(Exception):
    """Raise for unexpected column."""


class ColumnValues(Exception):
    """Raise for unallowed column values."""


class ComponentError(Exception):
    """Raise for component class errors"""