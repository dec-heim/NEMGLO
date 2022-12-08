# Quick Start
```{warning}
This material is outdated from a previous version of NEMGLO. To be updated soon!
- Add simple electrolyser exmaple
- Change existing one to advanced

```

## Simple Example
A simple example demonstrating the most basic functionally and structure of
**`nemglo`** is given below.

```python 
# This example demonstrates a very simple application of the nemglo package;
# extracting historical AEMO data of the NEM (price and renewable generation),
# defining load characteristics and PPA structures, then running the optimiser
# to find the operational load behaviour.

# This example uses plotly == 5.6.0 to plot results. Install with... 
# pip install plotly==5.6.0

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "browser"

from nemglo.data_fetch import nemosis_data
import nemglo.planner as lo

# Create a nemosis_data object to store the historical data for which we are
# running this modelling/analysis on.
inputdata = nemosis_data(intlength=30,local_cache=r'.\cache')

# Define a start and end interval timestamp, and NEM market region.
start = "03/01/2020 15:00"
end = "03/01/2020 18:00"
region = 'VIC1'

# Update object with each of the information described above, using <set_dates>
# and <set_region> functions of the nemosis_data module.
inputdata.set_dates(start, end)
inputdata.set_region(region)

# Retrieve the price and timestamp data from the nemosis_data module, using
# <get_prices> and <get_timestamp>.
price = inputdata.get_prices(valueonly=True)
time = inputdata.get_timestamp()

# Create a <P2G_plan> planner object which will be used to optimise the load
# operation. The market data is then loaded to this object using <load_market>.
P2G = lo.P2G_plan()
P2G.load_market(time, price)

# The H2 electrolyser load parameters can be defined as such...
P2G.add_h2e_load(capacity=100.0,min_power=0.0,max_power=100.0)

# The H2 production parameters defined a price incentive for the electrolyser to
# produce hydrogen, as such...
P2G.add_h2_production(h2_price_mwh=80)

# Finally running the optimisation by calling <optimise>.
P2G.optimise()

# Results from the optimisation can be extracted by calling <get_load> in this
# instance to retrive the MW load profile.
result_load = P2G.get_load()

# The above result_load produces the following...
# 	time		interval	value
# 0	2020-01-03 15:30:00	0	100.0
# 1	2020-01-03 16:00:00	1	100.0
# 2	2020-01-03 16:30:00	2	0.0
# 3	2020-01-03 17:00:00	3	0.0
# 4	2020-01-03 17:30:00	4	100.0
# 5	2020-01-03 18:00:00	5	0.0

# Recalling the earlier market price list (rounded to 2dp)...
# 0	65.75
# 1	72.19
# 2	86.63
# 3	83.32
# 4	78.49
# 5	92.43

# Hence the load does not operate when the energy spot price is above $80/MWh
# since it is not profitable.
```
![Simple Load Example Result](../../examples/figures/simple_load_operation.png "Example Figure")
