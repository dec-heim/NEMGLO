# Execution variables
import os
class CACHE():
    def __init__(self, FILEPATH):
        self.FILEPATH = FILEPATH

    def update_path(self, FILEPATH):
        self.FILEPATH = FILEPATH

DATA_CACHE = CACHE(os.path.join(os.getcwd(), "CACHE"))
LOG_FILEPATH = os.path.join(DATA_CACHE.FILEPATH, "nemglo_sim.log")

# Naming conventions
variable_names = ['vre_cap']
constraint_names = []

cost_variables_shadow = ['impact_emissions', 'relief_prd_tgt', 'msl_penalise', 'msl_relieve']
cost_variables_realized = ['h2e_load', 'h2_production', 'ppa_vre']

# Electrolyser Default SEC variable profiles and characteristics
# Beerbühl et.al, 2015, https://doi.org/10.1016/j.ejor.2014.08.039
SEC_PROFILE_AE = {'h2e_load_pct': [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    'nominal_sec_pct': [0.8068, 0.8068, 0.8181, 0.8523, 0.9091, 1.0]}
# Kopp et.al, 2017, https://doi.org/10.1016/j.ijhydene.2016.12.145
SEC_PROFILE_PEM = {'h2e_load_pct': [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    'nominal_sec_pct': [0.8383, 0.8383, 0.8598, 0.8982, 0.9315, 1.0101]}
SEC_NOMINAL_AE = 50
SEC_NOMINAL_PEM = 66
SEC_CONVERSION_AE = 1.0
SEC_CONVERSION_PEM = 1.0

PALETTE = ['#b2d4ee','#849db1','#4f6980',
           '#B4E3BC','#89AE8F','#638b66',
           '#ffb04f','#de9945','#af7635',
           '#ff7371','#d6635f','#b65551',
           '#AD134C','#cc688d','#ff82b0']