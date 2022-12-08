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