# Terminology

## Variable/Constraint Conventions

### Electrolyser
| variable_name     | Description   |
|-------------------|---------------|
| mw_load           | The load power consumption per interval (MW)              |
| mw_load_sum       | The sum of mw_load for all simulated intervals (MW)       |
| h2_produced       | Hydrogen production mass (kg) per interval                |
| h2_produced_sum   | The sum of h2_produced for all simulated intervals (kg)   |
| sec_w             | Weightings for the SOC-2 implementation of variable SEC   |
| msl_violate       | Violation amount for MSL, effectively distance from MSL (MW)|
| msl_penalise      | Binary indicator signifying MSL violation and enforcing penalty cost |
| msl_relieve       | Binary indicator signfinying load is at 0MW, relieving MSL penalty|
| mw_ramp           | The ramp rate (either up or down) in (MW) between intervals |
| production_target_{bound}_{period} | The sum of h2_produced within a defined period with corresponding constraint to enforce a defined production target |
| h2_stored         | 'Volume' of hydrogen in storage at any interval in time (kg) |
| h2_stored_initial | The defined starting 'volume' of stored hydrogen (kg)     |
| h2_stored_final   | The defined final 'volume' of stored hydrogen (kg)        |
| h2_stored_extflow | A defined external in-/out-flow (kg) of hydrogen to/from the hydrogen storage per interval |


|  constraint_name  | Description   |
|-------------------|---------------|
| mw_ramp_down      | Determines the ramp rate (MW) between intervals when ramping down occurs |
| mw_ramp_up        | Determines the ramp rate (MW) between intervals when ramping up occurs |
| mw_force          | Forces the electrolyser mw_load to a defined set value when applied |


Note: Most variable names also have a corresponding `constraint_name` which are not listed here.

### Generator
| variable_name     | Description   |
|-------------------|---------------|
| vre_cap           | The defined capacity of the renewable generator (MW)      |
| vre_avail         | The availability (MW trace) of the generator per interval |
| ppa_rec_sum       | Sum of vre_avail for all simulated intervals, signifying the amount of RECs received by the load (in MW units)                                              |




