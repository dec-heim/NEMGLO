# Data Connectors

## Historical NEM Data (via NEMOSIS)
[**NEMOSIS**](https://github.com/UNSW-CEEM/NEMOSIS) is the primary data connector used in NEMGLO which facilitates the process of downloading and process data from AEMO's Data Archive known as the [MMS database](https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/market-data-nemweb).

---

```{eval-rst}
.. autoclass:: nemglo.data_fetch.nemosis_data
   :members:
```

## Historical Emissions Data (via NEMED)
[**NEMED**](https://github.com/dec-heim/NEMED) is a data connector used in NEMGLO to extract historical NEM Emissions Data on a dispatch interval basis. These datasets are derived based on AEMO's Carbon Dioxide Equivalent Intensity Index ([CDEII Procedure](https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/market-operations/settlements-and-payments/settlements/))carbon-dioxide-equivalent-intensity-index), noting the official published datasets from AEMO are limited to daily resolution not dispatch intervals.

---

```{eval-rst}
.. autoclass:: nemglo.data_fetch.nemed_data
   :members:
```