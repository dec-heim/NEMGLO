# NEMGLO
NEMGLO, the Green-energy Load Optimisation tool for the Australian National Electricity Market (NEM) is a python tool allows users to counterfactually assess flexible load operating strategies in the NEM using historical market data. Specifically, the tool enables users to test various scenarios with differing PPA contracting structures and green-certificate schemes, a primary case study being the ability to 'validate' the integrity of "green"-hydrogen produced via grid-connected electrolysis with bundled PPA contracting. Although the tool was developed contextually for electrolyser loads, the functionalities can inherently be abstracted to commercial/industrial flexible loads seeking to consume green-energy. 

**Customisable Tool Features:**
- Load Operating Characteristics (Min stable load, ramp rates, etc.)
- Multiple Power Purchase Agreements (contract volume, strike, floor, etc.)
- Renewable Energy Certificate procurement + surrender (with/without temporal matching)
- Shadow Pricing of Grid Emissions Intensity (average & marginal emissions)
- Constrain to Green Energy Certification Standards for H2 (max tCO2 content per tH2) 

**Project Links**
- Project Homepage: https://www.nemglo.org/
- Python Github Repository: https://github.com/dec-heim/NEMGLO
- Python Documentation: https://nemglo.readthedocs.io/en/latest/about.html
- NEMGLO App Github Repository: https://github.com/dec-heim/NEMGLO-app


## Installation
```bash
pip install nemglo
```
For guidance on installation and usage of [NEMGLO-app](https://github.com/dec-heim/NEMGLO-app) (a graphical user interface), please refer to its project page.

## Usage
For guidance on `NEMGLO` usage, see the [Examples]() section of the documentation.

## Contributing
Interested in contributing? Check out the [contributing guidelines](CONTRIBUTING.md).
Please note that this project is released with a [Code of Conduct](). By contributing to this project, you agree to abide by its terms.

## License
`NEMGLO` was created by Declan Heim as a Master's Project at UNSW. It is licensed under the terms of the [BSD 3-Clause license](LICENSE).

## Credits
The `NEMGLO` project is affilitated with the [UNSW Collaboration on Energy and Environmental Markets](https://www.ceem.unsw.edu.au/). The [`NEMGLO-app`](https://github.com/dec-heim/NEMGLO-app) was financially supported by the [UNSW Digital Grid Futures Institute](https://www.dgfi.unsw.edu.au/) to Dec-2022.

`NEMGLO` incorporates functionality from a suite of UNSW-CEEM tools, namely, [`NEMOSIS`](https://github.com/UNSW-CEEM/NEMOSIS) to extract historical market data and [`NEMED`](https://github.com/UNSW-CEEM/NEMED) to compute emissions data from AEMO's MMS databases respectively. The structure of the optimiser codebase is further adopted from sister tool [`nempy`](https://github.com/UNSW-CEEM/nempy) under [nempy licence](https://github.com/UNSW-CEEM/nempy/blob/master/LICENSE).

### Acknowlgements

Many thanks to:
- Jay Anand, co-developer of `NEMGLO-app`. 
- Nick Gorman, Abhijith Prakash, and Shayan Naderi for pointers on `NEMGLO` code development (including adopted NEM tools).
- Iain MacGill, Anna Bruce, Rahman Daiyan, and Jack Sheppard for advice at various stages to this project.

## Contact
Questions and feedback are very much welcomed. Please reach out by email to [declanheim@outlook.com](mailto:declanheim@outlook.com)
