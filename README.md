# NEMGLO
NEMGLO is a Green-energy Load Optimisation tool for the Australian National Electricity Market (NEM).
This python tool allows users to counterfactually assess flexible load operating strategies in the NEM using historical market data. Specifically, the tool enables users to test various scenarios with differing PPA contracting structures and green-certificate schemes, a primary case study being the ability to 'validate' the integrity of "green"-hydrogen produced via grid-connected electrolysis with bundled PPA contracting. Although the tool was developed contextually for electrolyser loads, the functionalities can inherently be abstracted to commercial/industrial flexible loads seeking to consume green-energy. 

**Customisable Tool Features:**
- Load Operating Characteristics (Min stable load, ramp rates, etc.)
- Multiple Power Purchase Agreements (contract volume, strike, floor, etc.)

**Features Coming Soon:**
- Renewable Energy Certificate procurement + surrender (with/without temporal matching)
- Shadow Pricing of Grid Emissions Intensity (average & marginal emissions)
- Constrain to Green Energy Certification Standards for H2 (max tCO2 content per tH2) 

[*Read more about the project here*](https://nemglo.readthedocs.io/en/latest/about.html)

## Installation
```bash
pip install nemglo
```

## Future Development

Thanks for checking out this **beta** version! Future work will expand the functions and capabilities of modelling with more customability/settings for electrolyser operation, PPAs and certificate-trading/emissions considerations. A graphical user interface is also being developed. Check back for future information and releases. 

## Usage
For guidance on `NEMGLO` usage, see the [Examples]() section of the documentation.

## Contributing
Interested in contributing? Check out the [contributing guidelines](CONTRIBUTING.md).
Please note that this project is released with a [Code of Conduct](). By contributing to this project, you agree to abide by its terms.

## License
`NEMGLO` was created by Declan Heim as a Master's Project at UNSW. It is licensed under the terms of the [BSD 3-Clause license](LICENSE).

## Credits
This project is affilitated with the [UNSW Collaboration on Energy and Environmental Markets](https://www.ceem.unsw.edu.au/) and was further supported by the [UNSW Digital Grid Futures Institute](https://www.dgfi.unsw.edu.au/).

`NEMGLO` incorporates functionality from a suite of UNSW-CEEM tools, namely, [`NEMOSIS`](https://github.com/UNSW-CEEM/NEMOSIS) to extract historical market data and [`NEMED`](https://github.com/UNSW-CEEM/NEMED) to compute emissions data from AEMO's MMS databases respectively. The structure of the optimiser codebase is further adopted from sister tool [`nempy`](https://github.com/UNSW-CEEM/nempy) under [nempy licence](https://github.com/UNSW-CEEM/nempy/blob/master/LICENSE).

### Acknowlgements

Many thanks to:
- Jay Anand, co-developer of the `NEMGLO` interactive web tool. 
- Nick Gorman and Abhijith Prakash for pointers on `NEMGLO` code development.
- Iain MacGill, Anna Bruce, Rahman Daiyan, and Jack Sheppard as project advisors.

## Contact
Questions and feedback are very much welcomed. Please reach out by email to [declanheim@outlook.com](mailto:declanheim@outlook.com)