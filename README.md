# budget

A collection of tools to compute the Spectral Energy Budget of a dry non-hydrostatic
Atmosphere. This package is developed for application to numerical simulations on regional domains.
SEBA is implemented based on the formalism developed by Augier and Lindborg (2013). The analysis supports data sampled on a regular (equally spaced in longitude and latitude) or gaussian (equally spaced in longitude, latitudes located at roots of ordinary Legendre polynomial of degree nlat) as well as Cartesian horizontal grids. The vertical grid should be on geometric heights.

References:
- Augier, P., & Lindborg, E. (2013). A spectral model of the atmospheric energy
  cascade. Journal of the Atmospheric Sciences, 70(7), 2055-2073. https://doi.org/10.1175/JAS-D-12-0216.1

## Install
```bash
pip install .
```
