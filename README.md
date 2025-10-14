# budget

A collection of tools to compute the Spectral Energy Budget of a dry non-hydrostatic Atmosphere. 
This package is developed for application to numerical simulations on regional domains following 
Peng et al. (2014). The analysis supports data sampled on a regular (equally spaced in longitude 
and latitude) or Cartesian horizontal grids (metric-aware horizontal derivatives).
The vertical grid should be on geometric heights.


### Features

- Horizontal kinetic-energy spectra and budget terms (transfer, vertical flux/divergence, 
pressure work, conversion, divergence term)
- Chunk-friendly xarray/dask implementation (out-of-core)
- NetCDF output with CF-style metadata (horizontal coordinates replaced with the wavenumber in rad m⁻¹)

References:
- J. Peng, L. Zhang, and J. Guan (2015). Applications of a Moist Nonhydrostatic Formulation of the
  Spectral Energy Budget to Baroclinic Waves. J. Atmos. Sci., 70(7), 2055-2073.
  https://doi.org/10.1175/JAS-D-14-0306.1
- Wang, Y., L. Zhang, J. Peng, and S. Liu, 2018: Mesoscale Horizontal Kinetic Energy Spectra of a Tropical Cyclone.
  J. Atmos. Sci., 75, 3579–3596, https://doi.org/10.1175/JAS-D-17-0391.1.

## Install
```bash
# clone repository
git clone https://github.com/deterministic-nonperiodic/sbudget.git
cd sbudget

# Install clean environment (recommended)
conda env create -f environment.yml
conda activate budget

pip install .
```

## Usage
```bash
  budget --help
  budget inspect examples/config.yaml
  budget compute examples/config.yaml
```