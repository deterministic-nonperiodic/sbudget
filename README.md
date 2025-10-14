# budget

Tools to compute the spectral energy budget of a dry, non-hydrostatic atmosphere on regional domains.
The package targets model output on either regular lon–lat (equi-angular) or Cartesian horizontal grids, with a geometric-height vertical coordinate.

Analyses primarily follow Peng et al. and subsequent applications to mesoscale flows (see References).

### Features

- Horizontal kinetic-energy (HKE) spectra and budget terms (nonlinear spectral transfer, vertical 
  pressure and momentum fluxes/divergence, conversion from APE to HKE, divergence term)
- FFT backed chunk-friendly xarray/dask implementation (out-of-core)
- NetCDF output with CF-style metadata (horizontal coordinates are replaced with wavenumber in rad m⁻¹)

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