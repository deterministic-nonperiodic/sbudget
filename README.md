# budget

### Spectral transfer mode (--mode spectral)

Tools to compute the spectral energy budget of a dry, non-hydrostatic atmosphere on regional domains.
The energy budget is computed in horizontal wavenumber space using FFTs, following the 
formulation of [Peng et al. (2015)](https://doi.org/10.1175/JAS-D-14-0306.1) and [Wang et al. 
(2018)](https://doi.org/10.1175/JAS-D-17-0391.1). The package targets model output on either 
regular lon–lat (equiangular) or Cartesian horizontal grids, with a geometric-height vertical coordinate.

#### Features

- Horizontal kinetic-energy (HKE) spectra and budget terms (nonlinear spectral transfer, vertical 
  pressure and momentum fluxes/divergence, conversion from APE to HKE, divergence term)
- FFT backed chunk-friendly xarray/dask implementation (out-of-core)
- NetCDF output with CF-style metadata (horizontal coordinates are replaced with wavenumber in rad m⁻¹)

### Inter-scale transfer mode (--mode physical)
By default, the tool computes spectral budgets in wavenumber space using FFTs. This mode is suitable for
studying energy transfers across spectral scales. The user can also switch to physical mode, 
which computes scale-to-scale transfers at specified wavelengths based on third-order structure 
functions. This code is largely based on [LoSSETT](https://github.com/ElliotMG/LoSSETT) (the Local 
Scale-to-Scale Energy Transfer Tool). The energy transfer from scales larger than $\ell$ to 
scales smaller than $\ell$ is derived in [Duchon & Robert (2000)](https://iopscience.iop.org/article/10.1088/0951-7715/13/1/312):

$$\mathcal{D}_{\ell} := \frac{1}{4} \int \nabla G _\ell(\mathbf{r}) \cdot \delta \mathbf{u} |\delta \mathbf{u}|^2 \mathrm{d}^d \mathbf{r}.$$

### References:
- J. Peng, L. Zhang, and J. Guan (2015). Applications of a Moist Nonhydrostatic Formulation of the
  Spectral Energy Budget to Baroclinic Waves. J. Atmos. Sci., 70(7), 2055-2073.
  https://doi.org/10.1175/JAS-D-14-0306.1
- Wang, Y., L. Zhang, J. Peng, and S. Liu, 2018: Mesoscale Horizontal Kinetic Energy Spectra of a Tropical Cyclone.
  J. Atmos. Sci., 75, 3579–3596, https://doi.org/10.1175/JAS-D-17-0391.1.

### Install
```bash
# clone repository
git clone https://github.com/deterministic-nonperiodic/sbudget.git
cd sbudget

# Install clean environment (recommended)
conda env create -f environment.yml
conda activate budget

pip install .
```

### Examples
```bash
  # Quick help
  budget --help
 ``` 
Inspect configuration file
  ```bash
  budget inspect examples/config.yaml
  ```
Compute budget based on configuration file
  ```bash  
  budget compute examples/config.yaml
```
Inspect with on-the-fly overrides (without editing config file)
```bash
  budget inspect examples/config.yaml \
  --input-path ./data/model_output.nc \
  --dims z,lat,lon \
  --engine h5netcdf
```
Write to a different file or store type (NetCDF/Zarr)
```bash
  budget compute examples/config.yaml \
  --output-path ./out/budget.nc \
  --store netcdf --overwrite
```

Switch to physical mode and define wavelength bands (meters). This mode calculates inter-scale 
transfers at specified wavelengths based on third-order structure functions.
```bash
  budget compute examples/config.yaml \
  --mode physical \
  --scales 1000,5000,10000
```
Tip: For best FFT performance, keep spatial axes as single chunks (the tool can enforce this with --rechunk-spatial)
and parallelize across time/z.

```bash
  budget compute examples/config.yaml --rechunk-spatial
```