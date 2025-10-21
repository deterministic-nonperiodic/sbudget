# sbudget

### Spectral transfer mode (```--mode spectral_budget```)

Tools to compute the spectral kinetic energy budget of a dry, non-hydrostatic atmosphere on
regional domains. By default, this tool computes spectral budgets in wavenumber space using FFTs.
This mode is suitable for studying spectral energy transfers across scales. The budget is
computed following the formulation of [Peng et al. (2015)](https://doi.org/10.1175/JAS-D-14-0306.1)
and [Wang et al.(2018)](https://doi.org/10.1175/JAS-D-17-0391.1). The package targets model
outputs on either regular lon–lat (equiangular) or Cartesian horizontal grids, with a
geometric-height vertical coordinate. The spectral budget is formulated as follows:

$$\partial_t E_h(k)=T_h(k)+\partial_z F_{\uparrow}(k)+C_{A\to h}(k)+\mathrm{Div}_h(k)+H_h(k)+J_h(k)
+D_h(k).$$

* $E_h(k)$ — isotropic spectrum of HKE at wavenumber $k$.
* $T_h(k)$ — nonlinear **spectral transfer** of HKE (across scales).
* $F_{\uparrow}(k)$ — net **vertical flux** of HKE + pressure **pressure-work** flux
* $C_{A\to h}(k)$ — **conversion** from APE to HKE.
* $\mathrm{Div}_h(k)$ — tendency from **3-D divergence** processes.
* $H_h(k)$ — **diabatic** tendency (heating/cooling).
* $J_h(k)$ — **adiabatic nonconservative** tendency.
* $D_h(k)$ — **diffusive/dissipative** tendency (viscosity, filters).

Terms $H_h(k)$, $J_h(k)$, and $D_h(k)$ are model-physics dependent and therefore omitted here.

#### Features

- FFT backed chunk-friendly xarray/dask implementation (out-of-core). Fully parallel along 
  non-horizontal spatial dimensions
- NetCDF output with CF-style metadata (horizontal coordinates replaced with wavenumber in rad/m)

### Inter-scale transfer mode (```--mode scale_transfer```)

The user can switch to this mode, which computes local scale-to-scale transfers at specified
wavelengths based on third-order structure functions. This code is largely based on
[LoSSETT](https://github.com/ElliotMG/LoSSETT). The energy transfer from scales larger than
$\ell$ to scales smaller than $\ell$ is derived in
[Duchon & Robert (2000)](https://iopscience.iop.org/article/10.1088/0951-7715/13/1/312) as:

$$\mathcal{T}_{\ell} := \frac{1}{4} \int \nabla G _\ell(\mathbf{r}) \cdot \delta \mathbf{u}
|\delta \mathbf{u}|^2 \mathrm{d}^d \mathbf{r},$$

where, $\delta\mathbf{u}:=\mathbf{u}(x + r)-\mathbf{u}(x)$ is a velocity increment,
and $G_{\ell}(r)$
is a filter kernel with characteristic length scale $\ell$. See refences for more details.

### References:

- J. Peng, L. Zhang, and J. Guan (2015). Applications of a Moist Nonhydrostatic Formulation of the
  Spectral Energy Budget to Baroclinic Waves. J. Atmos. Sci., 70(7), 2055-2073.
  https://doi.org/10.1175/JAS-D-14-0306.1
- Wang, Y., L. Zhang, J. Peng, and S. Liu, 2018: Mesoscale Horizontal Kinetic Energy Spectra of a
  Tropical Cyclone.
  J. Atmos. Sci., 75, 3579–3596, https://doi.org/10.1175/JAS-D-17-0391.1.

- J. Duchon, and R. Robert (2000). Inertial energy dissipation for weak solutions of
  incompressible Euler and Navier-Stokes equations. Nonlinearity, 13(1), 249.
  https://doi.org/10.1088/0951-7715/13/1/312

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

Quick help
```bash
  sbudget --help
 ``` 

Inspect configuration file
  ```bash
  sbudget inspect examples/config.yaml
  ```

Compute budget based on configuration file
  ```bash  
  sbudget compute examples/config.yaml
```

**Tip**: For best FFT performance, keep spatial axes as single chunks (enforce this with --rechunk-spatial)
and parallelize across time/z.
```bash
  sbudget compute examples/config.yaml --rechunk-spatial
```

Inspect input file(s)
```bash
  sbudget inspect examples/config.yaml \
  --input-path ./data/model_output.nc \
  --dims z,lat,lon \
  --engine h5netcdf
```

Write to a different file or store type (NetCDF/Zarr)
```bash
  sbudget compute examples/config.yaml \
  --output-path ./out/budget.nc \
  --store netcdf --overwrite
```

Switch to scale transfer mode in physical space and define wavelengths (meters). This mode 
calculates inter-scale transfers at specified wavelengths based on third-order structure functions.
```bash
  sbudget compute examples/config.yaml \
  --mode scale_transfer \
  --scales 1000,5000,10000
```