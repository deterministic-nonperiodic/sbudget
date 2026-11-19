# sbudget

Tools for computing the spectral kinetic energy budget of a dry, fully compressible,
non-hydrostatic atmosphere on regional domains. Designed to analyze energy transfers across
scales in both wavenumber and physical space. The package targets model outputs on either
regular lon–lat (equiangular) or Cartesian horizontal grids, with a geometric-height vertical
coordinate.

### Spectral transfer mode (```--mode spectral_budget```)

This mode computes the budget of horizontal kinetic energy (HKE) in wavenumber space using FFTs,
suitable for studying spectral energy transfers across scales. The budget is formulated following
[Peng et al. (2015)](https://doi.org/10.1175/JAS-D-14-0306.1),
and [Wang et al.(2018)](https://doi.org/10.1175/JAS-D-17-0391.1) as:

$$\partial_t E_h(k)=T_h(k)+L_h(k)+\partial_z F_{h\uparrow}(k)+J_h(k)+\mathrm{Div}_h(k)
+\partial_z F_{p\uparrow}(k)+J_p(k)+C_{A\to h}(k)+\mathrm{Div}_p(k)+N_h(k)+H_h(k)+D_h(k),$$
and the budget of available potential energy (APE) as:
$$\partial_t E_A(k)=-C_A(k)+T_A(k)+\partial_z F_{A\uparrow}(k)+J_A(k)+\mathrm{Div}_A(k)+H_A
(k)+D_A(k).$$

Where subscripts $h$ and $A$ denote terms for HKE and APE, respectively, and $(a,b)_k$ is the
co-spectrum summed over the wavenumber shell $k$:
* $E_h(k), E_A(k)$ — isotropic spectrum of energy at wavenumber $k$.
* $T_h(k), T_A(k)$ — nonlinear **spectral transfer** of energy (across scales); sums to zero over
  $k$ on a periodic domain.
* $L_h(k)$ — linear **spectral transfer** of HKE due to Coriolis.
* $F_{h\uparrow} = -\tfrac12(\mathbf{u}, w\mathbf{u})_k$, $F_{A\uparrow}$ — **vertical flux** of energy
  (positive downward, so $\partial_z F > 0$ is a gain).
* $F_{p\uparrow} = -c_p\bar\theta(w,\pi')_k$ — vertical **pressure** flux.
* $J_h = F_{h\uparrow}\partial_z\ln\bar\rho$, $J_A = F_{A\uparrow}\partial_z\ln(\bar\rho/\gamma)$,
  $J_p = F_{p\uparrow}\partial_z\ln(\bar\rho/\bar\theta)$ — **reference-profile** terms; with them,
  $\partial_z F + J = \bar\rho^{-1}\partial_z(\bar\rho F)$ (HKE), so density-weighted layer
  integrals reduce to boundary fluxes.
* $\mathrm{Div}_X = \tfrac12(X, X D)_k$, $\mathrm{Div}_p = c_p\bar\theta(\pi', D)_k$ — **compressibility**
  terms, with $D = \bar\rho^{-1}\nabla\cdot(\bar\rho\mathbf{v})$ (zero for anelastic flow).
* $C_{A\to h} = c_p\bar\theta(w,\partial_z\pi')_k$ — **conversion** from APE to HKE (pressure work);
  $C_A = (g/\bar\theta)(w,\theta')_k$ is the total APE→KE conversion.
* $N_h = -c_p(\mathbf{u}, \theta'\nabla_h\pi')_k$ — pressure work by $\theta'$ (and
  $N_z = -c_p(w, \theta'\partial_z\pi')_k$ for VKE).
* $H_h(k), H_A(k)$ — **diabatic** tendency (heating/cooling).
* $D_h(k), D_A(k)$ — **diffusive/dissipative** tendency (viscosity, filters).

With these definitions the advective and pressure terms reproduce the exact per-unit-mass
tendencies (tested in `tests/test_synthetic_budget.py`). Terms $H_{h,A}(k)$ and $D_{h,A}(k)$ are
model-physics dependent and therefore omitted here; $D$ is the budget residual.

#### Features

- FFT backed chunk-friendly xarray/dask implementation (out-of-core). Fully parallel along
  non-horizontal spatial dimensions
- NetCDF output with CF-style metadata (horizontal coordinates replaced with wavenumber in rad/m)

### Inter-scale transfer mode (```--mode scale_transfer```)

This mode computes local scale-to-scale transfers at specified wavelengths based on third-order
structure functions. This code is largely based on [LoSSETT](https://github.com/ElliotMG/LoSSETT)
. The energy transfer from scales larger than $\ell$ to scales smaller than $\ell$ is derived in
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
# Install package directly via git
pip install git+https://github.com/deterministic-nonperiodic/sbudget.git
```

Install manually with clean environment (recommended)

```bash
# clone repository
git clone https://github.com/deterministic-nonperiodic/sbudget.git
cd sbudget

# Install clean environment
conda env create -f environment.yml
conda activate budget

# editable install
pip install -e .
```

### Examples

Quick help

```bash
  sbudget compute --help
```

Compute budget based on configuration file

```bash  
  sbudget compute examples/config.yaml
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

Perform analysis on selected levels and scales. Note, level selection is ignored in
"spectral_budget" mode since continuous sampling is required for computing vertical gradients

```bash
  sbudget compute examples/config.yaml --mode scale_transfer \ 
  --levels 60e3 --scales 20e3,35e3
```
