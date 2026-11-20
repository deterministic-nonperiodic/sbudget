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
* $F_{h\uparrow} = -\tfrac12(\mathbf{u}, w\mathbf{u})_k$, $F _{A\uparrow}$ — **vertical flux** of energy
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

#### Vertical discretization on staggered model grids

Models such as ICON use a Lorenz-staggered vertical grid: $u, v, \theta', \pi'$ on full levels
(layer centres) and $w$ on half levels (interfaces). When the input stores $w$ on its own
half-level dimension, the reader keeps it as `w_half` (coordinate `z_half`, metres) next to the
interpolated full-level `w`, and the spectral budget uses the grid's own operators:

* $\partial_z w$ (in all $\mathrm{Div}$ terms) is the compact difference of `w_half` across each layer;
* the vertical fluxes $F_{h\uparrow}, F_{z\uparrow}, F_{A\uparrow}, F_{p\uparrow}$ are formed at the
  interfaces with the native $w$ (full-level fields interpolated there) and their divergences are
  compact differences, so layer sums telescope exactly to interface fluxes; the fluxes are
  reported on full levels as the mean of the bounding interfaces;
* the pressure-work conversions $C_{A\to h}, C_{A\to z}$ and $N_z$ use the compact $\partial_z\pi'$
  at the interfaces, co-located with $w$, and are averaged to full levels.

The nonlinear transfers keep the full-level form (their antisymmetric vertical part conserves
energy exactly). Inputs without a half-level $w$ use centred differences on full levels.

Why: interpolating $w$ to full levels damps it by $\cos(m\Delta z/2)$, and a centred difference
over $2\Delta z$ by $\sin(m\Delta z)/(m\Delta z)$. Against analytic vertical waves
(`dev/vertical_discretization_study.py`, tested in `tests/test_synthetic_budget.py`), at 5 levels
per vertical wavelength the error of $\partial_z w$ falls from 39% to 6.5% and that of
$C_{A\to h}$ from 39% to 6%; at 8 levels, from 17% to 2–3%. The cubic flux terms are limited by
unresolved harmonics of the product (≈40% at 5 levels for every formulation), so the operator
choice matters there only for telescoping and closure.

#### Conservative transfers and periodic axes (`--transfer-form conservative`, `--periodic-axes`)

Every nonlinear transfer sums to zero over wavenumbers on a periodic domain. On the grid this
holds only if the advection of a quantity $a$ is written in skew-symmetric form (e.g. Morinishi
et al. 1998),

$$\tfrac12\left[(\mathbf{u}\cdot\nabla)a + \nabla\cdot(\mathbf{u}\,a)\right],$$

with a centred difference that wraps around: `transfer_form: conservative` (the default) uses
this form for the HKE, VKE and APE transfers, and its sum is zero to round-off. The `flux` form,
$(\mathbf{u}\cdot\nabla)a + \tfrac12\delta\,a$, relies on the product rule and leaves a
residual of a few per cent of $\max|\Pi|$ for smooth fields (more if there is energy near the
grid scale); both forms converge to the same transfer.

Periodic axes are detected from the data by default (`periodic_axes: auto`): the RMS jump
between the last and first points is compared with the RMS neighbour difference. It is about 1
on a periodic axis, about 0 when the last point repeats the first (that point is then dropped),
and well above 2 on a bounded axis. `periodic_axes: [x]`, `[x, y]` or `none` override the test;
a global longitude is always periodic. Along periodic axes derivatives wrap around, and spectra
are neither tapered nor detrended. The scale-transfer mode uses the same test: periodic axes are
padded by wrapping and other axes by reflection (explicit `x_coord_boundary`/`y_coord_boundary`
keywords take precedence). Along bounded axes the default is Errico (1985) end-point
detrending (the mean and the linear trend joining the two edges are removed, the mean restored
at $k = 0$), which removes the edge jump of the periodic extension and its leakage tail, and no
taper. A taper (`SPECTRAL_CFG["taper"]`) weights the domain: its transfers include the energy
flux into the window,

$$\frac{\langle \mathbf{F}\cdot\nabla W^2\rangle}{\langle W^2\rangle},
\qquad \mathbf{F} = \tfrac12\,\mathbf{u}\,|\mathbf{u}|^2,$$

which projects onto the largest scales, grows with the mean wind along the tapered axis and can
exceed the inter-scale transfer itself.

Defaults: `transfer_form: conservative`, `periodic_axes: auto`, Errico detrending along bounded
axes, no taper. They give exact conservation on periodic axes and leakage-free spectra on
bounded ones without any setting; on undeclared periodic data the transfers equal those with the
axes declared (`tests/test_synthetic_budget.py`).

```bash
sbudget compute config.yaml --mode spectral_budget                       # defaults
sbudget compute config.yaml --mode spectral_budget --periodic-axes x     # declare x periodic
```

On a domain bounded in $y$, the sum over all wavenumbers ($k = 0$ included) is the energy flux
through the walls,

$$\sum_k T(k) = -\frac{1}{L_y}\Big[\big\langle \tfrac12\, v\,|\mathbf{u}|^2 \big\rangle_x\Big]_{y_S}^{y_N},$$

and $\Pi(k_{min})$ is the part of it that reaches $k > 0$.

#### Features

- FFT backed chunk-friendly xarray/dask implementation (out-of-core). Fully parallel along
  non-horizontal spatial dimensions
- NetCDF output with CF-style metadata (horizontal coordinates replaced with wavenumber in rad/m)

### Inter-scale transfer mode (```--mode scale_transfer```)

This mode computes local scale-to-scale transfers at specified wavelengths based on third-order
structure functions. The energy transfer from scales larger than $\ell$ to scales smaller than
$\ell$ is derived in
[Duchon & Robert (2000)](https://iopscience.iop.org/article/10.1088/0951-7715/13/1/312) as:

$$\mathcal{T}_{\ell} := \frac{1}{4} \int \nabla G _\ell(\mathbf{r}) \cdot \delta \mathbf{u}
|\delta \mathbf{u}|^2 \mathrm{d}^d \mathbf{r},$$

where, $\delta\mathbf{u}:=\mathbf{u}(x + r)-\mathbf{u}(x)$ is a velocity increment,
and $G_{\ell}(r)$
is a filter kernel with characteristic length scale $\ell$. See references for more details.

The first implementation followed [LoSSETT](https://github.com/ElliotMG/LoSSETT), which evaluates
the integral as an explicit loop over shifts. sbudget now evaluates the same lattice sum by FFT
correlation: the result is unchanged, but the cost is no longer proportional to the number of
shifts (10-23 times faster on the grids measured below), which makes large domains and many
scales affordable.

#### How $\mathcal{T}_\ell$ is computed: exact lattice sum by FFT correlation

On the grid, the integral is the sum over the lattice of horizontal shifts $\mathbf{s}$ (cell area
$\Delta A$), with the mollifier normalized on the same lattice:

$$\mathcal{T}_\ell(\mathbf{x}) = \sum_{\mathbf{s}} \sum_{i\in\{x,y\}} K_i(\mathbf{s})\,
\delta u_i \,|\delta\mathbf{u}|^2, \qquad
K_i(\mathbf{s}) = \frac{\tfrac14\, G_\ell'(|\mathbf{s}|)\,\hat{s}_i\,\Delta A}
{\sum_{\mathbf{s}'} G_\ell(|\mathbf{s}'|)\,\Delta A}.$$

Writing $\mathbf{a} = \mathbf{u}(\mathbf{x}+\mathbf{s})$ and $\mathbf{b} = \mathbf{u}(\mathbf{x})$, each
term of $(a_i - b_i)\sum_j (a_j - b_j)^2$ is a product of a field at $\mathbf{x}$ and a field at
$\mathbf{x}+\mathbf{s}$, so the sum over shifts reduces to correlations
$(K_i \star F)(\mathbf{x}) = \sum_{\mathbf{s}} K_i(\mathbf{s})F(\mathbf{x}+\mathbf{s})$:

$$\mathcal{T}_\ell = \sum_i K_i\star(u_i|\mathbf{u}|^2) -
2\sum_j b_j \sum_i K_i\star(u_i u_j) + |\mathbf{b}|^2 \sum_i K_i\star u_i -
\sum_i b_i\, K_i\star|\mathbf{u}|^2 + 2\sum_{i,j} b_i b_j\, K_i\star u_j -
|\mathbf{b}|^2 \sum_i b_i \sum_{\mathbf{s}} K_i(\mathbf{s}),$$

with $i$ over the horizontal and $j$ over all velocity components. Each plane is padded with the
boundary extension (`reflect`, `wrap`, `symmetric`, `edge`) by the largest shift, so the circular
FFT correlation of the padded plane equals the linear one at every interior point: the result is
the exact lattice sum, for any boundary treatment of that kind. Plane means are removed first
($\delta\mathbf{u}$ is unchanged) to limit cancellation; transforms are float64.

Cost per plane and scale: 12 inverse real FFTs of the padded plane (11 forward FFTs of products
of the fields per plane, kernel spectra cached per process), i.e. $O(n_\ell\, N \log N)$ instead
of $O(N_{\rm shifts}\, N)$ for a direct sum over shifts. Measured with 16 cores:
10× faster than the direct sum on a 267×267 grid (20,592 shifts, 24 scales) and 23× faster on a
267×889 grid (174,000 shifts, 66 scales). Tested against a brute-force lattice sum (agreement
better than $10^{-5}$) for periodic, reflect, mixed and edge/symmetric boundaries. A
`constant` (NaN-filled) boundary falls back to a direct sum over shift-distance bins.

#### Counterpart of the spectral transfer: $\mathcal{T}^*_\ell$ (`--divergence-correction`, `--vertical-advection`)

$\mathcal{T}_\ell$ assumes non-divergent flow advected horizontally. To compare it scale by scale
with the spectral budget in divergent flow with vertical motion (e.g. gravity waves), the
generalized transfer

$$\mathcal{T}^*_\ell = \mathcal{T}_\ell - C_\ell + V_\ell, \qquad
C_\ell = \tfrac14\,\delta\,(\mathbf{u}\cdot\bar{\mathbf{u}}) + \tfrac14\,\mathbf{u}\cdot\overline{\delta\mathbf{u}} -
\tfrac14\,|\mathbf{u}|^2\,\bar\delta - \tfrac14\,\delta\,\overline{|\mathbf{u}|^2}, \qquad
V_\ell = \tfrac12\,w\left(\bar{\mathbf{u}}\cdot\partial_z\mathbf{u} - \mathbf{u}\cdot\partial_z\bar{\mathbf{u}}\right)$$

($\delta$ the horizontal divergence, bar the filter of $\mathcal{T}_\ell$) has a domain mean equal
to the flux-form spectral transfer seen through the same filter, $\sum_k T(k)\,[1-\hat G(k\ell)]$.
It is written to `energy_transfer_generalized`; `energy_transfer` is unchanged. $C _\ell$ is the
divergence part of the point-split energy balance of compressible turbulence (Galtier & Banerjee
2011; Banerjee & Galtier 2014; Eyink & Drivas 2018, Eq. 131), taken at unit density and without
the pressure terms.

```bash
sbudget compute config.yaml --mode scale_transfer --divergence-correction --vertical-advection
```

The increments contain $u, v$ by default, so $\mathcal{T} _\ell$ and $\mathcal{T}^{\ast} _\ell$ are HKE
transfers and $w$ enters only as advecting velocity; `--increment-components u,v,w` adds the
transfer of vertical kinetic energy (the counterpart of the spectral HKE + VKE transfers).

The same options are `compute:` keys in the YAML (`divergence_correction`, `vertical_advection`,
`perturbation`, `increment_components`). Unlike $\mathcal{T}_\ell$, which is Galilean invariant
(Aluie et al. 2018), the correction (like the spectral transfer) depends on the frame: a mean
wind adds wave–mean-flow exchange, which `--perturbation` removes. Closed-form and synthetic tests are in
[tests/test_synthetic_budget.py](tests/test_synthetic_budget.py).

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
- S. Galtier, and S. Banerjee (2011). Exact relation for correlation functions in compressible
  isothermal turbulence. Phys. Rev. Lett., 107, 134501.
  https://doi.org/10.1103/PhysRevLett.107.134501
- S. Banerjee, and S. Galtier (2014). A Kolmogorov-like exact relation for compressible
  polytropic turbulence. J. Fluid Mech., 742, 230-242. https://doi.org/10.1017/jfm.2013.657
- G. L. Eyink, and T. D. Drivas (2018). Cascades and dissipative anomalies in compressible fluid
  turbulence. Phys. Rev. X, 8, 011022. https://doi.org/10.1103/PhysRevX.8.011022
- H. Aluie, M. Hecht, and G. K. Vallis (2018). Mapping the energy cascade in the North Atlantic
  Ocean: The coarse-graining approach. J. Phys. Oceanogr., 48(2), 225-244.
  https://doi.org/10.1175/JPO-D-17-0100.1
- R. M. Errico (1985). Spectra computed from a limited area grid. Mon. Wea. Rev., 113, 1554-1562.
  https://doi.org/10.1175/1520-0493(1985)113%3C1554:SCFALA%3E2.0.CO;2
- Y. Morinishi, T. S. Lund, O. V. Vasilyev, and P. Moin (1998). Fully conservative higher order
  finite difference schemes for incompressible flow. J. Comput. Phys., 143, 90-124.
  https://doi.org/10.1006/jcph.1998.5962

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

Switch to scale transfer mode in physical space and define filter scales $\ell$ (meters). This mode
calculates inter-scale transfers across each filter scale based on third-order structure functions;
the output also carries the equivalent wavelength $\lambda_{eq} \approx 5.58\,\ell$ (half-power point
of the filter) for comparison with spectral fluxes.

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

### Choosing a scheduler

`compute.scheduler` (or `--scheduler`) selects how Dask runs:

* `null` (default) — automatic: `distributed` under SLURM, `threads` otherwise.
* `threads` — one process; recommended on workstations.
* `distributed` — several worker processes. Needed to span nodes; recommended on HPC. Without
  SLURM or explicit `n_workers`/`threads_per_worker`, cores are grouped into workers of 4 threads.
  Under SLURM, workers and threads follow `SLURM_NTASKS` and `SLURM_CPUS_PER_TASK`.

Resources are sized automatically. Under SLURM, the allocation is used (minus a 4 GB / 10%
reserve). On a workstation, sbudget uses the physical cores and 85% of the memory available at
start-up (`threads` mode: at least 0.5 GB per thread; `distributed`: at least 4 GB per process).
`--n-workers` and `--threads-per-worker` override the core count.

Measured on a workstation (i7-14650HX, 16 physical / 24 logical cores, 62 GB), 267×267
lon–lat grid, 12 levels × 8 times, 24 scales (wall time incl. start-up and I/O, peak memory of
all processes; identical results in every configuration):

| `scale_transfer` | 8 threads | 12 threads | **16 threads (default)** | 24 threads | `distributed` 6×4 |
|---|---|---|---|---|---|
| wall time | 19.3 s | 17.0 s | **17.0 s** | 18.3 s | 20.7 s |
| peak memory | 1.8 GB | 2.4 GB | **3.0 GB** | 4.3 GB | 4.9 GB |

Larger runs with the defaults: 50 levels × 8 times (400 planes) 71 s and 4.6 GB; the 267×889
nest with 66 scales, 2 levels × 8 times, 48 s and 11.7 GB. `spectral_budget` (50 levels, before
the automatic sizing; 14 threads/workers): `threads` 87 s / 6.9 GB, `distributed` 14×1 59 s /
14.0 GB, 4×4 75 s / 10.4 GB.

In `scale_transfer` mode each block needs whole horizontal planes (shifts span the plane), so the
planner splits only time and levels into blocks of one plane each (at least one block per
thread), and stops with a `MemoryError` if a single plane does not fit in the memory available
per thread; reduce the number of scales or the threads per worker in that case. Memory per block
is about `(12 × n_scales + 320 × pad_factor)` bytes per grid point and plane, where `pad_factor`
is the area growth from padding the plane by the largest shift (≈ 3–4 for the default scales);
each process also caches the kernel spectra (≤ 2 GB). Peak memory is therefore about the number
of threads times one plane's working set, independent of the number of times and levels, and
run time is linear in the number of planes.
