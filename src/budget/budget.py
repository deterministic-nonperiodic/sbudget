from dataclasses import dataclass
from typing import Union

import numpy as np
import xarray as xr

from .constants import cp, ps, chi, earth_radius
from .cf_coords import _coord_is_degrees, _is_lon, _is_lat, get_spatial_dims, infer_resolution


# --------------------------------------------------------------------------------------------------
# Global spectral options (to avoid plumbing flags everywhere)
# --------------------------------------------------------------------------------------------------
@dataclass
class _SpectralOptions:
    allow_rechunk: bool = True
    rechunk_spatial: bool = True


_OPTIONS = _SpectralOptions()


# --------------------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------------------
def _ensure_spatial_single_chunk(da: xr.DataArray, dims: tuple[str, str, str]) -> xr.DataArray:
    """Rechunk so each core spatial dimension is a single Dask chunk."""
    _, y, x = dims
    try:
        return da.chunk({y: -1, x: -1})
    except Exception:
        return da


def global_mean(da: xr.DataArray) -> xr.DataArray:
    """
    cos(lat) weighted mean over (y,x).
    Supports lat as 1-D (lat) or 2-D lat(y,x). Falls back to plain mean if no lat.
    """
    y, x = get_spatial_dims(da)

    if _is_lat(y, da.coords):
        lat = da["lat"]
        lat_rad = np.deg2rad(lat) if _coord_is_degrees(lat) else lat

        return da.weighted(np.cos(lat_rad)).mean(dim=(y, x))
    else:
        # unexpected lat layout
        return da.mean(dim=(y, x))


# --------------------------------------------------------------------------------------------------
# --- metric-aware horizontal derivatives in physical meters ---
# --------------------------------------------------------------------------------------------------
def differentiate_metric(da: xr.DataArray, dim: str, delta: float | None = None) -> xr.DataArray:
    """
    Metric-aware first derivative **in meters** along `dim`.

    - If `delta` is given (meters), assumes constant spacing along `dim`.
    - If `dim` is longitude/latitude (per `_is_lon` / `_is_lat`), applies the spherical metric:
        d/dx = (π/180)/(R cos φ) * d/dλ   (for longitude, if coord in degrees)
        d/dy = (π/180)/R          * d/dφ   (for latitude,  if coord in degrees)
      (If coords are already in radians, omit π/180.)
    - Otherwise (Cartesian dims, incl. `z`), uses `.differentiate(dim)` and converts km→m if needed.
    """
    if dim not in da.dims:
        raise ValueError(f"differentiate_metric: dim '{dim}' not in {tuple(da.dims)}")

    # Fetch the coordinate
    coord = da.coords.get(dim, None)

    if coord is None:
        if delta is None:
            raise ValueError(
                f"differentiate_metric: No coordinate found for dim '{dim}' "
                f"and no 'delta' provided; cannot determine metric spacing."
            )
        # index-based derivative (spacing=1) scaled by constant delta [m]
        return da.differentiate(dim, edge_order=2) / float(delta)

    # Longitude
    if _is_lon(dim, da.coords):
        # Need latitude for cos(phi)
        lat = da.coords['lat']
        phi = np.deg2rad(lat) if _coord_is_degrees(lat) else lat
        cos_phi = xr.ufuncs.cos(phi)
        cos_phi = xr.where(cos_phi < 1e-12, 1e-12, cos_phi)

        # convert from coord-units to per-meter
        d_lam = (np.pi / 180.0) if _coord_is_degrees(coord) else 1.0
        factor = d_lam / (earth_radius * cos_phi)  # rad/m

        return factor * da.differentiate(dim, edge_order=2)

    # Latitude
    if _is_lat(dim, da.coords):
        d_phi = (np.pi / 180.0) if _coord_is_degrees(coord) else 1.0
        factor = d_phi / earth_radius  # rad/m
        return factor * da.differentiate(dim, edge_order=2)

    # Cartesian (incl. 'z'): convert km→m if the coord says 'km'
    units = str(getattr(coord, "units", "")).lower()
    if "km" in units and "m" not in units:
        return (1e-3) * da.differentiate(dim, edge_order=2)

    return da.differentiate(dim, edge_order=2)


# --------------------------------------------------------------------------------------------------
# Spectral primitives (unchanged API; now read global options)
# --------------------------------------------------------------------------------------------------
def _fft2_shifted(a, norm=None):
    """2-D FFT over the last two axes, with optional normalization."""
    a_sc = np.fft.fftn(a, axes=(-2, -1), norm=norm)
    a_sc = np.fft.fftshift(a_sc, axes=(-2, -1))

    # Normalize FFT by total number of points
    if norm is None:
        a_sc = a_sc / np.prod(a.shape)

    return a_sc


def _real_fft2_shifted(a, norm=None):
    """2-D real-input FFT over the last two axes with ky-shift and rFFT half-plane weighting.

    - rFFT on (y, x): last axis returns non-negative kx (half-plane).
    - Legacy scaling when norm is None: divide by Ny * Nx.
    - Apply √2 on interior kx columns so that |F|^2 doubles there (DC and Nyquist stay 1).
    """
    a = np.asanyarray(a).real

    # rFFT core: half-plane along x
    a_sc = np.fft.rfftn(a, axes=(-2, -1), norm=norm)
    # center ky only (symmetric)
    a_sc = np.fft.fftshift(a_sc, axes=(-2,))

    # normalization matching legacy (forward scaled)
    if norm is None:
        a_sc = a_sc / np.prod(a.shape)

    # --- rFFT half-plane amplitude weights (√2 on interior kx) ---
    nx = a.shape[-1]
    nkx = nx // 2 + 1
    has_nyq = (nx % 2) == 0  # even Nx has explicit Nyquist column at the end
    idx = np.arange(nkx)

    # 1 at DC (i==0) and Nyquist (if present), √2 elsewhere
    weights = np.where(idx == 0, 1.0, np.where(has_nyq & (idx == nkx - 1), 1.0, np.sqrt(2.0)))

    return a_sc * weights


def scalar_spectrum(field: xr.DataArray, norm: str | None = None) -> xr.DataArray:
    """Return 2-D power spectrum |F(k)|^2 over the last two spatial axes.

    rFFT path:
      - _fft2_shifted returns (ky shifted, kx non-negative) complex spectrum
      - we compute |F|^2, then apply interior-kx ×2 in isotropize (not here)
    """

    dims = get_spatial_dims(field)  # your helper: e.g., ("y","x")

    # output spectral sizes: ky = Ny; kx = Nx//2 + 1
    ny = field.sizes[dims[0]]
    nx = field.sizes[dims[1]]
    nx_pos = nx // 2 + 1

    def _pow(a):
        spec = _real_fft2_shifted(a, norm)  # rFFT core
        return (spec.conj() * spec).real  # (…, ky, kx_rfft)

    power = xr.apply_ufunc(
        _pow, field,
        input_core_dims=[list(dims)],
        output_core_dims=[["ky", "kx"]],
        dask="parallelized",
        vectorize=True,
        keep_attrs=True,
        dask_gufunc_kwargs={
            "allow_rechunk": _OPTIONS.allow_rechunk,
            "output_sizes": {"ky": ny, "kx": nx_pos},
            "meta": np.array((), dtype=np.float64),
        }
    )

    # ensure spectral dim names
    if "ky" not in power.dims or "kx" not in power.dims:
        power = power.rename({dims[0]: "ky", dims[1]: "kx"})

    return power


def cross_spectrum(field1: xr.DataArray, field2: xr.DataArray,
                   norm: str | None = None) -> xr.DataArray:
    """Return 2-D cross-spectrum F1*(k) F2(k) over the horizontal dims.

    rFFT path (half-plane in kx, ky shifted). Interior-kx ×2 will be applied
    in isotropize (not here) to preserve total variance consistently.
    """
    dims = get_spatial_dims(field1)

    ny = field1.sizes[dims[0]]
    nx = field1.sizes[dims[1]]
    nx_pos = nx // 2 + 1

    def _cross(a, b):
        spec1 = _real_fft2_shifted(a, norm)
        spec2 = _real_fft2_shifted(b, norm)
        return (spec1.conj() * spec2).real

    power = xr.apply_ufunc(
        _cross, field1, field2,
        input_core_dims=[list(dims), list(dims)],
        output_core_dims=[["ky", "kx"]],
        dask="parallelized",
        vectorize=True,
        keep_attrs=True,
        dask_gufunc_kwargs={
            "allow_rechunk": _OPTIONS.allow_rechunk,
            "output_sizes": {"ky": ny, "kx": nx_pos},
            "meta": np.array((), dtype=np.float64),
        }
    )

    if "ky" not in power.dims or "kx" not in power.dims:
        power = power.rename({dims[0]: "ky", dims[1]: "kx"})

    return power


def _prep_bins(nx: int, ny: int, dx: float, dy: float, nyquist=True):
    """
    Precompute bin index. Non-overlapping, variance-conserving radial bins.

    rFFT-aware:
      - kx is non-negative via rfftfreq(nx), while ky is symmetric
      - Δ = 2π / min(dx*Nx_full, dy*ny)  (legacy center spacing)
      - centers at nΔ (drop inner < Δ/2 ring), identical to legacy layout
    """
    # reconstruct physical Nx from rFFT spectral size
    nx_pos = 2 * (nx - 1)

    kx = np.fft.rfftfreq(nx_pos, dx / (2 * np.pi))  # length nx (half-plane)
    ky = np.fft.fftshift(np.fft.fftfreq(ny, dy / (2 * np.pi)))  # length ny (centered)

    kh_grid = np.hypot(*np.meshgrid(kx, ky, indexing="xy")).astype(np.float64)

    delta = 2.0 * np.pi / min(dx * nx_pos, dy * ny)

    # Explicit Nyquist cutoff (if requested)
    nyq = np.pi / max(dx, dy)
    k_cut = min(nyq, float(kh_grid.max())) if nyquist else float(kh_grid.max())

    start = 0.5 * delta
    n_bins = int(np.floor((k_cut - start) / delta + 1e-12))
    if (k_cut - start) - n_bins * delta > 1e-12:
        n_bins += 1

    # bin edges and centers
    edges = start + np.arange(0, n_bins, dtype=np.float64) * delta
    edges = np.concatenate([edges, [max(k_cut + 1e-15 * delta, start + n_bins * delta)]])

    centers = 0.5 * (edges[:-1] + edges[1:])
    centers = delta * np.rint(centers / delta)

    # 2-D bin index array
    idx2d = np.digitize(kh_grid, edges, right=False) - 1
    idx2d[(idx2d < 0) | (idx2d >= n_bins)] = -1
    return centers, idx2d


def _azimuthal_bincount(block: np.ndarray, bin_idx2d: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Sum values into fine non-overlapping bins via bincount.

    Parameters
    ----------
    block : (ny, nx) spectrum values for one (time,z,...) slice
    bin_idx2d : (ny, nx) 2-D bin index array
    n_bins: number of final bins

    Returns
    -------
        binned spectrum
    """
    flat = block.reshape(-1)
    bins = bin_idx2d.reshape(-1)
    valid = bins >= 0

    binned_block = np.bincount(np.where(valid, bins, 0),
                               weights=np.where(valid, flat, 0.0),
                               minlength=n_bins)
    return binned_block


def isotropize(spectrum: xr.DataArray, dx: float, dy: float, nyquist: bool = True) -> xr.DataArray:
    """Variance-conserving azimuthally average a 2-D spectrum to a 1-D isotropic spectrum.

    Accepts spectra with dims ('ky','kx'), where kx is the rFFT half-plane.
      - doubles interior kx columns (except DC and Nyquist)

    Returns
    -------
    xr.Dataset
        Contains wavenumber coordinate ``k`` and the 1‑D spectrum in
        variable ``spectrum_1d``. Leading dims (e.g., time, z) are preserved.
    """
    if not {"ky", "kx"} <= set(spectrum.dims):
        # allow physical dims as a fallback, but you should be feeding spectra here
        y, x = get_spatial_dims(spectrum)
        spectrum = spectrum.rename({y: "ky", x: "kx"})

    # Get kappa bins and 2D bin index array
    nky, nkx = int(spectrum.sizes["ky"]), int(spectrum.sizes["kx"])

    # bins & index
    wavenumber, idx2d = _prep_bins(nkx, nky, dx, dy, nyquist=nyquist)

    spec1d = xr.apply_ufunc(
        _azimuthal_bincount,
        spectrum,
        xr.DataArray(idx2d, dims=("ky", "kx")),
        wavenumber.size,
        input_core_dims=[("ky", "kx"), ("ky", "kx"), []],
        output_core_dims=[["wavenumber"]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={
            "output_sizes": {"wavenumber": wavenumber.size},
            "allow_rechunk": False,
            "meta": np.array((), dtype=spectrum.dtype),
        },
        keep_attrs=True,
    )

    spec1d = spec1d.rename(spec1d.name or "spectrum_1d")
    return spec1d.assign_coords(wavenumber=("wavenumber", wavenumber))


# --------------------------------------------------------------------------------------------------
# Energy budget terms
# --------------------------------------------------------------------------------------------------
# --- vector helpers for invariant-form terms ---

def stack_vector(u: xr.DataArray, v: xr.DataArray, name: str | None = None) -> xr.DataArray:
    """Stack two horizontal components into a 2-component DataArray with dim ``comp``.

    ``comp`` coordinate is ["u", "v"]. Leading dims are broadcast as needed.
    """
    comp = xr.DataArray(["u", "v"], dims="comp", name="comp")
    return xr.concat([u, v], dim=comp).rename(name) if name else xr.concat([u, v], dim=comp)


def rotate_vector(vec: xr.DataArray) -> xr.DataArray:
    """Rotate a 2-component horizontal vector 90° counterclockwise: [u, v] → [-v, u].

    Expects a ``comp`` dimension with values ["u", "v"].
    """
    if "comp" not in vec.dims:
        raise ValueError("rotate_vector expects a DataArray with a 'comp' dimension")
    u = vec.sel(comp="u")
    v = vec.sel(comp="v")
    comp = xr.DataArray(["u", "v"], dims="comp", name="comp")
    return xr.concat([-v, u], dim=comp)


def vector_cross_spectrum(vec1: xr.DataArray, vec2: xr.DataArray,
                          norm: str | None = None) -> xr.DataArray:
    """Sum of cross-spectra of matching components of two 2D vectors.

    Returns ⟨u1, u2⟩ + ⟨v1, v2⟩ in spectral space.
    """
    if "comp" not in vec1.dims or "comp" not in vec2.dims:
        raise ValueError("vector_cross_spectrum expects inputs with 'comp' dimension")
    u_term = cross_spectrum(vec1.sel(comp="u"), vec2.sel(comp="u"), norm)
    v_term = cross_spectrum(vec1.sel(comp="v"), vec2.sel(comp="v"), norm)
    return u_term + v_term


def compute_divergence(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """Horizontal divergence."""
    y, x = get_spatial_dims(u)
    return differentiate_metric(u, x) + differentiate_metric(v, y)


def compute_vorticity(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """Vertical vorticity."""
    y, x = get_spatial_dims(u)
    return differentiate_metric(v, x) - differentiate_metric(u, y)


def kinetic_energy_spectra(u: xr.DataArray, v: xr.DataArray, norm: str | None = None,
                           name="hke") -> xr.DataArray:
    """Horizontal kinetic energy per unit mass spectrum: ½(|Û|² + |V̂|²)."""
    hke = 0.5 * (scalar_spectrum(u, norm) + scalar_spectrum(v, norm))
    return hke.rename(name)


def nonlinear_hke_transfer_flux(
        u: xr.DataArray,
        v: xr.DataArray,
        w: xr.DataArray,
        divergence: Union[xr.DataArray | None] = None,
        norm: str | None = None,
        name="pi_nke"
) -> xr.DataArray:
    """Nonlinear transfer term for horizontal kinetic energy (HKE), vectorized.

    Uses the compact form:
        T = -⟨U, A⟩ + ⟨∂z U, w U⟩
    where U = (u, v),
          A = (u ∂x u + v ∂y u + ½ div·u + ½ w ∂z u,
               u ∂x v + v ∂y v + ½ div·v + ½ w ∂z v)

    All horizontal derivatives use `differentiate_metric` (metric-aware on lon/lat).
    Vertical derivatives use `.differentiate("z")`.
    """
    y, x = get_spatial_dims(u)  # e.g., ("y","x") or ("lat","lon")

    # Horizontal & vertical derivatives
    dxu = differentiate_metric(u, x)
    dxv = differentiate_metric(v, x)
    dyu = differentiate_metric(u, y)
    dyv = differentiate_metric(v, y)
    dzu = u.differentiate("z", edge_order=2)
    dzv = v.differentiate("z", edge_order=2)

    # Divergence (compute if absent)
    if divergence is None:
        divergence = dxu + dyv

    # Stack vectors
    wind = stack_vector(u, v, name="wind")  # (comp=2, z, y, x, …)
    wind_shear = stack_vector(dzu, dzv, name="shear")

    # Advection-like vector A and transport vector wU
    adv_u = (u * dxu + v * dyu) + 0.5 * (divergence * u)
    adv_v = (u * dxv + v * dyv) + 0.5 * (divergence * v)
    advection = stack_vector(adv_u, adv_v, name="advection") + 0.5 * w * wind_shear

    # Spectral vector inner products (sum over components)
    t_adv = - vector_cross_spectrum(wind, advection, norm=norm)  # -⟨U, A⟩
    t_shear = vector_cross_spectrum(wind_shear, w * wind, norm=norm)  # ⟨∂z U, w U⟩

    pi_nke = t_adv + t_shear

    return pi_nke.rename(name)


def nonlinear_hke_transfer_invariant(u: xr.DataArray, v: xr.DataArray, w: xr.DataArray,
                                     divergence: xr.DataArray, vorticity: xr.DataArray,
                                     norm: str | None = None, name="pi_nke") -> xr.DataArray:
    """Invariant-form nonlinear KE transfer (Augier & Lindborg 2013, Eq. A2).

    Works in the current workflow using xarray stacking and spectral primitives:
    - Build horizontal vectors with a ``comp={u,v}`` dimension
    - Use physical-space gradients and shears
    - Convert to spectral space via component-wise cross spectra and sum
    """
    y, x = get_spatial_dims(u)

    # Physical-space kinetic energy and its horizontal gradient (vector)
    hke_phys = 0.5 * (u ** 2 + v ** 2)
    grad_hke = stack_vector(differentiate_metric(hke_phys, x),
                            differentiate_metric(hke_phys, y),
                            name="grad_hke")

    # Wind vector and vertical shear vector
    wind = stack_vector(u, v, name="wind")
    wind_shear = stack_vector(u.differentiate('z', edge_order=2),
                              v.differentiate('z', edge_order=2),
                              name="wind_shear")

    # Divergence and vorticity. Compute if any is absent for consistency.
    if divergence is None or vorticity is None:
        divergence = compute_divergence(u, v)
        vorticity = compute_vorticity(u, v)

    # Rotational form advection vector
    advection = grad_hke + vorticity * rotate_vector(wind)
    advection += (divergence * wind + w * wind_shear) / 2.0

    # Nonlinear spectral transfer
    adv_transfer = - vector_cross_spectrum(wind, advection, norm)
    vertical_transport = vector_cross_spectrum(wind_shear, w * wind, norm)

    pi_nke = adv_transfer + vertical_transport

    return pi_nke.rename(name)


def nonlinear_vke_transfer(
        u: xr.DataArray,
        v: xr.DataArray,
        w: xr.DataArray,
        divergence: Union[xr.DataArray | None] = None,
        norm: str | None = None,
        name="pi_vke"
) -> xr.DataArray:
    """Nonlinear transfer term for vertical kinetic energy (VKE), vectorized.

    Uses the compact form:
        T = -⟨w, Aw⟩ + ⟨∂z w, w^2⟩
    where U = (u, v),
          A = (u ∂x w + v ∂y w + ½ div·w + ½ w ∂z w) / 2

    All horizontal derivatives use `differentiate_metric` (metric-aware on lon/lat).
    Vertical derivatives use `.differentiate("z")`.
    """
    y, x = get_spatial_dims(u)  # e.g., ("y","x") or ("lat","lon")

    # Horizontal & vertical derivatives
    dxw = differentiate_metric(w, x)
    dyw = differentiate_metric(w, y)
    dzw = w.differentiate("z", edge_order=2)

    # Divergence (compute if absent)
    if divergence is None:
        divergence = compute_divergence(u, v)

    # Advection-like vector A and transport vector wU
    advection_w = (u * dxw + v * dyw) + 0.5 * (divergence * w) + 0.5 * (w * dzw)

    # Spectral vector inner products (sum over components)
    t_adv = - cross_spectrum(w, advection_w, norm=norm)  # -⟨w, Aw⟩
    t_shear = vector_cross_spectrum(dzw, w * w, norm=norm)  # ⟨∂z w, w^2⟩

    pi_nke = t_adv + t_shear

    return pi_nke.rename(name)


def turbulent_hke_flux(u: xr.DataArray, v: xr.DataArray, w: xr.DataArray,
                       norm: str | None, name="vf_hke") -> xr.DataArray:
    """Vertical flux of HKE: -½⟨u, w u⟩ - ½⟨v, w v⟩ in spectral space."""

    vf_hke = -0.5 * (cross_spectrum(u, w * u, norm) + cross_spectrum(v, w * v, norm))
    return vf_hke.rename(name)


def turbulent_vke_flux(w: xr.DataArray, norm: str | None, name="vf_vke") -> xr.DataArray:
    """Vertical flux of HKE: -½⟨u, w u⟩ - ½⟨v, w v⟩ in spectral space."""

    vf_vke = -0.5 * cross_spectrum(w, w * w, norm)
    return vf_vke.rename(name)


def pressure_flux(theta: xr.DataArray, w: xr.DataArray, exner: xr.DataArray,
                  norm: str | None, name="vf_pres") -> xr.DataArray:
    """Vertical pressure work flux term: -cp·θ·⟨w, exner⟩."""
    p_flux = -cp * global_mean(theta) * cross_spectrum(w, exner, norm)

    return p_flux.rename(name)


def conversion_ape_to_dke(theta: xr.DataArray, w: xr.DataArray, exner: xr.DataArray,
                          norm: str | None, name="cad") -> xr.DataArray:
    """APE→DKE conversion term: cp·θ·⟨w, ∂z exner⟩."""
    dz_exner = exner.differentiate('z', edge_order=2)

    cad = cp * global_mean(theta) * cross_spectrum(w, dz_exner, norm)

    return cad.rename(name)


def divergence_hke(u: xr.DataArray, v: xr.DataArray, w: xr.DataArray,
                   divergence: xr.DataArray, norm: str | None, name="div_hke") -> xr.DataArray:
    """Horizontal divergence contribution to the HKE budget."""
    if divergence is None:
        divergence = compute_divergence(u, v)

    s = w.differentiate("z", edge_order=2) + divergence
    div_hke = 0.5 * (cross_spectrum(u, u * s, norm) + cross_spectrum(v, v * s, norm))

    return div_hke.rename(name)


def accumulate(da: xr.DataArray) -> xr.DataArray:
    """Cumulative integral toward low wavenumbers along ``k`` → ``wavenumber``."""
    sorted_da = da.sortby("wavenumber", ascending=False)
    return sorted_da.cumsum("wavenumber").sortby("wavenumber")


def compute_budget(ds: xr.Dataset, cfg) -> xr.Dataset:
    """Compute the spectral non‑hydrostatic energy budget.

    Produces 1‑D isotropic spectra (k) for each budget term and, optionally,
    cumulative integrals toward low wavenumbers (default True via
    ``cfg.compute.cumulative``).

    Expected variables in ``cfg.variables``:
      - u, v, w  (required)
      - theta    (preferred). If absent, compute from ``pressure`` and ``temperature``.
      - pressure, temperature (optional; used if theta missing)
      - div (optional). If missing, computed as du/dx + dv/dy and added to the pipeline.
      - vorticity (optional). If missing, computed as dv/dx − du/dy for future use.

    Grid:
      - ``cfg.input.dims`` must list **[z, lat, lon]** (lon‑lat) or **[z, y, x]** (Cartesian). The
        vertical dimension name is taken directly from this list (no inference).
    """
    # Apply global spectral options from config
    _OPTIONS.allow_rechunk = getattr(cfg.compute, "dask_allow_rechunk", True)
    _OPTIONS.rechunk_spatial = getattr(cfg.compute, "rechunk_spatial", True)

    # --- dims & spacing ---
    # Expect cfg.input.dims as [z, y, x] or [z, lat, lon]
    space_dims = ("z",) + get_spatial_dims(ds)
    print(f"Resolved spatial dimensions {space_dims}")

    # After open_dataset(), variable names are normalized to logical names.
    u = ds["u"]
    v = ds["v"]
    w = ds["w"]

    # Validate that provided dims exist on w
    missing = [d for d in space_dims if d not in w.dims]
    if missing:
        raise ValueError(f"Configured dims {space_dims} not all found in 'w' dims {tuple(w.dims)}")

    # dx, dy infer if not set
    if cfg.compute.dx is None or cfg.compute.dy is None:
        dx, dy = infer_resolution(ds)
        print(f"Estimated resolution: dx = {dx:.4f} m, dy = {dy:.4f} m")
    else:
        dx, dy = cfg.compute.dx, cfg.compute.dy
        print(f"Specified resolution: dx = {dx:.4f} m, dy = {dy:.4f} m")

    # --- potential temperature ---
    theta = ds.get("theta")
    pressure = ds.get("pressure")
    if theta is None and pressure is not None:
        temperature = ds.get("temperature")
        if temperature is None:
            raise ValueError("Provide either 'theta' or both 'pressure' and 'temperature'.")
        theta = temperature / (pressure / ps) ** chi

    # Ensure spatial single-chunking for FFT inputs (configurable)
    if _OPTIONS.rechunk_spatial:
        u = _ensure_spatial_single_chunk(u, space_dims)
        v = _ensure_spatial_single_chunk(v, space_dims)
        w = _ensure_spatial_single_chunk(w, space_dims)

    # --- spectra: HKE 2D and isotropic 1D ---
    hke_2d = kinetic_energy_spectra(u, v, norm=cfg.compute.norm)
    hke_1d = isotropize(hke_2d, dx, dy)

    # ----- Calculate nonlinear spectral transfer → π(HKE) -----
    divergence = ds.get("divergence")
    vorticity = ds.get("vorticity")

    # in compute_budget:
    mode = getattr(cfg.compute, "transfer_form", "invariant")  # "invariant" | "flux"

    if mode == "invariant":  # Rotational form of the horizontal advection term
        transfer_2d = nonlinear_hke_transfer_invariant(u, v, w,
                                                       divergence, vorticity,
                                                       norm=cfg.compute.norm)
    else:
        transfer_2d = nonlinear_hke_transfer_flux(u, v, w, divergence, norm=cfg.compute.norm)

    transfer_1d = isotropize(transfer_2d, dx, dy)

    # --- vertical flux of HKE and its divergence ---
    fh_2d = turbulent_hke_flux(u, v, w, cfg.compute.norm)
    fh_1d = isotropize(fh_2d, dx, dy)
    vfd_dke_1d = isotropize(fh_2d.differentiate('z', edge_order=2), dx, dy).rename("vfd_dke")

    # --- pressure work & conversion (if pressure available) ---
    exner = (pressure / ps) ** chi
    if _OPTIONS.rechunk_spatial:
        exner = _ensure_spatial_single_chunk(exner, space_dims)
    fp_2d = pressure_flux(theta, w, exner, cfg.compute.norm)
    vf_pres_1d = isotropize(fp_2d, dx, dy)
    vfd_pres_1d = isotropize(fp_2d.differentiate('z', edge_order=2), dx, dy).rename("vfd_pres")

    cad_2d = conversion_ape_to_dke(theta, w, exner, cfg.compute.norm)
    cad_1d = isotropize(cad_2d, dx, dy)

    # --- horizontal KE divergence term ---
    div_hke_2d = divergence_hke(u, v, w, divergence, cfg.compute.norm)
    div_hke_1d = isotropize(div_hke_2d, dx, dy)

    # --- accumulation (optional; defaults True) ---
    do_acc = getattr(cfg.compute, "cumulative", True)

    def maybe_accumulate(da):
        return accumulate(da) if do_acc else da

    pi_nke = maybe_accumulate(transfer_1d)
    vfd_dke = maybe_accumulate(vfd_dke_1d)
    vf_pres = maybe_accumulate(vf_pres_1d)
    vfd_pres = maybe_accumulate(vfd_pres_1d)
    cad = maybe_accumulate(cad_1d)
    div_hke = maybe_accumulate(div_hke_1d)
    vf_hke = maybe_accumulate(fh_1d)

    # --- assemble ---
    fields = [hke_1d, pi_nke, vfd_dke, div_hke, vf_hke, vf_pres, vfd_pres, cad]
    fluxes = xr.Dataset({da.name: da for da in fields if da is not None})

    # attrs
    fluxes.wavenumber.attrs.update({'standard_name': 'wavenumber',
                                    'long_name': 'horizontal wavenumber',
                                    'axis': 'X', 'units': 'rad / m'})

    units = "watt / kilogram"
    meta = {
        "cad": ("conversion_ape_dke",
                "conversion from available potential energy to divergent kinetic energy"),
        "pi_nke": ("hke_transfer",
                   "spectral transfer of horizontal kinetic energy (cumulative if enabled)"),
        "vfd_dke": ("vertical_dke_flux_divergence",
                    "vertical flux divergence of horizontal kinetic energy"),
        "vf_hke": ("vertical_dke_flux", "vertical flux of horizontal kinetic energy"),
        "vf_pres": ("hke_pressure_vertical_flux",
                    "vertical flux of horizontal kinetic energy (pressure work)"),
        "vfd_pres": ("pressure_flux_divergence", "vertical divergence of pressure work"),
        "div_hke": ("horizontal_ke_divergence", "horizontal divergence contribution to HKE budget"),
    }

    # apply attrs only to variables that are present and recognized
    for var in fluxes.data_vars:
        if var in meta:
            std_name, long_name = meta[str(var)]
            fluxes[var].attrs.update(
                {"standard_name": std_name, "long_name": long_name, "units": units}
            )

    return fluxes
