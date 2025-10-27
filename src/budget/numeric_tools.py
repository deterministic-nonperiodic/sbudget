import numpy as np
import xarray as xr
from xarray import set_options

from .cf_coords import _coord_is_degrees, _is_geographic, get_spatial_dims
from .constants import earth_radius

set_options(keep_attrs=True)


# --------------------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------------------
def domain_mean(da: xr.DataArray) -> xr.DataArray:
    """
    cos(lat) weighted mean over spatial dimensions (y, x).

    Supports latitude coordinate identified as the Y-dimension (y) via CF attributes,
    regardless of dimensionality (1D or 2D). Falls back to a plain mean if a
    latitude coordinate cannot be identified.
    """
    # 1. Get the names of the spatial dimensions to reduce over
    y_dim, x_dim = get_spatial_dims(da)

    # Get the coordinate associated with the y-dimension name
    # This coordinate may be 1D ('lat') or 2D ('y' with latitude data)
    y_coord = da.coords[y_dim]

    # Check if the y-coordinate is a geographic latitude
    if _is_geographic(y_coord, "lat"):
        lat = y_coord

        # Convert to radians if units are detected as degrees
        lat_rad = np.deg2rad(lat) if _coord_is_degrees(lat) else lat

        # Calculate the weighted mean over the spatial dimensions
        return da.weighted(np.cos(lat_rad)).mean(dim=(y_dim, x_dim))
    else:
        # Fallback for non-geographic, projected, or unknown grids
        return da.mean(dim=(y_dim, x_dim))


# --------------------------------------------------------------------------------------------------
# --- metric-aware horizontal derivatives in physical meters ---
# --------------------------------------------------------------------------------------------------
def differentiate_metric(da: xr.DataArray, dim: str, delta: float | None = None) -> xr.DataArray:
    """
    Metric-aware first derivative **in meters** along `dim`.

    - If `delta` is given (meters), assumes constant spacing along `dim`.
    - If `dim` is longitude/latitude (per `_is_lon` / `_is_lat`), applies the spherical metric:
        d/dx = (π/180)/(a cos φ) * d/dλ   (for longitude, if coord in degrees)
        d/dy = (π/180)/a         * d/dφ   (for latitude,  if coord in degrees)
      where "a" is Earth's radius. If coords are already in radians, omit π/180.
    - Otherwise (Cartesian dims, incl. `z`), uses `.differentiate(dim)` and converts km→m if needed.
    """
    if dim not in da.dims:
        raise ValueError(f"differentiate_metric: dim '{dim}' not in {tuple(da.dims)}")

    # Fetch the coordinate
    coord = da.coords.get(dim, None)

    if coord is None:
        if delta is not None:
            # index-based derivative (spacing=1) scaled by constant delta [m]
            delta = xr.full_like(da, fill_value=float(delta))
            return da.differentiate(coord=dim, edge_order=2) / delta
        else:
            raise ValueError(
                f"differentiate_metric: No coordinate found for dim '{dim}' "
                f"and no 'delta' provided; cannot determine metric spacing.")

    # Longitude
    if _is_geographic(coord, "lon"):
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
    if _is_geographic(coord, "lat"):
        d_phi = (np.pi / 180.0) if _coord_is_degrees(coord) else 1.0
        factor = d_phi / earth_radius  # rad/m
        return factor * da.differentiate(dim, edge_order=2)

    # Cartesian (incl. 'z'): convert km→m if the coord says 'km'
    units = str(getattr(coord, "units", "")).lower()
    if "km" in units and "m" not in units:
        return 1e-3 * da.differentiate(dim, edge_order=2)

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
            "allow_rechunk": True,
            "output_sizes": {"ky": ny, "kx": nx_pos},
            "meta": np.array((), dtype=np.float64),
        }
    )

    # ensure spectral dim names
    if "ky" not in power.dims or "kx" not in power.dims:
        power = power.rename({dims[0]: "ky", dims[1]: "kx"})

    return power


def scalar_cross_spectrum(field1: xr.DataArray, field2: xr.DataArray,
                          norm: str | None = None) -> xr.DataArray:
    """Return 2-D cross-spectrum F1*(k) F2(k) over the horizontal dims.

    Parameters
    ----------
    field1 : xr.DataArray
        First input field.
    field2 : xr.DataArray
        Second input field.
    norm : str | None
        FFT normalization ('ortho', 'backward', or None for legacy).
    Returns
    -------
    xr.DataArray
        2-D cross-spectrum over the horizontal dims.

    Notes
    -----
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
            "allow_rechunk": True,
            "output_sizes": {"ky": ny, "kx": nx_pos},
            "meta": np.array((), dtype=np.float64),
        }
    )

    if "ky" not in power.dims or "kx" not in power.dims:
        power = power.rename({dims[0]: "ky", dims[1]: "kx"})

    return power


def _prep_bins(nx_fft: int, ny: int, dx: float, dy: float, nyquist=True):
    """
    Precompute bin index. Non-overlapping, variance-conserving radial bins.

    Parameters
    ----------
    nx_fft : int
        Spectral size along x (rFFT half-plane): Nx//2 + 1
    ny : int
        Spectral size along y: Ny
    dx : float
        Physical spacing along x (meters)
    dy : float
        Physical spacing along y (meters)
    nyquist : bool
        Whether to explicitly cut off at the Nyquist wavenumber.
    Returns
    -------
    centers : (n_bins,) float64
        Bin center wavenumbers (radians/meter), aligned to legacy spacing.
    bin_idx2d : (ny, nx_fft) int32
        2-D bin index array; -1 for out-of-bounds.
    Notes
    -----
      - rFFT-aware
      - kx is non-negative via rfftfreq(nx), while ky is symmetric
      - Δ = 2π / min(dx*Nx_full, dy*ny)  (legacy center spacing)
      - centers at nΔ (drop inner < Δ/2 ring), identical to legacy layout
    """
    # reconstruct physical Nx from rFFT spectral size
    nx = 2 * (nx_fft - 1)

    kx = np.fft.rfftfreq(nx, dx / (2 * np.pi))  # length nx (half-plane)
    ky = np.fft.fftshift(np.fft.fftfreq(ny, dy / (2 * np.pi)))  # length ny (centered)

    kh_grid = np.hypot(*np.meshgrid(kx, ky, indexing="xy")).astype(np.float64)

    delta = 2.0 * np.pi / min(dx * nx, dy * ny)

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
    bin_idx2d = np.digitize(kh_grid, edges, right=False) - 1
    bin_idx2d[(bin_idx2d < 0) | (bin_idx2d >= n_bins)] = -1

    return centers, bin_idx2d


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


def accumulate(da: xr.DataArray) -> xr.DataArray:
    """Cumulative integral toward low wavenumbers along ``k`` → ``wavenumber``."""
    sorted_da = da.sortby("wavenumber", ascending=False)
    return sorted_da.cumsum("wavenumber").sortby("wavenumber")


def isotropize(spectrum: xr.DataArray, dx: float, dy: float,
               nyquist: bool = True, cumulative: bool = False) -> xr.DataArray:
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
        y_dim, x_dim = get_spatial_dims(spectrum)

        spectrum = spectrum.rename({y_dim: "ky", x_dim: "kx"})

    # Get kappa bins and 2D bin index array
    nky, nkx = int(spectrum.sizes["ky"]), int(spectrum.sizes["kx"])

    # bins & index
    wavenumber, idx2d = _prep_bins(nkx, nky, dx, dy, nyquist=nyquist)
    idx2d = xr.DataArray(idx2d, dims=("ky", "kx"))

    # Apply binning of 2D to 1D spectrum
    spec1d = xr.apply_ufunc(
        _azimuthal_bincount,
        spectrum, idx2d, wavenumber.size,
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

    # assign coords and name: Should retain name from input if present
    spec1d = spec1d.rename(spec1d.name or "spectrum_1d")
    spec1d = spec1d.assign_coords(wavenumber=("wavenumber", wavenumber))

    if cumulative:
        spec1d = accumulate(spec1d)

    return spec1d


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
    u_term = scalar_cross_spectrum(vec1.sel(comp="u"), vec2.sel(comp="u"), norm)
    v_term = scalar_cross_spectrum(vec1.sel(comp="v"), vec2.sel(comp="v"), norm)
    return u_term + v_term


def compute_divergence(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """Horizontal divergence."""
    y_dim, x_dim = get_spatial_dims(u)
    return differentiate_metric(u, x_dim) + differentiate_metric(v, y_dim)


def compute_vorticity(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """Vertical vorticity."""
    y_dim, x_dim = get_spatial_dims(u)
    return differentiate_metric(v, x_dim) - differentiate_metric(u, y_dim)
