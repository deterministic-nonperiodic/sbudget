from typing import Tuple

import numpy as np
import xarray as xr
from scipy.signal.windows import dpss as _dpss

from .cf_coords import get_spatial_dims, is_geographic_grid, _coord_is_degrees
from .constants import epsilon
from .numeric_tools import domain_mean

# -----------------------------
# Global spectral configuration
# -----------------------------
SPECTRAL_CFG = {
    "taper": {
        "enabled": True,
        # Per-axis taper specification; accepts string or dict.
        # String options: {"hann","hanning","hamming","cosine","none","dpss"}
        # Dict form for DPSS (recommended for FFTs):
        #   {"kind":"dpss", "NW": 2.5, "Kmax": 1, "periodic": True}
        #   - NW: time-bandwidth product (>= 0.5)
        #   - Kmax: number of DPSS tapers to generate (we use the first to keep API unchanged)
        #   - periodic=True uses FFT-friendly tapers (SciPy sym=False)
        "x": "hanning",
        "y": "hanning",
    },
    "area_weighting": {
        "enabled": True,  # enable sqrt(area) weighting on lon–lat grids
        "normalize_row_mean": True  # divide by <cos(phi)> so mean-squared window is consistent
    },
    "detrend": {
        "remove_mean": "global",  # one of {"global","zonal","none"}
    },
}


# -----------------------------
# Utilities (internal)
# -----------------------------
def _get_weighted_mean(field: xr.DataArray) -> xr.DataArray:
    """Remove mean from field according to configured mode."""
    mode = SPECTRAL_CFG.get("detrend", {}).get("remove_mean", "none")
    y_dim, x_dim = get_spatial_dims(field)
    if mode == "global":
        return domain_mean(field)
    if mode == "zonal":  # remove zonal means (kx=0)
        return field.mean(dim=x_dim)

    return xr.zeros_like(field)


def _make_taper(n: int, kind_or_spec) -> np.ndarray:
    """Build a 1D window of length ``n``.

    Accepts either a string shorthand ("hann"/"hanning", "hamming", "cosine",
    "none", "dpss") or a dict spec for DPSS, e.g. {"kind":"dpss", "NW":2.5,
    "Kmax":1, "periodic":True}.

    Notes
    -----
    - For DPSS we return **only the first taper** to keep the public API unchanged.
      (Multitaper averaging would require expanding a new dimension.)
    - ``periodic=True`` yields FFT-friendly tapers (SciPy's ``sym=False``), which
      avoid endpoint duplication under the FFT's periodic extension.
    """
    # Dict form (e.g., DPSS detailed spec)
    if isinstance(kind_or_spec, dict):
        kind = (kind_or_spec.get("kind") or kind_or_spec.get("name") or "").lower()
        if kind == "dpss":
            if _dpss is None:
                raise RuntimeError("DPSS requested but SciPy is not available. Install scipy>=1.5.")
            nw = float(kind_or_spec.get("NW", 2.5))
            Kmax = int(kind_or_spec.get("Kmax", 1))
            periodic = bool(kind_or_spec.get("periodic", True))
            tapers = _dpss(M=n, NW=nw, Kmax=max(1, Kmax), sym=not periodic)
            return np.asarray(tapers[0], dtype=np.float64)
        # fall through to string handling if unknown dict kind
        # no-op
    else:
        kind = str(kind_or_spec).lower()
        if kind in {"hann", "hanning"}:
            return np.hanning(n)
        if kind == "hamming":
            return np.hamming(n)
        if kind == "cosine":
            return np.sin(np.pi * (np.arange(n) + 0.5) / n)
        if kind == "dpss":
            # sensible defaults if only "dpss" string is provided
            tapers = _dpss(M=n, NW=2.5, Kmax=1, sym=False)
            return np.asarray(tapers[0], dtype=np.float64)
        if kind == "none":
            return np.ones(n, dtype=np.float64)

    # default fallback
    return np.ones(n, dtype=np.float64)


def _taper_2d(field: xr.DataArray) -> tuple[xr.DataArray, float]:
    """Return separable 2-D taper W(lat,lon) and its mean-square for normalization.

    If disabled, returns W=1 and ms=1.
    """
    y_dim, x_dim = get_spatial_dims(field)
    ny, nx = field.sizes[y_dim], field.sizes[x_dim]

    if not SPECTRAL_CFG.get("taper", {}).get("enabled", False):
        return xr.DataArray(np.ones((ny, nx)), dims=(y_dim, x_dim)), 1.0

    taper_cfg = SPECTRAL_CFG.get("taper", {})
    ty_spec = taper_cfg.get("y", "hann")
    tx_spec = taper_cfg.get("x", "hann")

    ty = _make_taper(ny, ty_spec)
    tx = _make_taper(nx, tx_spec)

    # <Ty^2>*<Tx^2> (simple, unweighted mean)
    ms = float((ty ** 2).mean() * (tx ** 2).mean())

    w = xr.DataArray(ty[:, None] * tx[None, :], dims=(y_dim, x_dim),
                     coords={y_dim: field[y_dim], x_dim: field[x_dim]})
    return w, ms


def _sqrt_area_weight(field: xr.DataArray) -> xr.DataArray:
    """Return row-wise sqrt(area) weighting s(phi) for lon–lat grids; else ones.

    s(phi) = sqrt( cos(phi) / <cos(phi)> )  (if enabled)
    """
    y_dim, x_dim = get_spatial_dims(field)
    x_coord = field.coords.get(x_dim)
    y_coord = field.coords.get(y_dim)

    if not (SPECTRAL_CFG.get("area_weighting", {}).get("enabled", True) and
            x_coord is not None and y_coord is not None and is_geographic_grid(x_coord, y_coord)):
        return xr.DataArray(np.ones((field.sizes[y_dim],), dtype=np.float64),
                            dims=(y_dim,), coords={y_dim: field[y_dim]})

    phi = np.deg2rad(y_coord) if _coord_is_degrees(y_coord) else y_coord
    cos_phi = np.cos(phi).clip(min=epsilon)

    if SPECTRAL_CFG["area_weighting"].get("normalize_row_mean", True):
        cos_phi = cos_phi / float(cos_phi.mean())

    s = np.sqrt(cos_phi)
    return s.rename(y_dim)


def _apply_weighting(field: xr.DataArray) -> tuple[xr.DataArray, float, xr.DataArray]:
    """Apply taper and sqrt(area) weighting, detrend, and return weighted field, taper ms, and (mean)^2."""

    # Get window and sqrt(area) weighting
    window_1d, taper_ms = _taper_2d(field)
    window_2d = window_1d * _sqrt_area_weight(field)

    # Detrending (Restored for leakage control)
    mode = SPECTRAL_CFG.get("detrend", {}).get("remove_mean", "none")

    field_zero = field.copy()

    drop_dims = get_spatial_dims(field)
    mu_restore = domain_mean(xr.zeros_like(field).drop_vars(drop_dims, errors='ignore'))

    if mode == "global":
        ms_total_window = domain_mean(window_2d ** 2)
        ms_total_window = float(ms_total_window)
        ms_total_window = xr.where(ms_total_window < epsilon, 1.0, ms_total_window)

        # T^2-weighted mean: mu_w2 = <f * T^2> / <T^2>
        mu_w2 = domain_mean(field * window_2d ** 2) / ms_total_window

        # Remove the weighted mean
        field_zero = field - mu_w2

        # Calculate the energy to restore: the square of the area-weighted mean of the ORIGINAL field
        mu_restore = domain_mean(field)

    # Return detrended, windowed field, taper mean-square, and the mean-square to restore
    return field_zero * window_2d, taper_ms, mu_restore


# -----------------------------------------------------------------------------
# Spectral primitives (unchanged API; now read global options)
# -----------------------------------------------------------------------------

def _real_fft2_shifted(a) -> np.ndarray:
    """2-D real-input FFT over the last two axes with ky-shift and rFFT half-plane weighting.

    - rFFT on (y, x): last axis returns non-negative kx (half-plane).
    - Legacy scaling when norm is None: divide by Ny * Nx.
    - Apply √2 on interior kx columns so that |F|^2 doubles there (DC and Nyquist stay 1).
    """
    a = np.asanyarray(a).real

    # rFFT core: half-plane along x. forward normalization to fulfill Parseval's theorem
    a_sc = np.fft.rfftn(a, axes=(-2, -1), norm="forward")
    # center ky only (symmetric)
    a_sc = np.fft.fftshift(a_sc, axes=(-2,))

    # --- rFFT half-plane amplitude weights (√2 on interior kx) ---
    nx = a.shape[-1]
    nkx = nx // 2 + 1
    has_nyq = (nx % 2) == 0  # even Nx has explicit Nyquist column at the end
    idx = np.arange(nkx)

    # 1 at DC (i==0) and Nyquist (if present), √2 elsewhere
    weights = np.where(idx == 0, 1.0, np.where(has_nyq & (idx == nkx - 1), 1.0, np.sqrt(2.0)))

    return a_sc * weights


def horizontal_wavenumbers(nx: int, ny: int, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return 1D horizontal wavenumber components (ky, kx) for the last two spatial axes,
    assuming the Fourier transform convention used for rFFT.

    Args:
        nx (int): Size of the x-dimension.
        ny (int): Size of the y-dimension.
        dx (float): Grid spacing in x (spatial units).
        dy (float): Grid spacing in y (spatial units).

    Returns:
        Tuple[np.ndarray, np.ndarray]: (kx, ky) arrays.
    """
    # kx: Real FFT half-plane, non-negative wavenumbers (length nx//2 + 1)
    # The factor of 2*pi converts spatial units to wavenumber (k = 2*pi / L).
    # np.fft.rfftfreq handles the half-plane correctly.
    kx = np.fft.rfftfreq(nx, dx / (2 * np.pi))

    # ky: Full FFT, shifted (centered at zero) for a standard 2D representation (length ny)
    ky = np.fft.fftshift(np.fft.fftfreq(ny, dy / (2 * np.pi)))

    return kx, ky


def horizontal_wavenumber_magnitude(nx: int, ny: int, dx: float, dy: float) -> np.ndarray:
    """
    Return the 2D horizontal wavenumber magnitude (kh) grid.

    kh = sqrt(kx^2 + ky^2)

    Args:
        nx (int): Size of the x-dimension.
        ny (int): Size of the y-dimension.
        dx (float): Grid spacing in x (spatial units).
        dy (float): Grid spacing in y (spatial units).

    Returns:
        np.ndarray: 2D array of wavenumber magnitudes (kh_grid) with shape (ny, nx//2 + 1).
    """
    # Use the helper function to get the 1D wavenumber components
    kx, ky = horizontal_wavenumbers(nx, ny, dx, dy)

    # Create 2D grids of kx and ky, and calculate the magnitude
    # np.meshgrid with indexing="xy" ensures the output shape is (len(ky), len(kx))
    kh_grid = np.hypot(*np.meshgrid(kx, ky, indexing="xy")).astype(np.float64)

    return kh_grid


def truncate(spectrum, truncation_scale=None):
    """Truncate a 1-D isotropic spectrum at the given truncation scale."""

    if truncation_scale is None:
        return spectrum

    if np.isscalar(truncation_scale):
        truncation_scale = float(truncation_scale)
        return spectrum.sel({"wavenumber": slice(2.0 * np.pi / truncation_scale)})

    raise ValueError("Unknown type for truncation_scale. Expecting a float.")


# -----------------------------
# Spectra (scalar / cross / vector) with pre-FFT weighting (unchanged API)
# -----------------------------

def scalar_spectrum(field: xr.DataArray) -> xr.DataArray:
    """Return 2-D power spectrum |F(k)|^2 over the last two spatial axes.

    rFFT path:
      - _real_fft2_shifted returns (ky shifted, kx non-negative) complex spectrum
      - we compute |F|^2
      - interior-kx doubling is already encoded by √2 amplitude in _real_fft2_shifted
    """
    dims = get_spatial_dims(field)  # e.g., ("y","x")
    ny, nx = field.sizes[dims[0]], field.sizes[dims[1]]
    nx_pos = nx // 2 + 1

    # Pre-FFT: taper and sqrt(area) weighting
    field_w, ms_normalize, mu_removed = _apply_weighting(field)

    def _pow(a):
        spec = _real_fft2_shifted(a)
        pwr = (spec.conj() * spec).real  # power
        return pwr

    power = xr.apply_ufunc(
        _pow, field_w,
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

    # Undo taper energy loss (Parseval consistency). Area sqrt-weight is self-normalizing.
    if ms_normalize != 0:
        power = power / ms_normalize

    # ensure dimensions exist
    if "ky" not in power.dims or "kx" not in power.dims:
        power = power.rename({dims[0]: "ky", dims[1]: "kx"})

    # Restore field mean in case detrended fields
    mean_product = mu_removed * mu_removed
    if mean_product.any():
        # Identify the coordinate values for k=0
        kx_dc = power['kx'].values[0]
        ky_dc = power['ky'].values[ny // 2]  # Use the middle index for the shifted axis

        # create index mask
        kdc_mask = (power['kx'] == kx_dc) & (power['ky'] == ky_dc)

        # Add the mean product only where the mask is True.
        power = power + mean_product.where(kdc_mask, other=0.0)

    return power


def scalar_cross_spectrum(field1: xr.DataArray, field2: xr.DataArray) -> xr.DataArray:
    """Return 2-D cross-spectrum Re{F1*(k) F2(k)} over the horizontal dims.

    Notes
    -----
    rFFT path (half-plane in kx, ky shifted). Interior-kx ×2 is implicitly handled
    via the √2 amplitude in _real_fft2_shifted.
    """
    dims = get_spatial_dims(field1)
    ny, nx = field1.sizes[dims[0]], field1.sizes[dims[1]]
    nx_pos = nx // 2 + 1

    field1_w, ms1, mu_removed_1 = _apply_weighting(field1)
    field2_w, ms2, mu_removed_2 = _apply_weighting(field2)

    def _cross(a, b):
        s1 = _real_fft2_shifted(a)
        s2 = _real_fft2_shifted(b)
        return (s1.conj() * s2).real

    power = xr.apply_ufunc(
        _cross, field1_w, field2_w,
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

    # Apply window correction
    ms_normalize = (ms1 * ms2) ** 0.5 if (ms1 > epsilon and ms2 > epsilon) else 1.0

    if ms_normalize != 0:
        power = power / ms_normalize

    # ensure dimensions exist
    if "ky" not in power.dims or "kx" not in power.dims:
        power = power.rename({dims[0]: "ky", dims[1]: "kx"})

    # Restore field mean in case detrended fields
    mean_product = mu_removed_1 * mu_removed_2
    if mean_product.any():
        # Identify the coordinate values for k=0
        kx_dc = power['kx'].values[0]
        ky_dc = power['ky'].values[ny // 2]  # Use the middle index for the shifted axis

        # create index mask
        kdc_mask = (power['kx'] == kx_dc) & (power['ky'] == ky_dc)

        # Add the mean product only where the mask is True.
        power = power + mean_product.where(kdc_mask, other=0.0)

    return power


def vector_cross_spectrum(vec1: xr.DataArray, vec2: xr.DataArray) -> xr.DataArray:
    """Sum of cross-spectra of matching components of two 2D vectors.

    Returns ⟨u1, u2⟩ + ⟨v1, v2⟩ in spectral space.
    """
    if "comp" not in vec1.dims or "comp" not in vec2.dims:
        raise ValueError("vector_cross_spectrum expects inputs with 'comp' dimension")
    u_term = scalar_cross_spectrum(vec1.sel(comp="u"), vec2.sel(comp="u"))
    v_term = scalar_cross_spectrum(vec1.sel(comp="v"), vec2.sel(comp="v"))
    return u_term + v_term


def _prep_bins(nx_fft: int, ny: int, dx: float, dy: float, nyquist=True, include_dc=False):
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

    kh_grid = horizontal_wavenumber_magnitude(nx, ny, dx, dy)

    delta = 2.0 * np.pi / min(dx * nx, dy * ny)

    # Explicit Nyquist cutoff (if requested)
    nyq = np.pi / max(dx, dy)
    k_cut = min(nyq, float(kh_grid.max())) if nyquist else float(kh_grid.max())

    # build edges for k >= Δ/2
    start = 0.5 * delta
    n_bins = int(np.floor((k_cut - start) / delta + epsilon))
    if (k_cut - start) - n_bins * delta > epsilon:
        n_bins += 1

    # bin edges and centers
    edges = start + np.arange(0, n_bins, dtype=np.float64) * delta
    edges = np.concatenate([edges, [max(k_cut + epsilon * delta, start + n_bins * delta)]])

    centers = 0.5 * (edges[:-1] + edges[1:])
    centers = delta * np.rint(centers / delta)

    # 2-D bin index array
    bin_idx2d = np.digitize(kh_grid, edges, right=False) - 1

    if include_dc:
        # Insert DC bin at index 0 for the single pixel (ky0, kx0)
        ky0 = ny // 2  # ky is centered
        # Shift existing non-negative rings up by +1
        bin_idx2d += (bin_idx2d >= 0)
        # Place DC at 0
        bin_idx2d[ky0, 0] = 0
        # Prepend DC center 0.0
        centers = np.concatenate([[0.0], centers])

    n_bins = centers.size
    # **Sanitize**: anything outside [0, n_bins-1] → -1
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
               nyquist: bool = True, cumulative: bool = False,
               include_dc: bool = False) -> xr.DataArray:
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
    wavenumber, idx2d = _prep_bins(nkx, nky, dx, dy, nyquist=nyquist, include_dc=include_dc)
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
