from typing import Tuple

import numpy as np
import xarray as xr

from .cf_coords import get_spatial_dims, is_geographic_grid, _coord_is_degrees
from .constants import epsilon
from .numeric_tools import domain_mean

# -----------------------------
# Global spectral configuration
# -----------------------------
SPECTRAL_CFG = {
    "taper": {
        "enabled": True,
        # one of {"hann","hamming","cosine","none"}
        "x": "hanning",
        "y": "hanning",
    },
    "area_weighting": {
        "enabled": True,  # enable sqrt(area) weighting on lon–lat grids
        "normalize_row_mean": True,  # divide by <cos(phi)> so Parseval matches
    },
    "detrend": {
        "remove_mean": "global",  # one of {"global","zonal","none"}
    },
}


# -----------------------------
# Utilities (internal)
# -----------------------------
def _detrend_mean(field: xr.DataArray) -> xr.DataArray:
    """Remove mean from field according to configured mode."""

    mode = SPECTRAL_CFG.get("detrend", {}).get("remove_mean", "none")

    y_dim, x_dim = get_spatial_dims(field)
    if mode == "global":
        return field - domain_mean(field)
    if mode == "zonal":  # remove zonal means (kx=0)
        return field - field.mean(dim=x_dim)
    return field


def _make_taper(n: int, kind: str) -> np.ndarray:
    if kind == "none":
        return np.ones(n, dtype=np.float64)
    if kind == "hanning":
        return np.hanning(n)
    if kind == "hamming":
        return np.hamming(n)
    if kind == "cosine":
        return np.sin(np.pi * (np.arange(n) + 0.5) / n)
    # default fallback
    return np.ones(n, dtype=np.float64)


def _taper_2d(field: xr.DataArray) -> tuple[xr.DataArray, float]:
    """Return separable 2-D taper W(lat,lon) and its mean-square for normalization.

    If disabled, returns W=1 and ms=1.
    """
    if not SPECTRAL_CFG.get("taper", {}).get("enabled", True):
        return xr.ones_like(field), 1.0

    y_dim, x_dim = get_spatial_dims(field)
    ny, nx = field.sizes[y_dim], field.sizes[x_dim]
    ty = _make_taper(ny, SPECTRAL_CFG["taper"].get("y", "hann"))
    tx = _make_taper(nx, SPECTRAL_CFG["taper"].get("x", "hann"))
    ms = float((ty ** 2).mean() * (tx ** 2).mean())  # <Ty^2>*<Tx^2>

    W = xr.DataArray(ty[:, None] * tx[None, :], dims=(y_dim, x_dim),
                     coords={y_dim: field[y_dim], x_dim: field[x_dim]})
    return W, ms


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


def _apply_pre_fft_weights(field: xr.DataArray) -> tuple[xr.DataArray, float]:
    """Apply taper and sqrt(area) weighting; return weighted field and taper ms.

    The sqrt(area) factor is row-wise and broadcast across x; the 2-D taper is separable.
    We return `ms = <Ty^2>*<Tx^2>` to undo taper energy loss after power/cross-power.
    """

    # Detrend data first
    field = _detrend_mean(field)

    # Taper and area weighting
    W, ms = _taper_2d(field)
    s = _sqrt_area_weight(field)
    W_full = W * s.broadcast_like(W)

    return field * W_full, ms


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
    field_w, taper_ms = _apply_pre_fft_weights(field)

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
    if taper_ms != 0:
        power = power / taper_ms

    if "ky" not in power.dims or "kx" not in power.dims:
        power = power.rename({dims[0]: "ky", dims[1]: "kx"})

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

    f1_w, ms1 = _apply_pre_fft_weights(field1)
    f2_w, ms2 = _apply_pre_fft_weights(field2)
    taper_ms = (ms1 * ms2) ** 0.5 if (ms1 and ms2) else 1.0

    def _cross(a, b):
        s1 = _real_fft2_shifted(a)
        s2 = _real_fft2_shifted(b)
        return (s1.conj() * s2).real

    power = xr.apply_ufunc(
        _cross, f1_w, f2_w,
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

    if taper_ms != 0:
        power = power / taper_ms

    if "ky" not in power.dims or "kx" not in power.dims:
        power = power.rename({dims[0]: "ky", dims[1]: "kx"})

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

    kh_grid = horizontal_wavenumber_magnitude(nx, ny, dx, dy)

    delta = 2.0 * np.pi / min(dx * nx, dy * ny)

    # Explicit Nyquist cutoff (if requested)
    nyq = np.pi / max(dx, dy)
    k_cut = min(nyq, float(kh_grid.max())) if nyquist else float(kh_grid.max())

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
