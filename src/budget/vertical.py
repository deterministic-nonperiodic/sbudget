"""
Vertical operators for Lorenz-staggered model grids (e.g. ICON).

Full-level fields (u, v, θ', π', ...) live on ``z`` (layer centres) and w lives on ``z_half``
(interfaces), with ``z[k]`` between ``z_half[k]`` and ``z_half[k + 1]`` (either orientation).

Using the model's native staggering instead of interpolating w to full levels and taking
centred differences there is much more accurate for vertical gradients of short vertical waves:
at 5 levels per vertical wavelength, ∂w/∂z is wrong by ~40% with interpolation + centred
differences and by ~6% with the compact difference below (dev/vertical_discretization_study.py).
Fluxes formed at half levels and differenced compactly to full levels also telescope exactly:
Σ_k Δz_k ∂F/∂z = F(top interface) − F(bottom interface) on any grid.
"""
from typing import Tuple

import numpy as np
import xarray as xr

FULL, HALF = "z", "z_half"


def is_lorenz_staggered(z_full: xr.DataArray, z_half: xr.DataArray) -> bool:
    """True if there is one more half level than full levels and each full level lies strictly
    between its two neighbouring half levels."""
    zf = np.asarray(z_full.values, dtype=float)
    zh = np.asarray(z_half.values, dtype=float)
    if zh.size != zf.size + 1 or zf.size < 2:
        return False
    lo, hi = np.minimum(zh[:-1], zh[1:]), np.maximum(zh[:-1], zh[1:])
    return bool(np.all((zf > lo) & (zf < hi)))


def _layer_thickness(z_half: xr.DataArray, z_full: xr.DataArray) -> xr.DataArray:
    """Signed z_half[k+1] - z_half[k] on full levels."""
    zh = np.asarray(z_half.values, dtype=float)
    return xr.DataArray(zh[1:] - zh[:-1], dims=FULL, coords={FULL: z_full.values})


def _upper_lower(da: xr.DataArray, z_full: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
    """The half-level values at z_half[k+1] and z_half[k], relabelled onto full level k."""
    hi = da.isel({HALF: slice(1, None)}).rename({HALF: FULL})
    lo = da.isel({HALF: slice(None, -1)}).rename({HALF: FULL})
    hi = hi.drop_vars(FULL, errors="ignore").assign_coords({FULL: z_full.values})
    lo = lo.drop_vars(FULL, errors="ignore").assign_coords({FULL: z_full.values})
    return hi, lo


def half_to_full_mean(da: xr.DataArray, z_full: xr.DataArray) -> xr.DataArray:
    """Average of the two half levels bounding each full level."""
    hi, lo = _upper_lower(da, z_full)
    return 0.5 * (hi + lo)


def diff_half_to_full(da: xr.DataArray, z_full: xr.DataArray) -> xr.DataArray:
    """Compact derivative of a half-level field on full levels:
    (a[k+1/2] - a[k-1/2]) / (z_half[k+1] - z_half[k])."""
    hi, lo = _upper_lower(da, z_full)
    return (hi - lo) / _layer_thickness(da[HALF], z_full)


def _bracketing(z_full: np.ndarray, z_half: np.ndarray):
    """Full-level indices (i0, i1 = i0 + 1) around every half level and the linear weight.
    Interior half level k lies between full levels k-1 and k; the end interfaces use the
    nearest pair (linear extrapolation)."""
    n = z_full.size
    k = np.arange(z_half.size)
    i0 = np.clip(k - 1, 0, n - 2)
    i1 = i0 + 1
    wgt = (z_half - z_full[i0]) / (z_full[i1] - z_full[i0])
    return i0, i1, wgt


def interp_full_to_half(da: xr.DataArray, z_half: xr.DataArray) -> xr.DataArray:
    """Linear interpolation (extrapolation at the end interfaces) of a full-level field to all
    half levels."""
    zf = np.asarray(da[FULL].values, dtype=float)
    zh = np.asarray(z_half.values, dtype=float)
    i0, i1, wgt = _bracketing(zf, zh)
    a0 = da.isel({FULL: xr.DataArray(i0, dims=HALF)})
    a1 = da.isel({FULL: xr.DataArray(i1, dims=HALF)})
    w = xr.DataArray(wgt, dims=HALF)
    out = a0 + w * (a1 - a0)
    return out.drop_vars(FULL, errors="ignore").assign_coords({HALF: zh})


def diff_full_to_half(da: xr.DataArray, z_half: xr.DataArray) -> xr.DataArray:
    """Compact derivative of a full-level field on half levels:
    (a[k] - a[k-1]) / (z[k] - z[k-1]); the end interfaces take the nearest interior value."""
    zf = np.asarray(da[FULL].values, dtype=float)
    zh = np.asarray(z_half.values, dtype=float)
    i0, i1, _ = _bracketing(zf, zh)
    a0 = da.isel({FULL: xr.DataArray(i0, dims=HALF)})
    a1 = da.isel({FULL: xr.DataArray(i1, dims=HALF)})
    dz = xr.DataArray(zf[i1] - zf[i0], dims=HALF)
    out = (a1 - a0) / dz
    return out.drop_vars(FULL, errors="ignore").assign_coords({HALF: zh})
