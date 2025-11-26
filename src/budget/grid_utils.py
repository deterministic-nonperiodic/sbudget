"""
increments.py

Utilities for generating scale increments, separation vectors, angular
weights, and r-bin masks for structure-function analysis on Cartesian
or geographic grids.

This module guarantees full consistency with compensated-kernel
integration rules.
"""

from typing import Optional, Union, Tuple, List

import numpy as np
import xarray as xr
from pint import Quantity
from pyproj import Geod

from .cf_coords import is_geographic_grid, _is_global_longitude
from .constants import epsilon

_GEODE = Geod(ellps="WGS84")


# ============================================================================
# Section 1 — Small Utilities (Internal functions omitted for brevity)
# ============================================================================

def _ensure_1d_numeric(arr) -> Optional[np.ndarray]:
    """ Convert arr to a 1D float numpy array if arr is not None. """
    if arr is None:
        return None
    arr = np.atleast_1d(arr)
    return arr.astype(float, copy=False)


# ============================================================================
# Section 2 — Grid Metrics (Internal functions omitted for brevity)
# ============================================================================

def _get_spacing(
        coord: xr.DataArray,
        center: float,
        use_geode: bool,
        axis: str,
) -> float:
    """ Compute approximate grid spacing between first two coordinate entries. """
    if not use_geode:
        return float(np.abs(np.median(np.diff(coord.values))))

    if axis == "x":
        _, _, dist = _GEODE.inv(coord[0].item(), center, coord[1].item(), center)
    else:
        _, _, dist = _GEODE.inv(center, coord[0].item(), center, coord[1].item())

    return float(dist)


def infer_boundary_conditions(x_coord: xr.DataArray, **kwargs) -> tuple[str, str]:
    """ Infer boundary conditions (periodic for global domains). """
    x_boundary = kwargs.get("x_coord_boundary", "reflect")

    if x_boundary != "periodic" and _is_global_longitude(x_coord):
        print(f"[boundary_condition] Global domain detected; "
              f"overriding user defined {x_boundary} with {'periodic'}")
        x_boundary = "periodic"

    y_boundary = kwargs.get("y_coord_boundary", "reflect")
    return x_boundary, y_boundary


def compute_geodesic_domain_lengths(
        x: xr.DataArray,
        y_min: float,
        y_max: float,
        x_center: float,
        y_center: float,
) -> Tuple[float, float]:
    """ Compute domain extent (Lx, Ly) in meters on a geographic grid. """

    if _is_global_longitude(x):
        _, _, half = _GEODE.inv(0, y_center, 180, y_center)
        lx = 2 * half
    else:
        x_min, x_max = float(x.min()), float(x.max())
        _, _, lx = _GEODE.inv(x_min, y_center, x_max, y_center)

    _, _, ly = _GEODE.inv(x_center, y_min, x_center, y_max)
    return float(lx), float(ly)


def compute_grid_metrics(
        x: xr.DataArray,
        y: xr.DataArray,
) -> Tuple[bool, Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """ Compute grid metrics (geode flag, resolution, center, domain size). """
    use_geode = is_geographic_grid(x, y)

    # grid center
    x_center, y_center = float(x.mean()), float(y.mean())

    # grid spacing
    dx = max(_get_spacing(x, y_center, use_geode, axis="x"), epsilon)
    dy = max(_get_spacing(y, x_center, use_geode, axis="y"), epsilon)

    # domain size
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    if use_geode:
        lx, ly = compute_geodesic_domain_lengths(x, y_min, y_max, x_center, y_center)
    else:
        lx, ly = (x_max - x_min, y_max - y_min)

    return use_geode, (dx, dy), (x_center, y_center), (lx, ly)


# ============================================================================
# Section 3 — Kernel Grid Generation (r-values and scales)
# ============================================================================

def _sanitize_scales(
        scales: Optional[Union[List[float], np.ndarray, float]],
        r_step: float,
        verbose: bool = False,
) -> Optional[np.ndarray]:
    """ Sanitize user-provided scale list. """
    if scales is None:
        return None

    arr = _ensure_1d_numeric(scales)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]

    if arr.size == 0:
        if verbose:
            print(f"[GRID] All provided scales invalid → using r_step={r_step:.2f} m.")
        return np.array([r_step], dtype=float)

    arr = np.maximum(arr, r_step)
    return np.unique(arr)


def compute_kernel_grid(
        grid_res: Tuple[float, float],
        domain_size: Tuple[float, float],
        scales: Optional[Union[float, Quantity, List[float], np.ndarray]] = None,
        max_distance: Optional[Union[float, Quantity]] = None,
        verbose: bool = False,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Compute the radial separation grid `r` and the filter-scale coordinate `scale`.
    """
    r_step = max(grid_res)
    half_domain = min(domain_size) / 2.0

    # Normalize max_distance
    max_r = None
    if max_distance is not None:
        if isinstance(max_distance, Quantity):
            max_r = float(max_distance.to("meter").magnitude)
        else:
            max_r = float(max_distance)
        if max_r <= 0: raise ValueError("max_distance must be positive.")

    # Sanitize scales
    scales_arr = _sanitize_scales(scales, r_step, verbose)
    max_scale = float(scales_arr.max()) if scales_arr is not None else None

    # Determine r_max (Effective max radial separation)
    eff_max_r = half_domain
    if max_r is not None:
        eff_max_r = min(eff_max_r, max_r)
    if scales_arr is not None:
        eff_max_r = min(eff_max_r, 2 * max_scale)  # Kernel support condition

    if eff_max_r < r_step:
        raise ValueError(f"Effective r_max ({eff_max_r:.2f}) < r_step ({r_step:.2f}).")

    # Create r-values
    r_values = np.arange(r_step, eff_max_r + r_step * 0.51, r_step)
    if r_values.size == 0: raise ValueError("Failed to generate r-values.")

    # Final scale trimming
    if scales_arr is None:
        final_scales = r_values.copy()
    else:
        mask = (scales_arr >= r_step) & (scales_arr <= eff_max_r)
        final_scales = scales_arr[mask]

        if final_scales.size == 0:
            max_requested = scales_arr.max()
            fallback = max(min(max_requested, eff_max_r), r_step)
            if verbose: print(f"[GRID] Requested scales outside of usable range → "
                              f"using {fallback} m")
            final_scales = np.array([fallback], dtype=float)

    # Create kernel coordinates
    r_da = xr.DataArray(r_values, name="r", dims="r",
                        attrs={"standard_name": "radial_distance",
                               "long_name": "Separation distance", "units": "m"})

    scale_da = xr.DataArray(final_scales, name="scale", dims="scale",
                            attrs={"standard_name": "horizontal_scale",
                                   "long_name": "horizontal scale", "units": "m"})

    return r_da, scale_da


# ============================================================================
# Section 4 — Shift Grid Construction
# ============================================================================

def build_shift_grid(
        max_r: float,
        grid_res: Tuple[float, float]
) -> Tuple[xr.DataArray, xr.DataArray]:
    """ Construct integer shift grids and their 1D coordinates. """
    max_nx = int(np.ceil(max_r / grid_res[0]))
    max_ny = int(np.ceil(max_r / grid_res[1]))

    # Grid search space centered at 0
    nx_values = np.arange(-max_nx, max_nx + 1, 1, dtype=int)
    ny_values = np.arange(-max_ny, max_ny + 1, 1, dtype=int)

    nx = xr.DataArray(nx_values, dims="nx")
    ny = xr.DataArray(ny_values, dims="ny")

    return nx, ny


# ============================================================================
# Section 5 — Distance and Angle Computation
# ============================================================================

def compute_cartesian_distance_angle(
        nx_da: xr.DataArray,
        ny_da: xr.DataArray,
        grid_res: Tuple[float, float],
) -> Tuple[xr.DataArray, xr.DataArray]:
    """ Compute distances and angles from index offsets on Cartesian grids. """
    dx, dy = grid_res
    dxg = nx_da * dx
    dyg = ny_da * dy

    dist = np.sqrt(dxg ** 2 + dyg ** 2)
    ang = np.arctan2(dyg, dxg)

    return (
        xr.DataArray(dist, dims=("ny", "nx"), name="distance_grid"),
        xr.DataArray(ang, dims=("ny", "nx"), name="angle_grid"),
    )


def compute_geodesic_distance_angle(
        nx_da: xr.DataArray,
        ny_da: xr.DataArray,
        grid_res: Tuple[float, float],
        center: Tuple[float, float],
) -> Tuple[xr.DataArray, xr.DataArray]:
    """ Compute distances and angles using geodesic geometry. """
    dx, dy = grid_res
    lon0, lat0 = center

    dxm = nx_da * dx
    dym = ny_da * dy
    approx_dist = np.sqrt(dxm ** 2 + dym ** 2)

    cart_ang = np.arctan2(dym, dxm)
    az_deg = 90 - np.rad2deg(cart_ang)

    lon_f, lat_f, _ = _GEODE.fwd(
        np.full_like(approx_dist, lon0),
        np.full_like(approx_dist, lat0),
        az_deg,
        approx_dist,
    )

    az1, _, dist = _GEODE.inv(
        np.full_like(lon_f, lon0),
        np.full_like(lat_f, lat0),
        lon_f,
        lat_f,
    )

    math_angle = np.deg2rad(90 - az1)

    return (
        xr.DataArray(dist, dims=("ny", "nx"), name="distance_grid"),
        xr.DataArray(math_angle, dims=("ny", "nx"), name="angle_grid"),
    )


def compute_distance_angle(
        nx_da: xr.DataArray,
        ny_da: xr.DataArray,
        grid_res: Tuple[float, float],
        use_geode: bool,
        center: Tuple[float, float],
) -> Tuple[xr.DataArray, xr.DataArray]:
    """ Dispatch Cartesian or geodesic distance calculation. """
    # 2D grids with dims ("ny", "nx")
    ny_grid, nx_grid = xr.broadcast(ny_da, nx_da)

    if use_geode:
        return compute_geodesic_distance_angle(nx_grid, ny_grid, grid_res, center)

    # fallback: Cartesian
    return compute_cartesian_distance_angle(nx_grid, ny_grid, grid_res)


# ============================================================================
# Section 6 — Angle Weights
# ============================================================================

def compute_angle_weights(angle_vals: np.ndarray) -> xr.DataArray:
    """ Compute normalized weights for each angle based on multiplicity. """
    flat = angle_vals.flatten().round(10)
    uniq, counts = np.unique(flat, return_counts=True)
    inv = {u: 1.0 / c for u, c in zip(uniq, counts)}

    weights = np.vectorize(inv.get)(angle_vals.round(10))
    return xr.DataArray(weights, dims=("ny", "nx"), name="angle_weight")


# ============================================================================
# Section 7 — r-Bin Mask
# ============================================================================

def compute_r_mask(
        distances: np.ndarray,
        r_values: np.ndarray,
        nx: xr.DataArray,
        ny: xr.DataArray,
) -> xr.DataArray:
    """ Build a boolean mask selecting which shifts fall into which r-bin. """
    r_step = np.diff(r_values)[0] if r_values.size > 1 else r_values[0]  # Handle single r-step case

    lower = r_values[:, None, None] - r_step / 2
    upper = r_values[:, None, None] + r_step / 2

    mask = (distances[None] >= lower) & (distances[None] < upper)

    coords = {"r": r_values, "ny": ny, "nx": nx}
    return xr.DataArray(mask, dims=("r", "ny", "nx"), coords=coords, name="r_mask")


# ============================================================================
# Section 8 — Post-Filtering Scale Adjustment
# ============================================================================

def adjust_scales_after_r_filter(
        scale_da: xr.DataArray,
        r_da: xr.DataArray,
        verbose: bool = False,
) -> xr.DataArray:
    """ Adjust scale coordinate after directional filtering of r. """

    r_min = float(r_da.min())
    r_max = float(r_da.max())
    kernel_support = r_max / 2.0

    # Trim to r-range and apply kernel support condition
    mask = (scale_da >= r_min) & (scale_da <= r_max) & (scale_da <= kernel_support)
    masked_scale = scale_da.where(mask, drop=True).values

    #  Deduplicate
    if masked_scale.size > 1:
        masked_scale = np.unique(masked_scale)

    # 3. Fallback: picking the best possible scale
    if masked_scale.size == 0:
        original = float(scale_da.max())
        fallback = max(min(original, kernel_support), r_min)
        masked_scale = [fallback]

        if verbose:
            print(f"[GRID] No valid scales after filtering; fallback scale={fallback:.2f} m.")

    # recover metadata
    return xr.DataArray(masked_scale, dims="scale", name="scale", attrs=scale_da.attrs)


# ============================================================================
# Section 9 — Directional Coverage Filter
# ============================================================================

def filter_by_directional_coverage(ds: xr.Dataset, min_valid_shifts: int = 10) -> np.ndarray:
    """ Determine which r-bins have enough directional sampling. """
    counts = ds["mask"].sum(("ny", "nx")).values
    return counts >= min_valid_shifts


# ============================================================================
# Section 10 — Main High-Level Function: scale_increments (ORCHESTRATOR)
# ============================================================================

def summarize_scales(values: np.ndarray, max_items: int = 7) -> str:
    """ Summarize scale values into a compact string representation. """
    values = np.asarray(values)

    if values.size <= max_items:
        return "[" + ", ".join(f"{v:g}" for v in values) + "]"

    head = "[" + ", ".join(f"{v:g}" for v in values[:2])
    tail = ", ".join(f"{v:g}" for v in values[-2:]) + "]"

    return head + ", ..., " + tail


def scale_increments(
        x_coord: xr.DataArray,
        y_coord: xr.DataArray,
        **kwargs,
) -> xr.Dataset:
    """
    Compute geometric separation quantities required for structure-function
    kernels by orchestrating all specialized helper functions.

    This function ensures full consistency with compensated integration rules:
        r_max ≥ 2*scale
    even after directional filtering.

    Parameters
    ----------
    x_coord, y_coord : xr.DataArray
        1D coordinate arrays.
    **kwargs :
        scales : float or array-like
            User-defined filter scales.
        max_r : float or Quantity
            Maximum separation limit.
        resolution_factor : int
            Shift-grid density factor.
        min_valid_shifts : int
            Required directional sampling per r-bin.
        verbose : bool
            Print diagnostics.

    Returns
    -------
    xr.Dataset
        Dataset containing: r, scale, mask, distance_grid, angle_grid, etc.
    """
    verbose = kwargs.get("verbose", False)
    min_shifts = kwargs.get("min_valid_shifts", 10)
    res_factor = kwargs.get("resolution_factor", 1)

    # Validation & Grid Metrics
    if not isinstance(x_coord, xr.DataArray) or x_coord.ndim != 1:
        raise TypeError("x_coord must be a 1D xarray.DataArray")
    if not isinstance(y_coord, xr.DataArray) or y_coord.ndim != 1:
        raise TypeError("y_coord must be a 1D xarray.DataArray")

    use_geode, grid_res, center, domain = compute_grid_metrics(x_coord, y_coord)

    # Kernel Coordinates (r, scale) - Defines the maximum extent
    r_da, scale_da = compute_kernel_grid(
        grid_res, domain,
        scales=kwargs.get("scales"),
        max_distance=kwargs.get("max_r"),
        verbose=verbose,
    )
    r_max_pre_filter = float(r_da.max())

    # Shift Grid Construction: Use res_factor for larger grid
    nx_da, ny_da = build_shift_grid(r_max_pre_filter * res_factor, grid_res)

    # Distance, Angle, and Weights Calculation
    dist_da, angle_da = compute_distance_angle(nx_da, ny_da, grid_res, use_geode, center)
    angle_w = compute_angle_weights(angle_da.values)

    # Mask for valid r-bins
    r_mask = compute_r_mask(dist_da.values, r_da.values, nx_da, ny_da)

    # Assemble Initial Dataset
    ds = xr.Dataset(
        coords={"r": r_da, "scale": scale_da},
        data_vars={
            "mask": r_mask,
            "distance_grid": dist_da,
            "angle_grid": angle_da,
            "angle_weight": angle_w,
            "delta_x_spacing": xr.DataArray(grid_res[0], attrs={"units": "m"}),
            "delta_y_spacing": xr.DataArray(grid_res[1], attrs={"units": "m"}),
        }
    )

    if verbose:
        print("====================  Scale Increments Summary  ====================")

    # Boundary Conditions
    x_boundary, y_boundary = infer_boundary_conditions(x_coord, **kwargs)
    if verbose:
        print(f"[GRID] Boundary conditions -> x: {x_boundary}, y: {y_boundary}")
    ds.attrs.update({"x_boundary_type": x_boundary, "y_boundary_type": y_boundary})

    # Directional Coverage Filtering (avoid poorly sampled r-bins)
    ds = ds.sel(r=filter_by_directional_coverage(ds, min_shifts))

    # Scale Adjustment after Filtering (Trimming the 'scale' coordinate)
    ds["scale"] = adjust_scales_after_r_filter(scale_da, ds["r"], verbose)

    # Diagnostics
    if verbose:
        print(f"[GRID] Domain extension   : Lx={domain[0]:.2f} m, Ly={domain[1]:.2f} m")
        print(f"[GRID] Grid resolution    : dx = {grid_res[0]:8.2f} m, dy = {grid_res[1]:8.2f} m")
        print(f"[GRID] Effective range    : [{float(ds.r.min()):.2f}, {float(ds.r.max()):.2f}] m")
        print(f"[GRID] Retained {ds.scale.size:2d} scales : {summarize_scales(ds.scale.values)} m")
        print("====================================================================")

    # Force computation to NumPy on client for efficient Dask serialization
    return ds.compute()
