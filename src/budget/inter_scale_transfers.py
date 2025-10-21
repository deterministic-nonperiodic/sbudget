from typing import Any, Optional

import numpy as np
import psutil
import xarray as xr
from pint import Quantity
from pyproj import Geod

from .budget import get_spatial_dims
from .constants import earth_radius

# Constants
GEODE = Geod(ellps="WGS84")

ALLOWED_UNITS = ["deg", "degrees", "degrees_north", "degrees_east",
                 "m", "meters", "km", "kilometers"]


def estimate_dataset_bytes(ds: xr.Dataset) -> int:
    """
    Estimate working-set size (bytes) for a dataset.

    - For Dask-backed vars: use the largest chunk along each dimension.
    - For NumPy-backed vars: use the full array size.
    """
    total = 0
    for var in ds.data_vars.values():
        item_size = np.dtype(var.dtype).itemsize
        chunks = getattr(getattr(var, "data", None), "chunks", None)
        if chunks is not None:
            # dask-backed → product of max chunk sizes per dim
            max_elems = 1
            for dim_chunks in chunks:
                max_elems *= max(dim_chunks)
            var_bytes = max_elems * item_size
        else:
            # eager numpy-backed → whole array
            var_bytes = int(var.size) * item_size
        total += var_bytes
    return int(total)


def fits_in_memory(ds: xr.Dataset,
                   memory_threshold_ratio: float = 0.6,
                   expansion_factor: int = 1) -> bool:
    """
    Check if the dataset (possibly expanded along length_scale) fits in memory.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to evaluate.
    memory_threshold_ratio : float, optional
        Fraction of currently available system memory that can be used.
        Default is 0.6 (i.e. 60%).

    Returns
    -------
    bool
        True if the estimated dataset size is below the threshold.
    """
    ds = ds if isinstance(ds, xr.Dataset) else ds.to_dataset()

    # Base size estimate
    dataset_size = estimate_dataset_bytes(ds) * max(1, int(expansion_factor))

    available_memory = psutil.virtual_memory().available
    max_memory = memory_threshold_ratio * available_memory

    print(f"  Estimated dataset size: {dataset_size / 1024 ** 2:.1f} MB")
    print(f"  Allowed memory (threshold {memory_threshold_ratio:.0%}):"
          f" {max_memory / 1024 ** 2:.1f} MB")

    return dataset_size < max_memory


def infer_boundary_conditions(x_coord, **kwargs):
    """
    Infers boundary conditions for x and y coordinates based on their range.

    It determines if the x-coordinate represents a full 360-degree global
    domain (e.g., 0 to 360 or -180 to 180), making it periodic.

    Args:
        x_coord (np.ndarray): The x-coordinate array (longitude). Can be 1D or 2D.
        **kwargs: Optional keyword arguments to override inference.
            x_coord_boundary (str): Manually set x-boundary ('periodic', 'fill', 'reflect',
            'nearest').
            y_coord_boundary (str): Manually set y-boundary ('reflect', etc.).

    Returns:
        tuple[str, str]: A tuple containing the inferred x and y boundary conditions.
    """
    # Check if the user has already specified the boundary conditions
    if "x_coord_boundary" in kwargs:
        x_boundary = kwargs["x_coord_boundary"]
    else:
        # Default to non-periodic and infer below
        x_boundary = "reflect"
        is_global_x = False

        # --- Inference Logic ---
        # This logic checks the *span* of the coordinates, so it works for
        # both 0-to-360 and -180-to-180 degree conventions.
        if x_coord.ndim == 1:
            # For 1D coordinates, calculate tolerance from grid spacing
            spacing = np.median(np.diff(x_coord))
            # Calculate the total span of the coordinate axis
            span = x_coord[-1] - x_coord[0] + spacing
            is_global_x = np.isclose(span, 360.0, atol=spacing)

        elif x_coord.ndim == 2:
            # For 2D coordinates (e.g., curvilinear grids), check along an axis
            spacing = np.median(np.diff(x_coord, axis=1))
            # Check if the longitude span is close to 360 for the middle latitude row
            middle_row_idx = x_coord.shape[0] // 2
            span = x_coord[middle_row_idx, -1] - x_coord[middle_row_idx, 0] + spacing
            is_global_x = np.isclose(span, 360.0, atol=spacing)

        if is_global_x:
            x_boundary = "periodic"

    # For y_boundary, default to "fill" as it is the most common case.
    y_boundary = kwargs.get("y_coord_boundary", "reflect")

    return x_boundary, y_boundary


# --- Kernel Implementation based on User Input ---
def _evaluate_mollifier_and_derivative(radial_positions: np.ndarray,
                                       length_scales: np.ndarray
                                       ) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the standard mollifier and its radial derivative for all combinations of
    radial distances and filter length scales, in a vectorized manner.

    Returns
    -------
    mollifier : np.ndarray of shape (n_scales, n_r)
    derivative : np.ndarray of shape (n_scales, n_r)
    """
    r_grid, ell_grid = np.meshgrid(radial_positions, length_scales, indexing="ij")
    ratio_squared = (r_grid / (2 * ell_grid)) ** 2
    inside_support = ratio_squared < 1
    denominator = 1 - ratio_squared

    mollifier = np.zeros_like(denominator)
    mollifier[inside_support] = np.exp(-1.0 / denominator[inside_support])

    derivative = np.zeros_like(denominator)
    derivative[inside_support] = (
            -r_grid[inside_support] / (2 * ell_grid[inside_support] ** 2)
            * np.exp(-1.0 / denominator[inside_support])
            / denominator[inside_support] ** 2
    )

    return mollifier.T, derivative.T  # final shape: (n_scales, n_r)


def _normalize_mollifier_2d(mollifier: np.ndarray,
                            radial_positions: np.ndarray,
                            method: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize each mollifier profile using the appropriate area weighting.

    Returns
    -------
    mollifier_normalized : np.ndarray of shape (n_scales, n_r)
    integrals : np.ndarray of shape (n_scales,)
    """
    if method == "2D":
        area_weights = 2 * np.pi * radial_positions
    elif method == "sphere":
        area_weights = 2 * np.pi * earth_radius * np.sin(radial_positions / earth_radius)
    elif method == "3D":
        area_weights = 4 * np.pi * radial_positions ** 2
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Broadcast area weights: shape (1, n_r)
    integrals = np.trapz(mollifier * area_weights[None, :], x=radial_positions, axis=1)
    mollifier_normalized = mollifier / integrals[:, np.newaxis]
    return mollifier_normalized, integrals


def get_integration_kernels(r_da: xr.DataArray, length_scales: np.ndarray, normalization="2D",
                            return_derivative=True) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Compute mollifier kernels and optionally their derivatives over a set of radial distances
    and filter length scales. Uses a fully vectorized implementation.

    Parameters
    ----------
    r_da : xr.DataArray
        1D array of radial distances
    length_scales : np.ndarray
        1D array of filter length scales
    normalization : str
        One of {"2D", "sphere", "3D"} to determine area weighting
    return_derivative : bool
        Whether to compute the derivative dG/dr

    Returns
    -------
    G : xr.DataArray
        Mollifier kernel, dims: (length_scale, r)
    dG : xr.DataArray
        Derivative of mollifier, dims: (length_scale, r)
    """
    radial_positions = r_da.values
    r_name = r_da.name or "r"
    length_scales = np.asarray(length_scales)

    mollifier, derivative = _evaluate_mollifier_and_derivative(radial_positions,
                                                               length_scales)
    mollifier_normalized, integrals = _normalize_mollifier_2d(mollifier,
                                                              radial_positions,
                                                              normalization)

    mollifier_da = xr.DataArray(
        mollifier_normalized,
        dims=["length_scale", r_name],
        coords={"length_scale": length_scales, r_name: radial_positions},
        name="G_kernel"
    )

    if return_derivative:
        derivative_normalized = (derivative.T / integrals).T
        derivative_da = xr.DataArray(
            derivative_normalized,
            dims=["length_scale", r_name],
            coords={"length_scale": length_scales, r_name: radial_positions},
            name="dG_dr_kernel"
        )
    else:
        derivative_da = xr.full_like(mollifier_da, np.nan, name="dG_dr_kernel")

    return mollifier_da, derivative_da


# --- Helper Functions ---
def delta_u_cubed_geographic(
        ds: xr.Dataset,
        ds_shifted: xr.Dataset,
        angle_grid: xr.DataArray
) -> xr.DataArray:
    """
    Compute (δu ⋅ r̂) ⋅ |δu|² using angle and distance from geographic scale increments.
    """
    ds_increment = ds_shifted - ds
    delta_u = ds_increment["u"]
    delta_v = ds_increment["v"]
    delta_w = ds_increment.get("w", None)

    # Magnitude of increment vector |δu|²
    mag_sq = delta_u ** 2 + delta_v ** 2
    if delta_w is not None:
        mag_sq += delta_w ** 2

    # Directional projection: δu ⋅ r̂ = δu cos(θ) + δv sin(θ)
    delta_dot_r = delta_u * np.cos(angle_grid) + delta_v * np.sin(angle_grid)

    return (delta_dot_r * mag_sq).rename("delta_u_cubed")


def roll_with_boundary_handling(
        data: xr.Dataset,
        n_x: int,
        n_y: int,
        x_dim: str,
        y_dim: str,
        x_boundary_type: str = "periodic",
        y_boundary_type: str = "periodic",
        fill_value: Any = np.nan
) -> xr.Dataset:
    """
    Rolls a Dataset along spatial dimensions with optional boundary fill, reflection, or nearest extrapolation.

    Parameters
    ----------
    data : xr.Dataset
        Dataset to roll.
    n_x : int
        Shift along x_dim (positive = right).
    n_y : int
        Shift along y_dim (positive = down).
    x_dim, y_dim : str
        Names of the spatial dimensions.
    x_boundary_type, y_boundary_type : {'periodic', 'fill', 'reflect', 'nearest'}
        Boundary condition type per axis.
    fill_value : Any
        Value to use when boundary_type == 'fill'.
    lazy : bool
        If True, preserve dask arrays and avoid immediate computation.

    Returns
    -------
    xr.Dataset
    """

    def process_dimension(ds: xr.Dataset, dim: str, shift: int, boundary: str) -> xr.Dataset:
        if shift == 0 or boundary == "periodic":
            return ds.roll({dim: -shift}, roll_coords=False)

        dim_size = ds.sizes[dim]
        pad_width = abs(shift) + 1  # +1 to ensure we cover the full range

        if boundary == "fill":
            # Normalize shift to within the dimension length
            if dim_size == 0 or shift == 0:
                return ds
            k = int(shift) % dim_size  # number of positions to shift (0..dim_size-1)

            # Use the same effective direction as your original: roll({dim: -shift})
            shifted = ds.shift({dim: -k})

            # Edge slice newly exposed by the shift
            if shift > 0:
                edge = slice(dim_size - k, dim_size)  # right edge
            else:
                edge = slice(0, k)  # left edge

            # 1D mask along the shifting dim: True everywhere, False on the new edge
            edge_mask = xr.DataArray(
                np.ones(dim_size, dtype=bool),
                coords={dim: ds[dim]},
                dims=[dim],
            )
            edge_mask.loc[{dim: edge}] = False

            # Fill only the introduced edge; preserve pre-existing NaNs
            return shifted.where(edge_mask, other=fill_value)

        elif boundary in ("reflect", "nearest"):
            mode = "reflect" if boundary == "reflect" else "edge"
            padded = ds.pad({dim: (pad_width, pad_width)}, mode=mode)
            rolled = padded.roll({dim: -shift}, roll_coords=False)
            return rolled.isel({dim: slice(pad_width, pad_width + dim_size)})

        else:
            raise ValueError(f"Unsupported boundary type: {boundary}")

    ds_rolled = process_dimension(data, x_dim, n_x, x_boundary_type)
    ds_rolled = process_dimension(ds_rolled, y_dim, n_y, y_boundary_type)

    return ds_rolled


def filter_by_directional_coverage(scale_incs: xr.Dataset,
                                   min_valid_shifts: int = 10) -> np.ndarray:
    """
    Returns a boolean mask over the 'r' dimension where each True value corresponds to
    a scale with at least `min_valid_shifts` directional sampling vectors.
    """
    valid_counts = scale_incs['mask'].sum(dim=('ny', 'nx')).values
    return valid_counts >= min_valid_shifts


# --- Core Computational Functions (Refactored) ---
def _infer_coordinate_units(coord: xr.DataArray, name: str) -> str:
    units = coord.attrs.get("units", "").lower()
    if not units:
        raise ValueError(f"Missing 'units' attribute for {name} coordinate.")
    if units not in ALLOWED_UNITS:
        raise ValueError(f"Invalid units for {name}: '{units}'. Allowed: {ALLOWED_UNITS}")
    return units


def _is_geographic(units_x: str, units_y: str) -> bool:
    return np.logical_and(
        any(u in units_x for u in ["deg", "degrees", "degrees_east"]),
        any(u in units_y for u in ["deg", "degrees", "degrees_north"])
    )


def _get_spacing(coord: xr.DataArray, center: float, use_geode: Optional[bool], axis: str) -> float:
    if coord.size < 2:
        return 1.0
    if not use_geode:
        return float(np.abs(np.median(np.diff(coord.values))))

    if axis == 'x':
        _, _, dist = GEODE.inv(coord[0].item(), center, coord[1].item(), center)
    else:
        _, _, dist = GEODE.inv(center, coord[0].item(), center, coord[1].item())
    return dist


def scale_increments(
        x_coord: xr.DataArray,
        y_coord: xr.DataArray,
        max_r_m: float,
        verbose: bool = False,
        resolution_factor: int = 1,
        min_valid_shifts: int = 10
) -> xr.Dataset:
    """Calculate separation vectors and geometric quantities for structure function computation."""

    for coord, name in zip([x_coord, y_coord], ["x", "y"]):
        if not isinstance(coord, xr.DataArray):
            raise TypeError(f"{name}_coord must be an xarray.DataArray")
        if coord.ndim != 1:
            raise ValueError(f"{name}_coord must be 1-dimensional")

    units_x = _infer_coordinate_units(x_coord, 'x')
    units_y = _infer_coordinate_units(y_coord, 'y')
    use_geode = _is_geographic(units_x, units_y)

    x_center = float(x_coord.mean())
    y_center = float(y_coord.mean())

    dx = max(_get_spacing(x_coord, y_center, use_geode=use_geode, axis='x'), 1e-6)
    dy = max(_get_spacing(y_coord, x_center, use_geode=use_geode, axis='y'), 1e-6)
    r_step = max(dx, dy)

    x_min = float(x_coord.min())
    x_max = float(x_coord.max())
    y_min = float(y_coord.min())
    y_max = float(y_coord.max())

    if use_geode:
        spacing = float(np.median(np.diff(x_coord)))
        span = x_max - x_min + spacing
        if np.isclose(span, 360.0, atol=spacing):
            _, _, lx = GEODE.inv(0, y_center, 180, y_center)
            lx *= 2
        else:
            _, _, lx = GEODE.inv(x_min, y_center, x_max, y_center)
        _, _, ly = GEODE.inv(x_center, y_min, x_center, y_max)
    else:
        lx, ly = x_max - x_min, y_max - y_min

    # Generate scales for the analysis
    effective_max_r = min(max_r_m, min(lx, ly) / 2.0)

    r_values = np.arange(0, effective_max_r + r_step / 2.0, r_step)

    if r_values.size < 1:
        raise ValueError("No valid scales generated by scale_increments.")

    r_coord_da = xr.DataArray(
        r_values, dims="r", name="r",
        attrs={"units": "m", "long_name": "Separation distance (scale)"}
    )

    max_nx = int(np.ceil(effective_max_r / dx))
    max_ny = int(np.ceil(effective_max_r / dy))
    nx_vals = np.linspace(-max_nx, max_nx, 2 * max_nx * resolution_factor + 1, dtype=int)
    ny_vals = np.linspace(-max_ny, max_ny, 2 * max_ny * resolution_factor + 1, dtype=int)

    da_nx = xr.DataArray(nx_vals, dims="nx")
    da_ny = xr.DataArray(ny_vals, dims="ny")
    ny_grid, nx_grid = xr.broadcast(da_ny, da_nx)

    if use_geode:
        azimuth = np.rad2deg(np.arctan2(ny_grid, nx_grid))
        distance_vals = np.sqrt((nx_grid * dx) ** 2 + (ny_grid * dy) ** 2)

        fwd_lon, fwd_lat, _ = GEODE.fwd(
            np.full_like(distance_vals, x_center),
            np.full_like(distance_vals, y_center),
            azimuth,
            distance_vals
        )

        _, _, true_distance_vals = GEODE.inv(
            np.full_like(fwd_lon, x_center),
            np.full_like(fwd_lat, y_center),
            fwd_lon,
            fwd_lat
        )
        angle_vals = np.deg2rad(azimuth).values
        distance_vals = true_distance_vals
    else:
        dx_grid = nx_grid * dx
        dy_grid = ny_grid * dy
        distance_vals = np.sqrt(dx_grid ** 2 + dy_grid ** 2).values
        angle_vals = np.arctan2(dy_grid, dx_grid).values

    distance = xr.DataArray(distance_vals, dims=("ny", "nx"), name="distance_grid")
    distance.attrs = {"units": "m", "long_name": "Distance from origin"}

    angle = xr.DataArray(angle_vals, dims=("ny", "nx"), name="angle_grid")
    angle.attrs = {"units": "radians", "long_name": "Angle of offset from origin"}

    flat_angles = angle_vals.flatten()
    unique_angles, counts = np.unique(flat_angles, return_counts=True)
    angle_weights = {a: (2 * np.pi / np.sum(1.0 / counts)) * (1.0 / c) for a, c in
                     zip(unique_angles, counts)}
    angle_weights_da = xr.DataArray(
        np.vectorize(angle_weights.get)(angle_vals),
        dims=("ny", "nx"), name="angle_weight"
    )

    lower_bound = r_values[:, np.newaxis, np.newaxis] - r_step / 2.0
    upper_bound = r_values[:, np.newaxis, np.newaxis] + r_step / 2.0
    mask = (distance_vals[np.newaxis, :, :] >= lower_bound) & (
            distance_vals[np.newaxis, :, :] < upper_bound)
    r_mask = xr.DataArray(
        mask, dims=("r", "ny", "nx"),
        coords={"r": r_coord_da, "ny": da_ny, "nx": da_nx}, name="r_mask"
    )

    # Create increments dataset
    increments = xr.Dataset(
        {"r": r_coord_da,
         "mask": r_mask,
         "distance_grid": distance,
         "angle_grid": angle,
         "angle_weight": angle_weights_da,
         "delta_x_spacing": xr.DataArray(dx, name="delta_x_spacing", attrs={"units": "m"}),
         "delta_y_spacing": xr.DataArray(dy, name="delta_y_spacing", attrs={"units": "m"})}
    )

    # Filter scales based on directional coverage. At least min_valid_shifts
    valid_mask = filter_by_directional_coverage(increments, min_valid_shifts=min_valid_shifts)
    increments = increments.sel(r=valid_mask)

    effective_min_r = increments["r"].min().values
    effective_max_r = increments["r"].max().values
    if verbose:
        print("================== Scale Increments Summary ==================")
        print(f"  Domain size         : Lx = {lx:8.2f} m, Ly = {ly:8.2f} m")
        print(f"  Grid spacing        : dx = {dx:8.2f} m, dy = {dy:8.2f} m")
        print(f"  Effective min scale : {effective_min_r:8.2f} m (dr = {r_step:8.2f} m)")
        print(f"  Effective max scale : {effective_max_r:8.2f} m (Requested: {max_r_m:.2f} m)")
        print("==============================================================")

    return increments


def scale_space_integral(
        integrand: xr.DataArray,
        name: str,
        length_scales: Optional[np.ndarray] = None,
        weighting: str = "2D",
        verbose: bool = False,
        length_scale_chunk: Optional[int] = -1
) -> xr.DataArray:
    if verbose: print(f"Calculating scale-space integral for '{name}'...")

    r_coord = integrand.r

    if r_coord.size == 0:
        if verbose: print("Warning: Integrand has no 'r' dimension or it's empty.")
        return xr.full_like(integrand, np.nan).mean("r", skipna=True).expand_dims(
            length_scale=0).drop_vars("r").assign_coords(length_scale=[]).rename(name)

    if length_scales is None:
        length_scales = r_coord.values[1:] if r_coord.size > 1 else r_coord.values[:1]
    else:
        if not isinstance(length_scales, np.ndarray):
            length_scales = np.array(length_scales)

        length_scales = length_scales[length_scales < r_coord.max().values]

        if verbose:
            print(f"Externally defined length_scales:")
            min_scale = length_scales.min()
            max_scale = length_scales.max()
            print(f"  Effective scale limits: {min_scale:8.2f} m - {max_scale:8.2f} m)")
            print("==============================================================")

    if not length_scales.size:
        if verbose: print("Warning: No valid length scales to use in integration.")
        return xr.full_like(integrand, np.nan).mean("r", skipna=True).expand_dims(
            length_scale=0).drop_vars("r").assign_coords(length_scale=[]).rename(name)

    # Compute dG/dr
    _, dg_dr = get_integration_kernels(r_coord, length_scales,
                                       normalization=weighting, return_derivative=True)

    # Broadcast r and dG_dr
    r_broadcasted, _ = xr.broadcast(r_coord, dg_dr)
    _, integrand_broadcasted = xr.broadcast(dg_dr, integrand)

    # Mask to truncate at r <= 2*ell
    integration_mask = r_broadcasted <= 2 * dg_dr.length_scale

    # Compute integrand: (dG/dr * r) * integrand, then apply mask
    term_to_integrate = (dg_dr * r_broadcasted) * integrand_broadcasted
    term_to_integrate_masked = term_to_integrate.where(integration_mask, other=0.0)

    # Correction: estimate the retained fraction of the integrand
    retention_fraction = term_to_integrate_masked.sum("r") / term_to_integrate.sum("r")
    retention_fraction = retention_fraction.where(retention_fraction > 1e-6, 1.0)

    # Integrate and normalize by retained fraction
    integral = term_to_integrate_masked.integrate("r") / retention_fraction
    integral = integral.rename(name).assign_coords(length_scale=dg_dr.length_scale)

    if hasattr(integral.data, "chunks"):
        integral = integral.chunk({"length_scale": length_scale_chunk})

    if verbose:
        print(f"Finished calculating scale-space integral '{name}'. Shape: {integral.shape}")

    return integral


def process_single_r_for_field_chunk(
        field_chunk_ds: xr.Dataset,
        r_scalar_val: float,
        scale_mask_for_r: xr.DataArray,
        scale_angle_grid: xr.DataArray,
        nx_shift_coords: xr.DataArray,
        ny_shift_coords: xr.DataArray,
        angle_weight_grid: xr.DataArray,
        x_dim: str,
        y_dim: str,
        x_boundary_type: str,
        y_boundary_type: str,
        transform_type: str,
) -> xr.DataArray:
    """
    Processes a single spatial chunk of the field for a specific scale r.
    Optimized version with reduced code redundancy, Dask compatibility,
    and angular weighting normalization.
    """
    if transform_type != "delta_u_cubed":
        raise ValueError(f"Transform_type: {transform_type}, not implemented.")

    valid_indices = np.argwhere(scale_mask_for_r.data)
    if not valid_indices.size:
        return xr.full_like(
            field_chunk_ds[list(field_chunk_ds.data_vars)[0]].isel({x_dim: 0, y_dim: 0}),
            0.0
        ).assign_coords(r=r_scalar_val)

    angles = scale_angle_grid.data[valid_indices[:, 0], valid_indices[:, 1]]
    weights = angle_weight_grid.data[valid_indices[:, 0], valid_indices[:, 1]]
    n_x_values = nx_shift_coords.data[valid_indices[:, 1]]
    n_y_values = ny_shift_coords.data[valid_indices[:, 0]]

    # Normalize weights to sum to 2π for consistency with continuous angular integration
    weights *= 2.0 * np.pi / np.sum(weights)

    weighted_sum = None
    total_weight = 0.0

    for phi, n_x, n_y, weight_val in zip(angles, n_x_values, n_y_values, weights):

        rolled = roll_with_boundary_handling(
            field_chunk_ds, int(n_x), int(n_y),
            x_dim, y_dim, x_boundary_type, y_boundary_type
        )
        result = delta_u_cubed_geographic(field_chunk_ds, rolled, phi)

        if weighted_sum is None:
            weighted_sum = result * weight_val
        else:
            weighted_sum += result * weight_val
        total_weight += weight_val

    if weighted_sum is None or total_weight == 0:
        return xr.full_like(
            field_chunk_ds[list(field_chunk_ds.data_vars)[0]].isel({x_dim: 0, y_dim: 0}),
            0.0
        ).assign_coords(r=r_scalar_val)

    integrand = weighted_sum / total_weight

    for coord_to_drop in [x_dim, y_dim]:
        if coord_to_drop in integrand.coords:
            integrand = integrand.drop_vars(coord_to_drop)

    # Assign field coordinates to the integrand for broadcasting
    integrand = integrand.assign_coords(**{'r': r_scalar_val,
                                           x_dim: field_chunk_ds[x_dim],
                                           y_dim: field_chunk_ds[y_dim]})
    return integrand


def build_map_blocks_template(field: xr.Dataset, transform_type: str, r_vals: np.ndarray,
                              x_dim: str, y_dim: str) -> xr.DataArray:
    """
    Build a properly-shaped template for use with `xr.map_blocks`.
    """
    # Use the first variable in the dataset as a shape reference
    first_var_name = list(field.data_vars)[0]
    base = xr.zeros_like(field[first_var_name].isel({x_dim: slice(None), y_dim: slice(None)}))
    template = base.expand_dims(r=r_vals)
    template.name = transform_type

    return template


def increment_integrand(
        field: xr.Dataset,
        increments: xr.Dataset,
        x_dim: str,
        y_dim: str,
        x_boundary_type: str = "periodic",
        y_boundary_type: str = "fill",
        verbose: bool = False,
        transform_type: str = "delta_u_cubed"
) -> xr.DataArray:
    """Dask-parallelized integrand calculation using map_blocks."""
    if verbose:
        engine = 'Dask' if hasattr(field, "chunks") else 'NumPy'
        print(f"Starting {engine}-based integrand calculation for '{transform_type}'...")

    r_vals = increments["r"].values

    def block_fn(field_chunk: xr.Dataset) -> xr.DataArray:
        return xr.concat(
            (process_single_r_for_field_chunk(
                field_chunk_ds=field_chunk,
                r_scalar_val=r,
                scale_mask_for_r=increments["mask"].sel(r=r),
                scale_angle_grid=increments["angle_grid"],
                nx_shift_coords=increments["nx"],
                ny_shift_coords=increments["ny"],
                angle_weight_grid=increments["angle_weight"],
                x_dim=x_dim,
                y_dim=y_dim,
                x_boundary_type=x_boundary_type,
                y_boundary_type=y_boundary_type,
                transform_type=transform_type,
            ) for r in r_vals),
            dim="r"
        )

    # Build template for map_blocks
    template = build_map_blocks_template(
        field=field,
        transform_type=transform_type,
        r_vals=r_vals,
        x_dim=x_dim,
        y_dim=y_dim
    )

    return xr.map_blocks(block_fn, field, template=template)


def ensure_nonspatial_chunking(
        ds: xr.Dataset,
        x_name: str,
        y_name: str,
        expansion_factor: int = 1,
        memory_threshold_ratio: float = 0.5,
        verbose: bool = False,
):
    """
    Ensure that a dataset is chunked along non-spatial dimensions if needed
    to keep memory use under control.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset (may already be chunked).
    x_name, y_name : str
        Names of the spatial dimensions (kept unchunked).
    expansion_factor : int, optional
        Factor by which the dataset will be expanded (e.g. len(r)).
        Used in fits_in_memory check.
    memory_threshold_ratio : float, optional
        Fraction of available memory allowed before rechunking.
    verbose : bool, optional
        If True, log what rechunking was applied.

    Returns
    -------
    ds_chunked : xr.Dataset
        Dataset with non-spatial rechunking applied if necessary.
    """
    if fits_in_memory(ds, memory_threshold_ratio=memory_threshold_ratio,
                      expansion_factor=expansion_factor):

        if verbose:
            print("Dataset fits in memory. No rechunking required.")
            print("==============================================================")

        return ds

    # Plan rechunking: only add "auto" for non-spatial dims not yet chunked
    non_spatial_dims = {}
    for d in ds.dims:
        if d in (x_name, y_name):
            continue
        if not hasattr(ds[list(ds.data_vars)[0]].data, "chunks"):
            # NumPy-backed → no chunks at all
            non_spatial_dims[d] = "auto"
        elif d not in ds.chunks:
            # Dask-backed but not chunked along this dim
            non_spatial_dims[d] = "auto"

    if non_spatial_dims:
        ds = ds.chunk(non_spatial_dims)

        if verbose:
            print("Rechunked non-spatial dimensions due to memory constraints:")
            for dim in non_spatial_dims:
                if dim in ds.chunks:
                    chunks = ds.chunks[dim]
                    n_chunks = len(chunks)
                    sizes = ", ".join(str(c) for c in chunks[:3])
                    if n_chunks > 3:
                        sizes += ", …"
                    print(f"  - {dim}: {n_chunks} chunks (sizes: {sizes})")
            print("==============================================================")

    return ds


def inter_scale_kinetic_energy_transfer(wind: xr.Dataset, **kwargs) -> xr.Dataset:
    """ Computes the inter-scale kinetic energy transfer rate using third-order structure functions.
    Parameters
    ----------
    wind : xr.Dataset
        Dataset containing 3D velocity components (u, v, w).
    **kwargs : dict
    """

    # Validate input dataset
    velocity_vars = [v for v in ["u", "v", "w"] if v in wind]

    # Ensure velocity components are float32 for memory efficiency
    wind = wind[velocity_vars].astype({v: "float32" for v in velocity_vars})

    # Check if the dataset has the required variables
    verbose = kwargs.get("verbose", False)

    # Determine spatial coordinate names
    x_name = kwargs.get("x_coord_name", None)
    y_name = kwargs.get("y_coord_name", None)
    length_scales = kwargs.get("scales", None)
    ls_chunk_size = kwargs.get("ls_chunk_size", -1)

    try:
        # Attempt to retrieve coordinates by name
        if x_name is None and y_name is None:
            # Infer coordinates using helper (assumes CF compliance)
            y_name, x_name = get_spatial_dims(wind)

        if x_name in wind and y_name in wind:
            x_coord = wind[x_name]
            y_coord = wind[y_name]
        else:
            raise KeyError(f"Specified coordinate names {x_name}, {y_name} not found in dataset.")
    except Exception:
        raise ValueError(
            "Could not infer spatial coordinates. For non-geographic data, "
            "please specify 'x_coord_name' and 'y_coord_name' explicitly in kwargs."
        )

    # Determine boundary conditions
    x_boundary, y_boundary = infer_boundary_conditions(x_coord.values, **kwargs)

    if verbose:
        print(f"Inferred boundary conditions -> x: {x_boundary}, y: {y_boundary}")

    # Handle max_r input with unit checking
    max_r_input = kwargs.get("max_r", 500e3 if length_scales is None else max(length_scales))

    if isinstance(max_r_input, Quantity):
        max_r_m = max_r_input.to("meter").magnitude
    else:
        max_r_m = float(max_r_input)

    # Compute scale increments
    increments = scale_increments(
        x_coord, y_coord, max_r_m,
        verbose=verbose, resolution_factor=1,
        min_valid_shifts=kwargs.get("min_valid_shifts", 10)
    )

    # Ensure the increments fit in memory or compute in chunks along non-spatial dimensions
    wind = ensure_nonspatial_chunking(wind, x_name, y_name,
                                      expansion_factor=len(increments.r),
                                      memory_threshold_ratio=0.5,
                                      verbose=verbose)

    # Compute third-order structure functions. Mask missing values in velocity components.
    nan_mask = xr.concat(
        [xr.ufuncs.isnan(wind[var]) for var in velocity_vars],
        dim='component'
    ).any(dim='component')

    integrand = increment_integrand(
        field=wind.fillna(0.0),
        increments=increments,
        x_dim=x_name,
        y_dim=y_name,
        x_boundary_type=x_boundary,
        y_boundary_type=y_boundary,
        verbose=verbose,
        transform_type="delta_u_cubed"
    ).where(~nan_mask)

    # Apply mollifier normalization
    energy_transfer_rate = scale_space_integral(
        integrand=integrand,
        name="energy_transfer",
        length_scales=length_scales,
        weighting="2D",
        verbose=verbose,
        length_scale_chunk=ls_chunk_size
    )

    energy_transfer_rate.attrs.update({
        'units': "W/kg",
        'standard_name': "specific_kinetic_energy_transfer_rate",
        'long_name': "Specific kinetic energy inter-scale transfer rate",
        'description': "Kinetic energy transfer across scales, "
                       "calculated using structure functions and mollifier-based filtering."
    })

    # reassign coordinates from data
    energy_transfer_rate = energy_transfer_rate.assign_coords(**{x_name: x_coord, y_name: y_coord})
    energy_transfer_rate[x_name].attrs = x_coord.attrs
    energy_transfer_rate[y_name].attrs = y_coord.attrs

    # Check if the result fits in memory (the result is length_scale times the input's size)
    energy_transfer_rate = energy_transfer_rate.to_dataset()

    # --- ADD: enforce one-scale-at-a-time tasks for reductions/writes ---
    if hasattr(energy_transfer_rate[list(energy_transfer_rate.data_vars)[0]].data, "chunks"):
        energy_transfer_rate = energy_transfer_rate.chunk({"length_scale": 1})

    if verbose:
        # Avoid triggering a full compute if Dask-backed (expensive convolutions)
        arr = energy_transfer_rate.energy_transfer.data
        if hasattr(arr, "chunks"):
            print("Finished computing energy transfer rate (lazy graph built). "
                  "Skipping domain total to avoid full computation.")
        else:
            domain_total = float(energy_transfer_rate.energy_transfer.sum())
            print(f"Finished computing energy transfer rate. "
                  f"Domain total: {domain_total:.3e} W/kg")

    return energy_transfer_rate
