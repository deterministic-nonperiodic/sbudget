from typing import Any, Optional, Union, List

import dask.array as da
import numpy as np
import xarray as xr
from pint import Quantity
from pyproj import Geod

from .budget import get_spatial_dims
from .cf_coords import is_geographic_grid, _is_global_longitude
from .constants import earth_radius
from .io_utils import ensure_optimal_chunking

# Constants
GEODE = Geod(ellps="WGS84")


def infer_boundary_conditions(x_coord: xr.DataArray, **kwargs) -> tuple[str, str]:
    """
    Infers boundary conditions for x and y coordinates based on their range.

    It determines if the x-coordinate represents a full 360-degree global
    domain (e.g., 0 to 360 or -180 to 180), making it periodic.

    Args:
        x_coord (xr.DataArray): The x-coordinate array (longitude). Can be 1D or 2D.
        **kwargs: Optional keyword arguments to override inference.
            x_coord_boundary (str): ('periodic', 'fill', 'reflect', 'nearest').
            y_coord_boundary (str): ('periodic', 'fill', 'reflect', 'nearest').

    Returns:
        tuple[str, str]: A tuple containing the inferred x and y boundary conditions.
    """

    # Check if the user has already specified the boundary conditions
    x_boundary = kwargs.get("x_coord_boundary",
                            "periodic" if _is_global_longitude(x_coord) else "reflect")

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


def get_integration_kernels(r_da: xr.DataArray, scales: np.ndarray, normalization="2D",
                            return_derivative=True) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Compute mollifier kernels and optionally their derivatives over a set of radial distances
    and filter length scales. Uses a fully vectorized implementation.

    Parameters
    ----------
    r_da : xr.DataArray
        1D array of radial distances
    scales : np.ndarray
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
    scales = np.asarray(scales)

    mollifier, derivative = _evaluate_mollifier_and_derivative(radial_positions, scales)
    mollifier_normalized, integrals = _normalize_mollifier_2d(mollifier,
                                                              radial_positions,
                                                              normalization)

    mollifier_da = xr.DataArray(
        mollifier_normalized,
        dims=["scale", r_name],
        coords={"scale": scales, r_name: radial_positions},
        name="G_kernel"
    )

    if return_derivative:
        derivative_normalized = (derivative.T / integrals).T
        derivative_da = xr.DataArray(
            derivative_normalized,
            dims=["scale", r_name],
            coords={"scale": scales, r_name: radial_positions},
            name="dG_dr_kernel"
        )
    else:
        derivative_da = xr.full_like(mollifier_da, fill_value=np.nan).rename("dG_dr_kernel")

    return mollifier_da, derivative_da


# --- Helper Functions ---
def delta_u_cubed_geographic(
        ds: xr.Dataset,
        ds_shifted: xr.Dataset,
        angle_grid: xr.DataArray
) -> xr.DataArray:
    """
    Compute (δu ⋅ r̂) |δu|² using angle and distance from geographic scale increments.
    """
    ds_increment = ds_shifted - ds
    delta_u = ds_increment["u"]
    delta_v = ds_increment["v"]
    delta_w = ds_increment.get("w", None)

    # Magnitude of increment vector |δu|²
    delta_u_squared = delta_u ** 2 + delta_v ** 2
    if delta_w is not None:
        delta_u_squared += delta_w ** 2

    # Directional projection: δu ⋅ r̂ = δu cos(θ) + δv sin(θ)
    delta_dot_r = delta_u * np.cos(angle_grid) + delta_v * np.sin(angle_grid)

    return (delta_dot_r * delta_u_squared).rename("delta_u_cubed")


def roll_with_boundary_handling(
        data: xr.Dataset,
        n_x: int,
        n_y: int,
        x_dim: str,
        y_dim: str,
        x_boundary_type: str = "periodic",
        y_boundary_type: str = "periodic",
        fill_value: Any = np.nan,
) -> xr.Dataset:
    """
    Roll a Dataset along spatial dimensions with selectable boundary handling.

    Parameters
    ----------
    data : xr.Dataset
        Dataset to roll.
    n_x, n_y : int
        Shifts along x_dim (right) and y_dim (down). Positive = forward.
    x_dim, y_dim : str
        Names of the spatial dimensions.
    x_boundary_type, y_boundary_type : {'periodic', 'constant', 'reflect', 'edge'}
        Boundary mode for each axis. See xarray.Dataset.pad documentation for all available modes.
    fill_value : Any
        Value to use when boundary_type == 'fill'.

    Returns
    -------
    xr.Dataset
    """
    valid_pad_modes = {"constant", "edge", "linear_ramp", "maximum", "mean",
                       "median", "minimum", "reflect", "symmetric", "wrap"}

    def process_dimension(ds: xr.Dataset, dim: str, shift: int, boundary: str) -> xr.Dataset:
        """Process rolling along a single dimension with specified boundary handling."""

        dim_size = int(ds.sizes[dim])
        if dim_size == 0 or shift == 0:
            return ds

        # Fast path for periodic wrapping. Same as native xarray "wrap" mode
        if boundary in ("periodic", "wrap"):
            # Circular shift; negative to match original direction
            return ds.roll({dim: -shift}, roll_coords=False)

        # --- Prepare padding parameters for non-periodic modes ---
        if boundary not in valid_pad_modes:
            raise ValueError(f"Unsupported boundary type: {boundary!r}")

        pad_kwargs = dict(mode=boundary)
        if boundary == "constant":
            pad_kwargs["constant_values"] = fill_value

        # --- Non-periodic modes: pad → roll → slice ---
        pad_width = abs(shift) + 1

        # Pad both sides by k, roll by -k, then crop back to original length
        padded = ds.pad({dim: (pad_width, pad_width)}, **pad_kwargs)
        rolled = padded.roll({dim: -shift}, roll_coords=False)

        return rolled.isel({dim: slice(pad_width, pad_width + dim_size)})

    # Process x and y dimensions
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


def get_max_radial_distance(
        length_scales: Optional[Union[np.ndarray, List[float]]],
        max_r_input: Union[float, Quantity, None] = None) -> float:
    """
    Determines the maximum radial distance (max_r_m) for structure function
    computation, constrained by twice the largest length scale.

    Parameters
    ----------
    length_scales : Optional[Union[np.ndarray, List[float]]]
        Array or list of filter length scales (in meters) being analyzed.
        Used to constrain max_r to 2 * max(length_scales).
    max_r_input : Optional[Union[float, pint.Quantity]]
        The user-provided maximum radial distance. Defaults to 500 km
        if length_scales is None, otherwise defaults to max(length_scales).

    Returns
    -------
    max_r_m : float
        The constrained maximum radial distance in meters.
    """

    # Set default max_r_input based on presence of length_scales
    if max_r_input is None:
        max_r_m_default = 500e3 if length_scales is None else max(length_scales)
    else:
        max_r_m_default = max_r_input

    if isinstance(max_r_m_default, Quantity):
        # Convert Quantity to meters
        max_r_m = max_r_m_default.to("meter").magnitude
    else:
        # Assume float is already in meters if coming from max(length_scales) or 500e3
        max_r_m = float(max_r_m_default)

    # Apply the 2 * l_max constraint
    if length_scales is not None:
        # Calculate the maximum required radial distance
        max_scale_limit = 2 * max(length_scales)

        # Enforce that max_r_m does not exceed 2 * l_max
        max_r_m = min(max_r_m, max_scale_limit)

    return max_r_m


def scale_increments(
        x_coord: xr.DataArray,
        y_coord: xr.DataArray,
        max_r_m: float,
        verbose: bool = False,
        resolution_factor: int = 1,
        min_valid_shifts: int = 10
) -> xr.Dataset:
    """Calculate separation vectors and geometric quantities for structure function computation."""

    # 1. Validation and Setup
    for coord, name in zip([x_coord, y_coord], ["x", "y"]):
        if not isinstance(coord, xr.DataArray):
            raise TypeError(f"{name}_coord must be an xarray.DataArray")
        if coord.ndim != 1:
            raise ValueError(f"{name}_coord must be 1-dimensional")

    # Flag for geographic grid handling
    use_geode = is_geographic_grid(x_coord, y_coord)

    # Calculate grid spacing and domain centers (assumes GEODE and _get_spacing are available)
    x_center = float(x_coord.mean())
    y_center = float(y_coord.mean())
    # Note: _get_spacing is a helper function assumed to be available
    dx = max(_get_spacing(x_coord, y_center, use_geode=use_geode, axis='x'), 1e-6)
    dy = max(_get_spacing(y_coord, x_center, use_geode=use_geode, axis='y'), 1e-6)

    r_step = max(dx, dy)  # Scale resolution is set by the maximum spacing

    x_min, x_max = float(x_coord.min()), float(x_coord.max())
    y_min, y_max = float(y_coord.min()), float(y_coord.max())

    # 2. Calculate Domain Lengths (lx, ly)
    if use_geode:
        spacing = float(np.median(np.diff(x_coord)))
        span = x_max - x_min + spacing

        # Check for periodic domain near 360 degrees
        if np.isclose(span, 360.0, atol=spacing):
            # Calculate full circumference at y_center
            _, _, lx = GEODE.inv(0, y_center, 180, y_center)
            lx *= 2
        else:
            _, _, lx = GEODE.inv(x_min, y_center, x_max, y_center)
        _, _, ly = GEODE.inv(x_center, y_min, x_center, y_max)
    else:
        lx, ly = x_max - x_min, y_max - y_min

    # 3. Define Analysis Scales (r_values). Effective max scale limited by half the domain size
    effective_max_r = min(max_r_m, min(lx, ly) / 2.0)

    # Ensure r_values starts at r_step (or a physically resolved scale)
    r_values = np.arange(r_step, effective_max_r + r_step / 2.0, r_step)

    if r_values.size < 1:
        raise ValueError("No valid scales generated by scale_increments.")

    r_coord_da = xr.DataArray(
        r_values, dims="r", name="r",
        attrs={"units": "m", "long_name": "Separation distance (scale)"}
    )

    # 4. Define Shift Indices (nx, ny)
    max_nx = int(np.ceil(effective_max_r / dx))
    max_ny = int(np.ceil(effective_max_r / dy))

    # Grid search space (2 * max_n * resolution_factor + 1, centered at 0)
    nx_vals = np.linspace(-max_nx, max_nx, 2 * max_nx * resolution_factor + 1, dtype=int)
    ny_vals = np.linspace(-max_ny, max_ny, 2 * max_ny * resolution_factor + 1, dtype=int)

    da_nx = xr.DataArray(nx_vals, dims="nx")
    da_ny = xr.DataArray(ny_vals, dims="ny")
    ny_grid, nx_grid = xr.broadcast(da_ny, da_nx)

    # 5. Calculate Distance (r) and Angle (phi) for all shifts (nx, ny)
    if use_geode:
        # Calculate metric shifts
        dx_shift = nx_grid * dx
        dy_shift = ny_grid * dy

        # Calculate the angle based on the metric shifts (standard math angle: 0=E, CCW)
        cartesian_angle_rad = np.arctan2(dy_shift, dx_shift)
        distance_approx = np.sqrt(dx_shift ** 2 + dy_shift ** 2)

        # pyproj.Geod.fwd requires azimuth (0=N, CW). Convert the standard math angle to azimuth.
        # Azimuth (deg) = 90 - Cartesian Angle (deg)
        azimuth_deg = 90.0 - np.rad2deg(cartesian_angle_rad)

        # Forward calculation from center to shifted point
        fwd_lon, fwd_lat, _ = GEODE.fwd(
            np.full_like(distance_approx, x_center),
            np.full_like(distance_approx, y_center),
            azimuth_deg,
            distance_approx
        )

        # Inverse calculation gives the true geodesic distance and angle (azimuth)
        # FIX: Ensure we capture the INITIAL azimuth (azimuth1) which is the first return value.
        angle_vals_deg, _, true_distance_vals = GEODE.inv(
            np.full_like(fwd_lon, x_center),
            np.full_like(fwd_lat, y_center),
            fwd_lon,
            fwd_lat
        )
        distance_vals = true_distance_vals

        # NEW FIX: Convert Azimuth (0=N, CW) to Math Angle (0=E, CCW)
        math_angle_deg = 90.0 - angle_vals_deg

        angle_vals = np.deg2rad(math_angle_deg)  # Angle in radians
    else:
        # Cartesian grid
        dx_grid = nx_grid * dx
        dy_grid = ny_grid * dy
        distance_vals = np.sqrt(dx_grid ** 2 + dy_grid ** 2).values
        angle_vals = np.arctan2(dy_grid, dx_grid).values

    # 6. Create Distance and Angle DataArrays
    distance = xr.DataArray(distance_vals, dims=("ny", "nx"), name="distance_grid")
    distance.attrs = {"units": "m", "long_name": "Distance from origin"}

    angle = xr.DataArray(angle_vals, dims=("ny", "nx"), name="angle_grid")
    angle.attrs = {"units": "radians", "long_name": "Angle of offset from origin"}

    # 7. Angle Weighting Calculation
    flat_angles = angle_vals.flatten()
    # Use float for unique to avoid precision issues if the angles are very close
    unique_angles, counts = np.unique(flat_angles.round(10), return_counts=True)

    # Calculate weight normalization factor
    sum_inv_counts = np.sum(1.0 / counts)
    angle_weights = {a: (2 * np.pi / sum_inv_counts) * (1.0 / c)
                     for a, c in zip(unique_angles, counts)}

    angle_weights_da = xr.DataArray(
        np.vectorize(angle_weights.get)(angle_vals.round(10)),  # Ensure keys match
        dims=("ny", "nx"), name="angle_weight"
    )

    # 8. Create r_mask (Bin distances into r_values)
    lower_bound = r_values[:, np.newaxis, np.newaxis] - r_step / 2.0
    upper_bound = r_values[:, np.newaxis, np.newaxis] + r_step / 2.0

    # Check if distance falls within the bin [r_i - dr/2, r_i + dr/2)
    mask = (distance_vals[np.newaxis, :, :] >= lower_bound) & \
           (distance_vals[np.newaxis, :, :] < upper_bound)

    r_mask = xr.DataArray(
        mask, dims=("r", "ny", "nx"),
        coords={"r": r_coord_da, "ny": da_ny, "nx": da_nx}, name="r_mask"
    )

    # 9. Create Increments Dataset
    increments = xr.Dataset(
        {"r": r_coord_da,
         "mask": r_mask,
         "distance_grid": distance,
         "angle_grid": angle,
         "angle_weight": angle_weights_da,
         "delta_x_spacing": xr.DataArray(dx, name="delta_x_spacing", attrs={"units": "m"}),
         "delta_y_spacing": xr.DataArray(dy, name="delta_y_spacing", attrs={"units": "m"})}
    )

    # 10. Filter scales based on directional coverage
    valid_mask = filter_by_directional_coverage(increments, min_valid_shifts=min_valid_shifts)
    increments = increments.sel(r=valid_mask)

    # 11. Verbose Output
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
        scale_chunk_size: Optional[int] = 1
) -> xr.DataArray:
    """
    Computes the scale-space integral (convolution) of the integrand.

    The core calculation is: Integral[0 to 2*ell] (dG/dr * r * integrand) dr,
    with a normalization correction for truncation.
    """
    if verbose: print(f"Calculating scale-space integral for '{name}'...")

    r_coord = integrand.r

    # --- Handle edge cases and length_scales initialization ---
    if r_coord.size == 0:
        raise ValueError("Warning: Integrand has no 'r' dimension or it's empty.")

    if length_scales is None:
        # Use all available r coordinates as the integration scales
        length_scales = r_coord.values

    if not isinstance(length_scales, np.ndarray):
        length_scales = np.array(length_scales)

    # Filter scales that exceed the maximum available r value
    length_scales = length_scales[length_scales <= r_coord.max().values]

    if verbose and length_scales.size:
        print(f"Externally defined length_scales:")
        min_scale = length_scales.min()
        max_scale = length_scales.max()
        print(f"  Effective scale limits: {min_scale:8.2f} m - {max_scale:8.2f} m)")
        print("==============================================================")

    if not length_scales.size:
        if verbose: print(f"Warning: No valid length scales to use in integration."
                          f"Using {r_coord.max().values:8.2f} m")

        length_scales = np.atleast_1d(r_coord.max().values)

    # --- Get Integration Kernel and Prepare Data ---

    # dg_dr has dimensions (scale, r)
    _, dg_dr = get_integration_kernels(r_coord, length_scales,
                                       normalization=weighting,
                                       return_derivative=True)

    # The r coordinate must be promoted to the same shape as dg_dr for the mask
    r_coord = r_coord.to_dataset(name='R', promote_attrs=False)['R']

    # Xarray automatically broadcasts R (dim: r) to match dg_dr (dims: scale, r)
    term_to_integrate = (dg_dr * r_coord) * integrand

    # Mask to truncate at r <= 2 * ell (where ell = dg_dr.scale)
    term_to_integrate_masked = term_to_integrate.where(r_coord <= 2 * dg_dr.scale, other=0.0)

    # Estimate the fraction of the integral retained after applying the 2*ell mask
    retention_fraction = term_to_integrate_masked.sum("r") / term_to_integrate.sum("r")

    # Safety: If the denominator is zero or near-zero, use 1.0 to prevent division errors.
    retention_fraction = retention_fraction.where(retention_fraction > 1e-6, 1.0)

    # Integrate the masked term and normalize by the retained fraction
    integral = term_to_integrate_masked.integrate("r") / retention_fraction

    # Assign final coordinates and rename
    integral = integral.rename(name).assign_coords(scale=dg_dr.scale)

    if hasattr(integral.data, "chunks"):
        # Explicitly chunk the 'scale' dimension if a chunk size is provided
        integral = integral.chunk(scale=scale_chunk_size)

    if verbose:
        print(f"Finished calculating scale-space integral '{name}'. Shape: {integral.shape}")

    return integral


def process_single_r_for_field_chunk_optimized(
        field_chunk_ds: xr.Dataset,  # This now contains u, v, w
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
    Optimized: Implements shift caching to compute pad/roll only once per unique
    shift vector (nx, ny), drastically reducing Dask graph overhead inside the
    angle loop.
    """
    if transform_type != "delta_u_cubed":
        raise ValueError(f"Transform_type: {transform_type}, not implemented.")

    # --- 1. Identify valid angle/shift combinations and extract parameters ---
    valid_mask = scale_mask_for_r.data.astype(bool)

    # --- 2. Extract and Normalize Parameters (NumPy) ---
    if not np.any(valid_mask):
        raise ValueError(f"No valid value found for {r_scalar_val} m.")

    angles = scale_angle_grid.data[valid_mask]
    weights = angle_weight_grid.data[valid_mask]
    ny_idx, nx_idx = np.where(valid_mask)
    nx_values = nx_shift_coords.data[nx_idx].astype(int)
    ny_values = ny_shift_coords.data[ny_idx].astype(int)

    # Normalize weights to sum to 2π
    weights *= 2.0 * np.pi / np.sum(weights)
    total_weight = weights.sum()

    # Combine all parameters into a list of tuples for iteration
    angle_params = list(zip(angles, nx_values, ny_values, weights))

    # Initialize weighted_sum DataArray to zero (using 'u' as the template)
    dims = field_chunk_ds["u"].dims
    coords = field_chunk_ds["u"].coords
    weighted_sum = xr.DataArray(
        da.zeros_like(field_chunk_ds["u"].data, dtype=np.float32),
        dims=dims,
        coords=coords,
        name="weighted_sum_integrand"
    )

    # Loop over all angles
    for phi, nx_shift, ny_shift, weight in angle_params:
        # Roll the data (DASK GRAPH RECONSTRUCTION)
        rolled_ds = roll_with_boundary_handling(field_chunk_ds,
                                                int(nx_shift), int(ny_shift), x_dim, y_dim,
                                                x_boundary_type, y_boundary_type)

        # Compute the cubed difference (Efficient positional subtraction)
        result = delta_u_cubed_geographic(field_chunk_ds, rolled_ds, phi)

        # Accumulate the weighted sum
        weighted_sum += result * weight

        # Cleanup (Crucial for memory safety)
        del rolled_ds, result

    # Final result is average over all angles (weighted sum / total weight)
    integrand = weighted_sum / total_weight.clip(min=1e-12)

    # Handle final coordinate assignment
    if 'r' in integrand.coords:
        integrand = integrand.drop_vars('r')

    integrand = integrand.rename(transform_type)
    integrand = integrand.expand_dims({'r': [r_scalar_val]})

    return integrand.transpose('r', ...)


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
            (process_single_r_for_field_chunk_optimized(
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

    # calculate the integrand using map_blocks
    integrand = xr.map_blocks(block_fn, field, template=template)

    # This prepares the array for efficient integration/reduction in the next step.
    integrand = integrand.chunk(r=1)

    return integrand


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
    allow_rechunking = kwargs.get("allow_rechunking", True)

    # Process length scales input
    if length_scales is None:
        pass
    elif isinstance(length_scales, (list, tuple, np.ndarray)) or np.isscalar(length_scales):
        length_scales = np.atleast_1d(length_scales).astype(np.float32)
    else:
        raise ValueError("scale_transfer: 'compute.scales' not provided or invalid."
                         "'scales' must be an iterable, or scalar of length scales in meters.")

    # Attempt to retrieve coordinates by name
    if x_name is None and y_name is None:
        # Infer coordinates using helper (assumes CF compliance)
        y_name, x_name = get_spatial_dims(wind)

    if x_name in wind and y_name in wind:
        x_coord = wind[x_name]
        y_coord = wind[y_name]
    else:
        raise KeyError(f"Specified coordinate names {x_name}, {y_name} not found in dataset.")

    # Determine boundary conditions
    x_boundary, y_boundary = infer_boundary_conditions(x_coord, **kwargs)

    if verbose:
        print(f"Inferred boundary conditions -> x: {x_boundary}, y: {y_boundary}")

    # Infer max_r input from user-defined length scales if given with (r <= 2 l_max) constraints
    max_r_m = get_max_radial_distance(
        length_scales=length_scales,
        max_r_input=kwargs.get("max_r", None)
    )

    # Compute scale increments
    increments = scale_increments(
        x_coord, y_coord, max_r_m,
        verbose=verbose, resolution_factor=1,
        min_valid_shifts=kwargs.get("min_valid_shifts", 10)
    )

    # Ensure the result fits in memory or compute in chunks along non-spatial dimensions
    # Spatial dimensions are only rechunked if spatial plane times scales does not fit in memory
    if allow_rechunking:
        # intermediate array size is increased by the sie of the radial distances (L / 2dx)
        scale_size = 1 if length_scales is None else len(length_scales)
        scale_size = increments.r.size * scale_size

        wind = ensure_optimal_chunking(wind, spatial_dims=(y_name, x_name),
                                       # preferred chunk sizes
                                       preferred={'z': 1, 'time': 1},
                                       # limit chunk size (MB)
                                       desired_chunk_size_mb=64,
                                       # Safer 50% threshold for Dask compute budget
                                       memory_threshold_ratio=0.5,
                                       # extra memory for temporary arrays, i.e., padding
                                       working_set_multiplier=42,
                                       # extra memory required for kernel radial distance
                                       output_scale_mult=scale_size,
                                       # No derivatives required here. Allow min z-chunk size = 1
                                       deriv_edge_order=0)

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

    # Apply normalized mollifier kernel
    energy_transfer_rate = scale_space_integral(
        integrand=integrand,
        name="energy_transfer",
        length_scales=length_scales,
        weighting="2D",
        verbose=verbose,
        scale_chunk_size=ls_chunk_size
    )

    energy_transfer_rate.attrs.update({
        'units': "W / kg",
        'standard_name': "specific_kinetic_energy_transfer",
        'long_name': "Specific transfer rate of kinetic energy across scales",
        'description': "Computed using third-order structure functions and mollifier kernels. "
                       "Positive means forward energy transfer towards smaller scales"
    })

    # reassign coordinates from input data
    energy_transfer_rate = energy_transfer_rate.assign_coords(**{x_name: x_coord, y_name: y_coord})
    energy_transfer_rate[x_name].attrs = x_coord.attrs
    energy_transfer_rate[y_name].attrs = y_coord.attrs

    # Check if the result fits in memory (the result is length_scale times the input's size)
    energy_transfer_rate = energy_transfer_rate.to_dataset()

    # add scale coordinate attributes
    energy_transfer_rate["scale"].attrs.update({
        "standard_name": "horizontal_scale",
        "long_name": "horizontal scale",
        "units": "m"
    })

    # transpose to have scale as the last dimension
    energy_transfer_rate = energy_transfer_rate.transpose(..., "scale")

    # --- enforce one-scale-at-a-time tasks for reductions/writes ---
    energy_transfer_rate = energy_transfer_rate.chunk(scale=1)

    if verbose:
        # Avoid triggering a full compute if Dask-backed (expensive convolutions)
        arr = energy_transfer_rate.energy_transfer.data
        if hasattr(arr, "chunks"):
            print("Finished computing energy transfer rate.")
        else:
            domain_total = float(energy_transfer_rate.energy_transfer.sum())
            print(f"Finished computing energy transfer rate. "
                  f"Domain total: {domain_total:.3e} W/kg")

    return energy_transfer_rate
