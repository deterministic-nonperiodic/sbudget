from typing import Any, Optional, Union, List

import dask
import numpy as np
import xarray as xr
from pint import Quantity
from pyproj import Geod
from scipy.integrate import trapezoid

from .budget import get_spatial_dims
from .cf_coords import is_geographic_grid, _is_global_longitude
from .chunking_tools import CacheManager
from .chunking_tools import DEFAULT_CHUNK_SIZE_MB
from .chunking_tools import ensure_optimal_chunking
from .constants import earth_radius, epsilon

# Constants
_GEODE = Geod(ellps="WGS84")


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
    x_boundary = kwargs.get("x_coord_boundary", "reflect")

    if x_boundary != "periodic" and _is_global_longitude(x_coord):
        print(f"[boundary_condition] Global domain detected; "
              f"overriding user defined {x_boundary} with {'periodic'}")
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
    integrals = trapezoid(mollifier * area_weights[None, :], x=radial_positions, axis=1)
    mollifier_normalized = mollifier / np.expand_dims(integrals, axis=-1).clip(epsilon, None)

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
        derivative_normalized = (derivative.T / integrals.clip(epsilon, None)).T
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
        x_dim: str = None,
        y_dim: str = None,
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
    if x_dim is None or y_dim is None:
        y_dim, x_dim = get_spatial_dims(data)

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
    if not use_geode:
        return float(np.abs(np.median(np.diff(coord.values))))

    if axis == 'x':
        _, _, dist = _GEODE.inv(coord[0].item(), center, coord[1].item(), center)
    else:
        _, _, dist = _GEODE.inv(center, coord[0].item(), center, coord[1].item())
    return dist


def get_max_radial_distance(
        length_scales: Optional[Union[np.ndarray, List[float]]] = None,
        max_r_input: Optional[Union[float, Quantity]] = None,
) -> Optional[float]:
    """
    Determine the maximum radial distance (max_r_m) for structure function
    computation, constrained by twice the largest length scale.

    Parameters
    ----------
    length_scales : array-like of float, optional
        Filter length scales (in meters). Used to constrain max_r to 2 * max(length_scales).
    max_r_input : float or pint.Quantity, optional
        User-provided maximum radial distance.

    Returns
    -------
    max_r_m : float or None
        Constrained maximum radial distance in meters, or None if both inputs are None.
    """

    # ---------------------------------------------------------------
    # 1. Handle both inputs being None → no information available
    # ---------------------------------------------------------------
    if length_scales is None and max_r_input is None:
        return None

    # ---------------------------------------------------------------
    # 2. Determine initial max_r_m from user input or from length_scales
    # ---------------------------------------------------------------
    if max_r_input is not None:
        if isinstance(max_r_input, Quantity):
            max_r_m = max_r_input.to("meter").magnitude
        else:
            max_r_m = float(max_r_input)
    else:
        # Use the maximum provided length scale as a baseline
        max_r_m = float(max(length_scales))

    # ---------------------------------------------------------------
    # 3. Enforce constraint: cannot exceed twice the largest scale
    # ---------------------------------------------------------------
    if length_scales is not None:
        max_scale_limit = 2.0 * float(max(length_scales))
        max_r_m = min(max_r_m, max_scale_limit)

    return max_r_m


def scale_increments(
        x_coord: xr.DataArray,
        y_coord: xr.DataArray,
        max_r_m: float | None = None,
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
            _, _, lx = _GEODE.inv(0, y_center, 180, y_center)
            lx *= 2
        else:
            _, _, lx = _GEODE.inv(x_min, y_center, x_max, y_center)
        _, _, ly = _GEODE.inv(x_center, y_min, x_center, y_max)
    else:
        lx, ly = x_max - x_min, y_max - y_min

    # 3. Define Analysis Scales (r_values). Effective max scale limited by half the domain size
    domain_half_size_m = min(lx, ly) / 2.0
    effective_max_r = min(max_r_m, domain_half_size_m) if max_r_m else domain_half_size_m

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
        fwd_lon, fwd_lat, _ = _GEODE.fwd(
            np.full_like(distance_approx, x_center),
            np.full_like(distance_approx, y_center),
            azimuth_deg,
            distance_approx
        )

        # Inverse calculation gives the true geodesic distance and angle (azimuth)
        # FIX: Ensure we capture the INITIAL azimuth (azimuth1) which is the first return value.
        angle_vals_deg, _, true_distance_vals = _GEODE.inv(
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
    requested_max_r = "None" if not max_r_m else f"{max_r_m:.2f} m"

    if verbose:
        print("================== Scale Increments Summary ==================")
        print(f"  Domain size         : Lx = {lx:8.2f} m, Ly = {ly:8.2f} m")
        print(f"  Grid spacing        : dx = {dx:8.2f} m, dy = {dy:8.2f} m")
        print(f"  Effective min scale : {effective_min_r:8.2f} m (dr = {r_step:8.2f} m)")
        print(f"  Effective max scale : {effective_max_r:8.2f} m (Requested: {requested_max_r})")
        print("==============================================================")

    return increments


def validate_length_scales(
        length_scales: np.ndarray | list | float | str | None,
        r_coord: xr.DataArray,
        verbose: bool = True,
        label: str = "scale-integral"
) -> np.ndarray:
    """
    Validate and sanitize user-provided length scales for integration.

    Scalars and strings are automatically converted to 1D arrays.
    Out-of-range values are clamped to nearest r_coord edge.

    Parameters
    ----------
    length_scales : array-like, float, str, or None
        Candidate physical scales (ℓ values). If None, defaults to r_coord.values.
    r_coord : xr.DataArray
        Coordinate array representing available separation distances (r).
    verbose : bool, optional
        If True, prints diagnostic information.
    label : str, optional
        Prefix used for log messages.

    Returns
    -------
    np.ndarray
        Validated, unique, and sorted array of length scales.
    """
    if r_coord.size == 0:
        raise ValueError("`r_coord` is empty — cannot determine valid scale range.")

    # --- Normalize input type ---
    if length_scales is None:
        length_scales = np.asarray(r_coord.values)
    else:
        length_scales = np.atleast_1d(length_scales)
        try:
            length_scales = length_scales.astype(float)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid length_scales: must be numeric, got {length_scales!r}")

    # --- Remove NaN and non-finite values ---
    length_scales = length_scales[np.isfinite(length_scales)]
    if length_scales.size == 0:
        r_max = float(r_coord.max().item())
        if verbose:
            print(f"[{label}] Warning: No valid numeric scales found; using r_max={r_max:8.2f} m")
        return np.atleast_1d(r_max)

    # --- Clamp to valid range ---
    r_min, r_max = float(r_coord.min().item()), float(r_coord.max().item())
    length_scales = np.clip(length_scales, r_min, r_max)

    # --- Remove duplicates (preserve order) ---
    _, unique_idx = np.unique(length_scales, return_index=True)
    length_scales = length_scales[np.sort(unique_idx)]

    # --- Sort ascending ---
    length_scales = np.sort(length_scales)

    # --- Ensure non-empty array ---
    if length_scales.size == 0:
        length_scales = np.atleast_1d(r_max)

    # --- Verbose report ---
    if verbose:
        min_scale, max_scale = length_scales.min(), length_scales.max()
        print(f"[{label}] Externally defined length_scales:")
        print(f"[{label}]   Effective scale limits: {min_scale:8.2f} m - {max_scale:8.2f} m")
        print(f"[{label}]   {length_scales.size} unique scales retained")
        print("==============================================================")

    return length_scales


def process_single_r_for_field_chunk(
        field_chunk_ds: xr.Dataset,
        increments: xr.Dataset,
        x_dim: str = "x",
        y_dim: str = "y",
        transform_type: str = "delta_u_cubed",
        cache_manager: CacheManager = None,
) -> xr.DataArray:
    """
    Compute the integrand contribution for a *single* scale r
    for one spatial Dask block.

    Parameters
    ----------
    field_chunk_ds : xr.Dataset
        A spatially-chunked block of the full wind field dataset.
    increments : xr.Dataset
        Slice of the increments dataset containing *only one r value*.
        Must include nx, ny, mask, angle_grid, etc.
    x_dim, y_dim : str
        Names of spatial dimensions.
    transform_type : str
        Transformation to compute ("delta_u_cubed").
    cache_manager : CacheManager, optional
        Worker-local, block-local cache. If None, a new cache manager
        is constructed (inside map_blocks safe, but slower).

    Returns
    -------
    xr.DataArray with shape (r=1, ...) giving the integrand at this r.
    """

    if transform_type != "delta_u_cubed":
        raise ValueError(f"Transform_type '{transform_type}' not implemented.")

    x_boundary_type = increments.attrs.get("x_boundary_type", "periodic")
    y_boundary_type = increments.attrs.get("y_boundary_type", "reflect")

    # ------------------------------------------------------------
    # Extract mask + shift directions for this r
    # ------------------------------------------------------------
    mask = increments.mask.data.astype(bool)
    ny_idx, nx_idx = np.where(mask)

    nx_values = increments.nx.data[nx_idx].astype(int)
    ny_values = increments.ny.data[ny_idx].astype(int)

    angles = increments.angle_grid.data[mask].astype(np.float32)
    weights = increments.angle_weight.data[mask].astype(np.float32)
    weights /= np.clip(np.sum(weights), epsilon, None)

    # ------------------------------------------------------------
    # Weighted sum over all discrete angles for this r
    # ------------------------------------------------------------
    weighted_shifts: list[xr.DataArray] = []

    for phi, nx, ny, w in zip(angles, nx_values, ny_values, weights):
        rolled_ds = roll_with_boundary_handling(
            field_chunk_ds,
            nx, ny,
            x_dim, y_dim,
            x_boundary_type, y_boundary_type
        )

        # Spill to memory/disk depending on avail RAM
        if cache_manager is not None:
            rolled_ds = cache_manager.persist(rolled_ds, key=f"{nx:+04d}_{ny:+04d}")

        delta_u_cubed = delta_u_cubed_geographic(field_chunk_ds, rolled_ds, phi)
        weighted_shifts.append(delta_u_cubed * w)

    integrand = sum(weighted_shifts).rename(transform_type)

    # Attach the r-value dimension
    r_val = float(increments["r"].item())
    return integrand.expand_dims(r=[r_val])


def _block_space_scale_integral(
        field_chunk: xr.Dataset,
        increments: xr.Dataset,
        x_dim: str,
        y_dim: str,
        transform_type: str,
        kernel_derivative: xr.Dataset
) -> xr.DataArray:
    """
    Performs scale-space integral per Dask spatial block.

    Workflow inside each block:
        1. loop over all r-values → integrand(r, ...)
        2. build mollifier kernels for all ℓ
        3. compute (dg/dr)(r;ℓ) * r * integrand
        4. perform truncated, normalized ∫ ... dr
        5. return final (scale, ...) DataArray
    """

    # ----------------------------------------------------------------
    # CacheManager: per block per worker
    # ----------------------------------------------------------------
    cache_manager = CacheManager(verbose=False, force_threshold=0.20, auto_cleanup=True)

    # ----------------------------------------------------------------
    # Compute projected cubed velocity differences for all r-values
    # ----------------------------------------------------------------
    r_coord = increments["r"]
    integrand_blocks = []

    for r_value in r_coord.values:
        block = process_single_r_for_field_chunk(
            field_chunk_ds=field_chunk,
            increments=increments.sel(r=r_value),
            x_dim=x_dim,
            y_dim=y_dim,
            transform_type=transform_type,
            cache_manager=None,
        )

        # Spill to memory/disk depending on avail RAM
        block = cache_manager.persist(block)

        integrand_blocks.append(block)

    # Merge integrand and rechunk to 1
    integrand = xr.concat(integrand_blocks, dim="r").chunk(r=1)

    # ----------------------------------------------------------------
    # Kernel-weighting / masking
    # ----------------------------------------------------------------
    weighted = (kernel_derivative * r_coord) * integrand
    masked = weighted.where(r_coord <= 2 * kernel_derivative.scale, 0.0)

    num = masked.sum("r")
    den = weighted.sum("r")
    retention_fraction = xr.where(den > epsilon, num / den, 1.0)

    integral = masked.integrate("r") / retention_fraction

    # ----------------------------------------------------------------
    # Add scale coordinate
    # ----------------------------------------------------------------
    integral = integral.assign_coords(scale=kernel_derivative.scale)

    return integral


def scale_transfer(
        field: xr.Dataset,
        increments: xr.Dataset,
        x_dim: str,
        y_dim: str,
        length_scales: Optional[np.ndarray] = None,
        name: str = "energy_transfer",
        transform_type: str = "delta_u_cubed",
        weighting: str = "sphere",
        verbose: bool = False
) -> xr.DataArray:
    """
    Compute the inter-scale energy transfer.

    This integrates the quantity:
        Du = ∫₀^{2ℓ} (dG/dr) (δu ⋅ r̂) |δu|²  dr
    with normalization for truncated kernels (r ≤ 2ℓ).

    Parameters
    ----------
    field : xr.Dataset
        Input dataset containing the 3D wind field (u, v, w).
    increments: xr.Dataset
        Grid information
    name : str
        Name for the resulting variable.
    x_dim, y_dim : str
        Names of the spatial dimensions.
    length_scales : np.ndarray, optional
        Target physical length scales ℓ for the integral.
        Defaults to all available r-values.
    transform_type : str
        Type of integrand, i.e., cubed velocity differences
    weighting : {"2D", "3D"}, optional
        Mollifier normalization type for the kernel.
    verbose : bool, optional
        If True, print progress and summary info.

    Returns
    -------
    xr.DataArray
        Integrated field with dimension 'scale' and same spatial dims as integrand.
    """
    if verbose:
        print(f"[scale-integral] Computing '{name}' ...")

    if "r" not in increments.dims:
        raise ValueError("[scale-integral] increments must contain dimension 'r'")

    r_coord = increments["r"]
    if r_coord.size == 0:
        raise ValueError("[scale-integral] increments contain no r-values")

    # Normalize length scales
    length_scales = validate_length_scales(length_scales, r_coord, verbose=verbose)

    # ------------------------------------------------------------
    # Build integration kernels G_ℓ(r)
    # ------------------------------------------------------------
    _, kernel_derivative = get_integration_kernels(
        r_coord, length_scales,
        normalization=weighting,
        return_derivative=True,
    )

    # ------------------------------------------------------------
    # The full scale transfer computation is performed for each chunks in parallel
    # ------------------------------------------------------------
    transfer = xr.map_blocks(
        _block_space_scale_integral,
        field,
        kwargs=dict(
            increments=increments,
            x_dim=x_dim,
            y_dim=y_dim,
            transform_type=transform_type,
            kernel_derivative=kernel_derivative
        ),
        # Build template: final output is (scale, spatial dims)
        template=field["u"].expand_dims(scale=length_scales).rename(name),
    )

    return transfer.rename(name)


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
    ls_chunk_size = kwargs.get("ls_chunk_size", 1)
    allow_rechunking = kwargs.get("allow_rechunking", True)
    chunk_size_mb = float(kwargs.get("chunk_size", DEFAULT_CHUNK_SIZE_MB))

    # Resetting dask defaults from user override
    dask.config.set({
        "array.chunk-size": f"{max(1.0, chunk_size_mb):.1f}MB",
        "array.slicing.split_large_chunks": False,
    })

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

    # Infer max_r input from user-defined length scales if given with (r <= 2 l_max) constraints
    max_r_m = kwargs.get("max_r", None)
    max_r_m = get_max_radial_distance(length_scales=length_scales, max_r_input=max_r_m)

    # Compute scale increments
    increments = scale_increments(
        x_coord, y_coord, max_r_m,
        verbose=verbose, resolution_factor=1,
        min_valid_shifts=kwargs.get("min_valid_shifts", 10)
    )

    # Determine boundary conditions
    x_boundary, y_boundary = infer_boundary_conditions(x_coord, **kwargs)
    increments.attrs.update({"x_boundary_type": x_boundary, "y_boundary_type": y_boundary})

    if verbose:
        print(f"[main] Inferred boundary conditions -> x: {x_boundary}, y: {y_boundary}")

    # Ensure the result fits in memory or compute in chunks along non-spatial dimensions
    # Spatial dimensions are only rechunked if spatial plane times scales does not fit in memory
    if allow_rechunking:
        wind = ensure_optimal_chunking(wind, spatial_dims=(y_name, x_name), vertical_dim="z",
                                       # largest chunk limit chunk size (MB)
                                       desired_chunk_size_mb=float(chunk_size_mb),
                                       # Data size increase by number of scales
                                       output_scale_mult=increments['r'].size)

    # Compute third-order structure functions. Mask missing values in velocity components.
    nan_mask = xr.concat(
        [xr.ufuncs.isnan(wind[var]) for var in velocity_vars], dim='component'
    ).any(dim='component')

    # Compute third-order structure functions for each radial distance
    energy_transfer_rate = scale_transfer(
        field=wind.fillna(0.0),
        increments=increments,
        name="energy_transfer",
        length_scales=length_scales,
        x_dim=x_name,
        y_dim=y_name,
        weighting="2D",
        verbose=verbose
    ).where(~nan_mask)

    # Generate metadata
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
    energy_transfer_rate = energy_transfer_rate.chunk(scale=ls_chunk_size)

    return energy_transfer_rate
