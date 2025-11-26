from typing import Any, Dict

import numpy as np
import xarray as xr
from dask.distributed import Future
from dask.distributed import get_client

from .cf_coords import get_spatial_dims
from .constants import earth_radius, epsilon
from .grid_utils import scale_increments
from .memory_manager import CacheManager, ensure_optimal_chunking


# --------------------------------------------------------------------------------------------------
# -----------------------          Kernel Implementation           ---------------------------------
# --------------------------------------------------------------------------------------------------
def _evaluate_mollifier(radial_positions: np.ndarray, length_scales: np.ndarray) -> np.ndarray:
    """
    Evaluate the standard mollifier kernel for all combinations of
    radial distances and filter length scales.

    Returns
    -------
    mollifier : np.ndarray of shape (n_scales, n_r)
    """
    r_grid, ell_grid = np.meshgrid(radial_positions, length_scales, indexing="ij")
    ratio_squared = (r_grid / (2 * ell_grid)) ** 2
    inside_support = ratio_squared < 1
    denominator = 1 - ratio_squared

    mollifier = np.zeros_like(denominator)
    mollifier[inside_support] = np.exp(-1.0 / denominator[inside_support])

    return mollifier.T  # final shape: (n_scales, n_r)


def _evaluate_mollifier_derivative(radial_positions: np.ndarray,
                                   length_scales: np.ndarray) -> np.ndarray:
    """
    Evaluate the radial derivative of the standard mollifier for all combinations of
    radial distances and filter length scales.

    Returns
    -------
    derivative : np.ndarray of shape (n_scales, n_r)
    """
    r_grid, ell_grid = np.meshgrid(radial_positions, length_scales, indexing="ij")
    ratio_squared = (r_grid / (2 * ell_grid)) ** 2
    inside_support = ratio_squared < 1
    denominator = 1 - ratio_squared

    derivative = np.zeros_like(denominator)
    derivative[inside_support] = (
            -r_grid[inside_support] / (2 * ell_grid[inside_support] ** 2)
            * np.exp(-1.0 / denominator[inside_support])
            / denominator[inside_support] ** 2
    )

    return derivative.T  # final shape: (n_scales, n_r)


def _get_mollifier_norm(mollifier: xr.DataArray, method: str, r_name: str = "r") -> xr.DataArray:
    """
    Normalization for mollifier kernel using the appropriate area weighting.

    Returns
    -------
    mollifier_normalized : xr.DataArray of dimensions (scales, r)
    """
    radial_positions = mollifier[r_name]

    if method == "2D":
        area_weights = 2 * np.pi * radial_positions
    elif method == "sphere":
        area_weights = 2 * np.pi * earth_radius * np.sin(radial_positions / earth_radius)
    elif method == "3D":
        area_weights = 4 * np.pi * radial_positions ** 2
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Broadcast area weights: shape (1, n_r)
    return (mollifier * area_weights).integrate(r_name)


def get_integration_kernels(r_da: xr.DataArray, scales: xr.DataArray,
                            normalization="2D", scaled=False,
                            return_derivative=True) -> xr.DataArray:
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
    scaled : bool
        Whether to scale the kernel by r for integration
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
    length_scales = scales.values

    r_name = r_da.name or "r"
    s_name = scales.name or "scale"

    # Get mollifier kernel
    kernel_name = "G_kernel"

    kernel = xr.DataArray(
        _evaluate_mollifier(radial_positions, length_scales),
        dims=[s_name, r_name], coords={s_name: scales, r_name: r_da}
    )

    # Get normalization factor: area-weighted integral over r
    kernel_norm = _get_mollifier_norm(kernel, normalization, r_name=r_name)

    if return_derivative:
        kernel_name = "dG_dr_kernel"
        # Get the mollifier kernel's derivative
        kernel = xr.DataArray(
            _evaluate_mollifier_derivative(radial_positions, length_scales),
            dims=[s_name, r_name], coords={s_name: scales, r_name: r_da}
        )

    # Normalize kernel:
    kernel = kernel / kernel_norm.clip(epsilon, None)

    # Scale by r for integration
    if scaled:
        kernel = kernel * kernel[r_name]

    return kernel.rename(kernel_name)


# --------------------------------------------------------------------------------------------------
# ----------------------      Scale-Space Integral Computation     ---------------------------------
# --------------------------------------------------------------------------------------------------
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
        Value to use when boundary_type == 'constant'.

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


def process_single_r_for_field_chunk(
        field_chunk_ds: xr.Dataset,
        r_value: float,
        increments_data: Dict,  # NOW a simple dictionary
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
    r_value: float
        The radial distance at which to compute the integrand.
    increments_data : Dict
        Slice of the increments dataset containing *only one r value*.
        Must include nx, ny, mask, angle_grid, etc.
    x_dim, y_dim : str
        Names of spatial dimensions.
    transform_type : str
        Transformation to compute ("delta_u_cubed").
    cache_manager : CacheManager, optional
        Worker-local, block-local cache.

    Returns
    -------
    xr.DataArray with shape (r=1, ...) giving the integrand at this r.
    """

    if transform_type != "delta_u_cubed":
        raise ValueError(f"Transform_type '{transform_type}' not implemented.")

    # ------------------------------------------------------------
    # Extract pre-indexed NumPy arrays (Fast dictionary lookup)
    # ------------------------------------------------------------
    grid_data = increments_data[r_value]

    # Boundary types are stored in the main dictionary attributes
    x_boundary_type = increments_data.get("x_boundary_type", "periodic")
    y_boundary_type = increments_data.get("y_boundary_type", "reflect")

    angle_parameters = zip(
        grid_data["angles"],
        grid_data["weights"],
        grid_data["nx_values"],
        grid_data["ny_values"]
    )

    # ------------------------------------------------------------
    # Weighted sum over all discrete angles for this r
    # ------------------------------------------------------------
    weighted_shifts: list[xr.DataArray] = []

    for phi, weight, nx, ny in angle_parameters:
        # Roll the field to the angle grid
        rolled_ds = roll_with_boundary_handling(
            field_chunk_ds, nx, ny, x_dim, y_dim,
            x_boundary_type, y_boundary_type
        )

        # Spill to memory/disk depending on avail RAM
        if cache_manager is not None:
            rolled_ds = cache_manager.persist(rolled_ds, key=f"{nx:+04d}_{ny:+04d}")

        delta_u_cubed = delta_u_cubed_geographic(field_chunk_ds, rolled_ds, phi)
        weighted_shifts.append(delta_u_cubed * weight)

    integrand = sum(weighted_shifts).rename(transform_type)

    # Attach the r-value dimension
    return integrand.expand_dims(r=[r_value])


def _resolve_increments_data(increments_ds: xr.Dataset | Future) -> Dict[
    float | str, Dict[str, np.ndarray]]:
    """
    Resolves the increments_ds (Future/scattered object) and executes the
    expensive slicing, masking, and NumPy array extraction loop synchronously
    on a Dask worker to produce the optimized increments_data dictionary.

    This function should only be called once per worker block.
    """

    # Resolve the input: If increments_ds is a Future, Dask resolves it to xr.Dataset here.
    if isinstance(increments_ds, Future):
        # The Future resolves to the scattered xr.Dataset object
        increments_ds = increments_ds.result()

    # --- Start of the expensive indexing/masking loop (Original client code) ---
    r_values = increments_ds["r"].values
    increments_data = dict()

    # r_values are needed for the outer loop in _block_space_scale_integral
    increments_data["r_values"] = r_values

    # Store boundary types once, as they are used repeatedly
    increments_data["x_boundary_type"] = increments_ds.attrs.get("x_boundary_type", "periodic")
    increments_data["y_boundary_type"] = increments_ds.attrs.get("y_boundary_type", "reflect")

    for r_value in r_values:
        r_slice = increments_ds.sel(r=r_value)

        # These data accesses are synchronous and must be run on a worker.
        mask = r_slice.mask.data.astype(bool)
        ny_idx, nx_idx = np.where(mask)
        weights = r_slice.angle_weight.data[mask].astype(np.float32)

        increments_data[r_value] = {
            "nx_values": r_slice.nx.data[nx_idx].astype(int),
            "ny_values": r_slice.ny.data[ny_idx].astype(int),
            "angles": r_slice.angle_grid.data[mask].astype(np.float32),
            "weights": weights / np.clip(np.sum(weights), epsilon, None),
        }

    return increments_data


def _block_space_scale_integral(
        field_chunk: xr.Dataset,
        increments: Dict | Future,
        x_dim: str,
        y_dim: str,
        transform_type: str,
        kernel_derivative: xr.Dataset
) -> xr.DataArray:
    """
    Performs scale-space integral per Dask spatial block.

    Parameters
    ----------
    field_chunk : xr.Dataset
        A spatially-chunked block of the full wind field dataset.
    increments : Dict | Future
        Grid information containing precomputed scale increments and angle grids.
    x_dim, y_dim : str
        Names of spatial dimensions.
    transform_type : str
        Transformation to compute ("delta_u_cubed").
    kernel_derivative : xr.DataArray
        Precomputed mollifier kernel derivatives dG/dr, dims: (scale, r).
    Returns
    -------
    xr.DataArray
        Integrated field with dimension 'scale' and same spatial dims as integrand.

    Workflow inside each block:
        1. loop over all r-values → integrand(r, ...)
        2. build mollifier kernels for all ℓ
        3. compute (dg/dr)(r;ℓ) * r * integrand
        4. perform truncated, normalized ∫ ... dr
        5. return final (scale, ...) DataArray
    """

    # ------------------------------------------------------------------------------------
    # CacheManager: per block per worker
    # ------------------------------------------------------------------------------------
    cache_manager = CacheManager.for_current_worker(verbose=False, force_threshold=0.70)

    # Run the indexing loop ONLY once per block to get the dictionary.
    # increments is now either the precomputed dictionary or a Future pointing to it.
    if isinstance(increments, Future):
        increments_data = increments.result()
    else:
        increments_data = increments

    radial_distance = increments_data["r_values"]  # Access without modifying the shared dict

    # ------------------------------------------------------------------------------------
    # Compute cubed velocity differences for all radial distances: Dask main graph
    # ------------------------------------------------------------------------------------
    integrand_blocks = [
        process_single_r_for_field_chunk(
            field_chunk_ds=field_chunk,
            r_value=float(r_value),
            increments_data=increments_data,
            x_dim=x_dim, y_dim=y_dim,
            transform_type=transform_type,
            cache_manager=cache_manager
        )
        for r_value in radial_distance
    ]

    # ------------------------------------------------------------------------------------
    # Merge integrand and rechunk to have all r-values in one chunk
    # ------------------------------------------------------------------------------------
    integrand = xr.concat(integrand_blocks, dim="r").chunk(r=-1)

    # --------   Adaptatively cache the large integrand array to RAM or Disk   -----------
    integrand = cache_manager.persist(integrand)

    # ------------------------------------------------------------------------------------
    # Kernel-weighting / masking (Kernel support r ≤ 2ℓ is guaranteed by construction)
    # ------------------------------------------------------------------------------------
    weighted = kernel_derivative * integrand
    masked = weighted.where(weighted.r <= 2 * kernel_derivative.scale, 0.0)

    num = masked.sum("r")
    den = weighted.sum("r")
    retention_fraction = xr.where(den > epsilon, num / den, 1.0)

    # Integral correction due to the truncated kernel
    integral = masked.integrate("r") / retention_fraction

    # ------------------------------------------------------------------------------------
    # Add scale coordinate
    # ------------------------------------------------------------------------------------
    integral = integral.assign_coords(scale=kernel_derivative.scale)

    # --- enforce one-scale-at-a-time tasks for reductions/writes ---
    integral = integral.chunk(scale=1)

    return integral


def scale_transfer(
        field: xr.Dataset,
        increments: Dict | Future,
        radial_distance: xr.DataArray,
        length_scale: xr.DataArray,
        x_dim: str,
        y_dim: str,
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
    increments: dict, Dask Future
        Grid information containing precomputed scale increments and angle grids.
    radial_distance : xr.DataArray
        1D array of radial distances r.
    length_scale : xr.DataArray
        1D array of filter length scales ℓ.
    name : str
        Name for the resulting variable.
    x_dim, y_dim : str
        Names of the spatial dimensions.
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
        print(f"[scale-integral] Building parallel graph for '{name}' ...")

    # ------------------------------------------------------------
    # Build integration kernels dG_ℓ(r)/dr for all ℓ and r
    # ------------------------------------------------------------
    kernel_derivative = get_integration_kernels(
        radial_distance, length_scale,
        normalization=weighting,
        scaled=True, return_derivative=True,
    ).compute()

    # ------------------------------------------------------------
    # Distribute computation over spatial blocks
    # ------------------------------------------------------------
    transfer = xr.map_blocks(
        _block_space_scale_integral,
        field,
        kwargs=dict(
            increments=increments,
            x_dim=x_dim, y_dim=y_dim,
            transform_type=transform_type,
            kernel_derivative=kernel_derivative
        ),
        # Build template: final output is (scale, ...)
        template=field["u"].expand_dims(scale=length_scale).rename(name),
    )

    # Recover scale attributes
    transfer['scale'].attrs.update(length_scale.attrs)

    return transfer.rename(name)


def inter_scale_kinetic_energy_transfer(wind: xr.Dataset, **kwargs) -> xr.Dataset:
    """ Computes the inter-scale kinetic energy transfer rate using third-order structure functions.
    Parameters
    ----------
    wind : xr.Dataset
        Dataset containing 3D velocity components (u, v, w).
    **kwargs : dict
        Additional keyword arguments passed to scale_increments and scale_transfer functions.
    Returns
    -------
    xr.Dataset
        Dataset containing the specific kinetic energy transfer rate across scales.
    """

    # Validate input dataset
    velocity_vars = [v for v in ["u", "v", "w"] if v in wind]

    # Ensure velocity components are float32 for memory efficiency
    wind = wind[velocity_vars].astype({v: "float32" for v in velocity_vars})

    # Check if the dataset has the required variables
    verbose = kwargs.get("verbose", False)

    # Determine spatial coordinate names (legacy support)
    x_name = kwargs.get("x_coord_name", None)
    y_name = kwargs.get("y_coord_name", None)

    # Attempt to retrieve coordinates by name
    if x_name is None and y_name is None:
        # Infer coordinates using helper (assumes CF compliance)
        y_name, x_name = get_spatial_dims(wind)

    if x_name in wind and y_name in wind:
        x_coord = wind[x_name]
        y_coord = wind[y_name]
    else:
        raise KeyError(f"Specified coordinate names {x_name}, {y_name} not found in dataset.")

    # ----------------------------------------------------------------------------
    # Calculate geometry synchronously on the client thread
    # ----------------------------------------------------------------------------
    increments = scale_increments(x_coord, y_coord, **kwargs)

    # Extract the coordinates metadata to be passed synchronously.
    radial_distance = increments['r']
    length_scale = increments['scale']

    # ----------------------------------------------------------------------------
    # Scatter the large geometry payload if a client is present
    # ----------------------------------------------------------------------------
    increments_data = _resolve_increments_data(increments)

    try:
        client = get_client()
    except ValueError:
        client = None

    if client:
        # Scatter increments to workers
        increments_data = client.scatter([increments_data], broadcast=True)[0]
        print(f"[transfer] Preprocessed geometry scattered to workers.")

    # ----------------------------------------------------------------------------
    # Ensure 'optimal' chunks along non-spatial dimensions for high parallelism
    # ----------------------------------------------------------------------------
    num_workers = kwargs.get("num_workers", None)

    wind = ensure_optimal_chunking(wind,
                                   spatial_dims=(y_name, x_name), vertical_dim="z",
                                   # Data size increases by the number of scales
                                   output_scale_mult=radial_distance.size,
                                   # Limit number of workers to avoid overheads
                                   num_workers=num_workers)

    # ----------------------------------------------------------------------------
    # Compute third-order structure functions for each radial distance
    # ----------------------------------------------------------------------------
    energy_transfer_rate = scale_transfer(
        field=wind.fillna(0.0),
        increments=increments_data,
        radial_distance=radial_distance,
        length_scale=length_scale,
        name="energy_transfer",
        x_dim=x_name, y_dim=y_name,
        weighting="sphere",
        verbose=verbose
    )

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

    # Promote to dataset and transpose to have 'scale' as the last dimension
    energy_transfer_rate = energy_transfer_rate.to_dataset().transpose(..., "scale")

    return energy_transfer_rate
