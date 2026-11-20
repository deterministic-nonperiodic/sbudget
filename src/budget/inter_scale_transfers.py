import threading
from typing import Any, Dict, NamedTuple

import numpy as np
import scipy.fft as sfft
import xarray as xr
from dask.distributed import Future
from dask.distributed import get_client

from .cf_coords import get_spatial_dims
from .constants import earth_radius, epsilon
from .grid_utils import scale_increments
from .memory_manager import CacheManager, ensure_optimal_chunking, get_current_worker_count
from .numeric_tools import horizontal_divergence, vertical_derivative


# --------------------------------------------------------------------------------------------------
# -----------------------          Kernel Implementation           ---------------------------------
# --------------------------------------------------------------------------------------------------
def _evaluate_mollifier(radial_positions: np.ndarray, length_scales: np.ndarray,
                        p: float = 1.0) -> np.ndarray:
    """
    Evaluate the standard mollifier kernel for all combinations of
    radial distances and filter length scales, generalized with a shape parameter p.

    Returns
    -------
    mollifier : np.ndarray of shape (n_scales, n_r)
    """
    r_grid, ell_grid = np.meshgrid(radial_positions, length_scales, indexing="ij")
    ratio_squared = (r_grid / (2 * ell_grid)) ** 2

    ratio_2p = ratio_squared ** p if p != 1 else ratio_squared
    inside_support = ratio_squared < 1
    denominator = 1 - ratio_2p

    mollifier = np.zeros_like(denominator)
    mollifier[inside_support] = np.exp(-1.0 / denominator[inside_support])

    return mollifier.T  # final shape: (n_scales, n_r)


def _evaluate_mollifier_derivative(radial_positions: np.ndarray,
                                   length_scales: np.ndarray,
                                   p: float = 1.0) -> np.ndarray:
    """
    Evaluate the radial derivative of the standard mollifier for all combinations of
    radial distances and filter length scales, generalized with a shape parameter p.

    Returns
    -------
    derivative : np.ndarray of shape (n_scales, n_r)
    """
    r_grid, ell_grid = np.meshgrid(radial_positions, length_scales, indexing="ij")
    ratio = r_grid / (2 * ell_grid)
    ratio_squared = ratio ** 2

    ratio_2p = ratio_squared ** p if p != 1 else ratio_squared
    inside_support = ratio_squared < 1
    denominator = 1 - ratio_2p

    f_prime = -2 * p * (ratio ** (2 * p - 1)) / (2 * ell_grid)

    derivative = np.zeros_like(denominator)
    derivative[inside_support] = (
            f_prime[inside_support]
            * np.exp(-1.0 / denominator[inside_support])
            / denominator[inside_support] ** 2
    )

    return derivative.T  # final shape: (n_scales, n_r)


def _compute_half_power_cutoff(p_val: float) -> float:
    """Dynamically compute the normalized half-power cutoff wavenumber (kc * ell)."""
    from scipy.special import j0
    from scipy.optimize import root_scalar

    # Use normalized ell=1 for calculation
    r = np.linspace(0, 2.0, 5000)

    # Get spatial kernel using the production function
    G = _evaluate_mollifier(r, np.array([1.0]), p=p_val)[0, :]

    # Normalize
    norm = np.trapz(G * 2 * np.pi * r, r)
    G /= norm

    # Function to find root of: |G_hat(k)|^2 - 0.5 = 0
    def power_diff(k):
        integrand = G * j0(k * r) * 2 * np.pi * r
        return np.trapz(integrand, r) ** 2 - 0.5

    res = root_scalar(power_diff, bracket=[0.1, 5.0], method='brentq')
    return float(res.root)


def lattice_kernel_weights(r: xr.DataArray, bin_area: xr.DataArray, scales: xr.DataArray,
                           cell_area: float, p: float = 1.0) -> xr.DataArray:
    """
    Quadrature weights W[scale, r] such that the Duchon-Robert flux is

        D_ell = 1/4 ∫ d²r ∇G_ell(r)·δu |δu|²  ≈  Σ_r W[ell, r] · S(r),

    where S(r) = Σ_{shifts in bin r} (δu·r̂)|δu|² ΔA is the area-weighted sum over the lattice
    shifts of each radial bin (computed per grid point).

    The kernel is normalized on the same lattice measure, Σ_shifts G_ell ΔA = 1 (including the
    zero shift), so normalization and derivative use one discretization; this keeps the result
    accurate down to ell ~ 1.5 grid spacings.
    """
    r_vals = np.asarray(r.values, dtype=float)
    ell = np.asarray(scales.values, dtype=float)
    area = np.asarray(bin_area.values, dtype=float)

    g = _evaluate_mollifier(r_vals, ell, p=p)  # (scale, r)
    g0 = _evaluate_mollifier(np.array([0.0]), ell, p=p)[:, 0]  # (scale,)
    dg = _evaluate_mollifier_derivative(r_vals, ell, p=p)  # (scale, r)

    norm = g0 * cell_area + g @ area  # Σ G ΔA over the lattice
    weights = 0.25 * dg / np.clip(norm, epsilon, None)[:, None]

    s_name = scales.name or "scale"
    return xr.DataArray(weights, dims=[s_name, "r"], coords={s_name: scales, "r": r},
                        name="lattice_kernel_weights")


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
                            return_derivative=True, p: int = 2) -> xr.DataArray:
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
    p : int
        Shape parameter for the mollifier kernel. Default is 1.

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
        _evaluate_mollifier(radial_positions, length_scales, p=p),
        dims=[s_name, r_name], coords={s_name: scales, r_name: r_da}
    )

    # Get normalization factor: area-weighted integral over r
    kernel_norm = _get_mollifier_norm(kernel, normalization, r_name=r_name)

    if return_derivative:
        kernel_name = "dG_dr_kernel"
        # Get the mollifier kernel's derivative
        kernel = xr.DataArray(
            _evaluate_mollifier_derivative(radial_positions, length_scales, p=p),
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

    The filter is horizontal, so only the horizontal increments are projected on r̂; w enters
    through |δu|² only. With w, D_ℓ is the transfer of horizontal plus vertical kinetic energy by
    horizontal advection (the counterpart of the HKE + VKE transfers; see transfer_correction).
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
    integrand = xr.zeros_like(field_chunk_ds["u"])

    for phi, weight, nx, ny in angle_parameters:
        # Roll the field to the angle grid
        rolled_ds = roll_with_boundary_handling(
            field_chunk_ds, nx, ny, x_dim, y_dim,
            x_boundary_type, y_boundary_type
        )

        delta_u_cubed = delta_u_cubed_geographic(field_chunk_ds, rolled_ds, phi)
        integrand = integrand + (delta_u_cubed * weight)

    integrand = integrand.rename(transform_type)
    return integrand.expand_dims(r=[r_value])


# Keyword arguments of scale_increments() that define the shift geometry
_GEOMETRY_KWARGS = ("scales", "max_r", "resolution_factor", "min_valid_shifts", "exact_factor",
                    "x_coord_boundary", "y_coord_boundary")

# Per-process cache of resolved geometries, keyed by a hash of the recipe
_GEOMETRY_CACHE: Dict[str, Dict] = {}
_GEOMETRY_LOCK = threading.Lock()


def _geometry_spec(x_coord: xr.DataArray, y_coord: xr.DataArray, kwargs: Dict) -> Dict:
    """Small, picklable recipe for the shift geometry (coordinates + defining keywords)."""
    import hashlib
    import pickle

    recipe = {
        "x": (x_coord.name, np.asarray(x_coord.values), dict(x_coord.attrs)),
        "y": (y_coord.name, np.asarray(y_coord.values), dict(y_coord.attrs)),
        "kwargs": {k: kwargs[k] for k in _GEOMETRY_KWARGS if k in kwargs},
    }
    recipe["key"] = hashlib.sha1(pickle.dumps(recipe)).hexdigest()
    return {"geometry_spec": recipe}


def _increments_from_spec(recipe: Dict) -> Dict:
    """Rebuild (once per process) and return the resolved geometry described by a recipe."""
    key = recipe["key"]
    cached = _GEOMETRY_CACHE.get(key)
    if cached is not None:
        return cached
    with _GEOMETRY_LOCK:
        if key not in _GEOMETRY_CACHE:
            coords = {}
            for axis in ("x", "y"):
                name, values, attrs = recipe[axis]
                coords[axis] = xr.DataArray(values, dims=name, name=name, attrs=attrs)
            increments = scale_increments(coords["x"], coords["y"], verbose=False,
                                          **recipe["kwargs"])
            _GEOMETRY_CACHE[key] = _resolve_increments_data(increments)
    return _GEOMETRY_CACHE[key]


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
    increments_data["r"] = r_values

    # Store boundary types once, as they are used repeatedly
    increments_data["x_boundary_type"] = increments_ds.attrs.get("x_boundary_type", "periodic")
    increments_data["y_boundary_type"] = increments_ds.attrs.get("y_boundary_type", "reflect")

    for r_value in r_values:
        r_slice = increments_ds.sel(r=r_value)

        # These data accesses are synchronous and must be run on a worker.
        mask = r_slice.mask.data.astype(bool)
        ny_idx, nx_idx = np.where(mask)

        # Lattice quadrature: each shift contributes its cell area (area-weighted SUM over the
        # shifts of the bin, not an average), see lattice_kernel_weights.
        weights = r_slice.shift_area.data[mask].astype(np.float32)

        increments_data[r_value] = {
            "nx_values": r_slice.nx.data[nx_idx].astype(int),
            "ny_values": r_slice.ny.data[ny_idx].astype(int),
            "angles": r_slice.angle_grid.data[mask].astype(np.float32),
            "weights": weights,
        }

    # Full shift lattice (every shift of every bin) for the FFT path: exact per-shift kernel
    used = increments_ds["mask"].any("r").values
    ny_grid, nx_grid = np.meshgrid(increments_ds["ny"].values, increments_ds["nx"].values,
                                   indexing="ij")
    increments_data["lattice"] = {
        "sy": ny_grid[used].astype(np.int64),
        "sx": nx_grid[used].astype(np.int64),
        "dist": increments_ds["distance_grid"].values[used].astype(np.float64),
        "angle": increments_ds["angle_grid"].values[used].astype(np.float64),
        "cell_area": float(increments_ds["cell_area"]),
    }

    return increments_data


def _block_increments(increments: Dict | Future) -> Dict:
    """
    Resolved geometry for one block: increments is the precomputed dictionary, a Future
    pointing to it, or a recipe rebuilt once per worker process (_geometry_spec).
    """
    if isinstance(increments, Future):
        return increments.result()
    if isinstance(increments, dict) and "geometry_spec" in increments:
        return _increments_from_spec(increments["geometry_spec"])
    return increments


def _block_space_scale_integral(
        field_chunk: xr.Dataset,
        increments: Dict | Future,
        x_dim: str,
        y_dim: str,
        transform_type: str,
        kernel_derivative: xr.Dataset,
        kernel_params: Dict | None = None
) -> xr.DataArray:
    """
    Performs scale-space integral per Dask spatial block.

    For boundary types with a fixed extension (wrap, reflect, symmetric, edge) the integral is
    evaluated exactly on the shift lattice by FFT correlation (_dell_fft_block); otherwise by
    streaming over r-bins (below).

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
        Lattice quadrature weights W[scale, r] (see lattice_kernel_weights).
    Returns
    -------
    xr.DataArray
        Integrated field with dimension 'scale' and same spatial dims as the block (float32).

    Workflow inside each block (streaming, memory independent of the number of r-bins):
        for each r-bin with any non-zero weight:
            S_r = Σ_{shifts in bin} (δu·r̂)|δu|² ΔA           (block-sized array)
            D[scale, ...] += W[scale, r] · S_r                 (accumulate, float64)
        return D as float32 (scale, ...)
    Peak memory per block ≈ (n_scales × 8 + a few × 4) bytes per grid point.
    """

    # Run the indexing loop ONLY once per block to get the dictionary.
    increments_data = _block_increments(increments)

    radial_distance = increments_data["r"]  # Access without modifying the shared dict

    weights = kernel_derivative.transpose("scale", "r")
    w_np = weights.values  # (scale, r)
    w_r = weights["r"].values
    scale_coords = weights["scale"].values

    template = field_chunk["u"]

    # r-bins that contribute to at least one requested scale (others lie beyond every 2ℓ)
    used = []
    for r_value in radial_distance:
        k = int(np.argmin(np.abs(w_r - r_value)))
        if np.any(w_np[:, k]):
            used.append((r_value, w_np[:, k]))
    if not used:
        raise ValueError("No radial distances to process.")

    x_mode = _NP_PAD_MODES.get(increments_data.get("x_boundary_type", "periodic"))
    y_mode = _NP_PAD_MODES.get(increments_data.get("y_boundary_type", "reflect"))

    p = float((kernel_params or {}).get("p", 1.0))
    fft_ok = (transform_type == "delta_u_cubed" and "lattice" in increments_data
              and x_mode in _FFT_PAD_MODES and y_mode in _FFT_PAD_MODES)

    if fft_ok:
        accumulator, work_dims = _dell_fft_block(field_chunk, increments_data["lattice"],
                                                 scale_coords, p, x_dim, y_dim, x_mode, y_mode)
    elif transform_type == "delta_u_cubed" and x_mode and y_mode:
        result = _stream_bins_padded(field_chunk, increments_data, used, scale_coords.size,
                                     x_dim, y_dim, x_mode, y_mode)
        work_dims = result.pop("dims")
        accumulator = result.pop("accumulator")
    else:
        # Fallback for boundary modes whose padding depends on the pad width (e.g. linear_ramp)
        work_dims = template.dims
        accumulator = np.zeros((scale_coords.size,) + template.shape, dtype=np.float64)
        expand = (slice(None),) + (None,) * template.ndim
        for r_value, w_col in used:
            bin_sum = process_single_r_for_field_chunk(
                field_chunk_ds=field_chunk, r_value=float(r_value),
                increments_data=increments_data, x_dim=x_dim, y_dim=y_dim,
                transform_type=transform_type, cache_manager=None
            ).transpose("r", *template.dims).values[0]
            accumulator += w_col[expand] * bin_sum

    coords = {"scale": scale_coords}
    coords.update({str(d): template.coords[d] for d in template.dims if d in template.coords})

    out = xr.DataArray(accumulator.astype(np.float32), dims=("scale",) + tuple(work_dims),
                       coords=coords)
    return out.transpose("scale", *template.dims)


# numpy.pad equivalents of the boundary types for which padding once by the largest shift and
# slicing gives exactly the same values as padding by |shift| + 1 and rolling per shift.
_NP_PAD_MODES = {"periodic": "wrap", "wrap": "wrap", "reflect": "reflect",
                 "symmetric": "symmetric", "edge": "edge", "constant": "constant"}

# Boundary extensions supported by the FFT path (no NaN fill)
_FFT_PAD_MODES = {"wrap", "reflect", "symmetric", "edge"}

# Per-process cache of kernel spectra, shared by the threads of a worker. Keyed by the padded
# plane shape, the lattice and the scale; bounded so it never dominates worker memory.
_KERNEL_CACHE: Dict[tuple, Any] = {}
_KERNEL_CACHE_BYTES: Dict[tuple, int] = {}
_KERNEL_CACHE_LOCK = threading.Lock()
_KERNEL_CACHE_MAX_BYTES = 2 * 1024 ** 3


def _cache_kernel(key: tuple, entry: Any, nbytes: int) -> None:
    """Keep a kernel spectrum in _KERNEL_CACHE unless that exceeds _KERNEL_CACHE_MAX_BYTES."""
    with _KERNEL_CACHE_LOCK:
        if sum(_KERNEL_CACHE_BYTES.values()) + nbytes <= _KERNEL_CACHE_MAX_BYTES:
            _KERNEL_CACHE[key] = entry
            _KERNEL_CACHE_BYTES[key] = nbytes


def _kernel_spectra(lattice: Dict, shape: tuple, ell: float, p: float) -> tuple:
    """
    conj(rFFT) of the two horizontal kernels K_x, K_y on the padded plane, and their sums:

        K_i(s) = 1/4 G'_ℓ(|s|) r̂_i(s) ΔA / Σ_s G_ℓ(|s|) ΔA      (Σ includes the zero shift)

    placed at index (s mod M) so that irfft2(rfft2(A) · conj(K̂)) is the correlation
    Σ_s K(s) A(x + s).
    """
    key = (shape, lattice["sy"].size, int(np.abs(lattice["sy"]).max()),
           int(np.abs(lattice["sx"]).max()), lattice["cell_area"], float(ell), float(p))
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    dist, ang, cell = lattice["dist"], lattice["angle"], lattice["cell_area"]
    g = _evaluate_mollifier(dist, np.array([ell]), p=p)[0]
    g0 = _evaluate_mollifier(np.array([0.0]), np.array([ell]), p=p)[0, 0]
    dg = _evaluate_mollifier_derivative(dist, np.array([ell]), p=p)[0]
    coeff = 0.25 * dg * cell / max((g.sum() + g0) * cell, epsilon)

    my, mx = shape
    iy, ix = lattice["sy"] % my, lattice["sx"] % mx
    spectra, sums = [], []
    for trig in (np.cos(ang), np.sin(ang)):
        kernel = np.zeros(shape, dtype=np.float64)
        np.add.at(kernel, (iy, ix), coeff * trig)
        spectra.append(np.conj(sfft.rfft2(kernel, workers=1)))
        sums.append(float((coeff * trig).sum()))
    entry = (spectra[0], spectra[1], sums[0], sums[1])
    _cache_kernel(key, entry, spectra[0].nbytes * 2)
    return entry


def _rfft2(a: np.ndarray) -> np.ndarray:
    """Real 2-D FFT over the last two axes, single-threaded (blocks run in parallel)."""
    return sfft.rfft2(a, workers=1)


class _Padding(NamedTuple):
    """
    Padding of a horizontal plane by the largest lattice shift with a boundary extension, so that
    the circular FFT correlation of the padded plane equals the linear one on every original point
    (no wrap-around).
    """
    py: int
    px: int
    ny: int
    nx: int
    y_mode: str
    x_mode: str

    @classmethod
    def for_lattice(cls, lattice: Dict, plane_shape: tuple, y_mode: str, x_mode: str):
        py, px = int(np.abs(lattice["sy"]).max()), int(np.abs(lattice["sx"]).max())
        return cls(py, px, *plane_shape, y_mode, x_mode)

    @property
    def shape(self) -> tuple:
        """Shape of the padded plane."""
        return self.ny + 2 * self.py, self.nx + 2 * self.px

    def pad(self, a: np.ndarray) -> np.ndarray:
        return _pad_plane(a, self.py, self.px, self.y_mode, self.x_mode)

    def inverse(self, spectra: np.ndarray) -> np.ndarray:
        """Inverse FFT of stacked half-spectra (n, ...), cropped to the original plane."""
        fields = sfft.irfft2(spectra, s=self.shape, axes=(-2, -1), workers=1)
        return fields[..., self.py:self.py + self.ny, self.px:self.px + self.nx]


def _block_planes(field_chunk: xr.Dataset, names: list, x_dim: str, y_dim: str) -> tuple:
    """Arrays {name: (..., y, x)} of a block, and their dims with the horizontal ones last."""
    lead = [d for d in field_chunk["u"].dims if d not in (y_dim, x_dim)]
    work_dims = tuple(lead) + (y_dim, x_dim)
    return {v: field_chunk[v].transpose(*work_dims).values for v in names}, work_dims


def _dell_fft_block(field_chunk: xr.Dataset, lattice: Dict, scales: np.ndarray, p: float,
                    x_dim: str, y_dim: str, x_mode: str, y_mode: str):
    """
    Exact lattice evaluation of D_ℓ = ¼ Σ_s ∇G_ℓ(s)·δu |δu|² ΔA by FFT correlation.

    With a = u(x+s), b = u(x) (mean removed; δu is unchanged), i over the horizontal and j over
    all velocity components, (a_i - b_i) Σ_j (a_j - b_j)² expands into products of a function of
    x and a function of x+s, so the shift sum becomes correlations (K_i ⋆ F)(x):

      D = Σ_i K_i⋆(u_i|u|²) - 2 Σ_j b_j Σ_i K_i⋆(u_i u_j) + |b|² Σ_i K_i⋆u_i
          - Σ_i b_i [K_i⋆|u|² + |b|² Σ_s K_i(s) - 2 Σ_j b_j K_i⋆u_j].

    Each plane is padded by the largest shift with the boundary extension (_Padding). Per plane,
    the 11 products F are transformed once; per scale, the 12 correlations are one stacked inverse
    FFT, independent of the number of shifts.

    Returns (accumulator (scale, ..., y, x) float64, work dims).
    """
    names = [v for v in ("u", "v", "w") if v in field_chunk]
    horizontal = ("u", "v")
    data, work_dims = _block_planes(field_chunk, names, x_dim, y_dim)
    shape = data["u"].shape
    padding = _Padding.for_lattice(lattice, shape[-2:], y_mode, x_mode)
    accumulator = np.zeros((len(scales),) + shape, dtype=np.float64)

    for idx in np.ndindex(*shape[:-2]):
        b = {v: data[v][idx].astype(np.float64) for v in names}
        b = {v: plane - plane.mean() for v, plane in b.items()}
        b2 = sum(b[v] ** 2 for v in names)

        # Spectra of the products F, keyed by the horizontal component i that K_i acts on
        a = {v: padding.pad(b[v]) for v in names}
        a2 = sum(a[v] ** 2 for v in names)
        f_u_a2 = {i: _rfft2(a[i] * a2) for i in horizontal}  # u_i |u|²
        f_uu = {j: {i: _rfft2(a[i] * a[j]) for i in horizontal} for j in names}  # u_i u_j
        f_a2 = _rfft2(a2)  # |u|²
        f_u = {j: _rfft2(a[j]) for j in names}  # u_j
        del a, a2

        for n, ell in enumerate(scales):
            kx, ky, sum_kx, sum_ky = _kernel_spectra(lattice, padding.shape, ell, p)
            k_hat, k_sum = {"u": kx, "v": ky}, {"u": sum_kx, "v": sum_ky}

            # The 12 correlations, summed over i where the formula sums them
            corr = {"K⋆u|u|²": f_u_a2["u"] * kx + f_u_a2["v"] * ky}
            corr.update({("K⋆u_i u_j", j): f_uu[j]["u"] * kx + f_uu[j]["v"] * ky for j in names})
            corr.update({("K_i⋆|u|²", i): f_a2 * k_hat[i] for i in horizontal})
            corr.update({("K_i⋆u_j", i, j): f_u[j] * k_hat[i] for i in horizontal for j in names})
            t = dict(zip(corr, padding.inverse(np.stack(list(corr.values())))))

            d = t["K⋆u|u|²"]
            d -= 2.0 * sum(b[j] * t["K⋆u_i u_j", j] for j in names)
            d += b2 * (t["K_i⋆u_j", "u", "u"] + t["K_i⋆u_j", "v", "v"])
            for i in horizontal:
                b_dot_ku = sum(b[j] * t["K_i⋆u_j", i, j] for j in names)
                d -= b[i] * (t["K_i⋆|u|²", i] + b2 * k_sum[i] - 2.0 * b_dot_ku)

            accumulator[(n,) + idx] = d

    return accumulator, work_dims


def _filter_spectrum(lattice: Dict, shape: tuple, ell: float, p: float) -> np.ndarray:
    """
    conj(rFFT) of the normalized lattice filter on the padded plane,

        G(s) = G_ℓ(|s|) ΔA / Σ_s G_ℓ(|s|) ΔA      (Σ and G include the zero shift),

    the same normalization as the D_ℓ kernels (_kernel_spectra), placed at (s mod M) so that
    irfft2(rfft2(A) · Ĝ) = Σ_s G(s) A(x + s) = Ā(x).
    """
    key = ("filter", shape, lattice["sy"].size, int(np.abs(lattice["sy"]).max()),
           int(np.abs(lattice["sx"]).max()), lattice["cell_area"], float(ell), float(p))
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    sy, sx, dist, cell = lattice["sy"], lattice["sx"], lattice["dist"], lattice["cell_area"]
    zero = (sy == 0) & (sx == 0)
    g = _evaluate_mollifier(dist[~zero], np.array([ell]), p=p)[0]
    g0 = _evaluate_mollifier(np.array([0.0]), np.array([ell]), p=p)[0, 0]
    norm = max((g.sum() + g0) * cell, epsilon)

    my, mx = shape
    kernel = np.zeros(shape, dtype=np.float64)
    np.add.at(kernel, (sy[~zero] % my, sx[~zero] % mx), g * cell / norm)
    kernel[0, 0] += g0 * cell / norm
    spectrum = np.conj(sfft.rfft2(kernel, workers=1))
    _cache_kernel(key, spectrum, spectrum.nbytes)
    return spectrum


def _correction_pad_modes(x_boundary: str, y_boundary: str) -> tuple:
    """
    numpy.pad modes (x, y) of the transfer correction for the given boundary types. The
    correction is evaluated by FFT only, so it needs a fixed boundary extension.
    """
    x_mode, y_mode = _NP_PAD_MODES.get(x_boundary), _NP_PAD_MODES.get(y_boundary)
    if x_mode not in _FFT_PAD_MODES or y_mode not in _FFT_PAD_MODES:
        raise NotImplementedError(
            "The transfer correction needs boundary types with a fixed extension (periodic, "
            f"wrap, reflect, symmetric or edge); got x={x_boundary}, y={y_boundary}.")
    return x_mode, y_mode


def _dilatational_exchange(b: Dict[str, np.ndarray], bar: Dict[str, np.ndarray],
                           names: list) -> np.ndarray:
    """
    Dilatational exchange on one plane, with u over the components `names`:

        C_ℓ = ¼ δ (u·ū) + ¼ u·(δu)‾ - ¼ |u|² δ̄ - ¼ δ (|u|²)‾

    b holds the unfiltered fields (components and "div"), bar the filtered ones under the keys
    of _correction_fft_block (components, "div*<component>", "div", "|u|^2").
    """
    delta = b["div"]
    c = delta * sum(b[v] * bar[v] for v in names)
    c += sum(b[v] * bar[f"div*{v}"] for v in names)
    c -= sum(b[v] ** 2 for v in names) * bar["div"]
    c -= delta * bar["|u|^2"]
    return 0.25 * c


def _vertical_exchange(b: Dict[str, np.ndarray], bar: Dict[str, np.ndarray],
                       names: list) -> np.ndarray:
    """
    Vertical exchange on one plane, with u over the components `names`:

        V_ℓ = ½ w (ū·∂z u - u·∂z ū)

    b holds the unfiltered fields (components, "w_adv" and "dz<component>"), bar the filtered
    ones under the same keys.
    """
    return 0.5 * b["w_adv"] * sum(bar[v] * b[f"dz{v}"] - b[v] * bar[f"dz{v}"] for v in names)


def _correction_fft_block(field_chunk: xr.Dataset, lattice: Dict, scales: np.ndarray, p: float,
                          components: list, divergence: bool, vertical: bool,
                          x_dim: str, y_dim: str, x_mode: str, y_mode: str):
    """
    Lattice evaluation of the transfer correction X_ℓ = V_ℓ - C_ℓ by FFT filtering.

    Both terms are products of unfiltered fields with filtered ones, Ā = Σ_s G(s) A(x + s), where
    G is the lattice filter of D_ℓ (_filter_spectrum: same shifts and normalization). Each plane is
    padded with the boundary extension of D_ℓ (_Padding) and transformed once; every
    scale then needs one stacked inverse FFT. Unlike in _dell_fft_block the plane mean is kept:
    C_ℓ and V_ℓ depend on it. Cost per plane and scale: up to 2 n + 4 inverse FFTs of the padded
    plane (n = number of components).

    Returns (accumulator (scale, ..., y, x) float64, work dims).
    """
    names = list(components)
    shear = [f"dz{v}" for v in names]
    extra = (["div"] if divergence else []) + (["w_adv"] + shear if vertical else [])
    data, work_dims = _block_planes(field_chunk, names + extra, x_dim, y_dim)
    shape = data["u"].shape
    padding = _Padding.for_lattice(lattice, shape[-2:], y_mode, x_mode)
    accumulator = np.zeros((len(scales),) + shape, dtype=np.float64)

    for idx in np.ndindex(*shape[:-2]):
        b = {v: data[v][idx].astype(np.float64) for v in names + extra}
        a = {v: padding.pad(b[v]) for v in b}

        # Fields to be filtered (ū is shared by both terms; names always include u and v)
        filtered = {v: a[v] for v in names}
        if divergence:
            filtered.update({f"div*{v}": a["div"] * a[v] for v in names})
            filtered["div"] = a["div"]
            filtered["|u|^2"] = sum(a[v] ** 2 for v in names)
        if vertical:
            filtered.update({k: a[k] for k in shear})
        spectra = np.stack([_rfft2(f) for f in filtered.values()])
        keys = list(filtered)
        del a, filtered

        for n, ell in enumerate(scales):
            g_hat = _filter_spectrum(lattice, padding.shape, ell, p)
            bar = dict(zip(keys, padding.inverse(spectra * g_hat)))

            if divergence:
                accumulator[(n,) + idx] -= _dilatational_exchange(b, bar, names)
            if vertical:
                accumulator[(n,) + idx] += _vertical_exchange(b, bar, names)

    return accumulator, work_dims


def _block_transfer_correction(
        field_chunk: xr.Dataset,
        increments: Dict | Future,
        x_dim: str,
        y_dim: str,
        scales: np.ndarray,
        components: list,
        divergence: bool,
        vertical: bool,
        kernel_params: Dict | None = None
) -> xr.DataArray:
    """
    Transfer correction X_ℓ = V_ℓ - C_ℓ per Dask block (see transfer_correction).

    Parameters
    ----------
    field_chunk : xr.Dataset
        A block of whole horizontal planes with the velocity components and the fields of the
        requested terms ("div"; "w_adv" and "dz<component>").
    increments : Dict | Future
        Grid information containing precomputed scale increments and angle grids.
    x_dim, y_dim : str
        Names of spatial dimensions.
    scales : np.ndarray
        Filter length scales ℓ.
    components : list
        Velocity components of the increments in D_ℓ.
    divergence, vertical : bool
        Include the dilatational exchange C_ℓ and the vertical exchange V_ℓ.
    Returns
    -------
    xr.DataArray
        Correction with dimension 'scale' and same spatial dims as the block (float32).
    """
    increments_data = _block_increments(increments)
    if "lattice" not in increments_data:
        raise NotImplementedError("The transfer correction needs the shift lattice.")
    x_mode, y_mode = _correction_pad_modes(increments_data.get("x_boundary_type", "periodic"),
                                           increments_data.get("y_boundary_type", "reflect"))
    p = float((kernel_params or {}).get("p", 1.0))

    accumulator, work_dims = _correction_fft_block(field_chunk, increments_data["lattice"],
                                                   scales, p, components, divergence, vertical,
                                                   x_dim, y_dim, x_mode, y_mode)

    template = field_chunk["u"]
    coords = {"scale": scales}
    coords.update({str(d): template.coords[d] for d in template.dims if d in template.coords})

    out = xr.DataArray(accumulator.astype(np.float32), dims=("scale",) + tuple(work_dims),
                       coords=coords)
    return out.transpose("scale", *template.dims)


def _pad_plane(a: np.ndarray, py: int, px: int, y_mode: str, x_mode: str) -> np.ndarray:
    """Pad a 2-D plane by (py, px) with numpy boundary extensions (y then x)."""
    a = np.pad(a, ((py, py), (0, 0)), mode=y_mode)
    return np.pad(a, ((0, 0), (px, px)), mode=x_mode)


def _stream_bins_padded(field_chunk: xr.Dataset, increments_data: Dict, used: list,
                        n_scales: int, x_dim: str, y_dim: str,
                        x_mode: str, y_mode: str) -> Dict[str, Any]:
    """
    Fast path of the scale integral for one block.

    Each velocity component is padded ONCE by the largest shift used (instead of padding,
    rolling and cropping every component for every shift). A shifted field is then a view
    into the padded array, and all per-shift arithmetic runs in preallocated NumPy buffers,
    which keeps the Python overhead per shift small and releases the GIL for the array work.

    Returns {"accumulator": (scale, ..., y, x) float64, "dims": (..., y, x)}.
    """
    names = [v for v in ("u", "v", "w") if v in field_chunk]
    lead = [d for d in field_chunk["u"].dims if d not in (y_dim, x_dim)]
    work_dims = tuple(lead) + (y_dim, x_dim)
    base = {v: np.ascontiguousarray(field_chunk[v].transpose(*work_dims).values,
                                    dtype=np.float32) for v in names}
    shape = base["u"].shape
    ny, nx = shape[-2], shape[-1]

    # Largest shifts actually needed
    py = max(int(np.abs(increments_data[r]["ny_values"]).max(initial=0)) for r, _ in used)
    px = max(int(np.abs(increments_data[r]["nx_values"]).max(initial=0)) for r, _ in used)

    def pad(a):
        lead_pad = [(0, 0)] * (a.ndim - 2)
        kw_y = {"constant_values": np.nan} if y_mode == "constant" else {}
        kw_x = {"constant_values": np.nan} if x_mode == "constant" else {}
        a = np.pad(a, lead_pad + [(py, py), (0, 0)], mode=y_mode, **kw_y)
        return np.pad(a, lead_pad + [(0, 0), (px, px)], mode=x_mode, **kw_x)

    padded = {v: pad(base[v]) for v in names}

    # Preallocated work buffers
    du, dv, sq, proj, tmp = (np.empty(shape, dtype=np.float32) for _ in range(5))
    bin_sum = np.empty(shape, dtype=np.float32)
    accumulator = np.zeros((n_scales,) + shape, dtype=np.float64)
    expand = (slice(None),) + (None,) * len(shape)

    for r_value, w_col in used:
        grid = increments_data[r_value]
        bin_sum.fill(0.0)
        for phi, weight, sx, sy in zip(grid["angles"], grid["weights"],
                                       grid["nx_values"], grid["ny_values"]):
            ys = slice(py + int(sy), py + int(sy) + ny)
            xs = slice(px + int(sx), px + int(sx) + nx)

            np.subtract(padded["u"][..., ys, xs], base["u"], out=du)
            np.subtract(padded["v"][..., ys, xs], base["v"], out=dv)
            np.multiply(du, du, out=sq)
            np.multiply(dv, dv, out=tmp)
            sq += tmp
            if "w" in padded:
                np.subtract(padded["w"][..., ys, xs], base["w"], out=tmp)
                np.multiply(tmp, tmp, out=tmp)
                sq += tmp

            # (δu·r̂) |δu|² ΔA,  r̂ = (cos φ, sin φ)
            np.multiply(du, np.float32(np.cos(phi)), out=proj)
            np.multiply(dv, np.float32(np.sin(phi)), out=tmp)
            proj += tmp
            proj *= sq
            proj *= np.float32(weight)
            bin_sum += proj

        accumulator += w_col[expand] * bin_sum

    return {"accumulator": accumulator, "dims": work_dims}


def scale_transfer(
        field: xr.Dataset,
        increments: Dict | Future,
        radial_distance: xr.DataArray,
        length_scale: xr.DataArray,
        x_dim: str,
        y_dim: str,
        name: str = "energy_transfer",
        transform_type: str = "delta_u_cubed",
        bin_area: xr.DataArray = None,
        cell_area: float = None,
        p: int = 1,
        verbose: bool = False
) -> xr.DataArray:
    """
    Compute the inter-scale energy transfer (Duchon & Robert 2000):

        D_ℓ = 1/4 ∫ d²r ∇G_ℓ(r) · δu |δu|²,

    evaluated as a lattice sum over grid shifts r (see lattice_kernel_weights), with G_ℓ
    normalized on the same lattice. Kernels truncated by r_max are renormalized.

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
    bin_area : xr.DataArray
        Lattice area of each radial bin (number of shifts × cell area), dims ("r",).
    cell_area : float
        Area of one grid cell (Δx Δy).
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
    # Lattice quadrature weights ¼ G'_ℓ(r) / Σ G_ℓ ΔA for all ℓ and r
    # ------------------------------------------------------------
    if bin_area is None or cell_area is None:
        raise ValueError("scale_transfer needs bin_area and cell_area from scale_increments().")

    kernel_derivative = lattice_kernel_weights(
        radial_distance, bin_area, length_scale, cell_area=float(cell_area), p=p
    )

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
            kernel_derivative=kernel_derivative,
            kernel_params={"p": p},
        ),
        # Build template: final output is (scale, ...)
        template=field["u"].expand_dims(scale=length_scale).rename(name),
    )

    # Recover scale attributes
    transfer['scale'].attrs.update(length_scale.attrs)

    return transfer.rename(name)


def transfer_correction(
        field: xr.Dataset,
        increments: Dict | Future,
        length_scale: xr.DataArray,
        x_dim: str,
        y_dim: str,
        components: list,
        divergence: bool = True,
        vertical: bool = False,
        name: str = "transfer_correction",
        p: float = 1,
        verbose: bool = False
) -> xr.DataArray:
    """
    Compute the correction X_ℓ = V_ℓ - C_ℓ that makes

        D*_ℓ = D_ℓ - C_ℓ + V_ℓ

    the physical-space counterpart of the flux-form spectral transfer T(k).

    D_ℓ (Duchon & Robert 2000) closes the local budget of the mixed energy ½ u·ū only for
    non-divergent u advected horizontally. T(k) is the spectrum of the skew-symmetric advection

        N = (U·∇)u + ½ δ u + ½ [w ∂z u + ∂z (w u)],    δ = ∂x u + ∂y v,  U = (u, v),

    with u the components of the increments,

    and the domain mean of D*_ℓ equals Σ_k T(k) [1 - Ĝ(kℓ)], Ĝ the transfer function of the
    filter of D_ℓ, with

        C_ℓ = ¼ δ (u·ū) + ¼ u·(δu)‾ - ¼ |u|² δ̄ - ¼ δ (|u|²)‾     (dilatational exchange)
        V_ℓ = ½ w (ū·∂z u - u·∂z ū)                             (vertical exchange)

    C_ℓ follows from the exact local identity -½ [ū·N_h + u·N̄_h] = -∇·J_ℓ - (D_ℓ - C_ℓ), J_ℓ the
    Duchon-Robert spatial flux; V_ℓ is the part of ū·N_v that is not a vertical flux divergence.
    C_ℓ vanishes for δ = 0 and V_ℓ for w = 0; both vanish as ℓ → 0. Unlike D_ℓ, they depend on a
    uniform background velocity, as T(k) does. With components ("u", "v"), D*_ℓ is the
    counterpart of the HKE transfer Π_h; with ("u", "v", "w"), of the HKE plus VKE transfers
    (the VKE transfer of the spectral budget has the same flux form, with w advected). In D_ℓ
    the filter and r̂ are horizontal, so w enters through |δu|² only. Closed-form tests:
    tests/test_synthetic_budget.py.

    Parameters
    ----------
    field : xr.Dataset
        Velocity components `components`, the horizontal divergence "div" (if divergence) and
        the vertical velocity "w_adv" with the shear "dz<component>" of every component (if
        vertical).
    increments: dict, Dask Future
        Grid information containing precomputed scale increments and angle grids.
    length_scale : xr.DataArray
        1D array of filter length scales ℓ.
    x_dim, y_dim : str
        Names of the spatial dimensions.
    components : list
        Velocity components of the increments in D_ℓ (u and v, optionally w).
    divergence : bool
        Include the dilatational exchange -C_ℓ.
    vertical : bool
        Include the vertical exchange V_ℓ.
    name : str
        Name for the resulting variable.
    p : float
        Shape parameter of the mollifier (as for D_ℓ).
    verbose : bool, optional
        If True, print progress and summary info.

    Returns
    -------
    xr.DataArray
        Correction with dimension 'scale' and the dims of field["u"].
    """
    if not (divergence or vertical):
        raise ValueError("transfer_correction: request at least one of divergence, vertical.")

    if verbose:
        print(f"[scale-integral] Building parallel graph for '{name}' ...")

    correction = xr.map_blocks(
        _block_transfer_correction,
        field,
        kwargs=dict(
            increments=increments,
            x_dim=x_dim, y_dim=y_dim,
            scales=length_scale.values,
            components=list(components),
            divergence=divergence, vertical=vertical,
            kernel_params={"p": p},
        ),
        # Build template: final output is (scale, ...)
        template=field["u"].expand_dims(scale=length_scale).rename(name),
    )

    # Recover scale attributes
    correction['scale'].attrs.update(length_scale.attrs)

    return correction.rename(name)


def _correction_inputs(wind: xr.Dataset, components: list, divergence: bool, vertical: bool,
                       kwargs: Dict) -> Dict[str, xr.DataArray]:
    """
    Fields of the transfer correction besides the velocity components:

        "div"                   δ, if divergence: kwargs["divergence"]; else the dataset's
                                divergence if it also holds the vorticity (the rule of the
                                spectral budget, so that both diagnostics use the same δ);
                                else centred differences (horizontal_divergence).
        "w_adv", "dz<c>"        w and ∂z of every component c, if vertical: the shear from
                                kwargs or the dataset (the reader differentiates the full column
                                before selecting levels), else centred differences along z
                                (vertical_derivative).
    """
    fields = {}
    if divergence:
        delta = kwargs.get("divergence", None)
        if delta is None and "divergence" in wind and "vorticity" in wind:
            delta = wind["divergence"]
        if delta is None:
            delta = horizontal_divergence(wind["u"], wind["v"])
        fields["div"] = delta

    if vertical:
        if "w" not in wind:
            raise ValueError("vertical_advection needs the vertical velocity w.")
        fields["w_adv"] = wind["w"]
        for v in components:
            key = f"dz{v}"
            if kwargs.get(key) is not None:
                fields[key] = kwargs[key]
            elif key in wind:
                fields[key] = wind[key]
            elif wind.sizes.get("z", 0) < 3:
                raise ValueError(f"vertical_advection: provide {key} (∂z {v}) or at least "
                                 "3 levels to differentiate.")
            else:
                fields[key] = vertical_derivative(wind[v], "z")

    return fields


def inter_scale_kinetic_energy_transfer(wind: xr.Dataset, **kwargs) -> xr.Dataset:
    """ Computes the inter-scale kinetic energy transfer rate using third-order structure functions.
    Parameters
    ----------
    wind : xr.Dataset
        Dataset containing 3D velocity components (u, v, w).
    **kwargs
        Additional keyword arguments passed to scale_increments and scale_transfer functions.
        For the generalized transfer D*_ℓ (see transfer_correction): divergence_correction,
        vertical_advection, perturbation (bool), increment_components (default: u, v),
        and optionally the fields divergence and dzu, dzv (dzw).
    Returns
    -------
    xr.Dataset
        Dataset containing the specific kinetic energy transfer rate across scales, and
        energy_transfer_generalized if a correction is requested.
    """

    # Validate input dataset: components advected in D_ℓ (in the increments). The default (u, v)
    # gives the HKE transfer, the counterpart of Π_h; w then enters only as advecting velocity
    # (V_ℓ). With w the result also contains the VKE transfer, a separate reservoir.
    present = [v for v in ["u", "v", "w"] if v in wind]
    velocity_vars = list(kwargs.get("increment_components") or ("u", "v"))
    if not {"u", "v"} <= set(velocity_vars) or not set(velocity_vars) <= set(present):
        raise ValueError(f"increment_components {velocity_vars} must contain u, v and be a "
                         f"subset of the velocity components present {present}.")

    # Optional correction D*_ℓ = D_ℓ - C_ℓ + V_ℓ (see transfer_correction)
    divergence_correction = bool(kwargs.get("divergence_correction", False))
    vertical_advection = bool(kwargs.get("vertical_advection", False))
    generalized = divergence_correction or vertical_advection

    # Perturbation frame: remove the level-mean horizontal wind (D_ℓ is unchanged; C_ℓ, V_ℓ and
    # the flux-form spectral transfer are not Galilean invariant). Given dzu, dzv must then be
    # derivatives of the perturbation (the reader removes the mean before differentiating).
    if bool(kwargs.get("perturbation", False)):
        y_d, x_d = get_spatial_dims(wind)
        wind = wind.assign(u=wind["u"] - wind["u"].mean((y_d, x_d)),
                           v=wind["v"] - wind["v"].mean((y_d, x_d)))

    correction_fields = _correction_inputs(wind, velocity_vars, divergence_correction,
                                           vertical_advection, kwargs) if generalized else {}

    # Ensure velocity components are float32 for memory efficiency
    wind = wind[velocity_vars].astype({v: "float32" for v in velocity_vars})
    wind = wind.assign({k: da.transpose(*wind["u"].dims).astype("float32")
                        for k, da in correction_fields.items()})

    # Check if the dataset has the required variables
    verbose = kwargs.get("verbose", False)

    # Determine spatial coordinate names (legacy support)
    y_infer, x_infer = get_spatial_dims(wind)
    x_name: str = kwargs.get("x_coord_name", x_infer)
    y_name: str = kwargs.get("y_coord_name", y_infer)

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

    # The correction is evaluated by FFT only: fail before building the graph
    if generalized:
        _correction_pad_modes(increments.attrs.get("x_boundary_type", "periodic"),
                              increments.attrs.get("y_boundary_type", "reflect"))

    # ----------------------------------------------------------------------------
    # Scatter the large geometry payload if a client is present
    # ----------------------------------------------------------------------------
    increments_data = _resolve_increments_data(increments)

    try:
        client = get_client()
    except ValueError:
        client = None

    if client:
        # With a distributed client, pass a small *recipe* for the geometry instead of a
        # scattered future: each worker process rebuilds it once from the coordinates and caches
        # it (_increments_from_spec). A scattered future embedded in the task arguments deadlocked
        # the scheduler on Levante (32 workers): the graph never reached the workers and nothing
        # was logged.
        increments_data = _geometry_spec(x_coord, y_coord, kwargs)
        print("[transfer] Shift geometry will be rebuilt once per worker process.")

    # ----------------------------------------------------------------------------
    # Ensure 'optimal' chunks along non-spatial dimensions for high parallelism
    # ----------------------------------------------------------------------------
    allow_rechunk = kwargs.get("allow_rechunk", True)

    if allow_rechunk:
        # Thread layout: each worker runs one block per thread at the same time
        n_workers = kwargs.get("num_workers", None) or get_current_worker_count(
            verbose=False, total=False)
        n_threads = max(n_workers, get_current_worker_count(verbose=False, total=True))
        tasks_per_worker = max(1, n_threads // max(1, n_workers))

        # Working set per grid point of one block, relative to the float32 input fields:
        #   float64 accumulator + float32 result copy over all scales (12 B per scale), and
        #   FFT path: per padded point ~320 B (11 half-spectra, 12 stacked inverse transforms,
        #   padded fields and products, all float64); plus 12 B per scale per grid point
        #   (float64 accumulator + float32 output). The shift path needs less.
        n_vars = len(velocity_vars)
        shifted = increments["mask"].any("r")
        max_ny = int(np.abs(increments["ny"].where(shifted.any("nx"))).max())
        max_nx = int(np.abs(increments["nx"].where(shifted.any("ny"))).max())
        pad_factor = ((wind.sizes[y_name] + 2 * max_ny) * (wind.sizes[x_name] + 2 * max_nx)
                      / (wind.sizes[y_name] * wind.sizes[x_name]))
        working_bytes = 12 * length_scale.size + 320 * pad_factor
        expansion = int(np.ceil(working_bytes / (4 * n_vars)))

        wind = ensure_optimal_chunking(wind,
                                       spatial_dims=(y_name, x_name), vertical_dim="z",
                                       output_scale_mult=expansion,
                                       num_workers=n_workers,
                                       tasks_per_worker=tasks_per_worker,
                                       # compute-bound: at least one block per thread, and
                                       # one plane per block, so memory per task does not grow
                                       # with the data volume (measured: per-plane time rises
                                       # 2.7x and memory 8x from 1 to 24 planes per block)
                                       preferred_num_blocks=n_threads,
                                       max_planes_per_block=1,
                                       min_chunk_mb=1.0,
                                       # whole horizontal planes are required (shifts span them)
                                       allow_spatial_tiling=False)

    # ----------------------------------------------------------------------------
    # Compute third-order structure functions for each radial distance
    # ----------------------------------------------------------------------------
    energy_transfer_rate = scale_transfer(
        field=wind[velocity_vars].fillna(0.0),
        increments=increments_data,
        radial_distance=radial_distance,
        length_scale=length_scale,
        name="energy_transfer",
        x_dim=x_name, y_dim=y_name,
        bin_area=increments["bin_area"],
        cell_area=float(increments["cell_area"]),
        p=kwargs.get("p", 1),
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
    coord_dict = {x_name: x_coord, y_name: y_coord}
    energy_transfer_rate = energy_transfer_rate.assign_coords(**coord_dict)

    # ----------------------------------------------------------------------------
    # Add equivalent spectral coordinates to allow direct comparison with spectral budgets
    # The 2D Hankel transform of the mollifier kernel drops to half-power at a
    # normalized wavenumber k_c * ell that depends on the kernel's shape parameter p.
    # ----------------------------------------------------------------------------
    p_val = float(kwargs.get("p", 1.0))
    kc_ell = _compute_half_power_cutoff(p_val)

    lambda_factor = 2 * np.pi / kc_ell

    equivalent_wavelength = lambda_factor * length_scale
    equivalent_wavenumber = 2 * np.pi / equivalent_wavelength

    energy_transfer_rate = energy_transfer_rate.assign_coords(
        equivalent_wavelength=("scale", equivalent_wavelength.data),
        equivalent_wavenumber=("scale", equivalent_wavenumber.data)
    )

    energy_transfer_rate["equivalent_wavenumber"].attrs.update({
        "units": "rad / m",
        "long_name": "Equivalent Spectral Cutoff Wavenumber",
        "description": f"Estimated half-power spectral cutoff wavenumber (k_c = {kc_ell} / scale) for the 2D mollifier (p={p_val})."
    })

    energy_transfer_rate["equivalent_wavelength"].attrs.update({
        "units": "m",
        "long_name": "Equivalent Spectral Wavelength",
        "description": "Equivalent spectral wavelength calculated as 2*pi / equivalent_wavenumber."
    })

    # Promote to dataset
    energy_transfer_rate = energy_transfer_rate.to_dataset()

    # ----------------------------------------------------------------------------
    # Physical-space counterpart of the flux-form spectral transfer (optional):
    #     D*_ℓ = D_ℓ - C_ℓ + V_ℓ    (see transfer_correction)
    # ----------------------------------------------------------------------------
    if generalized:
        correction = transfer_correction(
            field=wind.fillna(0.0),
            increments=increments_data,
            length_scale=length_scale,
            x_dim=x_name, y_dim=y_name,
            components=velocity_vars,
            divergence=divergence_correction,
            vertical=vertical_advection,
            p=kwargs.get("p", 1),
            verbose=verbose
        ).assign_coords(**coord_dict)

        terms = (["- C_ell (dilatational exchange)"] if divergence_correction else []) \
                + (["+ V_ell (vertical exchange)"] if vertical_advection else [])
        generalized_rate = energy_transfer_rate["energy_transfer"] + correction
        generalized_rate.attrs.update({
            'units': "W / kg",
            'long_name': "Generalized transfer rate of kinetic energy across scales",
            'description': f"D_ell {' '.join(terms)}, increments of {', '.join(velocity_vars)}: "
                           "counterpart of the flux-form spectral transfer (its domain mean is "
                           "the kernel-weighted spectral transfer). Positive means forward "
                           "energy transfer towards smaller scales"
        })
        energy_transfer_rate["energy_transfer_generalized"] = generalized_rate

    # Transpose to have 'scale' as the last dimension
    energy_transfer_rate = energy_transfer_rate.transpose(..., "scale")

    return energy_transfer_rate
