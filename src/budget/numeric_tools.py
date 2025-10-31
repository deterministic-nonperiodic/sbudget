from typing import Tuple

import numpy as np
import xarray as xr
from xarray import set_options

from .cf_coords import _coord_is_degrees
from .cf_coords import _is_geographic, get_spatial_dims, is_geographic_grid
from .constants import earth_radius, epsilon

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
    # Get the names of the spatial dimensions to reduce over
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
def _check_coordinate_consistency(*arrays: xr.DataArray, dims_to_check: Tuple[str, str]):
    """Verify that all input arrays share the same specified dimensions and coordinates."""
    if not arrays:
        return
    first_coords = arrays[0].coords
    first_sizes = {dim: arrays[0].sizes[dim] for dim in dims_to_check}

    for dim in dims_to_check:
        if dim not in first_coords:
            raise ValueError(
                f"Dimension '{dim}' not found as a coordinate in the first input array.")

    for i, da in enumerate(arrays[1:], start=2):
        for dim in dims_to_check:
            if dim not in da.dims:
                raise ValueError(f"Dimension '{dim}' missing from input array {i}.")
            if da.sizes[dim] != first_sizes[dim]:
                raise ValueError(f"Size mismatch for dimension '{dim}' in input array {i}. "
                                 f"Expected {first_sizes[dim]}, got {da.sizes[dim]}.")
            if dim not in da.coords:
                raise ValueError(f"Coordinate for dimension '{dim}' missing from input array {i}.")
            # Check if coordinates are numerically close
            if not np.allclose(first_coords[dim].values, da.coords[dim].values, equal_nan=True):
                raise ValueError(
                    f"Coordinate values for dimension '{dim}' mismatch in input array {i}.")


def first_derivative(da: xr.DataArray, dim: str, delta: float | None = None) -> xr.DataArray:
    """
    Metric-aware first derivative along `dim` in spherical or Cartesian coordinates.

    Returns
    -------
    xr.DataArray
        The partial derivative d(da)/ds where `s` is distance in **meters**
        along the axis `dim`.

    Rules
    -----
    - If `delta` is given (meters) and the coordinate is missing, compute
      index-based derivative and divide by `delta` to obtain per-meter.
    - If `dim` is longitude or latitude, convert to per-radian and then
      divide by the corresponding spherical metric so the result is per-meter:
        * lon:  (1 / (R * cos(phi))) * d(da)/d(lambda)
        * lat:  (1 / R)               * d(da)/d(phi)
      where `lambda` and `phi` are in radians.
    - Otherwise (Cartesian), we assume the coordinate is metric (meters) and
      return the standard coordinate derivative (already per meter). If this
      is not true in a given dataset, pass an explicit `delta` instead.
    """
    if dim not in da.dims:
        raise ValueError(f"first_derivative: dim '{dim}' not in {tuple(da.dims)}")

    coord = da.coords.get(dim, None)

    # --- Index-based derivative with explicit spacing (meters) ---
    if coord is None:
        if delta is None:
            raise ValueError(
                f"first_derivative: No coordinate for dim '{dim}' and no 'delta' provided; "
                f"cannot compute meters-based derivative.")
        # d/d[index] divided by physical spacing (meters)
        deriv_index = da.differentiate(dim, edge_order=2)

        return deriv_index / xr.full_like(deriv_index, fill_value=float(delta))

    # --- Coordinate-based derivative: start with d/d[coord] ---
    deriv_coord = da.differentiate(dim, edge_order=2)

    # Geographic longitude/latitude handling
    is_lon = _is_geographic(coord, "lon")
    is_lat = _is_geographic(coord, "lat")

    if is_lon or is_lat:
        # Convert coordinate to radians if provided in degrees
        if _coord_is_degrees(coord):
            # d/d[deg] * (deg per rad) = d/d[rad]
            deriv_rad = deriv_coord * xr.full_like(deriv_coord, fill_value=180.0 / np.pi)
        else:
            deriv_rad = deriv_coord

        if is_lon:
            # per-meter: (1 / (R * cos(phi))) * d/d(lambda)
            # We need latitude to compute cos(phi). Infer y-dim from data array.
            y_dim, x_dim = get_spatial_dims(da)
            # If dim is longitude, the latitude coordinate is the other spatial dim
            lat_coord = da.coords[y_dim]
            phi = np.deg2rad(lat_coord) if _coord_is_degrees(lat_coord) else lat_coord
            cos_phi = xr.where(np.abs(phi) > (np.pi / 2 - epsilon), epsilon, np.cos(phi))
            return deriv_rad / (earth_radius * cos_phi)

        # latitude case: (1 / R) * d/d(phi)
        return deriv_rad / xr.full_like(deriv_rad, fill_value=earth_radius)

    # Non-geographic: assume coordinate is metric (meters)
    return deriv_coord


def horizontal_divergence(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """
    Horizontal divergence of vector field (u, v), returned in s^-1.

    On the sphere (lon/lat):
        div = (1/(R cos(phi))) * ∂_λ u + (1/(R cos(phi))) * ∂_φ (v cos(phi))
            = du/dx + (1/cos(phi)) * d(v cos(phi))/dy
      because `first_derivative(u, x_dim)` returns du/dx = (1/(R cos(phi))) ∂_λ u
      and `first_derivative(v cos φ, y_dim)` returns (1/R) ∂_φ (v cos φ).
    """
    y_dim, x_dim = get_spatial_dims(u)
    x_coord = u.coords[x_dim]
    y_coord = u.coords[y_dim]

    _check_coordinate_consistency(u, v, dims_to_check=(y_dim, x_dim))

    du_dx = first_derivative(u, x_dim)  # per meter

    if is_geographic_grid(x_coord, y_coord):
        lat = y_coord
        phi = np.deg2rad(lat) if _coord_is_degrees(lat) else lat
        cos_phi = xr.where(np.abs(phi) > (np.pi / 2 - epsilon), epsilon, np.cos(phi))

        d_v_cos_dy = first_derivative(v * cos_phi, y_dim)  # = (1/R) ∂_φ (v cos φ)
        dv_dy = (d_v_cos_dy / cos_phi)
    else:
        # Cartesian fallback
        dv_dy = first_derivative(v, y_dim)

    # Horizontal divergence div = du/dx + dv/dy (both per meter)
    div = du_dx + dv_dy

    # Construct name and attributes
    div.name = "divergence"
    div.attrs.update({
        "units": "s-1",
        "standard_name": "divergence_of_wind",
        "long_name": "Horizontal divergence of wind",
    })
    return div


def relative_vorticity(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """
    Vertical component of relative vorticity, returned in s^-1.

    On the sphere:
        ζ = (1/(R cos φ)) ∂_λ v  - (1/(R cos φ)) ∂_φ (u cos φ)
          = dv/dx              - (1/cos φ) * d(u cos φ)/dy
      because `first_derivative(v, x_dim)` returns dv/dx = (1/(R cos φ)) ∂_λ v
      and `first_derivative(u cos φ, y_dim)` returns (1/R) ∂_φ (u cos φ).
    """
    y_dim, x_dim = get_spatial_dims(u)
    x_coord = u.coords[x_dim]
    y_coord = u.coords[y_dim]

    _check_coordinate_consistency(u, v, dims_to_check=(y_dim, x_dim))

    dv_dx = first_derivative(v, x_dim)  # per meter

    if is_geographic_grid(x_coord, y_coord):
        lat = y_coord
        phi = np.deg2rad(lat) if _coord_is_degrees(lat) else lat
        cos_phi = xr.where(np.abs(phi) > (np.pi / 2 - epsilon), epsilon, np.cos(phi))

        d_u_cos_dy = first_derivative(u * cos_phi, y_dim)  # = (1/R) ∂_φ (u cos φ)
        du_dy = (d_u_cos_dy / cos_phi)
    else:
        # Cartesian fallback
        du_dy = first_derivative(u, y_dim)

    # Relative vorticity ζ = dv/dx - du/dy (both per meter)
    vort = dv_dx - du_dy

    # Construct name and attributes
    vort.name = "relative_vorticity"
    vort.attrs.update({
        "units": "s-1",
        "standard_name": "relative_vorticity",
        "long_name": "Relative vorticity",
    })
    return vort


def horizontal_gradient(scalar: xr.DataArray, delta: float | None = None) -> Tuple[
    xr.DataArray, xr.DataArray]:
    """
    Horizontal gradient of a scalar field, returned as (dA/dx, dA/dy) in **per meter**.

    With `first_derivative` providing per-meter derivatives, we simply call it
    along each horizontal axis. For geographic coordinates, the internal metric
    handling is already applied inside `first_derivative`.

    Parameters
    ----------
    scalar : xr.DataArray
        Scalar field.
    delta : float, optional
        Constant spacing in meters for index-based differentiation when the
        corresponding coordinate is missing.
    """
    y_dim, x_dim = get_spatial_dims(scalar)

    da_dx = first_derivative(scalar, x_dim, delta=delta)  # per meter
    da_dy = first_derivative(scalar, y_dim, delta=delta)  # per meter

    return da_dx, da_dy


def horizontal_advection(scalar: xr.DataArray, u: xr.DataArray, v: xr.DataArray,
                         delta: float | None = None) -> xr.DataArray:
    """
    Calculate the horizontal advection of a scalar field A by a vector field (u, v).

    Formula: u * (dA/dx) + v * (dA/dy)

    Handles both geographic (latitude/longitude) and Cartesian coordinates correctly.
    Infers coordinates and dimensions from input DataArrays.

    Parameters
    ----------
    scalar : xr.DataArray
        The scalar field being advected (e.g., temperature, kinetic energy).
    u : xr.DataArray
        Zonal (Eastward) component of the advecting velocity field.
    v : xr.DataArray
        Meridional (Northward) component of the advecting velocity field.
    delta : float, optional
        Constant grid spacing in meters, used only if coordinates are missing
        from A for index-based differentiation via differentiate_metric.

    Returns
    -------
    xr.DataArray
        Horizontal advection of A (units: A_units * s^-1).
    """
    # Infer spatial dimensions and coordinates from 'A'

    y_dim, x_dim = get_spatial_dims(scalar)

    # Check consistency with 'u' and 'v'
    _check_coordinate_consistency(scalar, u, v, dims_to_check=(y_dim, x_dim))

    da_dx, da_dy = horizontal_gradient(scalar, delta=delta)

    # Calculate advection: u * dA/dx + v * dA/dy
    adv = u * da_dx + v * da_dy

    # Try to construct a meaningful name and attributes
    adv.name = f"{scalar.name or 'scalar'}_advection"
    adv_units = f"({scalar.attrs.get('units', 'unknown')}) s-1" if scalar.attrs.get(
        'units') else "s-1"
    adv.attrs.update({
        'units': adv_units,
        'long_name': f"Horizontal advection of "
                     f"{scalar.attrs.get('long_name', scalar.name or 'scalar')}",
        'standard_name': f"tendency_of_"
                         f"{scalar.attrs.get('standard_name', scalar.name or 'scalar')}"
                         f"_due_to_horizontal_advection"
    })
    return adv


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
