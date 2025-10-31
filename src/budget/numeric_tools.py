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
