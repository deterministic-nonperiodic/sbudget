import re
from typing import Tuple, Union, Dict, Any, List

import numpy as np
import pint
import xarray as xr
from pyproj import Geod

# --- Public API for external access ---
__all__: List[str] = [
    "_cf_guess",
    "_coord_is_degrees",
    "_is_geographic",
    "_is_z",
    "_coord_is_meter",
    "convert_units",
    "is_geographic_grid",
    "get_spatial_dims",
    "infer_grid_resolution",
]

# -----------------------------
# --- CF Convention Lookups ---
# -----------------------------
_CF_COORDS_LOOKUP: Dict[str, Dict[str, Any]] = {
    "lon": {
        "names": ("lon", "long", "longitude"),
        "units_hints": ("east", "degree", "degrees", "deg", "degree_east"),
        "standard_name": "longitude",
        "axis": "X",
    },
    "lat": {
        "names": ("lat", "latitude"),
        "units_hints": ("north", "degree", "degrees", "deg", "degree_north"),
        "standard_name": "latitude",
        "axis": "Y",
    },
}

_CF_VARS_LOOKUP: Dict[str, Dict[str, Any]] = {
    "u": {"standard_names": {"eastward_wind"}, "units": {"m s-1", "m/s"}},
    "v": {"standard_names": {"northward_wind"}, "units": {"m s-1", "m/s"}},
    "w": {"standard_names": {"upward_air_velocity", "vertical_velocity_in_air"},
          "units": {"m s-1", "Pa s-1"}},
    "pressure": {"standard_names": {"air_pressure"}, "units": {"Pa", "pascal"}},
    "temperature": {"standard_names": {"air_temperature"}, "units": {"K", "kelvin"}},
    "density": {"standard_names": {"air_density"}, "units": {"kg / m**3", "kg m-3"}},
    "theta": {"standard_names": {"air_potential_temperature"}, "units": {"K", "kelvin"}},
    "divergence": {"standard_names": {"divergence_of_wind"}, "units": {"s-1"}},
    "vorticity": {"standard_names": {"relative_vorticity"}, "units": {"s-1"}},
}

ALLOWED_UNITS: List[str] = ["deg", "degrees", "degrees_north", "degrees_east",
                            "m", "meters", "km", "kilometers"]

_METER_UNITS: set = {"m", "meter", "meters", "metre", "metres"}


# ----------------------
# CF-based var guessing
# ----------------------
def _cf_guess(ds: xr.Dataset, target: str) -> str | None:
    """
    Very light CF-based guess for a logical variable name.

    Looks at ``standard_name`` and common units to suggest a candidate
    when a configured variable is missing. Advisory only.
    """
    rule = _CF_VARS_LOOKUP.get(target)
    if rule is None:
        return None
    for name, da in ds.data_vars.items():
        std = str(da.attrs.get("standard_name", "")).strip()
        units = str(da.attrs.get("units", "")).strip()
        if std in rule["standard_names"] or any(u in units for u in rule["units"]):
            return name
    return None


# --------------------------
# Unit and coordinate checks
# --------------------------
expected_units = {
    "u": "m/s",
    "v": "m/s",
    "w": "m/s",
    "divergence": "1/s",
    "vorticity": "1/s",
    "temperature": "K",
    "pressure": "Pa",
    'lat': 'degrees_north',
    'lon': 'degrees_east',
}

# from Metpy
# Create a pint UnitRegistry object
UNITS_REG = pint.UnitRegistry()

# from Metpy
cmd = re.compile(r"(?<=[A-Za-z)])(?![A-Za-z)])(?<![0-9\-][eE])(?<![0-9\-])(?=[0-9\-])")


def _normalize_unit(units: Union[str, None]) -> str:
    """Normalize CF-ish units for robust checks."""
    units = (units or "").strip().lower()
    return units.replace("°", "degree").replace("-", "_")


def _get_units_str(c: xr.DataArray) -> str:
    """Extracts and normalizes the units string from a DataArray."""
    return _normalize_unit(c.attrs.get("units", ""))


def _parse_units(unit_str):
    if isinstance(unit_str, (pint.Quantity, pint.Unit)):
        return unit_str
    else:
        return UNITS_REG(cmd.sub('**', unit_str))


def equivalent_units(unit_1, unit_2):
    ratio = (_parse_units(unit_1) / _parse_units(unit_2)).to_base_units()
    return ratio.dimensionless and np.isclose(ratio.magnitude, 1.0)


def compatible_units(unit_1, unit_2):
    return _parse_units(unit_1).is_compatible_with(_parse_units(unit_2))


def get_conversion_components(from_units: str, to_units: str) -> Tuple[float, float]:
    """
    Calculates the multiplicative factor (M) and offset (O) for a linear unit conversion
    using Pint, such that Y = X * M + O.

    This handles offset units like Celsius/Kelvin.
    """
    # This implementation requires a Pint UnitRegistry instance 'u' to be accessible (e.g., u = UnitRegistry()).

    # 1. Calculate Offset (O): Convert 0 from source to destination (0 * M + O = O)
    q0 = pint.Quantity(0.0, _parse_units(from_units))
    y0 = q0.to(_parse_units(to_units)).magnitude
    offset = y0

    # 2. Calculate Multiplier (M): Convert 1 from source to destination (1 * M + O), then subtract offset
    q1 = pint.Quantity(1.0, _parse_units(from_units))
    y1 = q1.to(_parse_units(to_units)).magnitude
    multiplier = y1 - offset

    # Returns (M, O)
    return multiplier, offset


def convert_units(da: xr.DataArray, from_units: str, to_units: str) -> xr.DataArray:
    """
    Converts the units of a DataArray, respecting Dask chunking and Xarray immutability.
    """
    # Use the units attached to the DataArray as the source unit
    source_units = _get_units_str(da) or from_units

    if equivalent_units(source_units, to_units) or source_units == to_units:
        # Units are already equivalent; no conversion needed
        return da

    if compatible_units(source_units, to_units):
        # Calculate scalar conversion factor
        multiplier, offset = get_conversion_components(source_units, to_units)

        # Apply conversion formula Y = X * M + O lazily. This is Dask-aware.
        result_data = multiplier * da + offset

        # FIX: Create a new DataArray to ensure immutability and update the units attribute.
        new_da = da.copy(data=result_data.data)
        new_da.attrs.update({"units": to_units})

        return new_da

    # Units are incompatible
    raise ValueError(f"Cannot convert due to incompatible units: '{source_units}' to '{to_units}'!")


def check_convert_units(ds: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset, xr.DataArray]:
    """
    Checks and converts the units of variables within a Xarray Dataset or DataArray
    to match predefined expected units, while respecting Dask chunking.

    Assumes the existence of:
    - `expected_units` (dict[str, str]): Mapping of variable name to target unit string.
    - `expected_range` (dict[str, Tuple[float, float]]): Admitted value ranges.
    - `equivalent_units`, `compatible_units`, `_parse_units`, and `get_conversion_factor` (from pint or similar).

    :param ds: The input Xarray Dataset or DataArray.
    :return: The Xarray object with converted units and values.
    """

    # 1. Handle DataArray input
    is_array = isinstance(ds, xr.DataArray)
    if is_array:
        ds = ds.to_dataset(name=ds.name or '_data_array_var', promote_attrs=True)

    # --- Process variables ---
    for varname in ds.data_vars:

        if varname not in expected_units:
            continue

        # var_units = ds[varname].attrs.get("units")
        var_units = _get_units_str(ds[varname])
        expected_unit = expected_units[str(varname)]

        # --- Handle missing units (Unit inference based on value range) ---
        if var_units == "":
            print(f"Warning: Units not found for variable {varname}. Assumed '{expected_unit}'.")
            continue

        # --- Handle existing units needing Dask-compatible conversion ---
        ds[varname] = convert_units(ds[varname], var_units, expected_unit)

    # Return original type
    if is_array:
        # Convert back to DataArray. Use to_array() / squeeze / drop_vars for a clean return.
        return ds.to_array().squeeze('variable', drop=True)
    else:
        return ds


# ----------------------
# Compact CF-aware utils
# ----------------------
def _infer_coordinate_units(coord: xr.DataArray, name: str) -> str:
    """Infers and validates coordinate units against ALLOWED_UNITS."""
    units = coord.attrs.get("units", "").lower()
    if not units:
        raise ValueError(f"Missing 'units' attribute for {name} coordinate.")
    if units not in ALLOWED_UNITS:
        raise ValueError(f"Invalid units for {name}: '{units}'. Allowed: {ALLOWED_UNITS}")
    return units


def _coord_is_degrees(
        coord: xr.DataArray,
        allow_infer: bool = True,
        tol: float = 1e-12,
) -> bool:
    """
    True if `coord` uses degrees (CF-compliant).

    If units are absent/ambiguous, and we need to infer, treat as degrees
    when |values| exceed 2π (cannot be radians).
    """
    units = _get_units_str(coord)

    # Explicit units
    if "radian" in units:
        return False
    if units == "deg" or units.startswith("degree") or units.startswith("degrees"):
        return True

    # Heuristic inference when units missing/unknown
    if allow_infer:
        vals = np.asarray(coord.values)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            if float(np.nanmax(np.abs(vals))) > (2.0 * np.pi + tol):
                return True

    return False


def _is_z(cname: str, coords: Union[xr.DataArray, Any]) -> bool:
    """CF-ish vertical detection using name/units/standard_name/axis signals."""
    if not cname in coords:
        return False
    coord = coords[cname]
    name = cname.lower()
    units = _get_units_str(coord)
    standard_name = (coord.attrs.get("standard_name", "") or "").strip().lower()
    axis = (coord.attrs.get("axis", "") or "").strip().upper()

    name_ok = any(k in name for k in ("z", "height", "geometric_height", "altitude"))
    # accept metre variants; avoid overly-broad substring matches
    units_ok = any(tok in units for tok in ("metre", "meter", "metres", "meters")) or units == "m"
    std_ok = (standard_name in ("altitude", "height"))
    axis_ok = (axis == "Z" and ("metre" in units or "meter" in units or units == "m"))
    return name_ok or units_ok or std_ok or axis_ok


def _is_geographic(coord: xr.DataArray, coord_type: str) -> bool:
    """
    Performs CF-ish checks for a single coordinate (Lat or Lon) using a lookup dictionary.
    """
    lookup = _CF_COORDS_LOOKUP[coord_type]

    # 1. Attributes and Names
    name = str(coord.name).lower() if coord.name else ""
    attrs: Dict[str, Any] = coord.attrs
    units = _get_units_str(coord)

    standard_name = (attrs.get("standard_name", "") or "").strip().lower()
    axis = (attrs.get("axis", "") or "").strip().upper()

    # Name Check: Check if any of the target names are in the coordinate name
    name_ok = any(n in name for n in lookup["names"])

    # Unit Check: Must contain 'degree' AND one of the directional/generic unit hints
    units_ok = ("degree" in units and any(u in units for u in lookup["units_hints"]))

    # Standard Name Check
    std_ok = (standard_name == lookup["standard_name"])

    # Axis Check: Must match target axis AND contain a degree/directional unit hint
    axis_ok = (axis == lookup["axis"] and any(u in units for u in lookup["units_hints"]))

    return name_ok or units_ok or std_ok or axis_ok


def is_geographic_grid(coord_x: xr.DataArray, coord_y: xr.DataArray) -> bool:
    """
    Determines if the grid coordinates are geographic (Longitude/Latitude) based on
    names, units, and CF conventions.

    Args:
        coord_x: The x-axis coordinate DataArray (e.g., 'lon').
        coord_y: The y-axis coordinate DataArray (e.g., 'lat').

    Returns:
        True if both coordinates are determined to be geographic (Lon/Lat), False otherwise.
    """
    # Check if the X coordinate is Longitude-like
    is_x_lon = _is_geographic(coord_x, "lon")

    # Check if the Y coordinate is Latitude-like
    is_y_lat = _is_geographic(coord_y, "lat")

    # The grid is geographic if and only if both components are identified
    return np.logical_and(is_x_lon, is_y_lat)


# ----------------------
# Spatial dim resolution
# ----------------------
def get_spatial_dims(obj: Union[xr.Dataset, xr.DataArray]) -> Tuple[str, str]:
    """
    Return the horizontal dims to use for FFT/derivatives as (y, x).

    Priority:
    1) True geographic axes as dims (1-D lat & lon) → ('lat','lon')
    2) Projected axes as dims with 2-D auxiliary lat/lon(y,x) → ('y','x')
    3) Plain projected grid with ('y','x') dims → ('y','x')
    """
    ds = obj if isinstance(obj, xr.Dataset) else obj.to_dataset(name="_tmp")
    dims = set(ds.dims)

    # Case A: true geographic axes as dims (1-D lat & lon)
    if "lat" in dims and "lon" in dims and ds["lat"].ndim == 1 and ds["lon"].ndim == 1:
        return "lat", "lon"

    # Case B: projected axes with 2-D auxiliary lat/lon(y,x)
    if {"y", "x"} <= dims and ("lat" in ds.coords) and ("lon" in ds.coords):
        if ds["lat"].dims == ("y", "x") and ds["lon"].dims == ("y", "x"):
            return "y", "x"

    # Case C: plain projected grid
    if {"y", "x"} <= dims:
        return "y", "x"

    raise ValueError(
        "get_spatial_dims: Could not determine horizontal dims. "
        "Expected identifiable lon/lat or projected y/x. "
        f"Available dims: {tuple(ds.dims)} | coords: {tuple(ds.coords)}"
    )


# ----------------------
# Resolution inference
# ----------------------
def _coord_is_meter(c: xr.DataArray) -> bool:
    """Checks if the coordinate units are meter-like."""
    u = _get_units_str(c)
    return (u in _METER_UNITS) or any(tok in u for tok in ("metre", "meter"))


def infer_grid_resolution(ds: xr.Dataset) -> tuple[float, float]:
    """
    Infer horizontal grid spacing (dx, dy) in meters using robust metadata checks.

    - If dims are geographic (('lat','lon')), convert to meters using Earth radius
      and the *median* latitude.
    - Else if dims are projected (('y','x')) and units are meters, return the
      *median* spacing along each axis.
    """
    y, x = get_spatial_dims(ds)  # ('lat','lon') or ('y','x')
    if (y not in ds.coords) or (x not in ds.coords):
        raise ValueError(f"infer_resolution: coords '{y}' and/or '{x}' not found.")

    ycoord = ds[y]
    xcoord = ds[x]

    # Geographic lat/lon
    if is_geographic_grid(xcoord, ycoord):
        # NOTE: We rely on is_geographic_grid() to confirm degree units indirectly.
        geode = Geod(ellps="WGS84")

        # Use median point for robust distance estimation at domain center
        y_center = float(ycoord.median())
        x_center = float(xcoord.median())

        # Calculate dx: distance between two adjacent x-points at the y-center
        _, _, dx = geode.inv(xcoord[0].item(), y_center, xcoord[1].item(), y_center)
        # Calculate dy: distance between two adjacent y-points at the x-center
        _, _, dy = geode.inv(x_center, ycoord[0].item(), x_center, ycoord[1].item())

        return dx, dy

    # Projected / Cartesian axes in meters
    if _coord_is_meter(ycoord) and _coord_is_meter(xcoord):
        # Calculate the median difference along the axes
        dy = float(ycoord.diff(y).median())
        dx = float(xcoord.diff(x).median())
        return dx, dy

    raise ValueError(
        "infer_resolution: could not infer spacing. "
        f"Dims=({y},{x}), units=({_get_units_str(ycoord)}, {_get_units_str(xcoord)})")
