from typing import Tuple, Union, Dict, Any, List

import numpy as np
import xarray as xr
from pyproj import Geod

# --- Public API for external access ---
__all__: List[str] = [
    "_cf_guess",
    "_coord_is_degrees",
    "_is_geographic",
    "_is_z",
    "is_geographic_grid",
    "get_spatial_dims",
    "infer_resolution",
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


# ----------------------
# Compact CF-aware utils
# ----------------------
def _has(cname: str, coords: Dict[str, Any]) -> bool:
    """True if coordinate name exists in coords mapping."""
    return cname in coords


def _normalize_unit(units: Union[str, None]) -> str:
    """Normalize CF-ish units for robust checks."""
    units = (units or "").strip().lower()
    return units.replace("°", "degree").replace("-", "_").replace(" ", "_")


def _units_str(c: xr.DataArray) -> str:
    """Extracts and normalizes the units string from a DataArray."""
    return _normalize_unit(c.attrs.get("units", ""))


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
    units = _normalize_unit(coord.attrs.get("units", ""))

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
    if not _has(cname, coords):
        return False
    da = coords[cname]
    name = cname.lower()
    units = _normalize_unit(da.attrs.get("units", ""))
    standard_name = (da.attrs.get("standard_name", "") or "").strip().lower()
    axis = (da.attrs.get("axis", "") or "").strip().upper()

    name_ok = any(k in name for k in ("z", "height", "geometric_height", "altitude"))
    # accept metre variants; avoid overly-broad substring matches
    units_ok = any(tok in units for tok in ("metre", "meter", "metres", "meters")) or units == "m"
    std_ok = (standard_name in ("altitude", "height"))
    axis_ok = (axis == "Z" and ("metre" in units or "meter" in units or units == "m"))
    return name_ok or units_ok or std_ok or axis_ok


def _is_geographic(da: xr.DataArray, coord_type: str) -> bool:
    """
    Performs CF-ish checks for a single coordinate (Lat or Lon) using a lookup dictionary.
    """
    lookup = _CF_COORDS_LOOKUP[coord_type]

    # 1. Attributes and Names
    name = str(da.name).lower() if da.name else ""
    attrs: Dict[str, Any] = da.attrs
    units = _normalize_unit(attrs.get("units", ""))
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
def _is_meter_like(c: xr.DataArray) -> bool:
    """Checks if the coordinate units are meter-like."""
    u = _normalize_unit(_units_str(c))
    return (u in _METER_UNITS) or any(tok in u for tok in ("metre", "meter"))


def infer_resolution(ds: xr.Dataset) -> tuple[float, float]:
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
    if _is_meter_like(ycoord) and _is_meter_like(xcoord):
        # Calculate the median difference along the axes
        dy = float(ycoord.diff(y).median())
        dx = float(xcoord.diff(x).median())
        return dx, dy

    raise ValueError(
        "infer_resolution: could not infer spacing. "
        f"Dims=({y},{x}), units=({_units_str(ycoord)}, {_units_str(xcoord)})")
