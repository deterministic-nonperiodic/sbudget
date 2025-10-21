import numpy as np
import xarray as xr
from typing import Tuple, Union

from .constants import earth_radius

__all__ = [
    "_cf_guess",
    "_coord_is_degrees",
    "_is_lat",
    "_is_lon",
    "_is_z",
    "is_lonlat",
    "get_spatial_dims",
    "infer_resolution",
]


# ----------------------
# CF-based var guessing
# ----------------------
def _cf_guess(ds: xr.Dataset, target: str) -> str | None:
    """
    Very light CF-based guess for a logical variable name.

    Looks at ``standard_name`` and common units to suggest a candidate
    when a configured variable is missing. Advisory only.
    """
    cf_map = {
        "u": {"standard_names": {"eastward_wind"}, "units": {"m s-1", "m/s"}},
        "v": {"standard_names": {"northward_wind"}, "units": {"m s-1", "m/s"}},
        "w": {"standard_names": {"upward_air_velocity", "vertical_velocity_in_air"},
              "units": {"m s-1", "Pa s-1"}},
        "pressure": {"standard_names": {"air_pressure"}, "units": {"Pa", "pascal"}},
        "temperature": {"standard_names": {"air_temperature"}, "units": {"K", "kelvin"}},
        "theta": {"standard_names": {"air_potential_temperature"}, "units": {"K", "kelvin"}},
        "divergence": {"standard_names": {"divergence_of_wind"}, "units": {"s-1"}},
        "vorticity": {"standard_names": {"relative_vorticity"}, "units": {"s-1"}},
    }
    rule = cf_map.get(target)
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
def _has(cname: str, coords) -> bool:
    """True if coordinate name exists in coords mapping."""
    return cname in coords


def _norm_units(u: str) -> str:
    """Normalize CF-ish units for robust checks."""
    u = (u or "").strip().lower()
    return u.replace("°", "degree").replace("-", "_").replace(" ", "_")


def _coord_is_degrees(
        coord: xr.DataArray,
        allow_infer: bool = True,
        tol: float = 1e-12,
) -> bool:
    """
    True if `coord` uses degrees (CF-compliant).

    If units are absent/ambiguous and we need to infer, treat as degrees
    when |values| exceed 2π (cannot be radians).
    """
    units = _norm_units(coord.attrs.get("units", ""))

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


def _is_lat(cname: str, coords) -> bool:
    """CF-ish latitude detection using name/units/standard_name/axis signals."""
    if not _has(cname, coords):
        return False
    da = coords[cname]
    name = cname.lower()
    units = _norm_units(da.attrs.get("units", ""))
    stdn = (da.attrs.get("standard_name", "") or "").strip().lower()
    axis = (da.attrs.get("axis", "") or "").strip().upper()

    name_ok = ("lat" in name) or ("latitude" in name)
    units_ok = ("degree" in units and ("north" in units or units in ("degree", "degrees", "deg")))
    std_ok = (stdn == "latitude")
    axis_ok = (axis == "Y" and ("degree" in units or "north" in units))
    return name_ok or units_ok or std_ok or axis_ok


def _is_lon(cname: str, coords) -> bool:
    """CF-ish longitude detection using name/units/standard_name/axis signals."""
    if not _has(cname, coords):
        return False
    da = coords[cname]
    name = cname.lower()
    units = _norm_units(da.attrs.get("units", ""))
    stdn = (da.attrs.get("standard_name", "") or "").strip().lower()
    axis = (da.attrs.get("axis", "") or "").strip().upper()

    name_ok = ("lon" in name) or ("long" in name) or ("longitude" in name)
    units_ok = ("degree" in units and ("east" in units or units in ("degree", "degrees", "deg")))
    std_ok = (stdn == "longitude")
    axis_ok = (axis == "X" and ("degree" in units or "east" in units))
    return name_ok or units_ok or std_ok or axis_ok


def _is_z(cname: str, coords) -> bool:
    """CF-ish vertical detection using name/units/standard_name/axis signals."""
    if not _has(cname, coords):
        return False
    da = coords[cname]
    name = cname.lower()
    units = _norm_units(da.attrs.get("units", ""))
    stdn = (da.attrs.get("standard_name", "") or "").strip().lower()
    axis = (da.attrs.get("axis", "") or "").strip().upper()

    name_ok = any(k in name for k in ("z", "height", "geometric_height", "altitude"))
    # accept metre variants; avoid overly-broad substring matches
    units_ok = any(tok in units for tok in ("metre", "meter", "metres", "meters")) or units == "m"
    std_ok = (stdn in ("altitude", "height"))
    axis_ok = (axis == "Z" and ("metre" in units or "meter" in units or units == "m"))
    return name_ok or units_ok or std_ok or axis_ok


def is_lonlat(obj: Union[xr.Dataset, xr.DataArray], dims: tuple[str, str]) -> bool:
    """
    True if dims look like (lat, lon) by CF-ish signals on coords.
    Requires both dims to be present as coords.
    """
    y, x = dims
    coords = obj.coords if isinstance(obj, xr.DataArray) else obj
    if (y not in coords) or (x not in coords):
        return False
    return _is_lat(y, coords) and _is_lon(x, coords)


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
_METER_UNITS = {"m", "meter", "meters", "metre", "metres"}


def _units_str(c: xr.DataArray) -> str:
    return str(getattr(c, "units", "") or "").strip().lower()


def _is_meter_like(c: xr.DataArray) -> bool:
    u = _norm_units(_units_str(c))
    return (u in _METER_UNITS) or any(tok in u for tok in ("metre", "meter"))


def infer_resolution(ds: xr.Dataset) -> tuple[float, float]:
    """
    Infer horizontal grid spacing (dx, dy) in meters using robust metadata checks.

    - If dims are geographic (('lat','lon')) and both are in degrees
      (via `_coord_is_degrees`), convert to meters using Earth radius and the
      *median* latitude.
    - Else if dims are projected (('y','x')) and units are meters, return the
      *median* spacing along each axis.
    - If units are missing but dims are not degrees, treat diffs as meters.
    """
    y, x = get_spatial_dims(ds)  # ('lat','lon') or ('y','x')
    if (y not in ds.coords) or (x not in ds.coords):
        raise ValueError(f"infer_resolution: coords '{y}' and/or '{x}' not found.")

    ycoord = ds[y]
    xcoord = ds[x]

    y_in_degrees = _coord_is_degrees(ycoord)
    x_in_degrees = _coord_is_degrees(xcoord)

    # Geographic lat/lon in degrees
    if (y in ("lat", "latitude")) and (x in ("lon", "longitude")):
        if y_in_degrees and x_in_degrees:
            d_lat = np.deg2rad(float(ycoord.diff(y).median()))
            d_lon = np.deg2rad(float(xcoord.diff(x).median()))
            phi = np.deg2rad(float(ycoord.median()))
            dy = earth_radius * d_lat
            dx = earth_radius * np.cos(phi) * d_lon
            return dx, dy
        # else fall through

    # Projected / Cartesian axes in meters
    if _is_meter_like(ycoord) and _is_meter_like(xcoord):
        dy = float(ycoord.diff(y).median())
        dx = float(xcoord.diff(x).median())
        return dx, dy

    # Fallback: units missing or non-standard but not degrees → assume meters
    if (not y_in_degrees) and (not x_in_degrees):
        dy = float(ycoord.diff(y).median())
        dx = float(xcoord.diff(x).median())
        return dx, dy

    raise ValueError(
        "infer_resolution: could not infer spacing. "
        f"Dims=({y},{x}), units=({_units_str(ycoord)}, {_units_str(xcoord)})")
