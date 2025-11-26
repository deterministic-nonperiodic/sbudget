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
    "check_convert_units",
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
    "level": {
        "standard_name": {
            'altitude', 'height', 'depth', 'geopotential_height',
            'height_above_geopotential_datum',
            'height_above_mean_sea_level',
            'height_above_reference_ellipsoid',
            'atmosphere_hybrid_height_coordinate',
            'atmosphere_sigma_coordinate',
            'atmosphere_sleve_coordinate'
        },
        "units": ('meter', 'm', 'gpm', 'Pa', 'hPa', 'mb', 'millibar', '~'),
        "axis": ('Z', 'vertical')
    }
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


def _find_coordinate(ds: xr.Dataset, name: str,
                     raise_notfound: bool = True,
                     check_duplicates: bool = False) -> xr.DataArray | None:
    """
    Find a dimension coordinate in a Dataset using CF conventions.
    
    Searches for coordinates by:
    1. Exact name match
    2. CF convention attributes (standard_name, axis, units)
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset to search
    name : str
        Coordinate type to find ('lat', 'lon', 'level')
    raise_notfound : bool
        If True, raise ValueError when coordinate not found
    check_duplicates : bool
        If True, raise ValueError when multiple candidates found
        
    Returns
    -------
    xr.DataArray or None
        The found coordinate, or None if not found and raise_notfound=False
    """
    # Try exact name match first
    coord = ds.coords.get(name)
    if coord is not None:
        return coord

    if name not in _CF_COORDS_LOOKUP:
        raise ValueError(
            f"Unknown coordinate type: {name}. Must be one of {list(_CF_COORDS_LOOKUP.keys())}")

    criteria = _CF_COORDS_LOOKUP[name]

    # Build predicate function based on available criteria
    def matches_criteria(c: xr.DataArray) -> bool:
        # Check name
        if 'names' in criteria and c.name in criteria['names']:
            return True

        # Check standard_name attribute
        if 'standard_name' in criteria:
            std_name = c.attrs.get('standard_name', '').strip().lower()
            expected = criteria['standard_name']
            if isinstance(expected, str):
                if std_name == expected:
                    return True
            elif isinstance(expected, tuple):
                if std_name in expected:
                    return True

        # Check axis attribute
        if 'axis' in criteria:
            axis = c.attrs.get('axis', '').strip().upper()
            expected = criteria['axis']
            if isinstance(expected, str):
                if axis == expected:
                    return True
            elif isinstance(expected, tuple):
                if axis in expected:
                    return True

        # Check units hints
        if 'units_hints' in criteria:
            units = c.attrs.get('units', '').strip().lower()
            if any(hint in units for hint in criteria['units_hints']):
                return True

        # Check units (for level coordinate)
        if 'units' in criteria:
            units = c.attrs.get('units', '').strip().lower()
            expected = criteria['units']
            if isinstance(expected, tuple):
                if units in expected:
                    return True

        return False

    # Search through dimension coordinates
    candidates = [ds.coords[dim] for dim in ds.dims if matches_criteria(ds.coords[dim])]

    if check_duplicates and len(candidates) > 1:
        raise ValueError(f"Multiple {name} coordinates found: {[c.name for c in candidates]}")

    if not candidates:
        if raise_notfound:
            raise ValueError(
                f"The coordinate '{name}' is not in the dataset or is "
                f"inconsistent with CF conventions. Available dims: {list(ds.dims)}"
            )
        return None

    return candidates[0]


# from Metpy
# Create a pint UnitRegistry object
UNITS_REG = pint.UnitRegistry()

# from Metpy
cmd = re.compile(r"(?<=[A-Za-z)])(?![A-Za-z)])(?<![0-9\-][eE])(?<![0-9\-])(?=[0-9\-])")


def _get_units_str(c: xr.DataArray) -> str:
    """Extracts and normalizes the units string from a DataArray."""
    units = c.attrs.get("units", "").strip()
    return units


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


def _is_z(cname: str, coords: Union[xr.Dataset, xr.DataArray, Any]) -> bool:
    """
    Robust CF-compliant vertical coordinate detection for HEIGHT-based coordinates.
    
    Returns True only for height/altitude coordinates (in meters).
    Returns False for pressure/isobaric coordinates.
    
    Uses CF conventions to identify vertical coordinates by checking:
    - Coordinate name patterns (z, height, altitude, etc.)
    - CF standard_name attribute
    - axis='Z' attribute (with meter units)
    - Units (meters only, NOT pressure)
    
    Parameters
    ----------
    cname : str
        Coordinate name to check
    coords : Dataset, DataArray, or coordinate dict
        Container with coordinates
        
    Returns
    -------
    bool
        True if coordinate is a height-based vertical coordinate
    """
    if cname not in coords:
        return False

    coord = coords[cname]
    name = cname.lower()
    units = _get_units_str(coord).lower()
    standard_name = (coord.attrs.get("standard_name", "") or "").strip().lower()
    axis = (coord.attrs.get("axis", "") or "").strip().upper()

    # Exclude pressure coordinates explicitly
    pressure_units = ('pa', 'hpa', 'mb', 'millibar', 'bar')
    pressure_names = ('plev', 'pressure', 'pres', 'isobaric')
    pressure_std_names = ('air_pressure', 'atmosphere_ln_pressure_coordinate')

    # If it's clearly a pressure coordinate, return False
    if any(unit in units for unit in pressure_units):
        return False
    if any(pname in name for pname in pressure_names):
        return False
    if standard_name in pressure_std_names:
        return False

    # Check for height-based coordinates
    # Check axis='Z' with meter units (most reliable)
    meter_units = ('m', 'meter', 'meters', 'metre', 'metres', 'gpm')
    if axis == "Z" and any(unit in units for unit in meter_units):
        return True

    # Check standard_name (CF-compliant, height-based only)
    if standard_name in _CF_COORDS_LOOKUP['level']['standard_name']:
        return True

    # Check name patterns (height-related only)
    height_name_patterns = ('z', 'height', 'altitude', 'depth', 'zlev', 'z_')
    if any(pattern in name for pattern in height_name_patterns):
        # Verify it has meter units to avoid false positives
        if any(unit in units for unit in meter_units):
            return True

    # Check for generic 'lev' or 'level' with meter units
    if ('lev' in name or 'level' in name) and any(unit in units for unit in meter_units):
        return True

    return False


def _is_geographic(coord: xr.DataArray, coord_type: str) -> bool:
    """
    Robust CF-compliant check for latitude/longitude coordinates.
    
    Parameters
    ----------
    coord : xr.DataArray
        Coordinate to check
    coord_type : str
        Expected type: 'lat' or 'lon'
        
    Returns
    -------
    bool
        True if coordinate matches the expected geographic type
    """
    if coord_type not in ('lat', 'lon'):
        raise ValueError(f"coord_type must be 'lat' or 'lon', got {coord_type}")

    lookup = _CF_COORDS_LOOKUP.get(coord_type, {})
    if not lookup:
        return False

    name = (coord.name or "").lower()
    units = str(coord.attrs.get("units", "")).lower()
    standard_name = str(coord.attrs.get("standard_name", "")).lower()
    axis = str(coord.attrs.get("axis", "")).upper()

    # Check axis (most reliable for CF compliance)
    expected_axis = lookup.get("axis")
    if axis and axis == expected_axis:
        # Verify units are degree-like to avoid false positives
        if any(hint in units for hint in ('degree', 'deg')):
            return True

    # Check standard_name (CF-compliant)
    expected_std = lookup.get("standard_name")
    if standard_name and standard_name == expected_std:
        return True

    # Check name patterns
    name_ok = any(name == n or name.endswith(n) for n in lookup.get("names", ()))

    # Check units with direction-specific validation
    units_hints = lookup.get("units_hints", ())
    units_ok = any(hint in units for hint in units_hints)

    # Ensure direction-specific hints are not cross-matched
    if coord_type == "lon":
        if "north" in units or "degree_north" in units:
            return False  # This is latitude, not longitude
        if units_ok or name_ok:
            # Additional check: longitude values should be in reasonable range
            if coord.size > 0:
                vals = coord.values
                vals = vals[np.isfinite(vals)]
                if vals.size > 0:
                    # Longitude typically in [-180, 360] range
                    if np.abs(vals).max() > 400:
                        return False
            return True

    if coord_type == "lat":
        if "east" in units or "degree_east" in units:
            return False  # This is longitude, not latitude
        if units_ok or name_ok:
            # Additional check: latitude values should be in [-90, 90]
            if coord.size > 0:
                vals = coord.values
                vals = vals[np.isfinite(vals)]
                if vals.size > 0:
                    if np.abs(vals).max() > 90.5:  # Small tolerance
                        return False
            return True

    return False


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


def _is_global_longitude(x_coord: xr.DataArray) -> bool:
    """
    Determine if a longitude coordinate covers (nearly) the full globe.

    Strategy (robust to wrap-around and irregular spacing):
    - Normalize longitudes to [0, 360).
    - Sort along the x-direction and compute circular gaps between consecutive points,
      including the wrap gap (last→first + 360).
    - If the largest gap is no bigger than a small tolerance (~a few grid spacings),
      then coverage is effectively global.

    Ignores NaNs and duplicate endpoints (e.g., both 0 and 360 present).
    """

    def _normalize_deg(x):
        """Map to [0, 360) in degrees; ignore NaNs."""
        return np.asarray(x, dtype=np.float32) % 360.0

    lon = x_coord.values
    lon = _normalize_deg(lon[np.isfinite(lon)])

    # Sort unique longitudes (avoid duplicate endpoints like 0 and 360)
    lon = np.unique(np.sort(lon))
    if lon.size < 2:
        return False

    # Estimate a representative spacing (median nearest-neighbor gap on the circle)
    diffs = np.diff(lon)
    wrap_gap = (lon[0] + 360.0) - lon[-1]
    all_gaps = np.concatenate([diffs, [wrap_gap]])
    # If there are large holes, the largest gap will reflect that.
    max_gap = float(np.max(all_gaps))

    # Tolerance: allow a few grid spacings worth of slack (handles uneven grids)
    # Use the median gap as a spacing proxy; fall back to 360/N if needed.
    spacing = float(np.median(all_gaps)) if all_gaps.size else 360.0 / lon.size

    # “Global” if there is no big uncovered arc: i.e., largest gap ≲ tol=1.5 * spacing
    return max_gap <= 1.5 * spacing


# ----------------------
# Spatial dim resolution
# ----------------------
def get_spatial_dims(obj: Union[xr.Dataset, xr.DataArray]) -> Tuple[str, str]:
    """
    Robustly determine horizontal dimensions using CF conventions.

    Priority:
    1) CF-compliant lat/lon coordinates as 1-D dimensions
    2) Projected y/x coordinates with 2-D auxiliary lat/lon
    3) Plain y/x dimensions
    4) Fallback: last two dimensions if they look spatial
    
    Returns
    -------
    tuple of str
        (y_dim, x_dim) - the horizontal dimension names
    """
    ds = obj if isinstance(obj, xr.Dataset) else obj.to_dataset(name="_tmp")
    dims = set(ds.dims)

    # Case A: Try to find CF-compliant lat/lon as 1-D dimensions
    try:
        lat_coord = _find_coordinate(ds, 'lat', raise_notfound=False)
        lon_coord = _find_coordinate(ds, 'lon', raise_notfound=False)

        if lat_coord is not None and lon_coord is not None:
            # Check if they are 1-D dimension coordinates
            if (lat_coord.name in dims and lon_coord.name in dims and
                    lat_coord.ndim == 1 and lon_coord.ndim == 1):
                return str(lat_coord.name), str(lon_coord.name)
    except ValueError:
        pass

    # Case B: Check for standard lat/lon names as 1-D dims (fallback)
    if "lat" in dims and "lon" in dims:
        if ds["lat"].ndim == 1 and ds["lon"].ndim == 1:
            # Verify they look geographic
            if _is_geographic(ds["lat"], "lat") and _is_geographic(ds["lon"], "lon"):
                return "lat", "lon"

    # Case C: Projected axes with 2-D auxiliary lat/lon(y,x)
    if {"y", "x"} <= dims:
        # Check if there are 2-D lat/lon coordinates
        if "lat" in ds.coords and "lon" in ds.coords:
            if ds["lat"].dims == ("y", "x") and ds["lon"].dims == ("y", "x"):
                return "y", "x"
        # Plain y/x without auxiliary coords
        return "y", "x"

    # Case D: Fallback - use last two dimensions if they look spatial
    # (not time, not vertical)
    if len(ds.dims) >= 2:
        # Get all dims, filter out known non-spatial dims
        spatial_candidates = []
        for dim in ds.dims:
            # Skip if it's clearly time
            if str(dim).lower() in ('time', 't', 'date'):
                continue
            # Skip if it's clearly vertical
            if _is_z(str(dim), ds.coords):
                continue
            spatial_candidates.append(dim)

        # If we have at least 2 spatial candidates, use the last two
        if len(spatial_candidates) >= 2:
            # Convention: last two are (y, x) or (lat, lon)
            return spatial_candidates[-2], spatial_candidates[-1]

    raise ValueError(
        "get_spatial_dims: Could not determine horizontal dimensions. "
        "Expected CF-compliant lat/lon or projected y/x coordinates. "
        f"Available dims: {tuple(ds.dims)}, coords: {tuple(ds.coords)}"
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
    y_dim, x_dim = get_spatial_dims(ds)  # ('lat','lon') or ('y','x')
    if (y_dim not in ds.coords) or (x_dim not in ds.coords):
        raise ValueError(f"infer_resolution: coords '{y_dim}' and/or '{x_dim}' not found.")

    ycoord = ds[y_dim]
    xcoord = ds[x_dim]

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
        dy = float(ycoord.diff(y_dim).median())
        dx = float(xcoord.diff(x_dim).median())
        return dx, dy

    raise ValueError(
        "infer_resolution: could not infer spacing. "
        f"Dims=({y_dim},{x_dim}), units=({_get_units_str(ycoord)}, {_get_units_str(xcoord)})")
