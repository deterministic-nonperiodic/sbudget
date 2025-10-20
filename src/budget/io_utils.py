from pathlib import Path

import shutil
import numpy as np
import xarray as xr

from typing import Union


# ----------------------
# Compact CF-aware utils
# ----------------------
def _has(cname: str, coords) -> bool:
    """Return True if coordinate name exists."""
    return cname in coords


def _norm_units(u: str) -> str:
    """Normalize CF-ish units for robust checks."""
    u = (u or "").strip().lower()
    u = u.replace("°", "degree").replace("-", "_").replace(" ", "_")
    return u


def _coord_is_degrees(
        coord: xr.DataArray,
        allow_infer: bool = True,
        tol: float = 1e-12,
) -> bool:
    """
    True if `coord` uses degrees (CF-compliant). If units are absent/ambiguous and
    `is_lon` or `is_lat` is True, infer degrees when |values| exceed 2π (cannot be radians).
    """
    units = _norm_units(coord.attrs.get("units", ""))

    # Explicit units
    if "radian" in units:
        return False
    if units == "deg" or units.startswith("degree") or units.startswith("degrees"):
        return True

    # Heuristic inference for lon/lat when units missing/unknown
    if allow_infer:
        vals = np.asarray(coord.values)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            max_abs = float(np.nanmax(np.abs(vals)))
            if max_abs > (2.0 * np.pi + tol):
                return True  # exceeds radians range → treat as degrees

    return False


def _is_lat(cname: str, coords) -> bool:
    """CF-ish latitude detection with name/units/standard_name/axis signals."""
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
    """CF-ish longitude detection with name/units/standard_name/axis signals."""
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
    """CF-ish longitude detection with name/units/standard_name/axis signals."""
    if not _has(cname, coords):
        return False
    da = coords[cname]
    name = cname.lower()
    units = _norm_units(da.attrs.get("units", ""))
    stdn = (da.attrs.get("standard_name", "") or "").strip().lower()
    axis = (da.attrs.get("axis", "") or "").strip().upper()

    name_ok = ("z" in name) or ("height" in name) or ("geometric_height" in name) or ("altitude" in
                                                                                      name)
    units_ok = ("m" in units and ("meter" in units or units in ("meters",)))
    std_ok = (stdn == "altitude")
    axis_ok = (axis == "Z" and ("meter" in units))
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


def ensure_vertical_consistent(ds: xr.Dataset, target_name="z") -> xr.Dataset:
    """Interpolate to common vertical levels"""

    if target_name not in ds.coords:
        raise ValueError(f"'target_name' '{target_name}' is not a coordinate in the dataset.")

    # Candidate vertical dims: anything that's not the standardized horizontal/time or target_name
    excluded = {"time", "x", "y", target_name}
    z_candidate = [str(d) for d in ds.dims if _is_z(str(d), ds.coords) and (d not in excluded)]

    for z_dim in z_candidate:
        # Check for metadata consistent with vertical coordinate
        print(f"Interpolating vertical coord {z_dim} --> {target_name} ...")
        ds = ds.interp({z_dim: ds[target_name]}, method="linear")

        if z_dim in ds.coords:
            ds = ds.drop_vars(z_dim)

    return ds


def open_dataset(cfg) -> xr.Dataset:
    """Open input dataset and normalize variable names using cfg.variables.

    cfg.variables should map logical names to actual dataset variable names, e.g.::
        variables:
          u: U
          v: V
          w: W
          pressure: pres
          temperature: temp
          theta: theta     # optional; if absent, computed from pressure & temperature
          divergence: div  # optional, else computed
          vorticity: vor   # optional, else computed
    """
    p = cfg.input.path
    if str(p).endswith(".zarr"):
        ds = xr.open_zarr(p, chunks="auto")
    else:
        ds = xr.open_dataset(p, chunks="auto")

    # Rename dataset variables to logical names used by the code
    rename = {}
    for logical, actual in (cfg.variables or {}).items():
        if logical != actual and actual in ds:
            rename[actual] = logical
    if rename:
        ds = ds.rename(rename)

    # Normalize coordinate names to standard ones
    # cfg.input.dims: the ORIGINAL dimension names in the file, e.g. ["z_mc", "lat", "lon"] or ["z", "y", "x"]
    z_name, y_name, x_name = cfg.input.dims

    # standard target names
    if is_lonlat(ds, (y_name, x_name)):
        target_y, target_x = "lat", "lon"
        ds.attrs["grid_type"] = "lonlat"
    else:
        target_y, target_x = "y", "x"
        ds.attrs["grid_type"] = "cartesian"

    # Prepare renaming for dims/coords
    dim_rename, coord_rename = {}, {}

    if z_name != "z":
        dim_rename[z_name] = "z"
    if y_name != target_y:
        dim_rename[y_name] = target_y
    if x_name != target_x:
        dim_rename[x_name] = target_x

    # 1) Rename dimensions only
    if dim_rename:
        ds = ds.rename_dims(dim_rename)

    # Rename coordinate variables if they still exist under their OLD names
    #    (compute this AFTER rename_dims so we check current membership)
    coord_rename = {}
    if z_name in ds.coords and z_name != "z":
        coord_rename[z_name] = "z"
    if y_name in ds.coords and y_name != target_y:
        coord_rename[y_name] = target_y
    if x_name in ds.coords and x_name != target_x:
        coord_rename[x_name] = target_x

    if coord_rename:
        ds = ds.rename(coord_rename)

    # (Optional) ensure that standardized coords are dimension/index coords
    #    This avoids the “rename does not create an index” warning paths.
    for cname in ("z", target_y, target_x):
        if cname in ds.coords and cname in ds.dims:
            # Make sure the coord is indexed by its own dim
            # (no-op if already true; harmless otherwise)
            ds = ds.set_coords(cname)

    # Interpolate to consistent vertical coordinates
    ds = ensure_vertical_consistent(ds)

    # Apply consistent rechunking:
    ds = ds.chunk("auto")

    return ds


def write_dataset(ds: xr.Dataset, cfg) -> None:
    out = Path(cfg.output.path)
    if out.exists() and not cfg.output.overwrite:
        raise FileExistsError(f"{out} exists; set output.overwrite: true to replace")
    if cfg.output.store == "zarr":
        if out.exists():
            shutil.rmtree(out)
        ds.to_zarr(out, mode="w")
    elif cfg.output.store == "netcdf":
        print(ds)
        ds.to_netcdf(out)
    else:
        raise ValueError("output.store must be 'zarr' or 'netcdf'")
