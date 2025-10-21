import math
import shutil
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import xarray as xr
from .cf_coords import _is_z, is_lonlat


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


def ensure_optimal_chunking(
        ds: xr.Dataset,
        spatial_dims: Tuple[str, str] = ("lat", "lon"),  # e.g. ("y","x") or ("lat","lon")
        vertical_dim: str = "z",
        target_chunk_mb: int = 64,
        preferred: Optional[Dict[str, int]] = None,  # any extra dims you want to chunk
        quiet: bool = False,
) -> xr.Dataset:
    """
    Rechunk for fast 2-D FFTs (physical space).

    - Forces single chunks along `spatial_dims` (FFT axes).
    - Tiles over 'time' and `vertical_dim` to aim for ~`target_chunk_mb` per chunk.
    - `preferred` can add chunking for *other* dims (never overrides spatial dims).
    """
    preferred = dict(preferred or {})
    y, x = spatial_dims
    if y not in ds.dims or x not in ds.dims:
        raise ValueError(f"Spatial dims {spatial_dims} must exist in dataset dims {tuple(ds.dims)}")

    # element size (bytes): pick the max dtype across variables
    item_size = max(
        (int(getattr(v.data, "dtype", np.dtype("float64")).itemsize) for v in
         ds.data_vars.values()),
        default=8,
    )

    bytes_plane = item_size * ds.sizes[y] * ds.sizes[x]  # one full (y,x) plane
    target_bytes = int(target_chunk_mb * 1024 ** 2)
    budget_mult = max(1, target_bytes // max(1, bytes_plane))  # how many (time*z) we can pack

    # build chunk plan
    plan: Dict[str, int] = {y: -1, x: -1}  # single chunk along FFT axes

    # choose time/z tiles within budget
    t_guess = int(ds.sizes["time"]) if "time" in ds.dims else 1
    z_guess = int(ds.sizes[vertical_dim]) if "z" in ds.dims else 1

    if ("time" in ds.dims) and (vertical_dim in ds.dims):
        # start from hints; if over budget, redistribute near-sqrt
        t_chunk = t_guess
        z_chunk = z_guess
        if t_chunk * z_chunk > budget_mult:
            z_chunk = max(1, min(z_guess, int(math.sqrt(budget_mult))))
            t_chunk = max(1, min(t_guess, budget_mult // z_chunk))
        plan["time"] = max(1, t_chunk)
        plan[vertical_dim] = max(1, z_chunk)
    elif "time" in ds.dims:
        plan["time"] = max(1, min(int(t_guess), budget_mult))
    elif vertical_dim in ds.dims:
        plan[vertical_dim] = max(1, min(int(z_guess), budget_mult))

    # any extra dims from preferred (don’t override spatial/time/z decisions)
    for d, c in preferred.items():
        if d not in plan and d in ds.dims:
            plan[d] = max(1, min(int(c), ds.sizes[d]))

    out = ds.unify_chunks().chunk(plan)

    if not quiet:
        est = bytes_plane * max(1, plan.get("time", 1)) * max(1, plan.get(vertical_dim, 1))
        msg = ", ".join(f"{d}={'all' if c == -1 else c}" for d, c in plan.items())
        print(f"[chunking] ({y},{x}) single-chunk; plan: {msg} | "
              f"~{est / 1024 ** 2:.1f} MB/chunk (target {target_chunk_mb} MB)")

    return out


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
        ds = ds.rename_vars(coord_rename)

    # (Optional) ensure that standardized coords are dimension/index coords
    for cname in ("z", target_y, target_x):
        if cname in ds.coords and cname in ds.dims:
            # Make sure the coord is indexed by its own dim
            ds = ds.set_coords(cname)

    # Interpolate to consistent vertical coordinates
    ds = ensure_vertical_consistent(ds)

    # Apply consistent rechunking:
    ds = ensure_optimal_chunking(ds, spatial_dims=(y_name, x_name), target_chunk_mb=64)

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
        ds.to_netcdf(out)
    else:
        raise ValueError("output.store must be 'zarr' or 'netcdf'")
