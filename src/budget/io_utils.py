import shutil
from datetime import datetime
from pathlib import Path

import dask
import xarray as xr
from dask.base import is_dask_collection
from dask.diagnostics import ProgressBar

from .cf_coords import _is_z, is_geographic_grid, _coord_is_meter
from .cf_coords import convert_units, check_convert_units
from .chunking_tools import CacheManager

_global_attrs = {'source': 'git@github.com:deterministic-nonperiodic/sbudget.git',
                 'institution': 'Leibniz Institute for Atmospheric Physics (IAP)',
                 'history': datetime.today().strftime('Created on %c'),
                 'references': '', 'Conventions': 'CF-1.6'}


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

    # Check vertical coordinate units
    if target_name in ds.coords:
        z_coord = ds[target_name]
        if not _coord_is_meter(z_coord):
            # Assume km if not meters
            print(f"Converting vertical coordinate '{target_name}' to meters ...")
            ds = ds.assign_coords({target_name: convert_units(z_coord, "km", "m")})

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
          density: rho     # optional; if absent, computed from pressure & temperature
          theta: theta     # optional; if absent, computed from pressure & temperature
          divergence: div  # optional, else computed
          vorticity: vor   # optional, else computed
    """
    p = cfg.input.path
    if str(p).endswith(".zarr"):
        ds = xr.open_zarr(p, chunks="auto")
    else:
        engine = getattr(cfg.input, "engine", None)
        ds = xr.open_mfdataset(p, chunks="auto", engine=engine)

    # unify data type
    ds = ds.astype("float32")

    # Rename dataset variables to logical names used by the code
    rename = {}
    for logical, actual in (cfg.variables or {}).items():
        if logical != actual and actual in ds:
            rename[actual] = logical
    if rename:
        ds = ds.rename(rename)

    # Normalize coordinate names to standard ones
    z_name, y_name, x_name = cfg.input.dims

    # standard target names
    if is_geographic_grid(ds[x_name], ds[y_name]):
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

    # Ensure units consistency
    ds = check_convert_units(ds)

    # Interpolate to consistent vertical coordinates and convert to meters if needed
    ds = ensure_vertical_consistent(ds, target_name='z')

    # select specified vertical levels
    levels = getattr(cfg.compute, "levels", None)
    mode = str(cfg.compute.mode).strip()

    # select specified vertical levels
    if levels is not None and mode == "scale_transfer":
        ds = ds.sel(z=levels, method='nearest')
        print("[budget] Performing analysis on selected levels: ", ds.z.values)

    return ds


def write_dataset(ds: xr.Dataset, cfg) -> None:
    """
    Write a Dask-backed xarray.Dataset to disk efficiently.

    This function handles large, lazily-evaluated Dask datasets produced by
    the inter-scale energy transfer computation. It performs a parallel write
    to either Zarr or NetCDF format while ensuring that:
      - The dataset is not prematurely computed in memory.
      - Hybrid or on-disk cache directories remain available until I/O completes.
      - All temporary cache directories are cleaned up only after a successful write.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to write. Either an in-memory xarray object or a
        Dask-backed dataset containing lazy computations.
    cfg : Namespace or dict-like
        Configuration object specifying output options. Must define:
            - `output.path` (str or Path): Destination file path.
            - `output.store` (str): Either 'zarr' or 'netcdf'.
            - `output.overwrite` (bool): Whether to overwrite existing files.
            - Optionally, `input.engine` for NetCDF (e.g., 'h5netcdf', 'netcdf4').

    Notes
    -----
    - For very large datasets (>100 GB), Zarr is strongly recommended since
      it supports parallel, chunked writes.
    - The function avoids calling `.compute()` directly on the dataset to
      prevent exhausting system memory.
    - Dask tasks are executed only once during the write operation.
    - Cache cleanup occurs after I/O completion to ensure no data loss.

    Examples
    --------
    >>> cfg.output.path = "energy_transfer.zarr"
    >>> cfg.output.store = "zarr"
    >>> cfg.output.overwrite = True
    >>> write_dataset(ds, cfg)

    This writes the Dask-backed dataset to a Zarr store using parallel I/O and
    cleans up all temporary cache directories after completion.
    """

    out = Path(cfg.output.path)
    if out.exists() and not cfg.output.overwrite:
        raise FileExistsError(f"{out} exists; set output.overwrite: true to replace")

    ds.attrs.update(_global_attrs)

    store_type = cfg.output.store.lower()
    engine = getattr(cfg.input, "engine", "h5netcdf")
    scheduler = getattr(cfg.compute, "scheduler", "threads")

    print(f"[I/O] Writing output to {cfg.output.path} (store={store_type})")

    # --- Choose writing method ---
    if store_type == "zarr":
        if out.exists():
            shutil.rmtree(out)
        write_op = ds.to_zarr(out, mode="w", compute=False)
    elif store_type == "netcdf":
        # For Dask safety, use compute=False (requires recent xarray)
        encoding = {var: {'zlib': True, 'complevel': 4} for var in ds.data_vars}
        write_op = ds.to_netcdf(out, engine=engine, encoding=encoding, compute=False)
    else:
        raise ValueError("output.store must be 'zarr' or 'netcdf'")

    # --- Trigger computation explicitly and safely ---
    if is_dask_collection(write_op):
        print(f"[I/O] Executing parallel write with scheduler: {scheduler} ...")
        with ProgressBar():
            dask.compute(write_op, scheduler=scheduler)  # Ensures all I/O is done before cleanup
    else:
        print("[I/O] Direct write completed (non-lazy dataset).")

    print(f"[I/O] Wrote output file: {cfg.output.path}")

    # --- Automatic cache cleanup ---
    print(f"[cache] Cleaning up temporary cache directories")
    CacheManager(verbose=False).cleanup_all()
    print(f"[cache] Done.")
