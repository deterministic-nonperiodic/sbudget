import shutil
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

import dask
import xarray as xr
from dask.base import is_dask_collection
from dask.diagnostics import ProgressBar
from dask.distributed import progress

from .cf_coords import _cf_guess
from .cf_coords import _is_z, is_geographic_grid, _coord_is_meter
from .cf_coords import convert_units, check_convert_units
from .memory_manager import CacheManager

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


def _report_var_existence(raw: xr.Dataset, cfg) -> str:
    """Create a plain-text report showing whether configured variables exist in the raw dataset.

    Checks both the *actual* names provided under cfg.variables and, after renaming,
    validates the presence of the *logical* names our code expects.
    """
    lines = []
    mapping = cfg.variables or {}
    # 1) Check actual names against the raw dataset
    lines.append("[I/O] Configured variables (actual names) in file:\n")
    for logical, actual in mapping.items():
        exists = (actual in raw.data_vars) or (actual in raw.coords)
        status = "OK" if exists else "MISSING"
        hint = ""
        if not exists:
            guess = _cf_guess(raw, logical)
            if guess:
                hint = f"  (hint: possible '{logical}' → '{guess}' by CF)"
        lines.append(f"[I/O]  - {logical:11s} ← {actual:20s} : {status}{hint}\n")

    # 2) Minimal logical requirements after renaming
    required = ["u", "v", "w"]
    lines.append("[I/O] Logical requirements: ")
    for key in required:
        lines.append(f"  - {key} : required")
    lines.append("  - theta : preferred (else need pressure + temperature)")

    return " ".join(lines)


def open_dataset(cfg, verbose=False) -> xr.Dataset:
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

    if verbose:
        report = _report_var_existence(ds, cfg)
        print(report)

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
        print("[sbudget] Performing analysis on selected levels: ", ds.z.values)

    return ds


def _resolve_store_and_path(path: Path | str, store_type: Optional[str] = None) -> Tuple[Path, str]:
    """
    Validates and resolves the output store type and file path extension.

    1. Determines the target store (zarr or netcdf), defaulting to 'zarr'.
    2. Ensures the Path has the correct corresponding extension (.zarr or .nc).

    Parameters
    ----------
    path : Path or str
        The output path specified by the user.
    store_type : Optional[str]
        The store type specified in the configuration (e.g., 'netcdf').

    Returns
    -------
    Tuple[Path, str]
        The corrected Path object and the resolved store type.
    """
    if isinstance(path, str):
        path = Path(path)

    # 1. Determine Target Store
    store = (store_type or "").lower()

    # Infer store from extension if config_store_type is empty/none
    suffix = path.suffix.lower()

    if store not in {"netcdf", "zarr"}:
        if suffix in {".nc", ".nc4", ".cdf"}:
            store = "netcdf"
        elif suffix == ".zarr":
            store = "zarr"
        else:
            # Default preference if nothing is recognized
            store = "zarr"
            if suffix != "":
                print(f"Unrecognized output extension '{suffix}' -- defaulting to Zarr.")

    # 2. Ensure Extension Matches Resolved Store
    if store == "zarr":
        # Force .zarr suffix for clarity
        if suffix != ".zarr":
            path = path.with_suffix(".zarr")
            print(f"[I/O] Corrected output path to {path} (Zarr).")
    elif store == "netcdf":
        # Force .nc suffix for consistency
        if suffix not in {".nc", ".nc4", ".cdf"}:
            path = path.with_suffix(".nc")
            print(f"[I/O] Corrected output path to {path} (NetCDF).")

    return path, store


def write_dataset(ds: xr.Dataset, cfg, client=None) -> None:
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
    client : dask.distributed.Client, optional
        An optional Dask client for distributed computation. If provided,
        it will be used to manage the Dask graph execution.

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

    # Get output path and store. Overwrite settings
    # --- Validation and Path Resolution ---
    output_path = Path(getattr(cfg.output, "path", "output.zarr"))
    store_type = getattr(cfg.output, "store", None)

    # Determine the final path and store type requested by the user
    output_path, store_type = _resolve_store_and_path(output_path, store_type)

    temp_zarr_path = Path(str(output_path) + ".tmp.zarr")

    # The actual parallel write destination is always Zarr
    write_target = temp_zarr_path if store_type == "netcdf" else output_path

    # Clean up any remnants or requested final output before starting
    if write_target.exists() or temp_zarr_path.exists():
        if not cfg.output.overwrite:
            raise FileExistsError(f"{output_path} exists; set output.overwrite: true to replace")

        # Clean up both potential paths if overwrite is true
        if write_target.exists():
            shutil.rmtree(write_target)
        if temp_zarr_path.exists():
            shutil.rmtree(temp_zarr_path)

    # --- Setup Parallel Zarr Write ---
    ds.attrs.update(_global_attrs)

    engine = getattr(cfg.input, "engine", "netcdf4")
    print(f"[I/O] Starting computation ...")

    # Define the lazy Zarr write operation
    delayed_write_op = ds.to_zarr(
        write_target,
        mode="w",
        compute=False,
        zarr_format=3,
        consolidated=False,
    )

    # --- Execute Parallel Write (Triggers Dask Graph) ---
    if is_dask_collection(delayed_write_op):
        if client:
            print(f"[I/O] Executing parallel Dask graph ...")
            future = client.compute(delayed_write_op)
            progress(future, notebook=False)

            try:
                future.result()
            except Exception as e:
                print(f"\n[I/O] FATAL WRITE ERROR: Zarr computation failed on the cluster.")
                # Attempt to clean up temp store before raising
                if temp_zarr_path.exists(): shutil.rmtree(temp_zarr_path)
                raise e

        else:
            # Synchronous fallback
            print(f"[I/O] Executing synchronous Dask graph using local threads...")
            with dask.config.set(scheduler='threads'):
                with ProgressBar():
                    dask.compute(delayed_write_op)

    print(f"[I/O] Calculation completed")

    # --- Synchronous NetCDF Conversion (If Requested) ---
    if store_type == "netcdf":
        print(f"[I/O] Starting synchronous Zarr-to-NetCDF conversion...")

        # Load Zarr store (should be fast metadata read, data is already computed)
        ds_computed = xr.open_zarr(temp_zarr_path, consolidated=False)

        # Synchronous write to final NetCDF file (this resolves the HDF5 lock issue)
        ds_computed.to_netcdf(output_path, engine=engine)

        print(f"[I/O] Final output successfully written to: '{output_path}' (NetCDF).")

        # Clean up the temporary Zarr store immediately
        shutil.rmtree(temp_zarr_path)

    else:
        # Final output was Zarr, written in step 3.
        print(f"[I/O] Final output successfully written to: {output_path} (Zarr).")

    # --- Automatic cache cleanup ---
    print(f"[cache] Cleaning up temporary cache directories")
    CacheManager(verbose=False).cleanup_all()
    print(f"[cache] Done.")
