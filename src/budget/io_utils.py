import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Union

import numpy as np
import psutil
import xarray as xr

from .cf_coords import _is_z, is_geographic_grid, _coord_is_meter
from .cf_coords import convert_units, check_convert_units

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


# --------------------------------------------------------------------
# --- START OF MEMORY ESTIMATION UTILITIES ---
# --------------------------------------------------------------------
WORKING_SET_MULTIPLIER = 12  # Heuristic: 5x to 12x the output size in temporary working memory
TOTAL_SIZE_THRESHOLD = 0.8  # Fixed threshold (80%) for the initial total size check


def estimate_dataset_bytes(ds: xr.Dataset) -> int:
    """
    Estimate working-set size (bytes) for a dataset.

    - For Dask-backed vars: use the largest chunk along each dimension.
    - For NumPy-backed vars: use the full array size.
    """
    total = 0
    for var in ds.data_vars.values():
        item_size = np.dtype(var.dtype).itemsize
        chunks = getattr(getattr(var, "data", None), "chunks", None)
        if chunks is not None:
            # dask-backed → product of max chunk sizes per dim
            max_elems = 1
            for dim_chunks in chunks:
                max_elems *= max(dim_chunks)
            var_bytes = max_elems * item_size
        else:
            # eager numpy-backed → whole array
            var_bytes = int(var.size) * item_size
        total += var_bytes
    return int(total)


def fits_in_memory(ds: xr.Dataset,
                   expansion_factor: int = 1,
                   ratio_to_use: float = TOTAL_SIZE_THRESHOLD
                   ) -> Tuple[bool, int, int]:
    """
    Check if the dataset (possibly expanded along length_scale) fits in memory,
    using the specified ratio of total available memory.

    Returns
    -------
    (bool, dataset_size_bytes, max_allowed_bytes)
    """
    ds = ds if isinstance(ds, xr.Dataset) else ds.to_dataset()

    # Base size estimate
    dataset_size = estimate_dataset_bytes(ds) * max(1, int(expansion_factor))

    # NOTE: psutil is used here, but mocked above if unavailable.
    available_memory = psutil.virtual_memory().available

    # Calculate the memory limit based on the provided ratio
    max_memory = int(ratio_to_use * available_memory)

    return dataset_size < max_memory, dataset_size, max_memory


# --------------------------------------------------------------------
# --- START OF CHUNKING UTILITIES ---
# --------------------------------------------------------------------
def _balanced_chunks(n: int, target: int, min_size: int) -> Tuple[int, ...]:
    """
    Split length n into m nearly-equal chunks, all >= min_size,
    with average near `target`. Returns a tuple of chunk sizes.
    """
    if n <= 0:
        return ()
    # Choose number of chunks m so that each chunk >= min_size and near target
    # Start with m = ceil(n / target), but cap so that floor(n/m) >= min_size
    m = max(1, math.ceil(n / max(1, target)))
    while m > 1 and (n // m) < min_size:
        m -= 1
    # Now spread n across m chunks as evenly as possible (sizes differ by <= 1)
    base = n // m
    rem = n % m
    chunks = (base + 1,) * rem + (base,) * (m - rem)
    # Safety: ensure all >= min_size; if not, fall back to packing with min_size
    if any(c < min_size for c in chunks):
        m = max(1, n // min_size)
        base = n // m
        rem = n % m
        chunks = (base + 1,) * rem + (base,) * (m - rem)
    return chunks


def ensure_optimal_chunking(
        ds: xr.Dataset,
        spatial_dims: Tuple[str, str] = ("lat", "lon"),
        vertical_dim: str = "z",
        target_chunk_ratio: float = 0.005,  # Target chunk size as % of total output size
        memory_threshold_ratio: float = 0.25,  # Safer 25% threshold for Dask compute budget
        preferred: Optional[Dict[str, int]] = None,
        deriv_edge_order: int = 2,
        verbose: bool = True,
        rechunk_spatial: bool = False,  # Kept as a manual override
        output_scale_mult: int = 1,
        scale_dim: Optional[str] = None
) -> xr.Dataset:
    """
    Rechunk for fast 2-D FFTs, and ensures chunks are small enough to accommodate
    a multiplication factor (like a 'scale' dimension) in the final computation.

    Includes dynamic working set estimation to conditionally enable spatial chunking.
    """
    preferred = dict(preferred or {})
    y, x = spatial_dims
    if y not in ds.dims or x not in ds.dims:
        raise ValueError(f"Spatial dims {spatial_dims} must exist in dataset dims {tuple(ds.dims)}")

    # Required minimum chunk size for vertical dimension
    min_required_z = deriv_edge_order + 1

    # Determine the Dask worker compute memory budget (e.g., 25% of total available)
    available_memory = psutil.virtual_memory().available
    max_memory_budget = int(memory_threshold_ratio * available_memory)

    # 1. Initial Memory Check: Is the *entire* output too large? (Uses the less conservative 80% threshold)
    fits_total, total_output_bytes, max_allowed_total_bytes = fits_in_memory(
        ds,
        expansion_factor=output_scale_mult,
        ratio_to_use=TOTAL_SIZE_THRESHOLD
    )

    if not fits_total:
        raise MemoryError(
            f"The estimated total output size (factored by x{output_scale_mult}) "
            f"exceeds the allowed system memory ({max_allowed_total_bytes / 1024 ** 2:.1f} MB, "
            f"using {TOTAL_SIZE_THRESHOLD * 100:.0f}% threshold). Reduce data size."
        )

    # Calculate item size and spatial plane size for memory checks
    item_size = max(
        (int(getattr(v.data, "dtype", np.dtype("float64")).itemsize) for v in
         ds.data_vars.values()),
        default=8,
    )
    bytes_spatial_plane_in = item_size * ds.sizes[y] * ds.sizes[x]
    bytes_plane_out = bytes_spatial_plane_in * max(1,
                                                   output_scale_mult)  # One full (y,x) plane of OUTPUT data

    # 2. CRITICAL SPATIAL CHUNKING CHECK (LAST RESORT)
    # Check if the peak memory requirement for the ABSOLUTE SMALLEST T/Z CHUNK
    # on the full spatial plane (lat=-1, lon=-1) exceeds the budget.

    # Smallest T/Z: T=1 (if exists), Z=min_required_z (if exists)
    min_t_size = 1 if "time" in ds.dims else 1
    min_z_size = min_required_z if vertical_dim in ds.dims else 1

    # Peak Working Set = (Min T*Z planes) * (Output Plane Size) * (Working Set Multiplier)
    estimated_peak_min_tz = min_t_size * min_z_size * bytes_plane_out * WORKING_SET_MULTIPLIER

    # This comparison determines if we are forced to spatially tile
    needs_spatial_chunking = estimated_peak_min_tz > max_memory_budget

    # 3. Determine Spatial Chunking Plan (Prioritizing performance)
    if rechunk_spatial or needs_spatial_chunking:
        # Fallback to safer, but slower, spatial tiling
        spatial_chunks = {y: "auto", x: "auto"}
        if verbose and needs_spatial_chunking:
            print(f"[chunking] WARNING: Absolute minimum T/Z working set "
                  f"({estimated_peak_min_tz / 1024 ** 2:.1f} MB) "
                  f"exceeds allowed compute budget ({max_memory_budget / 1024 ** 2:.1f} MB). "
                  f"Enabling spatial chunking as a last resort to prevent OOM errors.")
    else:
        # High performance plan: no spatial chunking
        spatial_chunks = {y: -1, x: -1}

    plan: Dict[str, Union[int, Tuple[int, ...]]] = spatial_chunks

    # 4. T and Z Chunking (Maximizing non-spatial chunks to fit the target_chunk_ratio budget)
    # Calculate the target size (bytes) for a single chunk (5% of total output size)
    dynamic_target_bytes = int(total_output_bytes * target_chunk_ratio)
    target_bytes = max(int(64 * 1024 ** 2), dynamic_target_bytes)
    target_chunk_mb = target_bytes / 1024 ** 2

    # budget_mult is the max number of (time*z) planes we can fit into the target size
    budget_mult = max(1, target_bytes // max(1, bytes_plane_out))

    t_guess = ds.sizes.get("time", 1)
    z_guess = ds.sizes.get(vertical_dim, 1)

    needs_t = "time" in ds.dims
    needs_z = vertical_dim in ds.dims

    if needs_t and needs_z:
        # near-sqrt split within budget
        z_chunk_target = min(z_guess, max(1, int(math.sqrt(budget_mult))))
        t_chunk_target = max(1, budget_mult // max(1, z_chunk_target))
        t_chunk = min(t_guess, max(1, t_chunk_target))
    elif needs_t:
        t_chunk = min(t_guess, budget_mult)
        z_chunk_target = None
    elif needs_z:
        t_chunk = None
        z_chunk_target = min(z_guess, budget_mult)
    else:
        t_chunk = None
        z_chunk_target = None

    if needs_t:
        plan["time"] = max(1, int(t_chunk))

    # Build explicit vertical chunks so every chunk >= min_required_z
    if needs_z:
        if z_chunk_target is None:
            if needs_t and "time" in plan:
                z_chunk_target = max(1, budget_mult // max(1, plan["time"]))
            else:
                z_chunk_target = min(z_guess, budget_mult)

        z_chunk_target = max(min_required_z, int(z_chunk_target))

        z_chunks = _balanced_chunks(z_guess, z_chunk_target, min_required_z)
        plan[vertical_dim] = z_chunks  # explicit tuple of sizes

    # any extra dims from preferred (don’t override spatial/time/z decisions)
    for d, c in preferred.items():
        if d not in plan and d in ds.dims:
            plan[d] = max(1, min(int(c), ds.sizes[d]))

    # Add the new scale dimension to the plan, if it's the dimension we need to iterate over
    if scale_dim is not None and scale_dim not in plan:
        plan[scale_dim] = 1  # Assuming we want to iterate over the new dimension

    out = ds.unify_chunks().chunk(plan)

    if verbose:
        # --- Print estimation for debugging ---
        # Calculate effective chunk sizes for estimation
        z_eff = 1
        if needs_z:
            zv = plan[vertical_dim]
            z_eff = (sum(zv) / len(zv)) if isinstance(zv, (tuple, list)) else int(zv)
        t_eff = int(plan.get("time", 1))

        # Check if spatial dimensions were chunked (if so, estimate average chunk size)
        if isinstance(plan.get(y), str) and plan[y] == "auto":
            spatial_msg = "(auto-tiled)"
            est_out = target_bytes
        else:
            est_out = bytes_plane_out * max(1, t_eff) * max(1, z_eff)
            spatial_msg = "(full plane)"

        msg_parts: List[str] = []
        for d, c in plan.items():
            if isinstance(c, (tuple, list)):
                msg_parts.append(f"{d}={list(c)}")
            elif c == -1:
                msg_parts.append(f"{d}=all")
            else:
                msg_parts.append(f"{d}={c}")

        # Add the scale dimension to the message for context
        if output_scale_mult > 1:
            scale_msg = f"{scale_dim or 'scale'}=x{output_scale_mult} "
            msg_parts.append(scale_msg)

        print(
            f"[chunking] Dynamic Target ({target_chunk_ratio:.1%} of total) = {target_chunk_mb:.1f} MB. "
            f"Spatial policy: {spatial_msg}. Plan: {', '.join(msg_parts)} | "
            f"Output Chunk Est: ~{est_out / 1024 ** 2:.1f} MB")

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
        print("Calculating transfers on selected levels: ", ds.z.values)

    # Apply consistent rechunking:
    rechunk_spatial = getattr(cfg.compute, "rechunk_spatial", False)

    ds = ensure_optimal_chunking(ds, spatial_dims=(y_name, x_name), vertical_dim="z",
                                 rechunk_spatial=rechunk_spatial)

    return ds


def write_dataset(ds: xr.Dataset, cfg) -> None:
    out = Path(cfg.output.path)
    if out.exists() and not cfg.output.overwrite:
        raise FileExistsError(f"{out} exists; set output.overwrite: true to replace")

    # Add global attributes to output file
    ds.attrs.update(_global_attrs)

    if cfg.output.store == "zarr":
        if out.exists():
            shutil.rmtree(out)
        ds.to_zarr(out, mode="w")
    elif cfg.output.store == "netcdf":
        engine = getattr(cfg.input, "engine", None)
        ds.to_netcdf(out, engine=engine)
    else:
        raise ValueError("output.store must be 'zarr' or 'netcdf'")
