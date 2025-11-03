import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Union, Iterable

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
TOTAL_SIZE_THRESHOLD = 0.8  # Fixed threshold (80%) for the total size check


def estimate_dataset_bytes(ds: xr.Dataset,
                           exclude_dims: Union[str, Iterable[str], None] = None) -> int:
    """
    Estimate working-set size (bytes) for a dataset chunk.

    This function calculates the maximum memory footprint for a single Dask chunk
    or the full size for an eager NumPy array. Dimensions listed in `exclude_dims`
    are assumed to be iterated over, reducing their contribution to the chunk size
    to a factor of 1.

    - For Dask-backed vars: uses the largest chunk along each included dimension.
    - For NumPy-backed vars: uses the full array size (as it's loaded eagerly).
    """
    total = 0

    if isinstance(exclude_dims, str):
        exclude_dims = [exclude_dims]
    elif exclude_dims is None:
        exclude_dims = []

    # Convert to set for fast lookup
    exclude_set = set(exclude_dims)

    for var in ds.data_vars.values():
        item_size = np.dtype(var.dtype).itemsize

        # Get Dask chunk sizes. Fallback to full dimension sizes for NumPy-backed arrays.
        chunks = getattr(getattr(var, "data", None), "chunks", [var.sizes[dim] for dim in var.dims])

        # Product of max chunk sizes per dim, respecting exclusion
        max_elems = 1

        for dim_name, dim_chunks in zip(var.dims, chunks):

            if dim_name in exclude_set:
                # Excluded dimension is assumed to be iterated over (chunk size = 1)
                max_chunk_size = 1
            else:
                # Included dimension: use the largest chunk size
                # np.atleast_1d handles cases where dim_chunks is just an integer (NumPy fallback)
                max_chunk_size = max(np.atleast_1d(dim_chunks))

            max_elems *= max_chunk_size

        total += max_elems * item_size

    return int(total)


def fits_in_memory(ds: xr.Dataset,
                   expansion_factor: int = 1,
                   ratio_to_use: float = TOTAL_SIZE_THRESHOLD,
                   exclude_dims: Union[Iterable[str], str, None] = None,
                   ) -> Tuple[bool, int, int]:
    """
    Check if the estimated dataset working set (based on chunk size and expansion)
    fits within the memory budget defined by `ratio_to_use` of available system memory.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to estimate memory for.
    expansion_factor : int
        Multiplier for the estimated size, accounting for temporary arrays during computation.
    ratio_to_use : float
        The fraction of available memory to use as the budget (e.g., 0.8).
    exclude_dims : Union[Iterable[str], str, None]
        Dimensions to treat as size 1 for the working set estimation (iterated over).

    Returns
    -------
    (bool, dataset_size_bytes, max_allowed_bytes)
        A tuple indicating: (if it fits, estimated working size, maximum allowed size).
    """
    ds = ds if isinstance(ds, xr.Dataset) else ds.to_dataset()

    expansion_factor = max(1, int(expansion_factor))

    # Estimated size of the working set (chunk size * temporary multiplier)
    dataset_size = estimate_dataset_bytes(ds, exclude_dims=exclude_dims) * expansion_factor

    # Get available system memory (eager operation)
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
    # Safety: ensure all >= min_size
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
        memory_threshold_ratio: float = TOTAL_SIZE_THRESHOLD,
        working_set_multiplier: int = 1,
        preferred: Optional[Dict[str, int]] = None,
        deriv_edge_order: int = 2,
        verbose: bool = True,
        rechunk_spatial: bool = False,  # Kept as a manual override
        output_scale_mult: int = 1,
        scale_dim: Optional[str] = None
) -> xr.Dataset:
    """
    Rechunk the dataset to ensure chunks are small enough to fit the memory budget,
    prioritizing full spatial chunks for performance.

    The function is executed eagerly to determine the chunk plan, but the resulting
    dataset remains Dask-backed (lazy).
    """
    preferred = dict(preferred or {})
    y, x = spatial_dims

    if y not in ds.dims or x not in ds.dims:
        raise ValueError(f"Spatial dims {spatial_dims} must exist in dataset dims {tuple(ds.dims)}")

    # 1. Budget Calculation & Spatial Decision

    # Dimensions to exclude from memory estimation (T/Z/Other)
    exclude_dims = [str(d) for d in ds.dims if d not in spatial_dims]

    # Check if a single full spatial plane (multiplied by working set factor) fits
    spatial_fits, plane_output_bytes, max_memory_budget = fits_in_memory(
        ds, exclude_dims=exclude_dims,
        expansion_factor=output_scale_mult * working_set_multiplier,
        ratio_to_use=memory_threshold_ratio
    )

    # Determine Spatial Chunking Plan (Prioritizing high performance)
    plan: Dict[str, Union[str, int, Tuple[int, ...]]] = {}

    if not rechunk_spatial and spatial_fits:
        # High performance plan: no spatial chunking (-1 means one chunk along that dim)
        plan.update({y: -1, x: -1})
    else:
        # Fallback to safer, slower spatial tiling
        plan.update({y: "auto", x: "auto"})
        if verbose and not spatial_fits:
            print(f"[chunking] WARNING: Estimated working set for full spatial plane "
                  f"exceeds allowed compute budget ({max_memory_budget / 1024 ** 2:.1f} MB). "
                  f"Enabling spatial chunking ('auto') as a last resort.")

    # 2. T and Z Chunking (Balancing non-spatial chunks within budget)

    needs_t = "time" in ds.dims
    needs_z = vertical_dim in ds.dims

    t_guess = ds.sizes.get("time", 1)
    z_guess = ds.sizes.get(vertical_dim, 1)

    # Calculate the max number of T*Z planes (units) we can fit into the budget
    if plane_output_bytes > 0:
        budget_mult = max(1, max_memory_budget // plane_output_bytes)
    else:
        # Fallback for very small arrays or safety
        budget_mult = max(t_guess * z_guess, 1)

    t_chunk_final: Optional[int] = None
    z_chunk_final: Optional[int] = None

    # Balance T and Z chunks (Prioritize balance using near-sqrt split)
    if needs_t and needs_z:
        # Split budget by sqrt to keep aspect ratio near 1
        z_chunk_target = min(z_guess, max(1, int(math.sqrt(budget_mult))))
        t_chunk_target = max(1, budget_mult // max(1, z_chunk_target))
        t_chunk_final = min(t_guess, max(1, t_chunk_target))
        z_chunk_final = z_chunk_target
    elif needs_t:
        t_chunk_final = min(t_guess, budget_mult)
    elif needs_z:
        z_chunk_final = min(z_guess, budget_mult)

    # Apply 'time' plan
    if needs_t and "time" not in preferred:
        plan["time"] = max(1, int(t_chunk_final))  # type: ignore [Possibly Unbound]

    # 3. Build explicit vertical chunks (incorporating min_required_z)
    if needs_z and vertical_dim not in preferred:
        min_required_z = deriv_edge_order + 1

        z_chunk_target: int

        # Adjust z target if time was chunked (and not preferred)
        if needs_t and "time" not in preferred and t_chunk_final is not None and t_chunk_final < t_guess:
            z_budget = budget_mult // max(1, t_chunk_final)
            z_chunk_target = min(z_guess, max(1, z_budget))
        elif z_chunk_final is not None:
            z_chunk_target = z_chunk_final
        else:
            z_chunk_target = min(z_guess, budget_mult)

        z_chunk_target = max(min_required_z, int(z_chunk_target))

        z_chunks = _balanced_chunks(z_guess, z_chunk_target, min_required_z)
        plan[vertical_dim] = z_chunks  # explicit tuple of sizes

    # 4. Handle remaining dimensions (preferred/scale)

    # Add any extra dims from preferred, overriding T/Z/spatial if specified
    for d, c in preferred.items():
        if d in ds.dims:
            plan[d] = max(1, min(int(c), ds.sizes[d]))

    # Add the new scale dimension (usually iterated over)
    if scale_dim is not None and scale_dim in ds.dims and scale_dim not in plan:
        plan[scale_dim] = 1

    # 5. Execute chunking (LAZY OPERATION)
    out = ds.unify_chunks().chunk(plan)

    if verbose:
        # --- Print estimation for debugging ---
        # Note: These values rely on the calculated 'plan' dict
        z_eff = 1
        if needs_z and vertical_dim in plan:
            zv = plan[vertical_dim]
            z_eff = (sum(zv) / len(zv)) if isinstance(zv, (tuple, list)) else int(
                zv)  # type: ignore
        t_eff = int(plan.get("time", 1))

        if plan.get(y) == "auto":
            spatial_msg = "(auto-tiled)"
            # When spatial is 'auto', the output chunk size is limited by the budget
            est_out = max_memory_budget / working_set_multiplier
        else:
            # Estimate output based on full plane size * max non-spatial chunks
            # Note: plane_output_bytes already includes the output_scale_mult and working_set_multiplier
            est_out = plane_output_bytes * max(1, t_eff) * max(1, z_eff)
            # Revert est_out back to output size (not working set size) for logging clarity
            est_out /= working_set_multiplier
            spatial_msg = "(full plane)"

        msg_parts: List[str] = []
        for d, c in plan.items():
            if isinstance(c, (tuple, list)):
                msg_parts.append(f"{d}={list(c)}")
            elif c == -1:
                msg_parts.append(f"{d}=all")
            else:
                msg_parts.append(f"{d}={c}")

        if output_scale_mult > 1:
            scale_msg = f"Scale=x{output_scale_mult}"
            msg_parts.append(scale_msg)

        print(
            f"[chunking] Target Working Set ({memory_threshold_ratio:.1%} of total)"
            f" = {max_memory_budget / 1024 ** 2:.1f} MB. "
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
