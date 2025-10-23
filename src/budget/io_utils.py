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
        spatial_dims: Tuple[str, str] = ("lat", "lon"),  # e.g. ("y","x") or ("lat","lon")
        vertical_dim: str = "z",
        target_chunk_mb: int = 64,
        preferred: Optional[Dict[str, int]] = None,  # any extra dims you want to chunk
        deriv_edge_order: int = 2,  # minimum required per-chunk = edge_order + 1
        quiet: bool = False,
) -> xr.Dataset:
    """
    Rechunk for fast 2-D FFTs, and ensure vertical chunks all satisfy
    `chunk >= deriv_edge_order + 1` (for finite-difference edge_order constraints).

    - Single chunk along `spatial_dims`.
    - Tiles over 'time' and `vertical_dim` to target ~`target_chunk_mb`.
    - Builds explicit vertical chunk sizes so the *last* chunk is never too small.
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

    plan: Dict = {y: -1, x: -1}  # -1 → single chunk along FFT axes

    # guesses
    t_guess = int(ds.sizes.get("time", 1))
    z_guess = int(ds.sizes.get(vertical_dim, 1))

    needs_t = "time" in ds.dims
    needs_z = vertical_dim in ds.dims
    min_required = deriv_edge_order + 1

    # Distribute budget between time and z
    if needs_t and needs_z:
        # near-sqrt split within budget
        z_chunk_target = min(z_guess, max(1, int(math.sqrt(budget_mult))))
        t_chunk_target = max(1, budget_mult // max(1, z_chunk_target))
        t_chunk = min(t_guess, max(1, t_chunk_target))
        # We'll compute z chunks explicitly below; keep target for guidance
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

    # Build explicit vertical chunks so every chunk >= min_required
    if needs_z:
        # If no target proposed, aim to use budget as much as possible
        if z_chunk_target is None:
            # If time is chunked, try to keep product near budget
            if needs_t and "time" in plan:
                z_chunk_target = max(1, budget_mult // max(1, plan["time"]))
            else:
                z_chunk_target = min(z_guess, budget_mult)

        # Ensure the target itself respects the minimum
        z_chunk_target = max(min_required, int(z_chunk_target))

        z_chunks = _balanced_chunks(z_guess, z_chunk_target, min_required)
        plan[vertical_dim] = z_chunks  # explicit tuple of sizes

    # any extra dims from preferred (don’t override spatial/time/z decisions)
    for d, c in preferred.items():
        if d not in plan and d in ds.dims:
            plan[d] = max(1, min(int(c), ds.sizes[d]))

    out = ds.unify_chunks().chunk(plan)

    if not quiet:
        # Estimate size from avg z chunk (works even if plan[vertical_dim] is a tuple)
        z_eff = 1
        if needs_z:
            zv = plan[vertical_dim]
            z_eff = (sum(zv) / len(zv)) if isinstance(zv, (tuple, list)) else int(zv)
        t_eff = int(plan.get("time", 1))
        est = bytes_plane * max(1, t_eff) * max(1, z_eff)
        msg_parts = []
        for d, c in plan.items():
            if isinstance(c, (tuple, list)):
                msg_parts.append(f"{d}={list(c)}")
            else:
                msg_parts.append(f"{d}={'all' if c == -1 else c}")
        print(f"[chunking] ({y},{x}) single-chunk; plan: {', '.join(msg_parts)} | "
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

    # select specified vertical levels
    levels = getattr(cfg.compute, "levels", None)
    mode = str(cfg.compute.mode).strip()

    # select specified vertical levels
    if levels is not None and mode == "scale_transfer":
        ds = ds.sel(z=levels, method='nearest')
        print("Calculating transfers on selected levels: ", ds.z.values)

    # Interpolate to consistent vertical coordinates
    ds = ensure_vertical_consistent(ds)

    # Apply consistent rechunking:
    ds = ensure_optimal_chunking(ds, spatial_dims=(y_name, x_name), target_chunk_mb=128)

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
        engine = getattr(cfg.input, "engine", None)
        ds.to_netcdf(out, engine=engine)
    else:
        raise ValueError("output.store must be 'zarr' or 'netcdf'")
