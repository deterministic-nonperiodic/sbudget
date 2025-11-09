"""
chunking_utils.py
----------------
System- and memory-aware utilities for xarray/Dask computations.

This module centralizes:
    - Temporary on-disk caching of intermediate variables (Zarr-backed)
    - Estimation of dataset working-set sizes (per chunk or total)
    - Adaptive memory and chunking heuristics for Dask/xarray
    - Safe cleanup of temporary files and grouped stores
    - Recommendations for persistence strategy (memory vs. disk)
"""
import atexit
import hashlib
import math
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from threading import Lock
from typing import *

import numcodecs
import numpy as np
import psutil
import xarray as xr
from dask.distributed import get_client, get_worker
from filelock import FileLock

_ZARR_GROUP_LOCK = Lock()

import os
import json

# ---------------------------------------------------------------
# Global configuration constants
# ---------------------------------------------------------------
_MEMORY_RESERVE_RATIO = 0.85
_SMALL_DATA_THRESHOLD_MB = 128.0  # skip chunking for small datasets

_GLOBAL_ZARR_TMPDIR = None

# Global lock registry to prevent concurrent writes to same group
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_LOCK = threading.Lock()
_CACHE_INDEX_NAME = "cache_index.json"

# ---------------------------------------------------------------------
# Global cleanup registry (ensures all temp dirs removed on exit)
# ---------------------------------------------------------------------
_GLOBAL_CACHE_DIRS: set[str] = set()


def _get_worker_cache() -> dict:
    """Return a persistent cache dictionary stored on the current Dask worker."""
    try:
        worker = get_worker()
    except ValueError:
        # Running without Dask support(e.g., local debug)
        return {}
    if not hasattr(worker, "_shift_cache"):
        worker._shift_cache = {}
    return worker._shift_cache


class CacheManager:
    """Production-safe cache manager for temporary on-disk Zarr stores."""

    _GLOBAL_SESSION = None  # Singleton reuse
    _CACHE_INDEX_NAME = "cache_index.json"

    def __init__(self, base_dir=None, max_total_gb=20, verbose=False, auto_cleanup=True):
        """
        Parameters
        ----------
        base_dir : str or Path, optional
            Base cache directory (default ~/.cache/sbudget).
        max_total_gb : float
            Maximum allowed cache quota across sessions.
        verbose : bool
            Print diagnostics.
        auto_cleanup : bool
            If True, automatically remove cache dirs at process exit.
            If False, you must call cleanup_session() manually.
        """
        self.verbose = verbose
        self.auto_cleanup = auto_cleanup
        self._tracked_dirs: set[str] = set()  # ✅ initialize here

        # --- 1. Explicit session override ---
        session_override = os.getenv("SBUDGET_CACHE_SESSION")
        if session_override and os.path.isdir(session_override):
            self.session_dir = Path(session_override).resolve()
            self.base_dir = self.session_dir.parent
            if self.verbose:
                print(f"[cache] Using existing session cache: {self.session_dir}")
            return

        # --- 2. Base cache directory ---
        base_dir = (
                base_dir
                or os.getenv("SBUDGET_CACHE_DIR")
                or os.getenv("TMPDIR")
                or Path.home() / ".cache" / "sbudget"
        )
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # --- 3. Create per-session directory ---
        self.session_dir = self.base_dir / f"session_{os.getpid()}_{uuid.uuid4().hex[:6]}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # --- 4. Disk quota enforcement ---
        self.max_total_bytes = max_total_gb * 1024 ** 3

        # --- 5. Register cleanup ---
        if self.auto_cleanup:
            self._enforce_quota()
            atexit.register(self.cleanup_session)

        if self.verbose:
            print(f"[cache] Created session cache: {self.session_dir}")

    # ------------------------------------------------------------------
    # ✅ INDEX MANAGEMENT HELPERS
    # ------------------------------------------------------------------
    def _index_path(self, tmpdir: Path | None = None) -> Path:
        return Path(tmpdir or self.session_dir) / self._CACHE_INDEX_NAME

    def load_index(self, tmpdir: str | Path | None = None) -> dict:
        """Load or initialize a session cache index from tmpdir."""
        path = self._index_path(tmpdir)
        if not path.exists():
            return {}
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def save_index(self, index: dict, tmpdir: str | Path | None = None) -> None:
        """Write updated cache index safely."""
        path = self._index_path(tmpdir)
        try:
            with open(path, "w") as f:
                json.dump(index, f, indent=2)
        except Exception as e:
            print(f"[cache] WARNING: Failed to write cache index: {e}")

    def cleanup_stale_entries(self, tmpdir: str | Path | None = None) -> dict:
        """Remove orphaned files from a previous run."""
        tmpdir = Path(tmpdir or self.session_dir)
        index = self.load_index(tmpdir)
        changed = False
        for key, path in list(index.items()):
            if not os.path.exists(path) or not Path(path).is_dir():
                del index[key]
                changed = True
        if changed:
            self.save_index(index, tmpdir)
        return index

    # ------------------------------------------------------------------
    def new_var_cache(self, prefix="var_cache") -> Path:
        """Create a variable-specific cache directory and track it."""
        if CacheManager._GLOBAL_SESSION is None:
            CacheManager._GLOBAL_SESSION = self.session_dir

        tmp = tempfile.mkdtemp(prefix=f"{prefix}_", dir=CacheManager._GLOBAL_SESSION)
        path = Path(tmp)
        self._register_temp_dir(path)
        return path

    # ------------------------------------------------------------------
    def _register_temp_dir(self, path: str | Path):
        """Track and optionally schedule cleanup for a temporary directory."""
        path = str(path)
        self._tracked_dirs.add(path)

        # Do not register for cleanup unless requested
        if self.auto_cleanup:
            atexit.register(lambda p=path: shutil.rmtree(p, ignore_errors=True))

    # ------------------------------------------------------------------
    def cleanup_session(self):
        """Delete all tracked temp dirs and this session’s cache directory."""
        for p in list(self._tracked_dirs):
            shutil.rmtree(p, ignore_errors=True)
        self._tracked_dirs.clear()

        if self.session_dir.exists():
            shutil.rmtree(self.session_dir, ignore_errors=True)
            if self.verbose:
                print(f"[cache] Cleaned session cache: {self.session_dir}")

    def cleanup_all(self):
        """Delete all session cache directories in the base directory."""
        if not self.base_dir.exists():
            return
        for s in self.base_dir.glob("session_*"):
            if s.is_dir():
                shutil.rmtree(s, ignore_errors=True)
        if self.verbose:
            print(f"[cache] Cleaned all sessions under {self.base_dir}")

    def _enforce_quota(self):
        """Remove the oldest sessions if total size exceeds the quota."""
        if not self.base_dir.exists():
            return
        sessions = [p for p in self.base_dir.glob("session_*") if p.is_dir()]
        total_bytes = sum(
            sum(f.stat().st_size for f in s.rglob("*") if f.is_file())
            for s in sessions
        )
        if total_bytes <= self.max_total_bytes:
            return

        sessions.sort(key=lambda p: p.stat().st_mtime)
        if self.verbose:
            print(
                f"[cache] Quota exceeded ({total_bytes / 1e9:.2f} GB). Cleaning oldest sessions...")

        while sessions and total_bytes > self.max_total_bytes:
            victim = sessions.pop(0)
            size = sum(f.stat().st_size for f in victim.rglob("*") if f.is_file())
            shutil.rmtree(victim, ignore_errors=True)
            total_bytes -= size


def _persist_grid_shifts_disk(
        ds: xr.Dataset,
        cache: CacheManager,
        key: tuple[int, int],
        shift_cache: dict,
        compressor_type: str = "lz4"
) -> xr.Dataset:
    """Persist a rolled dataset to its own Zarr file."""
    tmpdir = cache.new_var_cache()

    # Reuse existing
    if key in shift_cache:
        z_path = shift_cache[key]
        if os.path.exists(z_path):
            try:
                reopened = xr.open_zarr(z_path, consolidated=False)
                if set(ds.data_vars) == set(reopened.data_vars):
                    return reopened
            except Exception:
                shutil.rmtree(z_path, ignore_errors=True)

    z_path = os.path.join(tmpdir, f"var_{key[0]}_{key[1]}_{uuid.uuid4().hex}.zarr")

    if compressor_type is not None:
        compressor = numcodecs.Blosc(cname=compressor_type, clevel=1, shuffle=2)
    else:
        compressor = None

    encoding = {v: {"dtype": str(ds[v].dtype), "compressor": compressor} for v in ds.data_vars}

    try:
        ds.to_zarr(
            z_path,
            mode="w",
            compute=True,
            consolidated=False,
            zarr_format=2,
            encoding=encoding,
        )

        reopened = xr.open_zarr(z_path, chunks=dict(ds.chunks), consolidated=False)
        reopened = reopened.transpose(*ds.dims, missing_dims="ignore")
        reopened = reopened.assign_coords(
            {dim: ds[dim] for dim in ds.dims if dim in reopened.dims}
        )

        shift_cache[key] = z_path
        index = cache.cleanup_stale_entries(tmpdir) if os.path.exists(tmpdir) else {}

        index[str(key)] = z_path
        cache.save_index(index, tmpdir)

        return reopened
    except Exception as e:
        print(f"[cache] WARNING: disk cache failed ({e}); falling back to in-memory persist.")
        return ds.persist()


def _persist_grid_shifts_grouped_rotating(
        ds: xr.Dataset,
        cache: CacheManager,
        key: tuple[int, int],
        shift_cache: dict,
        max_group_gb: float = 25.0,  # rotate if current zarr dir > 50 GB
        compressor_type: str = "lz4"
) -> xr.Dataset:
    """
    Persist rolled datasets into rotating grouped Zarr stores with LZ4 compression.

    Each (nx, ny) combination shares a Zarr store until it exceeds `max_group_gb`.
    Then a new store is automatically started (e.g., var_cache_grouped_001.zarr).

    Features:
      • Fast writes with LZ4 (Blosc)
      • Thread/process safe with FileLock
      • Auto-rotation to prevent unbounded disk growth
      • Auto-cleanup via CacheManager session
    """
    base_group = f"nx{key[0]}_ny{key[1]}"

    # --- Get or create state for this base group ---
    if base_group not in shift_cache:
        tmpdir = cache.new_var_cache(prefix="var_cache_grouped")
        shift_cache[base_group] = {
            "dir": tmpdir,
            "index": 0,
        }

    meta = shift_cache[base_group]
    tmpdir = meta["dir"]
    z_path = os.path.join(tmpdir, f"var_cache_grouped_{meta['index']:03d}.zarr")

    # --- Rotate if existing group exceeds size limit ---
    def _get_dir_size(path: str) -> float:
        return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / (1024 ** 3)

    if os.path.exists(z_path) and _get_dir_size(z_path) > max_group_gb:
        meta["index"] += 1
        tmpdir = cache.new_var_cache(prefix=f"var_cache_grouped_{meta['index']:03d}")
        z_path = os.path.join(tmpdir, f"var_cache_grouped_{meta['index']:03d}.zarr")

    subgroup = f"shift_{uuid.uuid4().hex[:8]}"
    full_group = f"{base_group}_g{meta['index']:03d}/{subgroup}"

    # --- Thread + process locking ---
    lock_id = f"{z_path}:{base_group}_g{meta['index']:03d}"
    lock_hash = hashlib.md5(lock_id.encode()).hexdigest()
    lock_file = os.path.join(tmpdir, f".lock_{lock_hash}")
    file_lock = FileLock(lock_file)

    with _LOCKS_LOCK:
        if lock_id not in _LOCKS:
            _LOCKS[lock_id] = threading.Lock()
        thread_lock = _LOCKS[lock_id]

    with thread_lock, file_lock:
        if compressor_type is not None:
            compressor = numcodecs.Blosc(cname=compressor_type, clevel=1, shuffle=2)
            encoding = {v: {"compressor": compressor} for v in ds.data_vars}
        else:
            encoding = None

        ds.to_zarr(
            z_path,
            group=full_group,
            mode="a",
            compute=True,
            consolidated=False,
            zarr_format=2,
            encoding=encoding,
        )

    # --- Reopen the subgroup we just wrote ---
    reopened = xr.open_zarr(z_path, group=full_group, consolidated=False)
    reopened = reopened.transpose(*ds.dims, missing_dims="ignore")
    reopened = reopened.assign_coords({dim: ds[dim] for dim in ds.dims if dim in reopened.dims})

    return reopened


# ---------------------------------------------------------------------
def _persist_grid_shifts(
        ds: xr.Dataset,
        key: tuple[int, int],
        persist_shifts: bool | str,
        shift_cache: dict | None = None,
        mem_limit: int = 25,
        mem_threshold: float = 0.3,
        auto_cleanup: bool = True,
        compressor: str | None = "lz4"
) -> xr.Dataset:
    """
    Persist or cache rolled datasets according to `persist_shifts` mode.

    persist_shifts:
        - False           → return as-is
        - True            → persist all in-memory
        - "smart"         → persist up to N cached shifts (heuristic)
        - "disk"          → store each shift in its own Zarr file
        - "hybrid"        → keep some in memory, spill to disk when memory is low
        - "disk_grouped"  → same as "disk" but shared .zarr per (nx, ny)

    Notes
    -----
    A safety cleanup hook is triggered after completion for disk-based modes
    ("disk", "fast"), unless auto_cleanup=False, to ensure residual cache directories are removed.
    """
    if shift_cache is None:
        shift_cache = {}

    # -------------------------------------------------------------
    # In-memory modes
    # -------------------------------------------------------------
    if persist_shifts is True:
        return ds.persist()

    if persist_shifts == "smart":
        if len(shift_cache) < mem_limit:
            cached = ds.persist()
            shift_cache[key] = cached
            return cached
        return ds

    # -------------------------------------------------------------
    # Disk-based modes
    # -------------------------------------------------------------
    cache = CacheManager(verbose=False, auto_cleanup=auto_cleanup)

    if persist_shifts == "disk":
        result = _persist_grid_shifts_disk(ds, cache, key, shift_cache, compressor_type=compressor)

    elif persist_shifts == "disk_grouped":
        result = _persist_grid_shifts_grouped_rotating(ds, cache, key, shift_cache,
                                                       compressor_type=compressor)

    # -------------------------------------------------------------
    # Hybrid (adaptive)
    # -------------------------------------------------------------
    elif persist_shifts == "hybrid":
        mem = psutil.virtual_memory()
        available_ratio = mem.available / mem.total

        if available_ratio < mem_threshold:
            result = _persist_grid_shifts_grouped_rotating(ds, cache, key, shift_cache,
                                                           compressor_type=compressor)
        else:
            cached = ds.persist()
            shift_cache[key] = cached
            result = cached

    # -------------------------------------------------------------
    # Default fallback
    # -------------------------------------------------------------
    else:
        result = ds

    # -------------------------------------------------------------
    # Safety cleanup for disk modes
    # -------------------------------------------------------------
    # Trigger cleanup when:
    # - using disk-based mode ("disk", "fast")
    # - no cached shifts remain (e.g., end of batch)
    # - CacheManager uses manual cleanup (auto_cleanup=False)
    # This ensures stale var_cache_* directories are removed.
    # -------------------------------------------------------------
    if persist_shifts in {"disk", "fast"} and not shift_cache:
        if auto_cleanup:
            try:
                cache.cleanup_session()
                if cache.verbose:
                    print(f"[cache] Safety cleanup: removed session cache at {cache.session_dir}")
            except Exception as e:
                print(f"[cache] WARNING: safety cleanup failed ({e})")

    return result


def auto_select_cache_mode(
        ds: xr.Dataset,
        memory_threshold_ratio: float = _MEMORY_RESERVE_RATIO,
        verbose: bool = True,
        working_set_multiplier: int = 1,
) -> str:
    """
    Automatically select the initial cache mode ("smart" or "hybrid")
    based on dataset size relative to available worker memory.
    The 'hybrid' mode will automatically spill to disk when memory
    drops below its runtime threshold.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to evaluate.
    spatial_dims : tuple[str, str], default=("lat", "lon")
        Spatial dimensions to exclude from memory heuristics.
    memory_threshold_ratio : float, default=0.9
        Fraction of total available memory usable before switching
        to disk-based caching.
    verbose : bool, default=True
        Print diagnostic info.
    working_set_multiplier : int, default=1
        Expansion factor for temporary working memory.

    Returns
    -------
    str
        Either "smart" or "hybrid".
    """
    # --- 1. Estimate dataset size (excluding spatial dims) ---
    ds_bytes = estimate_dataset_bytes(ds) * working_set_multiplier

    # --- 2. Get per-worker memory budget ---
    per_worker_budget, n_workers = get_worker_memory_budget(memory_threshold_ratio)
    available_bytes = per_worker_budget * memory_threshold_ratio

    usage_ratio = ds_bytes / max(1, int(available_bytes))

    # --- 3. Select mode ---
    # Below 50% of memory → fully in-memory
    # Above 50% → start hybrid to be safe
    mode = "smart" if usage_ratio < 0.5 else "hybrid"

    # --- 4. Diagnostics ---
    if verbose:
        print(
            f"[budget] Auto-selected cache mode: {mode} "
            f"(dataset={_fmt_bytes(ds_bytes)}, "
            f"budget/worker={_fmt_bytes(available_bytes)}, "
            f"usage≈{usage_ratio:.2f}×, workers={n_workers})"
        )

    return mode


# ---------------------------------------------------------------
# 2. Memory estimation utilities
# ---------------------------------------------------------------
def _fmt_bytes(nbytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while nbytes >= 1024 and i < len(units) - 1:
        nbytes /= 1024
        i += 1
    return f"{nbytes:.2f} {units[i]}"


def estimate_dataset_bytes(
        ds: xr.Dataset,
        exclude_dims: Iterable[str] | str | None = None,
        mode: str = "largest_chunk",
) -> int:
    """
    Estimate memory footprint (bytes) for a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to inspect.
    exclude_dims : iterable of str or str, optional
        Dimensions to exclude when computing chunk sizes (e.g., 'time').
    mode : {'largest_chunk', 'total'}
        - 'largest_chunk': estimate memory for the largest single chunk
          across variables (default, for working-set estimation).
        - 'total': sum of all variable sizes across all chunks.

    Returns
    -------
    int
        Estimated memory footprint in bytes (metadata-only, no computation).
    """
    if isinstance(exclude_dims, str):
        exclude_dims = [exclude_dims]
    exclude_dims = set(exclude_dims or [])

    total = 0

    for v in ds.data_vars.values():
        item_size = np.dtype(v.dtype).itemsize
        chunks = getattr(v, "chunksizes", None)

        if chunks:
            # List of chunk sizes per dimension
            dim_chunks = [chunks.get(d, (v.sizes[d],)) for d in v.dims]
        else:
            # Non-dask (fully in-memory) arrays
            dim_chunks = [(v.sizes[d],) for d in v.dims]

        if mode == "largest_chunk":
            # Estimate memory of largest chunk per variable
            elems = 1
            for d, ch in zip(v.dims, dim_chunks):
                elems *= 1 if d in exclude_dims else max(ch)
            total += elems * item_size

        elif mode == "total":
            # Estimate total dataset memory (sum of all chunks)
            elems = 1
            for d, ch in zip(v.dims, dim_chunks):
                if d in exclude_dims:
                    continue
                elems *= sum(ch)
            total += elems * item_size

        else:
            raise ValueError("mode must be either 'largest_chunk' or 'total'")

    return int(total)


def fits_in_memory(
        ds: xr.Dataset,
        expansion_factor: int = 1,
        ratio_to_use: float = _MEMORY_RESERVE_RATIO,
        exclude_dims: Iterable[str] | str | None = None,
) -> tuple[bool, int, int]:
    """Check if dataset fits into memory budget (returns fits, size, limit)."""
    dataset_size = estimate_dataset_bytes(ds, exclude_dims) * max(1, expansion_factor)
    try:
        client = get_client()
        limits = client.scheduler_info()["workers"]
        avail = min(w["memory_limit"] for w in limits.values()) if limits else None
    except Exception:
        avail = None

    available = avail or psutil.virtual_memory().available
    max_mem = int(ratio_to_use * available)
    return dataset_size < max_mem, dataset_size, max_mem


# ---------------------------------------------------------------
# 3. System resource helpers
# ---------------------------------------------------------------
def get_worker_memory_budget(reserve_ratio: float = _MEMORY_RESERVE_RATIO) -> tuple[int, int]:
    """Estimate memory budget (bytes) per worker/task."""
    total = psutil.virtual_memory().total
    env_keys = [
        "SLURM_NTASKS", "SLURM_NTASKS_PER_NODE",
        "DASK_WORKER_NPROCS", "DASK_WORKER_NTHREADS",
    ]
    for k in env_keys:
        if k in os.environ:
            try:
                n = int(os.environ[k])
                break
            except ValueError:
                continue
    else:
        n = psutil.cpu_count(logical=True) or 1

    per_worker = total * reserve_ratio / n
    return int(per_worker), int(n)


def optimal_batch_size(
        ds: xr.Dataset,
        items_total: int,
        exclude_dims: Iterable[str] | str | None = None,
        working_set_multiplier: int = 1,
        reserve_ratio: float = _MEMORY_RESERVE_RATIO,
        safety_factor: float = 0.85,
        verbose: bool = True,
) -> int:
    """
    Estimate a safe batch size for processing multiple scales or loop items
    based on available memory and dataset footprint.

    Automatically accounts for multi-worker (SLURM/Dask) environments.
    """
    # --- Estimate memory per dataset instance ---
    per_item_bytes = estimate_dataset_bytes(ds, exclude_dims=exclude_dims) * working_set_multiplier
    per_item_bytes = max(1, per_item_bytes)

    # --- Worker-aware memory budget ---
    per_worker_budget, n_workers = get_worker_memory_budget(reserve_ratio)
    usable_mem = per_worker_budget * safety_factor

    # --- Compute max safe items ---
    max_items_fit = max(1, int(usable_mem // per_item_bytes))
    batch_size = min(items_total, max_items_fit)

    # --- Verbose diagnostics ---
    if verbose:
        print(f"[batch] Estimating optimal batch size: "
              f"per-item ≈ {_fmt_bytes(per_item_bytes)} | usable/worker ≈ {_fmt_bytes(usable_mem)}"
              f" → batch_size = {batch_size} (out of {items_total}) | workers {n_workers}")

    return batch_size


# ---------------------------------------------------------------
# 4. Chunking helpers
# ---------------------------------------------------------------
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


def choose_z_chunk_size(n_z: int, deriv_edge_order: int, sys_mult: int | None = None) -> int:
    """
    Choose a balanced z-chunk size based on dataset size, derivative stencil, and system multiplier.
    Ensures we don't end up with trivially small (1-level) chunks unless absolutely necessary.
    """
    if n_z <= 4:
        return n_z  # tiny vertical domain, keep as-is

    # Estimate reasonable number of vertical chunks
    if sys_mult is None or sys_mult < 1:
        sys_mult = 1

    target_chunks = min(10, max(2, int(round(math.log2(sys_mult ** 0.25 * n_z / 8)))))

    base_chunk = max(1, n_z // target_chunks)

    # Align with derivative stencil width
    align = max(1, deriv_edge_order)
    if base_chunk % align != 0:
        base_chunk -= base_chunk % align

    # Avoid extreme imbalance
    while base_chunk > 1 and n_z % base_chunk != 0 and (n_z // base_chunk) < 4:
        base_chunk -= 1

    # Ensure we don't produce single-level chunks if unnecessary
    if base_chunk < n_z // 8:
        base_chunk = max(2, n_z // 8)

    return max(2, min(base_chunk, n_z))


# ---------------------------------------------------------------
# 5. High-level adaptive chunking interface
# ---------------------------------------------------------------
def ensure_optimal_chunking(
        ds: xr.Dataset,
        spatial_dims: Tuple[str, str] = ("lat", "lon"),
        vertical_dim: str = "z",
        memory_threshold_ratio: float = _MEMORY_RESERVE_RATIO,
        working_set_multiplier: int = 1,
        preferred: Optional[Dict[str, int]] = None,
        deriv_edge_order: int = 2,
        verbose: bool = True,
        rechunk_spatial: bool = False,
        output_scale_mult: int = 1,
        scale_dim: Optional[str] = None,
        desired_chunk_size_mb: Optional[float] = None,
        min_auto_rechunk_mb: float = _SMALL_DATA_THRESHOLD_MB,
) -> Tuple[xr.Dataset, Dict[str, Any]]:
    """
    Rechunk dataset to balance performance and memory usage.
    Keeps full horizontal planes unless they exceed memory budget.
    """
    preferred = dict(preferred or {})
    y_dim, x_dim = spatial_dims

    # ---- Small dataset shortcut ----
    ds_total = estimate_dataset_bytes(ds, mode="total")

    if ds_total < min_auto_rechunk_mb * 1024 ** 2:
        if verbose:
            print(f"[chunking] Dataset is small ({_fmt_bytes(ds_total)}); skipping rechunk.")
        info = {"plan": {}, "skipped": True, "persist_shifts": "smart"}
        return ds, info

    exclude_dims = [str(d) for d in ds.dims if d not in spatial_dims]

    spatial_fits, plane_bytes, max_mem = fits_in_memory(
        ds, exclude_dims=exclude_dims,
        expansion_factor=output_scale_mult * working_set_multiplier,
        ratio_to_use=memory_threshold_ratio,
    )

    plan: Dict[str, Any] = {}

    # ---- Spatial chunking ----
    if not rechunk_spatial and spatial_fits:
        plan.update({y_dim: -1, x_dim: -1})
    else:
        reduction = max(1.0, plane_bytes / max(1, max_mem))
        n_tiles = max(1, math.ceil(math.sqrt(reduction)))
        cy, cx = math.ceil(ds.sizes[y_dim] / n_tiles), math.ceil(ds.sizes[x_dim] / n_tiles)
        plan.update({y_dim: cy, x_dim: cx})
        if verbose:
            print(f"[chunking] Applying spatial tiling: ({n_tiles}×{n_tiles}) → {cy}×{cx}")

    # ---- Time / Vertical chunk balancing ----
    needs_t = "time" in ds.dims
    needs_z = vertical_dim in ds.dims
    t_guess, z_guess = ds.sizes.get("time", 1), ds.sizes.get(vertical_dim, 1)

    # Estimate the size of one horizontal plane
    bytes_per_plane = plane_bytes / (output_scale_mult * working_set_multiplier)
    target_bytes = (desired_chunk_size_mb or 128.0) * 1024 ** 2

    # Compute how many planes fit into the target
    n_planes_per_chunk = max(1, int(target_bytes // max(1, int(bytes_per_plane))))

    if verbose:
        print(f"[chunking] Target chunk budget: {desired_chunk_size_mb or 128:.0f} MB "
              f"→ {n_planes_per_chunk} planes per chunk")

    # ---- Choose chunking strategy ----
    if needs_t and needs_z:
        # Split budget roughly between z and time
        z_chunk = choose_z_chunk_size(z_guess, deriv_edge_order + 1, sys_mult=n_planes_per_chunk)
        t_chunk = max(1, min(t_guess, n_planes_per_chunk // max(1, z_chunk)))
        plan.update({"time": t_chunk, vertical_dim: z_chunk})
    elif needs_z:
        z_chunk = choose_z_chunk_size(z_guess, deriv_edge_order + 1, sys_mult=n_planes_per_chunk)
        plan[vertical_dim] = z_chunk
    elif needs_t:
        plan["time"] = min(t_guess, n_planes_per_chunk)

    # ---- Apply overrides ----
    for d, c in preferred.items():
        if d in ds.dims:
            plan[d] = max(1, min(int(c), ds.sizes[d]))

    if scale_dim and scale_dim in ds.dims and scale_dim not in plan:
        plan[scale_dim] = 1

    # ---- Convert numeric chunk hints into balanced tuples ----
    for dim, chunks in plan.items():
        if dim not in ds.dims:
            continue
        dim_size = ds.sizes[dim]
        if isinstance(chunks, int):
            if dim not in spatial_dims:
                n_chunks = max(1, math.ceil(dim_size / chunks))
                plan[dim] = _balanced_chunks(dim_size, dim_size // n_chunks, min_size=1)
            else:
                plan[dim] = min(chunks, dim_size)

    # ---- Final rechunk ----
    out = ds.unify_chunks().chunk(plan)
    total_est = estimate_dataset_bytes(out, mode="largest_chunk") * output_scale_mult

    # ---- Summary ----
    msg_parts: List[str] = []
    for d, c in plan.items():
        if isinstance(c, (tuple, list)):
            c_min, c_max = min(c), max(c)
            msg_parts.append(f"{d}={c_min}" if c_min == c_max else f"{d}=({c_min}, {c_max})")
        elif c == -1:
            msg_parts.append(f"{d}={out.sizes[d]} (full)")
        else:
            msg_parts.append(f"{d}={c}")
    if output_scale_mult > 1:
        msg_parts.append(f"Scale=x{output_scale_mult}")

    if verbose:
        print(f"[chunking] Target: {_fmt_bytes(target_bytes)} | "
              f"Plan: {', '.join(msg_parts)} | "
              f"Est. working set: {_fmt_bytes(total_est)}")

    info = {"plan": plan, "max_memory": max_mem, "plane_bytes": plane_bytes}
    return out, info
