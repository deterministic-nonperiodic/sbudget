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
import gc
import hashlib
import json
import os
import resource
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from threading import Lock
from typing import *
from typing import Dict, Optional

import numpy as np
import psutil
import xarray as xr
from dask.distributed import get_client
from dask.distributed import get_worker
from filelock import FileLock, BaseFileLock
from numcodecs import Blosc

# --- Increase open file limit ---
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(65535, hard), hard))

# ---------------------------------------------------------------
# Global configuration constants
# ---------------------------------------------------------------
_MEMORY_RESERVE_RATIO = 0.85
_SMALL_DATA_THRESHOLD_MB = 512.0  # skip chunking for small datasets
DEFAULT_CHUNK_SIZE_MB = 2048  # MB

# ---------------------------------------------------------------------
# Global cleanup registry (ensures all temp dirs removed on exit)
# ---------------------------------------------------------------------
# --- Global Lock Management ---

_LOCKS_LOCK = threading.Lock()
_LOCKS: Dict[str, threading.Lock] = {}


def _get_lock(lock_id: str, tmpdir: Path) -> tuple[Lock, BaseFileLock]:
    """Returns a (threading.Lock, FileLock) pair for thread and process safety."""

    # 1. Thread Lock (Intra-process)
    with _LOCKS_LOCK:
        if lock_id not in _LOCKS:
            _LOCKS[lock_id] = threading.Lock()
        thread_lock = _LOCKS[lock_id]

    # 2. File Lock (Cross-process)
    lock_hash = hashlib.md5(lock_id.encode()).hexdigest()
    lock_file = tmpdir / f".lock_{lock_hash}.lock"
    file_lock = FileLock(lock_file)

    return thread_lock, file_lock


# --- CacheManager Class ---

class CacheManager:
    """Worker-aware hybrid cache manager for Zarr and Joblib persistence.

    Provides:
      • Automatic in-memory vs. disk persistence based on memory pressure.
      • Separate cache sessions per worker (safe for Dask/distributed runs).
      • Automatic cleanup and disk quota enforcement.
      • Reuse of existing cache files when possible.
    """

    _GLOBAL_SESSION = None
    _CACHE_INDEX_NAME = "cache_index.json"

    def __init__(
            self,
            base_dir: Optional[str] = None,
            max_total_gb: float = 50.0,
            mem_threshold: float = 0.2,
            verbose: bool = False,
            auto_cleanup: bool = True,
            compressor: str = "lz4",
    ):
        self.verbose = verbose
        self.auto_cleanup = auto_cleanup
        self.mem_threshold = mem_threshold
        self.max_total_bytes = max_total_gb * 1024 ** 3
        self.compressor = Blosc(cname=compressor, clevel=1, shuffle=Blosc.SHUFFLE)

        self._tracked_dirs: set[str] = set()

        # ----------------------------------------------------------
        # 1. Resolve base directory (worker-aware)
        # ----------------------------------------------------------
        self.base_dir = self._resolve_base_dir(base_dir)

        # ----------------------------------------------------------
        # 2. Create unique session per worker
        # ----------------------------------------------------------
        worker_id = self._get_worker_id()
        session_suffix = f"session_{os.getpid()}_{uuid.uuid4().hex[:6]}"
        if worker_id:
            session_suffix += f"_worker_{worker_id}"

        # cleanup stale sessions before creating a new one
        self._cleanup_stale_sessions()
        self.session_dir = self.base_dir / session_suffix
        self.session_dir.mkdir(parents=True, exist_ok=True)

        if CacheManager._GLOBAL_SESSION is None:
            CacheManager._GLOBAL_SESSION = self.session_dir

        # ----------------------------------------------------------
        # 4. Register automatic cleanup
        # ----------------------------------------------------------
        if self.auto_cleanup:
            self._enforce_quota()
            atexit.register(self.cleanup_session)

    @classmethod
    def _resolve_base_dir(cls, base_dir: str | None = None) -> Path:
        """
        Resolve the cache base directory using the same logic everywhere.
        Honors SBUDGET_CACHE_DIR, TMPDIR, or defaults to ~/.cache/sbudget.
        """
        base_dir = (
                base_dir
                or os.getenv("SBUDGET_CACHE_DIR")
                or os.getenv("TMPDIR")
                or Path.home() / ".cache" / "sbudget"
        )
        path = Path(base_dir).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _get_worker_id(self) -> str | None:
        """
        Calls dask.distributed.get_worker.
        Returns the worker ID if running on a worker, otherwise returns None.
        """
        try:
            worker = get_worker()
            return getattr(worker, "name", None) or getattr(worker, "id", None)
        except ValueError:
            # Running locally or outside a Dask worker context
            return None

    # ==========================================================
    # Index and cleanup utilities
    # ==========================================================
    def _index_path(self, tmpdir: Path | None = None) -> Path:
        return Path(tmpdir or self.session_dir) / self._CACHE_INDEX_NAME

    def load_index(self, tmpdir: str | Path | None = None) -> dict:
        """Load or initialize a cache index from a directory."""
        path = self._index_path(tmpdir)
        if not path.exists():
            return {}
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def save_index(self, index: dict, tmpdir: str | Path | None = None) -> None:
        """Write updated cache index to disk."""
        path = self._index_path(tmpdir)
        try:
            with open(path, "w") as f:
                json.dump(index, f, indent=2)
        except Exception as e:
            if self.verbose:
                print(f"[cache] WARNING: Failed to write cache index: {e}")

    def cleanup_stale_entries(self, tmpdir: str | Path | None = None) -> dict:
        """Remove orphaned entries from cache index."""
        tmpdir = Path(tmpdir or self.session_dir)
        # Lock required before accessing/modifying the index file
        thread_lock, file_lock = _get_lock(str(tmpdir / self._CACHE_INDEX_NAME), self.session_dir)

        with thread_lock, file_lock:
            index = self.load_index(tmpdir)
            changed = False
            for key, path in list(index.items()):
                # If path doesn't exist (e.g., deleted by another process's quota enforcement)
                if not os.path.exists(path):
                    del index[key]
                    changed = True
            if changed:
                self.save_index(index, tmpdir)
            return index

    def _cleanup_stale_sessions(self):
        """Remove stale cache sessions from dead processes."""
        for s in self.base_dir.glob("session_*"):
            if not s.is_dir():
                continue
            # Extract PID from directory name (format: session_<pid>_<uuid>)
            try:
                pid = int(s.name.split("_")[1])
            except (IndexError, ValueError):
                continue

            # If that PID no longer exists, remove the directory
            if not psutil.pid_exists(pid):
                if self.verbose:
                    print(f"[cache] Removing stale cache session {s}")
                shutil.rmtree(s, ignore_errors=True)

    def cleanup_session(self):
        """Delete tracked temporary directories and the session directory."""
        # Note: atexit already handles cleanup, this is mainly for manual calls.
        for p in list(self._tracked_dirs):
            shutil.rmtree(p, ignore_errors=True)
        self._tracked_dirs.clear()
        if self.session_dir.exists():
            shutil.rmtree(self.session_dir, ignore_errors=True)
            if self.verbose:
                print(f"[cache] Cleaned session cache: {self.session_dir}")

    def cleanup_all(self):
        """Delete all cache sessions under the base directory."""
        for s in self.base_dir.glob("session_*"):
            if s.is_dir():
                shutil.rmtree(s, ignore_errors=True)
        if self.verbose:
            print(f"[cache] Cleaned all sessions under {self.base_dir}")

    # ==========================================================
    # Directory management
    # ==========================================================
    def new_var_cache(self, prefix: str = "var_cache", subdir: str | None = None) -> Path:
        """
        Create a variable-specific cache directory under this session.
        Respects the user-specified base_dir.
        """
        # ensure a global session
        if CacheManager._GLOBAL_SESSION is None:
            CacheManager._GLOBAL_SESSION = self.session_dir

        # always use this instance's session, not the class global one
        base_dir = self.session_dir

        # optionally add subdir (e.g. r40500)
        if subdir is not None:
            base_dir = os.path.join(base_dir, str(subdir))
            os.makedirs(base_dir, exist_ok=True)

        tmp = tempfile.mkdtemp(prefix=f"{prefix}_", dir=base_dir)
        path = Path(tmp)
        self._register_temp_dir(path)

        if self.verbose:
            print(f"[cache] Created var cache: {path}")

        return path

    def _register_temp_dir(self, path: str | Path):
        """Track a temporary directory for later cleanup."""
        path = str(path)
        self._tracked_dirs.add(path)
        # Note: atexit cleanup is handled in __init__ for the main session,
        # but registering here handles subdirectories created post-init.
        if self.auto_cleanup:
            atexit.register(lambda p=path: shutil.rmtree(p, ignore_errors=True))

    def _get_active_cache_dir(self, subdir=None) -> Path:
        """Return a persistent per-session cache directory for shifts."""
        if not hasattr(self, "_active_cache_dir"):
            self._active_cache_dir = self.new_var_cache(subdir=subdir)
        return Path(self._active_cache_dir)

    # ==========================================================
    # Memory & quota management
    # ==========================================================
    def _available_memory_ratio(self) -> float:
        mem = psutil.virtual_memory()
        return mem.available / mem.total

    def _enforce_quota(self):
        """Safely delete old sessions to stay within disk quota."""
        sessions = [p for p in self.base_dir.glob("session_*") if p.is_dir()]
        if not sessions:
            return

        # Never delete the active session
        sessions = [s for s in sessions if s != self.session_dir]

        total_bytes = sum(
            sum(f.stat().st_size for f in s.rglob("*") if f.is_file())
            for s in sessions
        )

        if total_bytes <= self.max_total_bytes:
            return

        sessions.sort(key=lambda p: p.stat().st_mtime)
        if self.verbose:
            print(f"[cache] Quota exceeded ({_fmt_bytes(total_bytes)}). Cleaning old sessions...")

        while sessions and total_bytes > self.max_total_bytes:
            victim = sessions.pop(0)
            try:
                size = sum(f.stat().st_size for f in victim.rglob("*") if f.is_file())
                shutil.rmtree(victim, ignore_errors=True)
                total_bytes -= size
                if self.verbose:
                    print(f"[cache] Removed old session {victim.name} ({_fmt_bytes(size)})")
            except Exception as e:
                if self.verbose:
                    print(f"[cache] WARNING: failed to remove {victim}: {e}")

    def _write_to_disk_zarr(self, ds: xr.Dataset, key: tuple[int, int]) -> xr.Dataset:
        """
        Efficiently write a Dataset to disk as a Zarr store.

        This version is optimized for single-process or per-session usage
        (e.g., one CacheManager per `r`). It supports fast writes with
        cached compression encoding and safe reuse of existing files.
        """

        # --- Active session directory ---
        cache_dir = self._get_active_cache_dir()
        nx, ny = key
        z_path = os.path.join(cache_dir, f"shift_{nx:+04d}_{ny:+04d}.zarr")

        # --- Try to reuse existing cache ---
        if os.path.exists(z_path):
            try:
                reopened = xr.open_zarr(z_path, consolidated=False, chunks=None)
                if set(ds.data_vars) == set(reopened.data_vars):
                    if self.verbose:
                        print(f"[CacheManager] Reusing cached shift {key} at {z_path}")
                    return reopened
            except Exception:
                # Remove corrupted or partial store
                shutil.rmtree(z_path, ignore_errors=True)

        # --- Compressor encoding cache (reused across calls) ---
        if not hasattr(self, "_encoding_cache"):
            self._encoding_cache = {v: {"compressor": self.compressor} for v in ds.data_vars}
        encoding = self._encoding_cache

        # --- Write to disk (directly, not via tmpdir) ---
        ds.to_zarr(z_path,
                   mode="w",
                   consolidated=False,
                   zarr_format=2,
                   encoding=encoding,
                   compute=False
                   ).compute()

        # --- Update cache index ---
        index = self.cleanup_stale_entries(cache_dir) if os.path.exists(cache_dir) else {}
        index[str(key)] = z_path
        self.save_index(index, cache_dir)

        # --- Reopen the stored dataset for use ---
        reopened = xr.open_zarr(z_path, consolidated=False, chunks=None)
        reopened = reopened.transpose(*ds.dims, missing_dims="ignore")
        reopened = reopened.assign_coords(
            {dim: ds[dim] for dim in ds.dims if dim in reopened.dims}
        )
        del ds
        gc.collect()

        if self.verbose:
            print(f"[CacheManager] Stored and reopened shift {key} at {z_path}")

        return reopened

    # ==========================================================
    # Unified hybrid persistence API
    # ==========================================================
    def persist(self, ds: xr.Dataset, key: tuple[int, int]) -> xr.Dataset | None:

        # Check available memory
        avail_ratio = self._available_memory_ratio()

        if avail_ratio > self.mem_threshold:
            if self.verbose:
                print(f"[CacheManager] Keeping {key} in memory ({avail_ratio:.1%} free RAM)")
            return ds.persist()

        if self.verbose:
            print(f"[CacheManager] Spilling {key} to disk ({avail_ratio:.1%} free RAM)")

        # enforce quota or cached files
        self._enforce_quota()
        return self._write_to_disk_zarr(ds, key)


# ---------------------------------------------------------------
# 2. Memory estimation utilities
# ---------------------------------------------------------------
def _fmt_bytes(n_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while n_bytes >= 1024 and i < len(units) - 1:
        n_bytes /= 1024
        i += 1
    return f"{n_bytes:.2f} {units[i]}"


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
        safety_factor: float = 0.5,
        verbose: bool = True,
) -> tuple[int, int]:
    """
    Estimate a safe batch size for processing multiple scales or loop items
    based on available memory and dataset footprint.

    Automatically accounts for multi-worker (SLURM/Dask) environments.
    """
    # --- Estimate memory per dataset instance ---
    per_item_bytes = estimate_dataset_bytes(ds, exclude_dims=exclude_dims, mode="total")
    per_item_bytes *= working_set_multiplier

    per_item_bytes = max(1, per_item_bytes)

    # --- Worker-aware memory budget ---
    per_worker_budget, n_workers = get_worker_memory_budget(reserve_ratio)
    usable_mem = per_worker_budget * safety_factor

    # --- Compute max safe items ---
    max_items_fit = max(1, int(usable_mem // per_item_bytes))
    batch_size = min(items_total, max_items_fit)

    n_batches = min(items_total, int(np.ceil(items_total / batch_size)))

    # --- Verbose diagnostics ---
    if verbose:
        print(f"[batch] Estimating optimal batch size: "
              f"per-item ≈ {_fmt_bytes(per_item_bytes)} "
              f"| usable/worker ≈ {_fmt_bytes(usable_mem)} ({n_workers} workers)"
              f"| Running {n_batches} batches of ≤{batch_size} (out of {items_total})")

    return batch_size, n_batches


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

    if target <= 0:
        raise ValueError(f"'target' must be positive, got {target}")

    if min_size <= 0:
        raise ValueError(f"'min_size' must be positive, got {min_size}")

    # --- Determine number of chunks m ---
    # m must be *int* after ceiling
    m = max(1, int(np.ceil(n / max(1, target))))

    # shrink m if too many chunks cause < min_size pieces
    while m > 1 and (n // m) < min_size:
        m -= 1

    # --- Compute base distribution ---
    base = n // m
    rem = n % m

    # create chunks (all integers)
    chunks = (base + 1,) * rem + (base,) * (m - rem)

    # --- Safety: ensure no chunk < min_size ---
    if any(c < min_size for c in chunks):
        # recompute m based on min_size constraint only
        m = max(1, n // min_size)
        base = n // m
        rem = n % m
        chunks = (base + 1,) * rem + (base,) * (m - rem)

    return tuple(chunks)


# ---------------------------------------------------------------
# 5. High-level adaptive chunking interface
# ---------------------------------------------------------------
def ensure_optimal_chunking(
        ds: xr.Dataset,
        spatial_dims: Tuple[str, str] = ("lat", "lon"),
        vertical_dim: str = "z",
        memory_threshold_ratio: float = _MEMORY_RESERVE_RATIO,
        deriv_edge_order: int = 2,
        verbose: bool = True,
        rechunk_spatial: bool = False,
        output_scale_mult: int = 1,
        desired_chunk_size_mb: Optional[float] = None,
        min_auto_rechunk_mb: float = _SMALL_DATA_THRESHOLD_MB
) -> xr.Dataset:
    """
    Rechunk dataset to balance performance and memory usage.
    Keeps full horizontal planes unless they exceed memory budget.
    """
    y_dim, x_dim = spatial_dims

    # ---- Small dataset shortcut ----
    est_output_size = estimate_dataset_bytes(ds, mode="total") * output_scale_mult

    if est_output_size < min_auto_rechunk_mb * 1024 ** 2:
        if verbose:
            print(f"[chunking] Estimated output dataset "
                  f"size is small ({_fmt_bytes(est_output_size)}); "
                  f"ensuring at least spatially contiguous.")
        # ensure spatially contiguous
        plan: Dict[str, Any] = {x_dim: -1, y_dim: -1}
        return ds.chunk(plan)

    exclude_dims = [str(d) for d in ds.dims if d not in spatial_dims]

    spatial_fits, bytes_per_plane, max_mem = fits_in_memory(
        ds, exclude_dims=exclude_dims,
        expansion_factor=output_scale_mult,
        ratio_to_use=memory_threshold_ratio,
    )

    plan: Dict[str, Any] = {}

    # ---- Spatial chunking ----
    if not rechunk_spatial and spatial_fits:
        plan.update({y_dim: -1, x_dim: -1})
    else:
        reduction = max(1.0, bytes_per_plane / max(1, max_mem))
        n_tiles = max(1, np.ceil(np.sqrt(reduction)))
        cy, cx = np.ceil(ds.sizes[y_dim] / n_tiles), np.ceil(ds.sizes[x_dim] / n_tiles)
        plan.update({y_dim: cy, x_dim: cx})
        if verbose:
            print(f"[chunking] Applying spatial tiling: ({n_tiles}×{n_tiles}) → {cy}×{cx}")

    # ---- Time / Vertical chunk balancing ----
    needs_t = "time" in ds.dims
    needs_z = vertical_dim in ds.dims
    t_guess, z_guess = ds.sizes.get("time", 1), ds.sizes.get(vertical_dim, 1)

    # Estimate the size of one horizontal plane
    target_bytes = (desired_chunk_size_mb or DEFAULT_CHUNK_SIZE_MB) * 1024 ** 2

    # Compute how many planes fit into the target
    n_planes_per_chunk = max(1, int(target_bytes // max(1, int(bytes_per_plane))))

    if verbose:
        print(f"[chunking] Target chunk budget: {desired_chunk_size_mb or 128:.0f} MB "
              f"→ {n_planes_per_chunk} planes per chunk")

    # ---- Choose chunking strategy ----
    z_target = int(np.ceil(np.sqrt(n_planes_per_chunk)))  # split planes into time and z
    min_z_chunk = int(deriv_edge_order + 1)

    if needs_t and needs_z:
        # Split budget roughly between z and time
        z_chunk = _balanced_chunks(z_guess, z_target, min_z_chunk)
        t_chunk = max(1, min(t_guess, n_planes_per_chunk // max(1, min(z_chunk))))

        plan.update({"time": t_chunk, vertical_dim: z_chunk})
    elif needs_z:
        z_chunk = _balanced_chunks(z_guess, n_planes_per_chunk, min_z_chunk)
        plan[vertical_dim] = z_chunk
    elif needs_t:
        plan["time"] = min(t_guess, n_planes_per_chunk)

    # ---- Convert numeric chunk hints into balanced tuples ----
    for dim, chunks in plan.items():
        if dim not in ds.dims:
            continue
        dim_size = ds.sizes[dim]
        if isinstance(chunks, int):
            if dim not in spatial_dims:
                n_chunks = max(1, np.ceil(dim_size / chunks))
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
              f"Est. output working set: {_fmt_bytes(total_est)}")

    return out
