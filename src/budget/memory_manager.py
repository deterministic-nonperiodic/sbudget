"""
DESCRIPTION
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
import os
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, ClassVar, Union, List, Tuple, Iterable, Dict, Any

import numpy as np
import psutil
import xarray as xr
from dask.base import tokenize
from dask.distributed import get_client, get_worker
from numcodecs import Blosc

# ======================================================================
# Global configuration parameters
# ======================================================================
_MEMORY_RESERVE_RATIO = 0.60
_SMALL_DATA_THRESHOLD_MB = 200.0  # skip chunking for small datasets
DEFAULT_CHUNK_SIZE_MB = 4096.0  # MB

# --- CONFIGURATION CONSTANT ---
TARGET_WORKER_MEM_GB = DEFAULT_CHUNK_SIZE_MB / 1024


# ======================================================================
# Utility
# ======================================================================
def _simple_tokenize(obj):
    """Uses Dask's tokenize which is fast but not content-aware for NumPy arrays."""
    return tokenize(obj)


def _fmt_bytes(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


# ======================================================================
# Async, Worker-Aware CacheManager
# ======================================================================

class CacheManager:
    """
    High-performance hybrid persistence manager.

    Features:
    ----------
    • Worker-aware singleton instance (one per Dask worker process).
    • Automatic memory/disk spill based on free RAM.
    • Asynchronous Zarr-v2 writing via a dedicated writer thread.
    • Reuse of existing cached files + hit ratio tracking.
    • Optional auto cleanup + disk quota enforcement.
    """

    _GLOBAL_SESSION: ClassVar[Optional[Path]] = None
    _CACHE_INDEX_NAME: ClassVar[str] = "cache_index.json"

    # Worker-aware singleton storage
    _INSTANCES: ClassVar[dict] = {}
    _INSTANCE_LOCK: ClassVar[threading.Lock] = threading.Lock()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
            self,
            base_dir: str | None = None,
            max_total_gb: float = 50.0,
            force_threshold: float = _MEMORY_RESERVE_RATIO,
            verbose: bool = False,
            auto_cleanup: bool = True,
            compressor: str = "lz4",
            max_workers: int = 4,  # Increased default to 4 for better I/O concurrency
    ):
        self.verbose = verbose
        self.auto_cleanup = auto_cleanup
        self.force_threshold = force_threshold
        self.max_total_bytes = max_total_gb * 1024 ** 3
        self.compressor = Blosc(cname=compressor, clevel=1, shuffle=Blosc.SHUFFLE)

        # Metrics
        self.total_processed_files = 0
        self.total_reused_files = 0

        # Resolve global base directory
        self.base_dir = self._resolve_base_dir(base_dir)

        # Worker-aware session directory
        worker_id = self._get_worker_id()

        # Use a deterministic name for workers to potentially simplify reuse/debugging
        if worker_id:
            session_suffix = f"session_worker_{worker_id}_{os.getpid()}"
        else:
            session_suffix = f"session_{os.getpid()}_{uuid.uuid4().hex[:6]}"

        # Cleanup any stale sessions
        self._cleanup_stale_sessions()

        # Create this session directory
        self.session_dir = Path(self.base_dir) / session_suffix
        self.session_dir.mkdir(parents=True, exist_ok=True)

        if CacheManager._GLOBAL_SESSION is None:
            CacheManager._GLOBAL_SESSION = self.session_dir

        # Active var-cache root (created on demand)
        self._active_cache_dir = None
        self._tracked_dirs: set[str] = set()

        # Encoding cache (set on first write)
        self._encoding_cache = None

        # Async executor for Zarr writes
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._pending_writes = {}  # key → Future
        atexit.register(self._shutdown_executor)

        # Enforce quota if desired
        if self.auto_cleanup:
            self._enforce_quota()
            atexit.register(self.cleanup_session)

    # --- FACTORY METHOD ---
    @classmethod
    def for_current_worker(cls, **kwargs) -> "CacheManager":
        """
        Factory method to get or create a CacheManager for the current worker/process.
        Ensures only one instance exists per Dask worker process.
        """
        worker_id = cls._get_worker_id()

        # Use a key unique to the execution context
        key = f"worker_{worker_id}" if worker_id else f"pid_{os.getpid()}"

        with cls._INSTANCE_LOCK:
            if key not in cls._INSTANCES:
                if kwargs.get('verbose'):
                    print(f"[CacheManager] Initializing new instance for {key}")
                cls._INSTANCES[key] = cls(**kwargs)

            return cls._INSTANCES[key]

    # ==================================================================
    # Directory utilities
    # ==================================================================

    @classmethod
    def _resolve_base_dir(cls, base_dir: str | None = None) -> Path:
        """
        Resolve the cache base directory using the same logic everywhere.
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

    @staticmethod
    def _get_worker_id():
        """
        Returns the worker ID if running on a worker, otherwise returns None.
        """
        try:
            worker = get_worker()
            return getattr(worker, "name", None) or getattr(worker, "id", None)
        except Exception:
            return None

    def new_var_cache(self, prefix="var_cache", subdir=None):
        """
        Create a variable-specific cache directory under this session.
        """
        if CacheManager._GLOBAL_SESSION is None:
            CacheManager._GLOBAL_SESSION = self.session_dir

        base = self.session_dir
        if subdir:
            base = base / str(subdir)
            base.mkdir(parents=True, exist_ok=True)

        path = Path(tempfile.mkdtemp(prefix=f"{prefix}_", dir=base))
        self._register_temp_dir(path)

        if self.verbose:
            print(f"[cache] Created var cache: {path}")

        return path

    def _register_temp_dir(self, p: Path):
        p = str(p)
        self._tracked_dirs.add(p)
        if self.auto_cleanup:
            atexit.register(lambda path=p: shutil.rmtree(path, ignore_errors=True))

    def _get_active_cache_dir(self):
        if self._active_cache_dir is None:
            self._active_cache_dir = self.new_var_cache(prefix="zarr_cache")
        return self._active_cache_dir

    # ==================================================================
    # Cleanup utilities
    # ==================================================================

    def _cleanup_stale_sessions(self):
        """Remove stale cache sessions from dead processes, handling worker names."""
        for s in Path(self.base_dir).glob("session_*"):
            if not s.is_dir():
                continue
            try:
                # Attempt to parse PID, handles: session_<pid>_... or session_worker_<id>_<pid>
                parts = s.name.split("_")
                if "worker" in parts:
                    # format: session_worker_<id>_<pid>
                    pid = int(parts[-1])
                else:
                    # format: session_<pid>_<uuid>
                    pid = int(parts[1])
            except (IndexError, ValueError):
                continue

            # If that PID no longer exists, remove the directory
            if not psutil.pid_exists(pid):
                if self.verbose:
                    print(f"[cache] Removing stale session: {s}")
                shutil.rmtree(s, ignore_errors=True)

    def cleanup_session(self):
        for p in list(self._tracked_dirs):
            shutil.rmtree(p, ignore_errors=True)

        # Ensure the executor is shut down before attempting to remove the session dir
        self._shutdown_executor()

        if self.session_dir.exists():
            shutil.rmtree(self.session_dir, ignore_errors=True)
            if self.verbose:
                print(f"[cache] Cleaned session cache: {self.session_dir}")

    def cleanup_all(self):
        for s in Path(self.base_dir).glob("session_*"):
            shutil.rmtree(s, ignore_errors=True)
        if self.verbose:
            print(f"[cache] Cleaned all sessions under {self.base_dir}")

    # ==================================================================
    # Metrics
    # ==================================================================
    def get_reuse_fraction(self) -> float:
        """Returns the ratio of reused files to total processed files (0.0 to 1.0)."""
        if self.total_processed_files == 0:
            return 0.0
        return self.total_reused_files / self.total_processed_files

    # ==================================================================
    # Memory + quota utilities
    # ==================================================================

    def _available_memory_ratio(self):
        mem = psutil.virtual_memory()
        return mem.available / mem.total

    def _enforce_quota(self):
        sessions = [p for p in Path(self.base_dir).glob("session_*") if p.is_dir()]
        if not sessions:
            return

        sessions = [s for s in sessions if s != self.session_dir]

        total_bytes = sum(
            sum(f.stat().st_size for f in s.rglob("*") if f.is_file())
            for s in sessions
        )

        if total_bytes <= self.max_total_bytes:
            return

        sessions.sort(key=lambda p: p.stat().st_mtime)

        if self.verbose:
            print(f"[cache] Disk quota exceeded ({_fmt_bytes(total_bytes)})")

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

    # ==================================================================
    # Async executor shutdown
    # ==================================================================

    def _shutdown_executor(self):
        # Wait for all pending writes to finish before shutting down
        self.wait_for_all_writes()

        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    def wait_for_all_writes(self):
        """Public method to block until all pending asynchronous writes are complete."""
        keys_to_wait = list(self._pending_writes.keys())
        if self.verbose and keys_to_wait:
            print(f"[CacheManager] Waiting for {len(keys_to_wait)} pending writes to finish...")

        for key in keys_to_wait:
            self._ensure_finished(key, log_slow=False)

    # ==================================================================
    # Async write + lazy reopen utilities
    # ==================================================================

    def _async_write(self, ds: xr.Dataset, z_path: str, encoding: dict):
        """Target function for ThreadPoolExecutor."""
        # Note: compute=True is critical here to ensure write happens in the thread
        ds.to_zarr(
            z_path,
            mode="w",
            consolidated=False,
            zarr_format=2,
            encoding=encoding,
            compute=True,
        )
        return z_path

    def _ensure_finished(self, key: str, log_slow: bool = True):
        """Waits for a specific write job to complete and removes it from pending list."""
        fut = self._pending_writes.get(key)
        if fut is None:
            return

        start_time = time.time()
        try:
            # This will block until write is done (or raises an exception)
            fut.result()
        finally:
            del self._pending_writes[key]

        block_time = time.time() - start_time
        if self.verbose and log_slow and block_time > 0.1:  # Log if blocking is significant (> 100ms)
            print(
                f"[CacheManager] WARNING: Main thread blocked for {block_time:.3f}s waiting for {key}")

    def _open_lazy(self, z_path: str, ref: Union[xr.Dataset, xr.DataArray]):
        """
        Opens the Zarr store lazily and ensures dimensions/coordinates match the reference
        object (DataArray or Dataset) for downstream xarray use.
        """
        ds = xr.open_zarr(z_path, consolidated=False, zarr_format=2)

        if isinstance(ref, xr.DataArray):
            # If the original was a DataArray, we extract it and transpose
            var_name = ref.name or list(ds.data_vars)[0]
            da = ds[var_name]
            da = da.transpose(*ref.dims, missing_dims="ignore")
            da = da.assign_coords(
                {dim: ref[dim] for dim in ref.dims if dim in da.dims}
            )
            da.name = ref.name
            return da

        # Otherwise, assume it's a Dataset
        ds = ds.transpose(*ref.dims, missing_dims="ignore")
        ds = ds.assign_coords(
            {dim: ref[dim] for dim in ds.dims if dim in ref.dims}
        )
        return ds

    def _ensure_array_dimensions(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Ensure each variable has attribute `_ARRAY_DIMENSIONS`,
        required for xarray to reopen a Zarr v2 store safely.
        """
        fixed = {}
        for var, da in ds.data_vars.items():
            attrs = dict(da.attrs)
            # CRITICAL FIX: Ensure _ARRAY_DIMENSIONS is present
            attrs["_ARRAY_DIMENSIONS"] = list(da.dims)
            fixed[var] = da.assign_attrs(attrs)

        # Use .copy(deep=False) to preserve coordinate attributes/encoding
        return xr.Dataset(fixed, coords=ds.coords).copy(deep=False)

    # ==================================================================
    # Core I/O
    # ==================================================================
    def _write_to_disk_zarr(self, obj: Union[xr.Dataset, xr.DataArray], key: str):
        cache_dir = self._get_active_cache_dir()
        z_path = os.path.join(cache_dir, f"{key}.zarr")

        self.total_processed_files += 1

        # 1. Reuse (Cache Hit)
        if os.path.exists(z_path):
            self.total_reused_files += 1
            if self.verbose:
                print(f"[CacheManager] Reused cached entry {key} at {z_path}")
            return self._open_lazy(z_path, obj)

        # 2. Normalize to dataset (for internal processing)
        if isinstance(obj, xr.DataArray):
            ds_to_write = obj.to_dataset(name=obj.name or "data")
        else:
            ds_to_write = obj

        # 3. Fix Zarr metadata dimensions
        ds_to_write = self._ensure_array_dimensions(ds_to_write)

        # 4. Encoding cache initialization (if needed)
        if self._encoding_cache is None:
            self._encoding_cache = {
                v: {"compressor": self.compressor}
                for v in ds_to_write.data_vars
            }

        # 5. Submit async write
        fut = self._executor.submit(self._async_write, ds_to_write, z_path, self._encoding_cache)
        self._pending_writes[key] = fut

        if self.verbose:
            print(f"[CacheManager] Submitted async write for {key} to {z_path}")

        # 6. BLOCKING CALL: Wait for the write thread to finish before attempting to open the file.
        # This is necessary to prevent FileNotFoundError on immediate subsequent read.
        self._ensure_finished(key)

        # 7. Return lazy reader
        return self._open_lazy(z_path, obj)

    # ==================================================================
    # Hybrid persistence API
    # ==================================================================
    def persist(self, obj: xr.Dataset | xr.DataArray, key: str = None) -> Union[
        xr.Dataset, xr.DataArray]:

        # Check available memory
        if key is None:
            # Generate a dask-safe toke for dataset
            key = _simple_tokenize(obj)

        # Check if object fits in the 'force_threshold' fraction of the available memory
        fits, obj_size, mem_limit = fits_in_memory(obj=obj, ratio_to_use=self.force_threshold)

        if fits:
            if self.verbose:
                print(print(f"[CacheManager] Keeping {key} in memory "
                            f"(Obj size: {_fmt_bytes(obj_size)}, Limit: {_fmt_bytes(mem_limit)})"))
            return obj.persist()

        if self.verbose:
            print(f"[CacheManager] Spilling {key} to disk "
                  f"(Obj size: {_fmt_bytes(obj_size)} exceeds limit {_fmt_bytes(mem_limit)})")

        return self._write_to_disk_zarr(obj, key)


# ---------------------------------------------------------------
# 2. Memory estimation utilities
# ---------------------------------------------------------------
def optimal_batch_size(
        obj: Union[xr.Dataset, xr.Dataset],
        items_total: int,
        exclude_dims: Iterable[str] | str | None = None,
        reserve_ratio: float = _MEMORY_RESERVE_RATIO,
        target_size_bytes: int = DEFAULT_CHUNK_SIZE_MB * 1024 ** 2,
        verbose: bool = True) -> tuple[int, int]:
    """
    Estimate a safe batch size for processing multiple scales or loop items
    based on available memory and dataset footprint.

    Automatically accounts for multi-worker (SLURM/Dask) environments.
    """
    # --- Estimate memory per dataset instance ---
    per_item_bytes = estimate_dataset_bytes(obj, exclude_dims=exclude_dims, mode="total")

    per_item_bytes = max(1, per_item_bytes)

    # --- Worker-aware memory budget ---
    worker_limit = get_worker_memory_limit(mode="worker")
    usable_mem = int(worker_limit * reserve_ratio)
    n_workers = get_current_worker_count(verbose=False, total=False)

    # --- Compute max safe items ---
    max_items_fit = max(1, int(min(usable_mem, target_size_bytes) // per_item_bytes))

    batch_size = max(1, min(items_total, max_items_fit))

    n_batches = min(items_total, int(np.ceil(items_total / batch_size)))

    # --- Verbose diagnostics ---
    if verbose:
        print(f"[batch] Estimating optimal batch size: "
              f"per-item ≈ {_fmt_bytes(per_item_bytes)} "
              f"| usable/worker ≈ {_fmt_bytes(usable_mem)} ({n_workers} workers)"
              f"| Running {n_batches} batches of ≤{batch_size} (out of {items_total})")

    return batch_size, n_batches


def get_num_blocks(ds: xr.Dataset, ref_var_name: str = 'u') -> int:
    """
    Calculates the total number of Dask blocks (partitions) in the input dataset.

    This is equivalent to the total number of tasks Dask needs to run across
    all dimensions for the reference variable.

    Args:
        ds (xr.Dataset): The dataset containing Dask-backed DataArrays.
        ref_var_name (str): The name of the reference variable to use for
            counting blocks (e.g., 'u', 'v'). Defaults to 'u'.

    Returns:
        int: The total number of Dask blocks. Returns 0 if the dataset is not
             Dask-backed or the reference variable is missing.
    """
    # 1. Select the reference DataArray
    if ref_var_name in ds.data_vars:
        ref_field = ds[ref_var_name]
    elif ds.data_vars:
        # Fallback to the first variable if 'u' is missing
        ref_field = ds[list(ds.data_vars)[0]]
    else:
        # Dataset is empty
        return 0

    # Check if the variable is Dask-backed and return number of partitions
    if hasattr(ref_field.data, 'npartitions'):
        return ref_field.data.npartitions
    else:
        # Data is likely a NumPy array (fully loaded into memory/not chunked)
        return 1


def get_worker_memory_budget(reserve_ratio: float = 0.7) -> tuple[int, int]:
    """
    Estimate per‑worker memory budget using SLURM environment variables if
    available, otherwise fallback to detected CPU count.
    """
    total = psutil.virtual_memory().total
    env_keys = ["SLURM_NTASKS", "SLURM_NTASKS_PER_NODE",
                "DASK_WORKER_NPROCS", "DASK_WORKER_NTHREADS"]

    n = None
    for key in env_keys:
        if key in os.environ:
            try:
                n = int(os.environ[key])
                break
            except ValueError:
                pass

    if n is None:
        n = psutil.cpu_count(logical=True) or 1

    per_worker = total * reserve_ratio / n
    return int(per_worker), int(n)


# ---------------------------------------------------------------------------
# 1. Memory / Worker Utilities
# ---------------------------------------------------------------------------

def _parse_slurm_memory() -> Optional[float]:
    """
    Parses SLURM memory variables to determine total memory in GiB.
    """
    n_tasks = int(os.environ.get("SLURM_NTASKS", 1))

    # 1. SLURM_MEM_PER_NODE (Total memory for the whole job/node)
    if "SLURM_MEM_PER_NODE" in os.environ:
        return int(os.environ["SLURM_MEM_PER_NODE"]) / 1024.0

    # 2. SLURM_MEM_PER_CPU (Memory per task/CPU)
    if "SLURM_MEM_PER_CPU" in os.environ:
        mem_per_task = int(os.environ["SLURM_MEM_PER_CPU"]) / 1024.0
        return n_tasks * mem_per_task

    # 3. SLURM_MEM (General total memory request)
    if "SLURM_MEM" in os.environ:
        mem_str = os.environ["SLURM_MEM"].upper()
        # Handle units like 256G, 1000M
        import re
        match = re.search(r'(\d+)([MG])', mem_str)
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            if unit == 'G':
                return value
            elif unit == 'M':
                return value / 1024.0
        try:
            return float(os.environ["SLURM_MEM"]) / 1024.0
        except ValueError:
            pass
            
    return None


def get_worker_memory_limit(mode: str = "worker") -> int:
    """
    Robustly determine the memory limit for the current process/worker in bytes.
    
    Priority:
    1. Dask worker limit (if running on a worker)
    2. SLURM allocation (if in SLURM job)
    3. System RAM / N_workers (fallback)
    
    Parameters
    ----------
    mode : str
        "worker" (check Dask worker context) or "client" (global estimation)
        
    Returns
    -------
    int
        Memory limit in bytes
    """
    # 1. Check Dask Worker Context
    if mode == "worker":
        try:
            worker = get_worker()
            # Handle Dask deprecation: memory_limit moved to memory_manager.memory_limit
            if hasattr(worker, "memory_manager") and hasattr(worker.memory_manager, "memory_limit"):
                 return int(worker.memory_manager.memory_limit)
            elif hasattr(worker, "memory_limit") and worker.memory_limit:
                return int(worker.memory_limit)
        except (ValueError, ImportError):
            pass

    # 2. Check SLURM Allocation
    slurm_mem_gb = _parse_slurm_memory()
    if slurm_mem_gb is not None:
        # If we are in a worker context but couldn't get the worker object,
        # we need to divide the total SLURM memory by the number of tasks
        if mode == "worker":
             n_tasks = int(os.environ.get("SLURM_NTASKS", 1))
             return int((slurm_mem_gb * 1024**3) / n_tasks)
        return int(slurm_mem_gb * 1024**3)

    # 3. Fallback: System RAM / N_workers
    total_mem = psutil.virtual_memory().total
    n_workers = get_current_worker_count(verbose=False, total=False)
    
    # If we are just checking global capacity (mode="client"), return total
    if mode == "client":
        return total_mem
        
    return int(total_mem / n_workers)


def estimate_dataset_bytes(
        obj: Union[xr.Dataset, xr.Dataset],
        exclude_dims: Iterable[str] | str | None = None,
        mode: str = "largest_chunk",
) -> int:
    """
    Estimate memory footprint (in bytes) for an xarray Dataset or DataArray.

    Parameters
    ----------
    obj : xr.Dataset or xr.DataArray
        The object to analyze.
    exclude_dims : iterable of str or str, optional
        Dimensions to exclude when computing chunk sizes (e.g. 'time').
    mode : {'largest_chunk', 'total'}
        - 'largest_chunk': size of the largest single chunk.
        - 'total': sum of all chunks.

    Returns
    -------
    int
        Estimated memory footprint in bytes.
    """
    # Normalize exclude_dims
    if isinstance(exclude_dims, str):
        exclude_dims = [exclude_dims]
    exclude_dims = set(exclude_dims or [])

    def estimate_for_var(v: xr.DataArray) -> int:
        """Estimate byte size for a single DataArray."""
        item_size = np.dtype(v.dtype).itemsize
        chunks = getattr(v.data, "chunks", None)

        # Produce dim_chunks: list of tuples, each tuple = chunk sizes along that dim
        if chunks is not None:
            # Dask-backed
            dim_chunks = list(chunks)
        else:
            # In-memory array: one chunk equal to the full dimension size
            dim_chunks = [(v.sizes[d],) for d in v.dims]

        if mode == "largest_chunk":
            elems = 1
            for d, ch in zip(v.dims, dim_chunks):
                if d in exclude_dims:
                    elems *= 1
                else:
                    elems *= max(ch)
            return elems * item_size

        elif mode == "total":
            elems = 1
            for d, ch in zip(v.dims, dim_chunks):
                if d in exclude_dims:
                    continue
                elems *= sum(ch)
            return elems * item_size

        else:
            raise ValueError("mode must be 'largest_chunk' or 'total'.")

    # --- Handle Dataset by recursion ---
    if isinstance(obj, xr.Dataset):
        return int(sum(estimate_for_var(v) for v in obj.data_vars.values()))

    # --- Handle DataArray directly ---
    elif isinstance(obj, xr.DataArray):
        return int(estimate_for_var(obj))

    else:
        raise TypeError("Input must be an xarray Dataset or DataArray.")


def get_current_worker_count(verbose=True, total=True) -> int:
    """
    Return the total number of threads available for computation
    (workers * threads_per_worker).

    Logic:
    ------
    1. If SLURM_NTASKS is set → authoritative total thread count.
    2. Otherwise, check active Dask client for total threads used.
    3. Fall back to os.cpu_count() if client missing.

    Returns
    -------
    int
        Total number of threads (cores) ready for computation.
    """
    # --- SLURM authoritative override (Total Threads = Ntasks * Nthreads/task) ---
    if "SLURM_JOB_ID" in os.environ:
        if verbose:
            print("[chunking] SLURM environment detected: using SLURM_NTASKS for worker count.")
        try:
            n_tasks = int(os.environ.get("SLURM_NTASKS", 1))

            if total:
                # Default to 1 thread per task if SLURM_CPUS_PER_TASK is not set
                threads_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
                n_tasks = n_tasks * threads_per_task

            return max(1, n_tasks)

        except ValueError:
            pass  # Continue to Dask check if variable is corrupted

    # --- Dask client check (Total Threads = Sum of threads across all workers) ---
    try:
        client = get_client()
        workers = client.scheduler_info().get("workers", {})

        # Sum the number of threads value for all workers
        if total:
            n_tasks = sum(w.get("nthreads", 1) for w in workers.values())
        else:
            n_tasks = len(workers)

        # Ensure at least one thread is assumed if client is active
        if verbose:
            print("[chunking] Client detected: using Dask worker thread count.")
        return max(1, n_tasks)

    except Exception:
        # --- Fallback to local machine resources (Total logical CPUs) ---
        if verbose:
            print("[chunking] No client detected: fallback to local CPU count.")
        return os.cpu_count() or 4


def fits_in_memory(
        obj: xr.Dataset | xr.DataArray,
        expansion_factor: int = 1,
        ratio_to_use: float = 0.7,
        exclude_dims: Iterable[str] | str | None = None,
        mode: str = "worker",
) -> tuple[bool, int, int]:
    """
    Determine whether `obj` (possibly expanded by `expansion_factor`) fits into
    the available per-worker memory budget.

    Returns
    -------
    fits : bool
        True if dataset fits within allowed memory.
    size_bytes : int
        Estimated dataset size.
    limit_bytes : int
        Allowed memory budget.
    """
    size_bytes = estimate_dataset_bytes(obj, exclude_dims) * max(1, expansion_factor)

    # Use robust memory limit detection
    worker_limit = get_worker_memory_limit(mode=mode)
    
    # Apply safety ratio
    limit = int(worker_limit * ratio_to_use)

    return size_bytes < limit, size_bytes, limit


def _balanced_chunks(n: int, target: int, min_size: int) -> Tuple[int, ...]:
    """
    Split length n into chunks suitable for Zarr storage.
    
    Produces uniform chunks with a final chunk that may be different.
    Ensures:
    1. All chunks are >= min_size
    2. Final chunk <= first chunk (Zarr requirement)
    3. All non-final chunks are equal (Zarr requirement)
    
    Parameters
    ----------
    n : int
        Total length to split
    target : int
        Target chunk size
    min_size : int
        Minimum allowed chunk size (e.g., derivative_order + 1)
        
    Returns
    -------
    tuple of int
        Chunk sizes suitable for Zarr storage
    """
    # Start with target as the uniform chunk size
    chunk_size = max(min_size, target)

    # Calculate how many full chunks and remainder
    num_full_chunks = n // chunk_size
    remainder = n % chunk_size

    # Divides evenly - all chunks are uniform
    if remainder == 0:
        return (chunk_size,) * num_full_chunks

    # Remainder is valid (>= min_size and <= chunk_size)
    # This satisfies both constraints
    if remainder >= min_size:
        return (chunk_size,) * num_full_chunks + (remainder,)

    # Remainder < min_size
    # We need to increase chunk_size so that the final chunk is valid
    # Strategy: increase chunk_size until remainder >= min_size
    while remainder < min_size and remainder != 0:
        chunk_size += 1
        num_full_chunks = n // chunk_size
        remainder = n % chunk_size

        # Safety check: if chunk_size gets too large, just return single chunk
        if chunk_size >= n:
            return (n,)

    # After adjustment, remainder should be valid or zero
    if remainder == 0:
        return (chunk_size,) * num_full_chunks
    else:
        return (chunk_size,) * num_full_chunks + (remainder,)


def ensure_optimal_chunking(
        ds: xr.Dataset,
        spatial_dims: Tuple[str, str] = ("lat", "lon"),
        vertical_dim: str = "z",
        memory_threshold_ratio: float = _MEMORY_RESERVE_RATIO,
        deriv_edge_order: int = 0,
        verbose: bool = True,
        rechunk_spatial: bool = False,
        output_scale_mult: int = 1,
        num_workers: Optional[int] = None,
        preferred_num_blocks: Optional[int] = None,
        min_chunk_mb: float = 128.0,
) -> xr.Dataset:
    """
    Adaptive, HPC‑aware multi‑dimensional chunking for Xarray+Dask datasets.
    Optimized for the Distributed Scheduler by maximizing chunk size
    within the memory budget while ensuring high parallelism.
    """
    y_dim, x_dim = spatial_dims

    # ---- 0. Small dataset fast path ----
    est_total = estimate_dataset_bytes(ds, mode="total") * output_scale_mult
    if est_total < _SMALL_DATA_THRESHOLD_MB * 1024 ** 2:
        if verbose:
            print(f"[chunking] Dataset small {_fmt_bytes(est_total)} → "
                  f"keeping full spatial chunks.")
        # Ensure non-spatial dims are also chunked as -1 if they exist
        return ds.compute()

    # ---- Unify chunks to prevent inconsistencies ----
    ds = ds.unify_chunks()

    # ---- Check whether full spatial plane fits ----
    exclude = [str(d) for d in ds.dims if d not in spatial_dims]
    fits, plane_bytes, worker_limit = fits_in_memory(
        ds, exclude_dims=exclude,
        expansion_factor=output_scale_mult,
        ratio_to_use=memory_threshold_ratio,
        mode="worker",
    )

    plan: Dict[str, Any] = {}

    # ---- Spatial chunking decision ----
    if fits and not rechunk_spatial:
        plan[y_dim] = -1
        plan[x_dim] = -1

        if verbose:
            # Heuristic check: if we have 1 chunk in y/x, we are good
            print(f"[chunking] Full spatial slices fit in worker memory "
                  f"{_fmt_bytes(plane_bytes)} / {_fmt_bytes(worker_limit)}.")
    else:
        # Must tile spatial dims (logic remains correct for memory safety)
        reduction = max(1.0, plane_bytes / max(1, worker_limit))
        n_tiles = int(np.ceil(np.sqrt(reduction)))
        cy = int(np.ceil(ds.sizes[y_dim] / n_tiles))
        cx = int(np.ceil(ds.sizes[x_dim] / n_tiles))
        plan[y_dim] = cy
        plan[x_dim] = cx
        if verbose:
            print(f"[chunking] Spatial tiling → ({n_tiles}×{n_tiles}) → {cy}×{cx}")

    # ---- Non‑spatial dims (time & vertical) ----
    needs_t = "time" in ds.dims
    needs_z = vertical_dim in ds.dims
    t_size = ds.sizes.get("time", 1)
    z_size = ds.sizes.get(vertical_dim, 1)

    # Estimate bytes per spatial tile (single plane)
    isel_dict = {
        x_dim: slice(0, plan[x_dim] if plan[x_dim] != -1 else ds.sizes[x_dim]),
        y_dim: slice(0, plan[y_dim] if plan[y_dim] != -1 else ds.sizes[y_dim])
    }
    # Ensure we estimate for a single plane (t=1, z=1)
    if needs_t: isel_dict["time"] = slice(0, 1)
    if needs_z: isel_dict[vertical_dim] = slice(0, 1)

    tile = ds.isel(isel_dict, drop=True)
    tile_bytes = estimate_dataset_bytes(tile, mode="total")

    total_planes = t_size * z_size

    # Memory Constraint: Minimum blocks needed for safety (blocks_mem)
    # We must account for the expansion factor (output_scale_mult) when checking fit
    expanded_tile_bytes = tile_bytes * output_scale_mult
    planes_per_chunk_mem = max(1, int(worker_limit // max(1, expanded_tile_bytes)))
    blocks_mem = max(1, int(np.ceil(total_planes / planes_per_chunk_mem)))

    # Parallelism Constraint: Workers * Buffer (Goal: keep pipeline full)
    if num_workers is None:
        num_workers = get_current_worker_count(total=False)
    else:
        num_workers = max(1, int(num_workers))

    # Set parallelism buffer to 4. 2 caused Out-Of-Memory reboots in distributed workers 
    # operating under 7.5GB constraints, and 8 created far too many schedule tasks.
    PARALLEL_BUFFER = 4
    target_parallel_blocks = preferred_num_blocks or (num_workers * PARALLEL_BUFFER)
    target_parallel_blocks = int(np.ceil(target_parallel_blocks))

    # Efficiency Constraint: Don't create chunks smaller than min_chunk_mb just for parallelism
    # unless required by memory constraints.
    total_bytes_est = estimate_dataset_bytes(ds, mode="total") * output_scale_mult
    max_blocks_efficiency = max(1, int(total_bytes_est / (min_chunk_mb * 1024**2)))
    
    # Cap the parallelism target by efficiency
    # We want parallelism, but not if it makes chunks tiny.
    effective_parallel_target = min(target_parallel_blocks, max_blocks_efficiency)

    # Select Target Block Count: Use the largest count (smallest chunks) required by safety or parallelism.
    # We still use max() because we MUST satisfy the memory constraint (blocks_mem).
    blocks_target = max(blocks_mem, effective_parallel_target)

    # Calculate final planes per chunk based on the target block count
    planes_final = max(1, int(np.ceil(total_planes / blocks_target)))

    # ---- Choose chunking for time/Z ----
    # Aim for a square chunk in the t-z plane if possible for easier access/caching.
    z_target = int(np.ceil(np.sqrt(planes_final)))
    min_z = deriv_edge_order + 1

    if needs_t and needs_z:
        # Split vertical dim first, respecting the minimum stencil size
        z_chunks_tuple = _balanced_chunks(z_size, z_target, min_z)
        z_chunks_max = max(z_chunks_tuple)

        # Split time dim to use up remaining planes budget
        t_target = planes_final // max(1, z_chunks_max)
        t_chunks_tuple = _balanced_chunks(t_size, t_target, min_size=1)

        plan[vertical_dim] = z_chunks_tuple
        plan["time"] = t_chunks_tuple

    elif needs_z:
        plan[vertical_dim] = _balanced_chunks(z_size, planes_final, min_z)
    elif needs_t:
        plan["time"] = _balanced_chunks(t_size, planes_final, min_size=1)

    # ---- Final rechunk ----
    out = ds.unify_chunks().chunk(plan)

    n_blocks = get_num_blocks(out, ref_var_name="u")

    # ---- Summary ----
    if verbose:
        msg_parts: List[str] = []
        for d, c in plan.items():
            if isinstance(c, (tuple, list)):
                c_min, c_max = min(c), max(c)
                # Display target chunk size for non-spatial dims
                msg_parts.append(f"{d}={c_min}" if c_min == c_max else f"{d}=({c_min}, {c_max})")
            elif c == -1:
                msg_parts.append(f"{d}={out.sizes.get(d, 'N/A')} (full)")
            else:
                # Should not happen if _balanced_chunks is used correctly, but keep for robustness
                msg_parts.append(f"{d}={c}")
        if output_scale_mult > 1:
            msg_parts.append(f"Scale=x{output_scale_mult}")

        # Estimate largest chunk size after scaling
        largest = estimate_dataset_bytes(out, mode="largest_chunk") * output_scale_mult

        print(f"[chunking] Budget: {_fmt_bytes(worker_limit)} | workers: {num_workers} | "
              f"Plan: {', '.join(msg_parts)} | Partitions: {n_blocks} | "
              f"Est. largest chunk: {_fmt_bytes(largest)}")

    return out
