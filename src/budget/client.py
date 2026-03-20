import argparse
import os
import re
import resource
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import dask
import asyncio
import psutil
from dask.distributed import Client, LocalCluster

from .budget import compute_budget
from .config import load_config, apply_overrides
from .inter_scale_transfers import inter_scale_kinetic_energy_transfer
from .io_utils import open_dataset, write_dataset

# Suppress Dask warnings about large graphs
warnings.filterwarnings(
    "ignore",
    message=r"Sending large graph of size .* MiB\.",
    category=UserWarning,
    module=r'distributed\.client'
)

# New constant for maximum file descriptors
_MAX_FILE_DESCRIPTORS = 8192
_MIN_WORKER_MEM_GB = 4.0  # Minimum memory per worker in GiB



# --- Helper function definitions (omitted for brevity) ---
def _increase_fd_limit():
    """Attempts to increase the file descriptor limit for the current process."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < _MAX_FILE_DESCRIPTORS:
            new_soft = min(hard, _MAX_FILE_DESCRIPTORS)
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
            print(f"[FD] Increased file descriptor limit from {soft} to {new_soft}.")
    except Exception as e:
        print(f"[FD] Warning: Could not increase FD limit: {e}")


def _detect_resources(cfg: Optional[Any] = None) -> Dict[str, Union[int, float, bool]]:
    """Detects allocated resources, prioritizing SLURM environment variables."""
    from .memory_manager import get_current_worker_count, _parse_slurm_memory

    # Thread/Worker Allocation
    # Use centralized logic from memory_manager
    n_workers = get_current_worker_count(verbose=False, total=False)
    
    # Determine threads per worker
    if "SLURM_CPUS_PER_TASK" in os.environ:
        n_threads = int(os.environ["SLURM_CPUS_PER_TASK"])
    else:
        # Default to 1 thread per worker for GIL-bound workloads
        n_threads = 1

    # Override with config if provided
    if cfg is not None:
        if getattr(cfg.compute, "n_workers", None) is not None:
            n_workers = cfg.compute.n_workers
        if getattr(cfg.compute, "threads_per_worker", None) is not None:
            n_threads = cfg.compute.threads_per_worker

    # Total Memory Allocation (in GiB)
    # Use centralized logic from memory_manager
    slurm_mem = _parse_slurm_memory()
    if slurm_mem is not None:
        total_mem_gb = slurm_mem
    else:
        # Fallback to system memory
        total_mem_gb = psutil.virtual_memory().total / (1024 ** 3)

    return {"n_workers": n_workers, "threads_per_worker": n_threads, "total_mem_gb": total_mem_gb}


def auto_configure_dask(cfg: Optional[Any] = None) -> Tuple[int, int]:
    """HPC-aware configuration of Dask global settings and cluster parameters."""
    resources = _detect_resources(cfg)

    # --- Chunk Size Configuration ---
    # Only override Dask's default chunk size (usually 128MiB) if explicitly requested.
    # The previous heuristic (512MB/2048MB) was often too aggressive for 'auto' chunking.
    chunk_mb = getattr(cfg.compute, "chunk_size", None)
    
    dask_config = {
        "array.slicing.split_large_chunks": False,

        # Worker spilling ENABLED for stability
        "distributed.worker.memory.target": 0.65,  # Start garbage collection
        "distributed.worker.memory.spill": 0.85,   # Spilling to disk
        "distributed.worker.memory.pause": 0.95,   # Pause worker
        "distributed.worker.memory.terminate": 0.98, # Kill worker

        # DAG fusion optimizations
        "optimization.fuse.ave-width": 24,
        "optimization.fuse.max-width": 256,
    }

    if chunk_mb:
        dask_config["array.chunk-size"] = f"{chunk_mb}MB"

    # --- Configure Dask Global Settings (Memory Discipline) ---
    dask.config.set(dask_config)

    return int(resources["n_workers"]), int(resources["threads_per_worker"])


def init_dask_client(cfg: Any, scheduler_address: Optional[str] = None) -> Client:
    """
    Create and return a Dask client with adaptive memory configuration.
    """
    # Set glibc memory trimming threshold to release memory back to OS more aggressively
    os.environ["MALLOC_TRIM_THRESHOLD_"] = "65536"

    # Ensure sufficient file descriptor limit
    _increase_fd_limit()

    # Scheduler type
    scheduler = getattr(cfg.compute, "scheduler", "distributed")

    if scheduler not in {"distributed", "threads"}:
        raise ValueError(f"Unknown scheduler type: {scheduler}. Use 'threads' or 'distributed'.")

    # Get initial SLURM/Local core count and total memory pool
    # Note: auto_configure_dask now handles config overrides via cfg
    num_workers_detected, threads_per_worker_detected = auto_configure_dask(cfg)
    
    # Set threading environment variables to avoid oversubscription
    # This is critical when using numpy/scipy with Dask
    for env_var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        os.environ[env_var] = str(threads_per_worker_detected)

    total_mem_gb = _detect_resources(cfg)["total_mem_gb"]

    # Reserve memory for OS/Scheduler (e.g. 4GB or 10%, whichever is larger)
    # This prevents the cluster from consuming 100% of RAM and triggering OOM kills.
    reserved_mem = max(4.0, total_mem_gb * 0.10)
    available_mem_gb = max(0, total_mem_gb - reserved_mem)

    # Calculate the maximum number of workers that can safely fit (N_stable)
    # This determines the final worker count.
    max_workers_by_mem = int(available_mem_gb // _MIN_WORKER_MEM_GB)

    # Final worker count is the minimum of requested cores and memory capacity
    num_workers = min(num_workers_detected, max_workers_by_mem)

    if num_workers == 0:
        raise MemoryError(f"Insufficient memory ({available_mem_gb:.2f}GB available after reservation) "
                          f"to start even one worker with required minimum budget ({_MIN_WORKER_MEM_GB}GB).")
    # Finalize Configuration
    if scheduler == "distributed":
        # Distributed mode: We use multiple processes, so threads per worker is respected
        threads_per_worker = threads_per_worker_detected
        processes = True
    else:
        # Threads mode: We use one process, so threads per worker = total cores
        threads_per_worker = num_workers * threads_per_worker_detected
        num_workers = 1  # Force single worker process for thread mode
        processes = False

    # Calculate the Actual Worker Memory Limit. Distributing the *available* memory among workers
    actual_worker_mem_gb = available_mem_gb / num_workers

    # Override Dask config with the calculated actual limit
    dask.config.set({"distributed.worker.memory.limit": f'{actual_worker_mem_gb:.2f}GB'})

    print(f"[budget] Auto-Dask → processes={num_workers}, threads={threads_per_worker}")
    print(f"[budget] Available Memory Pool: {available_mem_gb:.2f} GiB")
    print(f"[budget] Actual Worker Limit: {actual_worker_mem_gb:.2f} GiB (Maximized)")

    # Configure cluster and return the Dask client
    if scheduler == "distributed" and scheduler_address:
        return Client(scheduler_address)

    cluster = LocalCluster(
        n_workers=num_workers,
        threads_per_worker=threads_per_worker,
        processes=processes,
        memory_limit=f'{actual_worker_mem_gb:.2f}GB',
        dashboard_address=":8787"  # Always enable dashboard
    )
    client = Client(cluster)

    if processes:
        # Wait for all workers to be ready to avoid race conditions in resource detection
        print(f"[budget] Waiting for {num_workers} workers to register...")
        client.wait_for_workers(n_workers=num_workers)

    print(f"[budget] Dask cluster: workers={num_workers}, threads/worker={threads_per_worker}")

    return client


def _cmd_compute(args) -> None:
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    # ==============================================
    # INITIALIZE DASK CLIENT
    # ==============================================
    use_client = not (getattr(args, 'client') is False)

    if use_client:
        client = init_dask_client(cfg, scheduler_address=None)
        print(f"[sbudget] Dask Dashboard: {client.dashboard_link}")
        print(f"[sbudget] Using client: {client}")
    else:
        client = None
        print(f"[sbudget] Running without Dask client (local threads).")

    # -------------------------------------------------------------------
    # Open dataset with normalized variable/dimension names
    ds = open_dataset(cfg, verbose=False)

    # Normalize/alias modes (backward compatibility)
    _MODE_ALIASES = {"spectral": "spectral_budget", "physical": "scale_transfer"}

    mode_in = str(cfg.compute.mode).strip()
    mode = _MODE_ALIASES.get(mode_in, mode_in)
    if mode_in in _MODE_ALIASES:
        print(f"[sbudget] NOTE: mode '{mode_in}' is deprecated; use '{mode}'")

    print(f"[sbudget] RUNNING MODE: {mode}")

    # store start time for profiling
    start_time = time.monotonic()

    try:
        if mode == "spectral_budget":
            print("[sbudget] Starting spectral budget calculation...")
            out = compute_budget(ds, cfg)
            print("[sbudget] Constructed dask graph for spectral budget calculation")
        elif mode == "scale_transfer":
            print("[sbudget] Starting inter-scale transfer calculation...")

            # Extract kwargs for scale_increments
            kwargs = {"scales": getattr(cfg.compute, "scales", None), "verbose": True}
            # Compute scale-transfers
            out = inter_scale_kinetic_energy_transfer(ds, **kwargs)
        else:
            raise ValueError(f"Unknown compute.mode='{cfg.compute.mode}'. "
                             f"Use 'spectral_budget' or 'scale_transfer'.")

        # Testing dry run: skip computation and I/O
        if getattr(args, 'dry_run', False):
            print("\n[budget] DRY RUN COMPLETE: Skipping Dask computation and I/O.")
            print(f"[budget] Graph for mode '{mode}' created successfully.")

            # Clean up the local cluster if created
            if client: client.close()
            return

        # Write output to disk. Up to here, computations are lazy
        write_dataset(out, cfg, client=client)

    except Exception:
        raise
    finally:
        # Clean up the local cluster if created
        if client:
            try:
                cluster = getattr(client, 'cluster', None)
                client.close()
                
                if cluster:
                    cluster.close()
            except (asyncio.CancelledError, Exception):
                pass  # Suppress shutdown noise

    # --- End Main Computation Block and Profiling ---
    end_time = time.monotonic()
    duration = end_time - start_time
    # Print the profiling information
    print(f"\n[sbudget] PROFILE: Total time elapsed: {duration:.2f} seconds")


def _add_bool_pair(p, name, dest, help_true, help_false):
    """Utility to create mutually exclusive tri-state boolean flags (True, False, None)."""
    g = p.add_mutually_exclusive_group()
    g.add_argument(f'--{name}', dest=dest, action='store_true', help=help_true)
    g.add_argument(f'--no-{name}', dest=dest, action='store_false', help=help_false)
    p.set_defaults(**{dest: None})


def _csv_or_list(s):
    # Accept "a,b,c" or space-separated "a b c"
    if s is None:
        return None
    if isinstance(s, (list, tuple)):
        return list(s)
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return s.split()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="budget",
                                     description="Non-hydrostatic spectral energy budget")
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- compute ----
    p_compute = sub.add_parser(
        "compute",
        help="Compute spectral energy budget or scale-to-scale energy transfers"
    )
    p_compute.add_argument("config", type=Path, help="Path to YAML configuration file.")

    # Input
    p_compute.add_argument("--input-path", help="Path to input dataset (overrides YAML).")
    p_compute.add_argument("--dims", type=_csv_or_list,
                           help="Comma- or space-separated list of dimensions, e.g. 'z,lat,lon'.")
    p_compute.add_argument("--engine", choices=["h5netcdf", "netcdf4", "scipy"],
                           help="NetCDF engine to use when reading files.")

    # Output
    p_compute.add_argument("--output-path", help="Output file path (overrides YAML).")
    p_compute.add_argument("--store", choices=["netcdf", "zarr"],
                           help="Output format for results.")
    _add_bool_pair(p_compute, "overwrite", "overwrite",
                   "Overwrite existing output file.", "Do not overwrite existing output.")

    # Compute
    p_compute.add_argument("--mode", choices=["spectral_budget", "scale_transfer"],
                           help="Select computation mode: 'spectral_budget' or 'scale_transfer'.")
    p_compute.add_argument("--scales", type=_csv_or_list,
                           help="Target horizontal wavelengths in meters, e.g. '1000,5000,10000'.")
    p_compute.add_argument("--levels", type=_csv_or_list,
                           help="Vertical levels in dataset units, e.g. '1000,5000,10000'.")

    p_compute.add_argument("--norm", choices=["ortho", "none"],
                           help="FFT normalization (use 'none' to disable normalization).")
    p_compute.add_argument("--dx", type=float, help="Grid spacing in x-direction (meters).")
    p_compute.add_argument("--dy", type=float, help="Grid spacing in y-direction (meters).")
    p_compute.add_argument("--transfer-form", choices=["invariant", "flux", "conservative"],
                           help="Formulation of transfer term to compute.")

    # user-exposed argument:
    p_compute.add_argument(
        "--chunk-size", type=float,
        help="Target per-chunk size in MB (approximate). Used for adaptive rechunking."
    )

    p_compute.add_argument(
        "--scheduler",
        choices=["threads", "distributed"],
        help="Execution backend for Dask computations."
    )
    
    p_compute.add_argument(
        "--n-workers", type=int,
        help="Number of Dask workers (overrides auto-detection)."
    )
    
    p_compute.add_argument(
        "--threads-per-worker", type=int,
        help="Number of threads per Dask worker (overrides auto-detection)."
    )

    _add_bool_pair(p_compute, "cumulative", "cumulative",
                   "Enable cumulative spectral sums.", "Disable cumulative spectral sums.")

    _add_bool_pair(p_compute, "rechunk-spatial", "rechunk_spatial",
                   "Force single spatial chunks for FFTs (recommended).",
                   "Skip spatial rechunking (faster, less stable for FFT).")

    _add_bool_pair(p_compute, "client", "client",
                   "Execute local cluster with active dask client (default).",
                   "Execute with local threads.")

    p_compute.add_argument("--dry-run", action="store_true",
                           help="Build Dask graph but skip final computation and I/O.")

    # Variable name overrides — added help for clarity
    var_help = "Override variable name in input dataset."
    p_compute.add_argument("--var-u", help=f"Zonal wind component. {var_help}")
    p_compute.add_argument("--var-v", help=f"Meridional wind component. {var_help}")
    p_compute.add_argument("--var-w", help=f"Vertical wind component. {var_help}")
    p_compute.add_argument("--var-theta", help=f"Potential temperature. {var_help}")
    p_compute.add_argument("--var-pressure", help=f"Atmospheric pressure. {var_help}")
    p_compute.add_argument("--var-density", help=f"Density of air. {var_help}")
    p_compute.add_argument("--var-temperature", help=f"Temperature of air. {var_help}")
    p_compute.add_argument("--var-divergence", help=f"horizontal divergence. {var_help}")
    p_compute.add_argument("--var-vorticity", help=f"Vertical component of "
                                                   f"relative vorticity. {var_help}")

    p_compute.set_defaults(func=_cmd_compute)

    args = parser.parse_args(argv)

    # optional echo of scheduler if present
    if hasattr(args, "scheduler") and args.scheduler:
        print(f"[sbudget] scheduler={args.scheduler}")

    args.func(args)


if __name__ == "__main__":
    main()
