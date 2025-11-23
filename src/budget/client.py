import argparse
import os
import re
import resource
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import dask
import psutil
from dask.distributed import Client, LocalCluster

from .budget import compute_budget
from .config import load_config, apply_overrides
from .inter_scale_transfers import inter_scale_kinetic_energy_transfer
from .io_utils import open_dataset, write_dataset
from .memory_manager import TARGET_WORKER_MEM_GB

# Suppress Dask warnings about large graphs
warnings.filterwarnings(
    "ignore",
    message=r"Sending large graph of size .* MiB\.",
    category=UserWarning,
    module=r'distributed\.client'
)

# New constant for maximum file descriptors
_MAX_FILE_DESCRIPTORS = 8192
_THREAD_LIMIT = 64  # Arbitrary threshold for switching to distributed


# --- Helper Function for SLURM Memory Detection ---
def _parse_slurm_memory() -> Optional[float]:
    """
    Parses SLURM memory variables (SLURM_MEM_PER_NODE, SLURM_MEM_PER_CPU, SLURM_MEM_PER_GPU)
    to determine the total memory allocated to the job, in GiB.

    Returns the total allocated memory in GiB, or None if no SLURM memory variable is found.
    """
    n_tasks = int(os.environ.get("SLURM_NTASKS", 1))

    # 1. SLURM_MEM_PER_NODE (Total memory for the whole job/node)
    if "SLURM_MEM_PER_NODE" in os.environ:
        # Value is usually in MiB
        return int(os.environ["SLURM_MEM_PER_NODE"]) / 1024.0

    # 2. SLURM_MEM_PER_CPU (Memory per task/CPU)
    if "SLURM_MEM_PER_CPU" in os.environ:
        # Value is usually in MiB
        mem_per_task = int(os.environ["SLURM_MEM_PER_CPU"]) / 1024.0
        return n_tasks * mem_per_task

    # 3. SLUM_MEM (General total memory request, typically in GB or MB)
    # This captures the --mem=256G request
    if "SLURM_MEM" in os.environ:
        mem_str = os.environ["SLURM_MEM"].upper()
        match = re.search(r'(\d+)([MG])', mem_str)
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            if unit == 'G':
                return value
            elif unit == 'M':
                return value / 1024.0
        # If it's a simple number (usually MB on some clusters)
        try:
            # Assume MB if no unit is given
            return float(os.environ["SLURM_MEM"]) / 1024.0
        except ValueError:
            pass

    return None


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


def _detect_resources() -> Dict[str, Union[int, float, bool]]:
    """Detects allocated resources, prioritizing SLURM environment variables."""
    is_slurm = "SLURM_JOB_ID" in os.environ

    # 1. Thread/Worker Allocation
    if is_slurm:
        n_tasks = int(os.environ.get("SLURM_NTASKS", 1))
        cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
        n_threads = cpus_per_task
        n_workers = n_tasks
    else:
        total_cores = psutil.cpu_count(logical=True) or 4
        n_workers = total_cores
        n_threads = 1

    # 2. Total Memory Allocation (in GiB)
    slurm_mem = _parse_slurm_memory()

    if slurm_mem is not None:
        # Use SLURM detected memory pool
        total_mem_gb = slurm_mem
        if is_slurm:
            print(f"[budget] Detected SLURM Memory Pool: {total_mem_gb:.2f} GiB")
    else:
        # Local Machine: Total system memory (Fallback)
        total_mem_gb = psutil.virtual_memory().total / (1024 ** 3)

    return {
        "n_workers": n_workers,
        "threads_per_worker": n_threads,
        "total_mem_gb": total_mem_gb,
        "is_slurm": is_slurm,
    }


def auto_configure_dask(cfg: Optional[Any] = None) -> Tuple[int, int]:
    """HPC-aware configuration of Dask global settings and cluster parameters."""
    resources = _detect_resources()

    # --- Chunk Size Heuristic ---
    total_mem_gb = resources["total_mem_gb"]
    if total_mem_gb < 64:
        default_chunk_mb = 512
    else:
        default_chunk_mb = 2048

    chunk_mb = float(getattr(cfg.compute, "chunk_size", None) or default_chunk_mb)

    # --- Configure Dask Global Settings (Memory Discipline) ---
    dask.config.set({
        "array.chunk-size": f"{chunk_mb}MB",
        "array.slicing.split_large_chunks": False,
        # Pass explicit memory limit to Dask config
        "distributed.worker.memory.limit": f'{TARGET_WORKER_MEM_GB:.2f}GB',
        "distributed.worker.memory.target": False,
        "distributed.worker.memory.spill": False,
        "distributed.worker.memory.pause": 0.95,
        "distributed.worker.memory.terminate": 0.985,
        "optimization.fuse.ave-width": 16,
        "optimization.fuse.max-width": 128,
    })

    return int(resources["n_workers"]), int(resources["threads_per_worker"])


def init_dask_client(cfg: Any, scheduler_address: Optional[str] = None) -> Client:
    """
    Create and return a Dask client, supporting 'threads' or 'distributed' modes.
    """
    _increase_fd_limit()  # Call the FD limit helper

    scheduler = getattr(cfg.compute, "scheduler", "threads")

    num_workers_detected, threads_per_worker_detected = auto_configure_dask(cfg)

    total_mem_gb = _detect_resources()["total_mem_gb"]

    # Calculate MAX workers allowed by the higher memory budget (3.0 GB)
    max_workers_by_mem = int(total_mem_gb // TARGET_WORKER_MEM_GB)

    # The final worker count must be the minimum of the requested count and the memory allowed.
    num_workers = min(num_workers_detected, max_workers_by_mem)
    threads_per_worker = threads_per_worker_detected

    if num_workers == 0:
        raise MemoryError(f"Insufficient memory ({total_mem_gb:.2f}GB) to start even one worker "
                          f"with required budget ({TARGET_WORKER_MEM_GB}GB).")

    print(
        f"[budget] Auto-Dask → workers={num_workers} | "
        f"threads={threads_per_worker} | "
        f"Total memory={total_mem_gb:.2f}GB ")

    # Calculate the total cores allocated/detected
    total_cores = num_workers * threads_per_worker

    if scheduler == "distributed":
        # MODE 1: DISTRIBUTED (HPC, large local scale-out)
        if scheduler_address:
            return Client(scheduler_address)

        cluster = LocalCluster(
            n_workers=num_workers,
            threads_per_worker=threads_per_worker,
            processes=True,
            memory_limit=f'{TARGET_WORKER_MEM_GB:.2f}GB'
        )
        return Client(cluster)

    elif scheduler == "threads":
        # MODE 2: THREADS (Small local, minimal overhead)
        if total_cores > _THREAD_LIMIT:
            print(
                f"[budget] WARNING: Total cores ({total_cores}) exceeds thread limit "
                f"({_THREAD_LIMIT}).")
            print(
                f"[budget]         Using 'distributed' scheduler instead for multiprocessing stability.")

            # Note: cfg.compute.scheduler change is handled in the original code logic path
            cfg.compute.scheduler = "distributed"

            # Recursively call to run the distributed logic with updated worker count
            return init_dask_client(cfg, scheduler_address)

        # Single process, multi-thread
        cluster = LocalCluster(
            n_workers=1,
            threads_per_worker=total_cores,
            processes=False,
            memory_limit=f'{TARGET_WORKER_MEM_GB:.2f}GB'
        )
        return Client(cluster)

    else:
        raise ValueError(
            f"Unknown or unsupported scheduler type: {scheduler}. Use 'threads' or 'distributed'.")


def _cmd_compute(args) -> None:
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    # ==============================================
    # INITIALIZE DASK CLIENT
    # ==============================================
    client = init_dask_client(cfg, scheduler_address=None)

    if client is not None:
        print(f"[sbudget] Dask Dashboard: {client.dashboard_link}")
        print(f"[sbudget] Using client: {client}")

    # -------------------------------------------------------------------
    # Open dataset with normalized variable/dimension names
    ds = open_dataset(cfg, verbose=False)

    # Normalize/alias modes (backward compatibility)
    _MODE_ALIASES = {
        "spectral": "spectral_budget",
        "physical": "scale_transfer",
    }
    mode_in = str(cfg.compute.mode).strip()
    mode = _MODE_ALIASES.get(mode_in, mode_in)
    if mode_in in _MODE_ALIASES:
        print(f"[sbudget] NOTE: mode '{mode_in}' is deprecated; use '{mode}'")

    print(f"[sbudget] RUNNING MODE: {mode}")

    # store start time for profiling
    start_time = time.monotonic()

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

    # Write output to disk. Up to here, computations are lazy
    write_dataset(out, cfg, client=client)

    # Clean up the local cluster if created
    if client and hasattr(client, 'cluster') and client.cluster:
        client.close()
        client.cluster.close()

    # --- End Main Computation Block and Profiling ---
    end_time = time.monotonic()
    duration = end_time - start_time
    # Print the profiling information
    print(f"\n[sbudget] PROFILE: Total time elapsed: {duration:.2f} seconds")


def _add_bool_pair(p, name, dest, help_true, help_false):
    g = p.add_mutually_exclusive_group()
    g.add_argument(f"--{name}", dest=dest, action="store_true", help=help_true)
    g.add_argument(f"--no-{name}", dest=dest, action="store_false", help=help_false)
    p.set_defaults(**{dest: None})  # tri-state: None means "no override"


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

    _add_bool_pair(p_compute, "cumulative", "cumulative",
                   "Enable cumulative spectral sums.", "Disable cumulative spectral sums.")
    p_compute.add_argument("--transfer-form", choices=["invariant", "flux", "conservative"],
                           help="Formulation of transfer term to compute.")

    _add_bool_pair(p_compute, "rechunk-spatial", "rechunk_spatial",
                   "Force single spatial chunks for FFTs (recommended).",
                   "Skip spatial rechunking (faster, less stable for FFT).")

    # user-exposed argument:
    p_compute.add_argument(
        "--chunk-size", type=float,
        help="Target per-chunk size in MB (approximate). Used for adaptive rechunking."
    )

    p_compute.add_argument(
        "--scheduler",
        choices=["threads", "processes", "distributed"],
        help="Execution backend for Dask computations."
    )

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
