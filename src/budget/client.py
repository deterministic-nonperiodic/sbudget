import argparse
import time
from pathlib import Path

import xarray as xr

from .budget import compute_budget
from .cf_coords import _cf_guess
from .chunking_tools import auto_select_cache_mode
from .config import load_config, apply_overrides
from .inter_scale_transfers import inter_scale_kinetic_energy_transfer
from .io_utils import open_dataset, write_dataset


def _report_var_existence(raw: xr.Dataset, cfg) -> str:
    """Create a plain-text report showing whether configured variables exist in the raw dataset.

    Checks both the *actual* names provided under cfg.variables and, after renaming,
    validates the presence of the *logical* names our code expects.
    """
    lines = []
    mapping = cfg.variables or {}
    # 1) Check actual names against the raw dataset
    lines.append("Configured variables (actual names) in file:\n")
    for logical, actual in mapping.items():
        exists = (actual in raw.data_vars) or (actual in raw.coords)
        status = "OK" if exists else "MISSING"
        hint = ""
        if not exists:
            guess = _cf_guess(raw, logical)
            if guess:
                hint = f"  (hint: possible '{logical}' → '{guess}' by CF)"
        lines.append(f"  - {logical:11s} ← {actual:20s} : {status}{hint}\n")

    # 2) Minimal logical requirements after renaming
    required = ["u", "v", "w"]
    lines.append("Logical requirements: ")
    for key in required:
        lines.append(f"  - {key} : required")
    lines.append("  - theta : preferred (else need pressure + temperature)")

    return " ".join(lines)


def _cmd_compute(args) -> None:
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    # Normalize/alias modes (backward compatibility)
    _MODE_ALIASES = {
        "spectral": "spectral_budget",
        "physical": "scale_transfer",
    }
    mode_in = str(cfg.compute.mode).strip()
    mode = _MODE_ALIASES.get(mode_in, mode_in)
    if mode_in in _MODE_ALIASES:
        print(f"[budget] NOTE: mode '{mode_in}' is deprecated; use '{mode}'")

    print(f"[budget] RUNNING MODE: {mode}")

    # store start time for profiling
    start_time = time.monotonic()

    # Open dataset with normalized variable/dimension names
    ds = open_dataset(cfg)

    if mode == "spectral_budget":
        print("[budget] Starting spectral budget calculation...")
        out = compute_budget(ds, cfg)
        print("[budget] Spectral budget calculation complete.")
    elif mode == "scale_transfer":
        print("[budget] Starting inter-scale transfer calculation...")

        # --- Auto-select cache mode if not explicitly given ---
        if not getattr(cfg.compute, "cache_mode", None):
            cfg.compute.cache_mode = auto_select_cache_mode(ds, working_set_multiplier=5)
        else:
            print(f"[budget] Using cache mode: {cfg.compute.cache_mode}")

        kwargs = {
            "scales": getattr(cfg.compute, "scales", None),
            "ls_chunk_size": 1,  # write one scale at a time to limit memory use
            "allow_rechunking": cfg.compute.dask_allow_rechunk,
            "chunksizes": cfg.compute.chunksizes,
            "cache_mode": cfg.compute.cache_mode,
            "verbose": True
        }

        out = inter_scale_kinetic_energy_transfer(ds, **kwargs)
        print("[budget] Inter-scale transfer calculation complete.")
    else:
        raise ValueError(f"Unknown compute.mode='{cfg.compute.mode}'. "
                         f"Use 'spectral_budget' or 'scale_transfer'.")

    # Write output to disk
    write_dataset(out, cfg)

    # --- End Main Computation Block and Profiling ---
    end_time = time.monotonic()
    duration = end_time - start_time

    # Print the profiling information
    print(f"\n[budget] PROFILE: Total time elapsed: {duration:.2f} seconds")


def _cmd_inspect(args) -> None:
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    # Open *raw* dataset to validate configured variable names
    p = cfg.input.path
    engine = getattr(cfg.input, "engine", None)
    if str(p).endswith(".zarr"):
        raw = xr.open_zarr(p, chunks="auto")
    elif str(p).endswith(".nc"):
        raw = xr.open_mfdataset(p, chunks="auto", engine=engine)
    else:
        raw = xr.open_mfdataset(p, chunks="auto")

    report = _report_var_existence(raw, cfg)

    # Also open the normalized (renamed) view
    ds = open_dataset(cfg)

    print(20 * "===" + "Input dataset (raw)" + 20 * "===")
    print(raw)
    print(20 * "===" + "Variable check" + 20 * "===")
    print(report)
    print(20 * "===" + "Dataset (normalized logical names)" + 20 * "===")
    print(ds)
    print(20 * "===" + "I/O configuration" + 20 * "===")
    print(cfg)


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
        help="Compute spectral or scale-dependent energy budgets and write results to disk."
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

    _add_bool_pair(p_compute, "dask-allow-rechunk", "dask_allow_rechunk",
                   "Allow Dask to rechunk automatically during computation.",
                   "Prevent automatic rechunking (for strict memory control).")

    # user-exposed argument:
    p_compute.add_argument(
        "--chunksizes", type=float,
        help="Target per-chunk size in MB (approximate). Used for adaptive rechunking."
    )

    p_compute.add_argument(
        "--cache-mode",
        choices=["smart", "disk", "disk_grouped", "hybrid"],
        default="hybrid",
        help=(
            "Caching strategy for intermediate fields. Scope --mode='scale_transfer':\n"
            "  'smart'        → Keep recent results in memory (fastest, but high RAM use).\n"
            "  'disk'         → Store each shift as a separate on-disk Zarr file (safe, slower I/O).\n"
            "  'disk_grouped' → Reuse shared Zarr stores for multiple shifts (efficient for large runs).\n"
            "  'hybrid'       → Adaptive mode: keep data in memory until nearing the memory limit, "
            "then spill to grouped on-disk cache automatically (recommended)."
        ),
    )

    p_compute.add_argument(
        "--scheduler",
        choices=["threads", "processes", "distributed"],
        help="Execution backend for Dask computations."
    )

    # Variable name overrides — added help for clarity
    var_help = "Override variable name in input dataset (if it differs from config)."
    p_compute.add_argument("--var-u", help=f"Zonal wind variable. {var_help}")
    p_compute.add_argument("--var-v", help=f"Meridional wind variable. {var_help}")
    p_compute.add_argument("--var-w", help=f"Vertical wind variable. {var_help}")
    p_compute.add_argument("--var-theta", help=f"Potential temperature variable. {var_help}")
    p_compute.add_argument("--var-pressure", help=f"Pressure variable. {var_help}")
    p_compute.add_argument("--var-density", help=f"Density variable. {var_help}")
    p_compute.add_argument("--var-temperature", help=f"Temperature variable. {var_help}")
    p_compute.add_argument("--var-divergence", help=f"Divergence variable. {var_help}")
    p_compute.add_argument("--var-vorticity", help=f"Vorticity variable. {var_help}")

    p_compute.set_defaults(func=_cmd_compute)

    # ---- inspect ----
    p_inspect = sub.add_parser("inspect", help="Print dataset and config summary")
    p_inspect.add_argument("config", type=Path, help="Path to YAML config")
    # same override flags help diagnose
    p_inspect.add_argument("--input-path")
    p_inspect.add_argument("--dims", type=_csv_or_list)
    p_inspect.add_argument("--engine", choices=["h5netcdf", "netcdf4", "scipy"])
    p_inspect.add_argument("--levels", type=_csv_or_list,
                           help="Levels in vertical axis units, e.g. '1000,5000,10000'")
    p_inspect.add_argument("--mode", choices=["spectral_budget", "scale_transfer"])

    p_inspect.set_defaults(func=_cmd_inspect)

    args = parser.parse_args(argv)

    # optional echo of scheduler if present
    if hasattr(args, "scheduler") and args.scheduler:
        print(f"[budget] scheduler={args.scheduler}")

    args.func(args)


if __name__ == "__main__":
    main()
