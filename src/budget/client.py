import argparse
import time
from pathlib import Path

import xarray as xr

from .budget import compute_budget
from .cf_coords import _cf_guess
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
        print("[budget] Constructed dask graph for spectral budget calculation")
    elif mode == "scale_transfer":
        print("[budget] Starting inter-scale transfer calculation...")

        # --- Auto-select cache mode if not explicitly given ---
        kwargs = {
            "scales": getattr(cfg.compute, "scales", None),
            "ls_chunk_size": 1,  # write one scale at a time to limit memory use
            "allow_rechunking": cfg.compute.dask_allow_rechunk,
            "chunk_size": cfg.compute.chunk_size,
            "verbose": True
        }

        out = inter_scale_kinetic_energy_transfer(ds, **kwargs)
    else:
        raise ValueError(f"Unknown compute.mode='{cfg.compute.mode}'. "
                         f"Use 'spectral_budget' or 'scale_transfer'.")

    # Write output to disk. Up to here, computations are lazy
    write_dataset(out, cfg)

    # --- End Main Computation Block and Profiling ---
    end_time = time.monotonic()
    duration = end_time - start_time

    # Print the profiling information
    print(f"\n[budget] PROFILE: Total time elapsed: {duration:.2f} seconds")


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

    _add_bool_pair(p_compute, "dask-allow-rechunk", "dask_allow_rechunk",
                   "Allow Dask to rechunk automatically during computation.",
                   "Prevent automatic rechunking (for strict memory control).")

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
        print(f"[budget] scheduler={args.scheduler}")

    args.func(args)


if __name__ == "__main__":
    main()
