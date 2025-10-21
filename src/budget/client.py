import argparse
from pathlib import Path

import xarray as xr

from .budget import compute_budget
from .config import load_config, apply_overrides
from .inter_scale_transfers import inter_scale_kinetic_energy_transfer
from .io_utils import open_dataset, write_dataset


def _cf_guess(ds: xr.Dataset, target: str) -> str | None:
    """Very light CF-based guess for a logical variable name.

    Looks at ``standard_name`` and common units to suggest a candidate when a
    configured variable is missing. This is *advisory* only.
    """
    cf_map = {
        "u": {"standard_names": {"eastward_wind"}, "units": {"m s-1", "m/s"}},
        "v": {"standard_names": {"northward_wind"}, "units": {"m s-1", "m/s"}},
        "w": {"standard_names": {"upward_air_velocity", "vertical_velocity_in_air"},
              "units": {"m s-1", "Pa s-1"}},
        "pressure": {"standard_names": {"air_pressure"}, "units": {"Pa", "pascal"}},
        "temperature": {"standard_names": {"air_temperature"}, "units": {"K", "kelvin"}},
        "theta": {"standard_names": {"air_potential_temperature"}, "units": {"K", "kelvin"}},
        "divergence": {"standard_names": {"divergence_of_wind"}, "units": {"s-1"}},
        "vorticity": {"standard_names": {"relative_vorticity"}, "units": {"s-1"}},
    }
    rule = cf_map.get(target)
    if rule is None:
        return None
    for name, da in ds.data_vars.items():
        std = str(da.attrs.get("standard_name", ""))
        units = str(da.attrs.get("units", ""))
        if std in rule["standard_names"] or any(u in units for u in rule["units"]):
            return name
    return None


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

    # Open dataset with normalized variable names
    ds = open_dataset(cfg)

    if cfg.compute.mode == "spectral":
        out = compute_budget(ds, cfg)
    else:
        control_dict = {
            "verbose": True,
            "scales": cfg.compute.scales,
            "ls_chunk_size": 1,  # write one scale at a time to limit memory use
        }
        out = inter_scale_kinetic_energy_transfer(ds, **control_dict)

        # rechunk output to match input dataset chunking
        if getattr(ds, "chunks", None) and "time" in ds.chunks:
            time_chunks = ds.chunks["time"]  # a tuple of chunk sizes
        else:
            time_chunks = "auto"

        out = out.chunk({"time": time_chunks, "length_scale": control_dict["ls_chunk_size"]})

    write_dataset(out, cfg)
    print(f"Wrote: {cfg.output.path}")


def _cmd_inspect(args) -> None:
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    # Open *raw* dataset to validate configured variable names
    p = cfg.input.path
    engine = getattr(cfg.input, "engine", None)
    if str(p).endswith(".zarr"):
        raw = xr.open_zarr(p, chunks="auto")
    elif str(p).endswith(".nc"):
        if engine:
            raw = xr.open_dataset(p, chunks="auto", engine=engine)
        else:
            raw = xr.open_dataset(p, chunks="auto")
    else:
        raw = xr.open_dataset(p, chunks="auto")

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
    p_compute = sub.add_parser("compute", help="Compute energy budget and write output")
    p_compute.add_argument("config", type=Path, help="Path to YAML config")

    # input
    p_compute.add_argument("--input-path")
    p_compute.add_argument("--dims", type=_csv_or_list, help="e.g. 'z,lat,lon' or 'z y x'")
    p_compute.add_argument("--engine", choices=["h5netcdf", "netcdf4", "scipy"])

    # output
    p_compute.add_argument("--output-path")
    p_compute.add_argument("--store", choices=["netcdf", "zarr"])
    _add_bool_pair(p_compute, "overwrite", "overwrite",
                   "Overwrite output", "Do not overwrite output")

    # compute
    p_compute.add_argument("--mode", choices=["spectral", "physical"], default="spectral")
    p_compute.add_argument("--scales", type=_csv_or_list,
                           help="Wavelengths in meters, e.g. '1000,5000,10000'")
    p_compute.add_argument("--norm", choices=["ortho", "none"],
                           help="FFT normalization ('none' to clear)")
    p_compute.add_argument("--dx", type=float)
    p_compute.add_argument("--dy", type=float)
    _add_bool_pair(p_compute, "cumulative", "cumulative",
                   "Enable cumulative spectra", "Disable cumulative spectra")
    p_compute.add_argument("--transfer-form", choices=["invariant", "flux", "conservative"])
    _add_bool_pair(p_compute, "rechunk-spatial", "rechunk_spatial",
                   "Ensure single spatial chunks for FFTs", "Do not rechunk spatial dims")
    _add_bool_pair(p_compute, "dask-allow-rechunk", "dask_allow_rechunk",
                   "Allow apply_ufunc to rechunk", "Disallow automatic rechunking")
    p_compute.add_argument("--scheduler", choices=["threads", "processes", "distributed"])

    # variables (name mapping overrides)
    p_compute.add_argument("--var-u")
    p_compute.add_argument("--var-v")
    p_compute.add_argument("--var-w")
    p_compute.add_argument("--var-theta")
    p_compute.add_argument("--var-pressure")
    p_compute.add_argument("--var-temperature")
    p_compute.add_argument("--var-divergence")
    p_compute.add_argument("--var-vorticity")

    p_compute.set_defaults(func=_cmd_compute)

    # ---- inspect ----
    p_inspect = sub.add_parser("inspect", help="Print dataset and config summary")
    p_inspect.add_argument("config", type=Path, help="Path to YAML config")
    # same override flags help diagnose
    p_inspect.add_argument("--input-path")
    p_inspect.add_argument("--dims", type=_csv_or_list)
    p_inspect.add_argument("--engine", choices=["h5netcdf", "netcdf4", "scipy"])
    p_inspect.set_defaults(func=_cmd_inspect)

    args = parser.parse_args(argv)

    # optional echo of scheduler if present
    try:
        sched = getattr(load_config(args.config), 'compute').scheduler  # your existing loader
        if sched:
            print(f"[budget] scheduler={sched}")
    except Exception:
        pass

    args.func(args)


if __name__ == "__main__":
    main()
