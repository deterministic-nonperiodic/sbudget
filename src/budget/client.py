import argparse
from pathlib import Path
import xarray as xr
from .config import load_config
from .budget import compute_budget
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
        "div": {"standard_names": {"divergence_of_wind"}, "units": {"s-1"}},
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
    if args.no_accumulate:
        # override from CLI
        cfg.compute.cumulative = False
    ds = open_dataset(cfg)
    out = compute_budget(ds, cfg)
    write_dataset(out, cfg)
    print(f"Wrote: {cfg.output.path}")


def _cmd_inspect(args) -> None:
    cfg = load_config(args.config)
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="budget",
                                     description="Non-hydrostatic spectral energy budget")
    sub = parser.add_subparsers(dest="command", required=True)

    p_compute = sub.add_parser("compute", help="Compute energy budget and write output")
    p_compute.add_argument("config", type=Path, help="Path to YAML config")
    p_compute.add_argument("--no-accumulate", action="store_true",
                           help="Disable cumulative spectra")
    p_compute.set_defaults(func=_cmd_compute)

    p_inspect = sub.add_parser("inspect", help="Print dataset and config summary")
    p_inspect.add_argument("config", type=Path, help="Path to YAML config")
    p_inspect.set_defaults(func=_cmd_inspect)

    args = parser.parse_args(argv)
    # echo scheduler choice if any
    try:
        sched = getattr(load_config(args.config), 'compute').scheduler
        if sched:
            print(f"[budget] scheduler={sched}")
    except Exception:
        pass
    args.func(args)


if __name__ == "__main__":
    main()
