from dataclasses import dataclass
import yaml


@dataclass
class IOConfig:
    path: str
    dims: list[str]  # [z, lat, lon] or [z, y, x]
    time: str | None = None
    engine: str | None = None  # optional: h5netcdf, netcdf4, or scipy for .nc files


@dataclass
class OutputConfig:
    path: str
    store: str = "netcdf"  # or "netcdf"
    overwrite: bool = False


@dataclass
class ComputeConfig:
    mode: str | None = "spectral"  # or "physical"
    scales: list[float] | None = None  # list of scales for physical-space mode
    norm: str | None = None  # None or "ortho"
    dx: float | None = None
    dy: float | None = None
    cumulative: bool = True  # default True per request
    rechunk_spatial: bool = True  # ensure (y,x) are single chunks before FFTs
    dask_allow_rechunk: bool = True  # allow apply_ufunc to rechunk as fallback
    scheduler: str | None = None
    transfer_form: str | None = "flux"  # e.g. "invariant", "flux"


@dataclass
class Config:
    input: IOConfig
    output: OutputConfig
    compute: ComputeConfig
    variables: dict  # mapping: logical name -> dataset var name

    def __repr__(self) -> str:
        return f"Config(input={self.input.path}, out={self.output.path})"


def load_config(path) -> Config:
    raw = yaml.safe_load(open(path))
    return Config(
        input=IOConfig(**raw["input"]),
        output=OutputConfig(**raw["output"]),
        compute=ComputeConfig(**raw.get("compute", {})),
        variables=raw.get("variables", {}),
    )


def _to_float_list(xs):
    if xs is None:
        return None
    out = []
    for x in xs:
        try:
            out.append(float(x))
        except Exception:
            pass
    return out


def _set_nested(obj, path, value):
    """Set obj['a']['b'] or obj.a.b to value; works for dicts or attribute objects."""
    if value is None:
        return
    cur = obj
    for key in path[:-1]:
        if isinstance(cur, dict):
            cur = cur.setdefault(key, {})
        else:
            cur = getattr(cur, key)
    last = path[-1]
    if isinstance(cur, dict):
        cur[last] = value
    else:
        setattr(cur, last, value)


def apply_overrides(cfg, args):
    """
    Apply argparse overrides into cfg (yaml loaded as dict or SimpleNamespace/dataclass).
    Returns the mutated cfg.
    """
    # Normalize norm: user may pass 'none' to clear
    norm = None if getattr(args, "norm", None) in (None, "none") else args.norm

    # input
    _set_nested(cfg, ["input", "path"], args.input_path)
    if args.dims:
        _set_nested(cfg, ["input", "dims"], args.dims)
    _set_nested(cfg, ["input", "engine"], args.engine)

    # output
    _set_nested(cfg, ["output", "path"], args.output_path)
    _set_nested(cfg, ["output", "store"], args.store)
    if args.overwrite is not None:
        _set_nested(cfg, ["output", "overwrite"], bool(args.overwrite))

    # compute
    _set_nested(cfg, ["compute", "mode"], args.mode)
    if args.scales:
        _set_nested(cfg, ["compute", "scales"], _to_float_list(args.scales))
    if norm is not None or getattr(args, "norm", None) == "none":
        _set_nested(cfg, ["compute", "norm"], norm)
    if args.dx is not None:
        _set_nested(cfg, ["compute", "dx"], float(args.dx))
    if args.dy is not None:
        _set_nested(cfg, ["compute", "dy"], float(args.dy))
    if args.cumulative is not None:
        _set_nested(cfg, ["compute", "cumulative"], bool(args.cumulative))
    _set_nested(cfg, ["compute", "transfer_form"], args.transfer_form)
    if args.rechunk_spatial is not None:
        _set_nested(cfg, ["compute", "rechunk_spatial"], bool(args.rechunk_spatial))
    if args.dask_allow_rechunk is not None:
        _set_nested(cfg, ["compute", "dask_allow_rechunk"], bool(args.dask_allow_rechunk))
    _set_nested(cfg, ["compute", "scheduler"], args.scheduler)

    # variables
    _set_nested(cfg, ["variables", "u"], args.var_u)
    _set_nested(cfg, ["variables", "v"], args.var_v)
    _set_nested(cfg, ["variables", "w"], args.var_w)
    _set_nested(cfg, ["variables", "theta"], args.var_theta)
    _set_nested(cfg, ["variables", "pressure"], args.var_pressure)
    _set_nested(cfg, ["variables", "temperature"], args.var_temperature)
    _set_nested(cfg, ["variables", "divergence"], args.var_divergence)
    _set_nested(cfg, ["variables", "vorticity"], args.var_vorticity)

    return cfg
