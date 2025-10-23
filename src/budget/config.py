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
    levels: list[float] | None = None  # list of levels for scale_transfer mode
    norm: str | None = None  # None or "ortho"
    dx: float | None = None
    dy: float | None = None
    transfer_form: str | None = "flux"  # e.g. "invariant", "flux"
    cumulative: bool = True  # default True per request
    rechunk_spatial: bool = True  # ensure (y,x) are single chunks before FFTs
    dask_allow_rechunk: bool = True  # allow apply_ufunc to rechunk as fallback
    scheduler: str | None = None


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


def _get(args, name, default=None):
    return getattr(args, name, default)


def apply_overrides(cfg, args):
    """Apply argparse overrides into cfg (yaml dict or object). Missing args are ignored."""
    norm_cli = _get(args, "norm", None)
    norm = None if norm_cli in (None, "none") else norm_cli

    # input
    _set_nested(cfg, ["input", "path"], _get(args, "input_path"))
    dims = _get(args, "dims")
    if dims: _set_nested(cfg, ["input", "dims"], dims)
    _set_nested(cfg, ["input", "engine"], _get(args, "engine"))

    # output
    _set_nested(cfg, ["output", "path"], _get(args, "output_path"))
    _set_nested(cfg, ["output", "store"], _get(args, "store"))
    ow = _get(args, "overwrite")
    if ow is not None:
        _set_nested(cfg, ["output", "overwrite"], bool(ow))

    # compute
    _set_nested(cfg, ["compute", "mode"], _get(args, "mode"))
    scales = _get(args, "scales")
    if scales: _set_nested(cfg, ["compute", "scales"], _to_float_list(scales))

    levels = _get(args, "levels")
    if levels: _set_nested(cfg, ["compute", "levels"], _to_float_list(levels))

    if norm is not None or norm_cli == "none":
        _set_nested(cfg, ["compute", "norm"], norm)
    dx = _get(args, "dx")
    dy = _get(args, "dy")
    if dx is not None: _set_nested(cfg, ["compute", "dx"], float(dx))
    if dy is not None: _set_nested(cfg, ["compute", "dy"], float(dy))

    cum = _get(args, "cumulative")
    if cum is not None: _set_nested(cfg, ["compute", "cumulative"], bool(cum))
    _set_nested(cfg, ["compute", "transfer_form"], _get(args, "transfer_form"))

    rs = _get(args, "rechunk_spatial")
    if rs is not None: _set_nested(cfg, ["compute", "rechunk_spatial"], bool(rs))
    dar = _get(args, "dask_allow_rechunk")
    if dar is not None: _set_nested(cfg, ["compute", "dask_allow_rechunk"], bool(dar))
    _set_nested(cfg, ["compute", "scheduler"], _get(args, "scheduler"))

    # variables
    _set_nested(cfg, ["variables", "u"], _get(args, "var_u"))
    _set_nested(cfg, ["variables", "v"], _get(args, "var_v"))
    _set_nested(cfg, ["variables", "w"], _get(args, "var_w"))
    _set_nested(cfg, ["variables", "theta"], _get(args, "var_theta"))
    _set_nested(cfg, ["variables", "pressure"], _get(args, "var_pressure"))
    _set_nested(cfg, ["variables", "temperature"], _get(args, "var_temperature"))
    _set_nested(cfg, ["variables", "divergence"], _get(args, "var_divergence"))
    _set_nested(cfg, ["variables", "vorticity"], _get(args, "var_vorticity"))

    return cfg
