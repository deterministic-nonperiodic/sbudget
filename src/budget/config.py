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
