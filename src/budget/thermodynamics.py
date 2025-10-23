import xarray as xr
import numpy as np
from . import constants as cn
from .cf_coords import _is_z

from typing import Union, Literal


# ---------------------------
# Helpers
# ---------------------------

def _infer_vertical_dim(pressure: xr.DataArray, temperature: xr.DataArray,
                        vertical_dim: str | None):
    """Pick the vertical dimension. If not given, infer from pressure (1D) or temperature."""
    if vertical_dim is not None:
        return vertical_dim
    if pressure.ndim == 1:
        # use its only dim (e.g., "level", "p", "z")
        return pressure.dims[0]
    # fallback: first shared dim
    common = [d for d in temperature.dims if
              d in pressure.dims and _is_z(str(d), temperature.coords)]
    if not common:
        raise ValueError("Cannot infer vertical dimension; please pass vertical_dim.")
    return common[0]


def _cumulative_trapezoidal(f: xr.DataArray, x: xr.DataArray,
                            dim: str, initial: float = 0.0):
    """
    Cumulative ∫ f(x) d(ln x) along dim using trapezoids, returning same length as input:
    result[0] = initial; result[i] = ∫_{x0}^{xi} f(x) d(ln x).
    """
    ln_x = np.log(x)

    # forward differences: Δ(x)_i = ln(p_{i+1}) - ln(p_i)
    dln_x = ln_x.shift({dim: -1}) - ln_x
    f_avg = 0.5 * (f + f.shift({dim: -1}))

    # increments aligned at i (exclude last NaN from shift)
    inc = (f_avg * dln_x).isel({dim: slice(0, -1)})

    # cumulative sum along dim and prepend the initial value at the first index
    cum = inc.cumsum(dim, skipna=True)

    # make a scalar "initial" with the same non-dim coords as a single slice
    init0 = xr.full_like(f.isel({dim: 0}), float(initial))
    out = xr.concat([init0, cum], dim=dim)

    # assign original coords for the output dim
    out = out.assign_coords({dim: f.coords[dim]})
    return out


# ---------------------------
# Core functions
# ---------------------------

def nh_pressure_derivative(f: xr.DataArray, p: xr.DataArray, zdim: str = "z",
                           edge_order: Literal[1, 2] = 2):
    # both f and p live on (..., z); this uses the actual dp/dz numerically
    df_dz = f.differentiate(zdim, edge_order=edge_order)
    dp_dz = p.differentiate(zdim, edge_order=edge_order)

    out = df_dz / dp_dz

    # optional: mask pathological spots where dp/dz ≈ 0 (inversions)
    tol = np.abs(dp_dz).median(dim=zdim) * 1e-6
    return out.where(np.abs(dp_dz) > tol)


def height_to_geopotential(height: Union[xr.DataArray | float]) -> Union[xr.DataArray | float]:
    r"""Φ = g R_e z / (R_e + z)"""
    return (cn.g * cn.earth_radius * height) / (cn.earth_radius + height)


def geopotential_to_height(geopotential: Union[xr.DataArray | float]) -> Union[
    xr.DataArray | float]:
    r"""z = Φ R_e / (g R_e - Φ)"""
    return (geopotential * cn.earth_radius) / (cn.g * cn.earth_radius - geopotential)


def exner_function(pressure: xr.DataArray, reference_pressure: float = cn.ps) -> xr.DataArray:
    r"""Π = (p / p0)^κ"""
    return (pressure / reference_pressure) ** cn.chi


def potential_temperature(pressure: xr.DataArray, temperature: xr.DataArray) -> xr.DataArray:
    r"""θ = T / Π"""
    return temperature / exner_function(pressure)


def static_stability(
        pressure: xr.DataArray,
        temperature: xr.DataArray,
        vertical_dim: str | None = None,
        edge_order: Literal[1, 2] = 2
) -> xr.DataArray:
    """
    Static stability σ.

    - If coordinate="z": σ = ( (Rd*T/p)**2 / g ) * d(ln θ)/dz
      (assumes the coordinate for `vertical_dim` is height in meters)
    - If coordinate="p": σ = - (Rd*T/p) * d(ln θ)/dp
    """
    vertical_dim = _infer_vertical_dim(pressure, temperature, vertical_dim)
    theta = potential_temperature(pressure, temperature)

    ddp_ln_theta = nh_pressure_derivative(np.log(theta), pressure,
                                          zdim=vertical_dim, edge_order=edge_order)

    return - cn.Rd * (temperature / pressure) * ddp_ln_theta


def density(pressure: xr.DataArray, temperature: xr.DataArray) -> xr.DataArray:
    r"""ρ = p / (R_d T)"""
    return pressure / (cn.Rd * temperature)


def specific_volume(pressure: xr.DataArray, temperature: xr.DataArray) -> xr.DataArray:
    r"""α = 1/ρ"""
    return 1.0 / density(pressure, temperature)


def vertical_velocity(omega: xr.DataArray, temperature: xr.DataArray,
                      pressure: xr.DataArray) -> xr.DataArray:
    r"""
    Convert ω = Dp/Dt (Pa s⁻¹) to w = Dz/Dt (m s⁻¹) via hydrostatic:
      ω ≈ - ρ g w  →  w ≈ - ω / (ρ g)
    """
    # Rely on xarray broadcasting by matching dims
    rho = density(pressure, temperature)
    return - omega / (cn.g * rho)


def pressure_vertical_velocity(w: xr.DataArray, temperature: xr.DataArray,
                               pressure: xr.DataArray) -> xr.DataArray:
    r"""
    Convert w = Dz/Dt (m s⁻¹) → ω = Dp/Dt (Pa s⁻¹) via hydrostatic:
      ω ≈ - ρ g w
    """
    rho = density(pressure, temperature)
    return - cn.g * rho * w


def lorenz_parameter(
        pressure: xr.DataArray,
        theta: xr.DataArray,
        vertical_dim: str | None = None,
        edge_order: Literal[1, 2] = 2,
) -> xr.DataArray:
    """
    Lorenz static stability parameter γ:
      γ = - R_d Π / (p ∂θ/∂p)
    """
    vertical_dim = _infer_vertical_dim(pressure, theta, vertical_dim)

    ddp_theta = nh_pressure_derivative(theta, pressure, zdim=vertical_dim, edge_order=edge_order)

    return - cn.Rd * exner_function(pressure) / (pressure * ddp_theta)


def brunt_vaisala_squared(
        pressure: xr.DataArray,
        temperature: xr.DataArray,
        vertical_dim: str | None = None,
        edge_order: Literal[1, 2] = 2,
) -> xr.DataArray:
    """
    N^2 = (g/θ)^2 / γ   (using Lorenz parameter γ)
    """
    theta = potential_temperature(pressure, temperature)
    gamma = lorenz_parameter(pressure, theta, vertical_dim=vertical_dim, edge_order=edge_order)
    return (cn.g / theta) ** 2 / gamma
