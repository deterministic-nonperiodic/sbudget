from typing import Union, Callable, Any

import numpy as np
import xarray as xr

from .cf_coords import get_spatial_dims, infer_grid_resolution, _is_geographic, _coord_is_degrees
from .constants import cp, Omega
from .numeric_tools import domain_mean, stack_vector, rotate_vector, isotropize
from .numeric_tools import horizontal_advection, horizontal_gradient
from .numeric_tools import horizontal_divergence, relative_vorticity
from .numeric_tools import horizontal_wavenumber_magnitude
from .numeric_tools import scalar_spectrum, scalar_cross_spectrum, vector_cross_spectrum
from .thermodynamics import potential_temperature, exner_function, density

# --------------------------------------------------------------------------------------------------
# Global spectral options
# --------------------------------------------------------------------------------------------------
# --- Global Metadata Configuration ---
_BUDGET_UNITS = "watt / kilogram"

_BUDGET_META = {
    "hke": ("horizontal_kinetic_energy_spectrum",
            "spectrum of horizontal kinetic energy per unit mass", "m**2 / s**2"),
    "rke": ("rotational_kinetic_energy_spectrum",
            "spectrum of rotational kinetic energy per unit mass", "m**2 / s**2"),
    "dke": ("divergent_kinetic_energy_spectrum",
            "spectrum of divergent kinetic energy per unit mass", "m**2 / s**2"),
    "cad": ("conversion_ape_dke",
            "conversion from available potential energy to kinetic energy", _BUDGET_UNITS),
    "pi_nke": ("nonlinear_hke_transfer",
               "Nonlinear spectral transfer of horizontal kinetic energy", _BUDGET_UNITS),
    "pi_lke": ("linear_hke_transfer",
               "Linear spectral transfer of horizontal kinetic energy due to Coriolis",
               _BUDGET_UNITS),
    "vfd_dke": ("vertical_dke_flux_divergence",
                "vertical flux divergence of horizontal kinetic energy", _BUDGET_UNITS),
    "vf_hke": ("vertical_dke_flux",
               "vertical flux of horizontal kinetic energy", _BUDGET_UNITS),
    "vfd_pres": ("pressure_flux_divergence",
                 "vertical divergence of pressure work", _BUDGET_UNITS),
    "vf_pres": ("hke_pressure_vertical_flux",
                "vertical flux of horizontal kinetic energy (pressure work)", _BUDGET_UNITS),
    "div_hke": ("hke_divergence",
                "divergence contribution to HKE budget", _BUDGET_UNITS),
    "j_hke": ("adiabatic_nonconservative",
              "adiabatic nonconservative contribution to HKE budget", _BUDGET_UNITS),
}


def budget_metadata(func: Callable) -> Callable:
    """
    Decorator that applies standard_name, long_name, and units attributes
    to the resulting DataArray based on the 'name' argument passed to the function.
    """

    def wrapper(*args: Any, **kwargs: Any) -> xr.DataArray:
        # Execute the wrapped function to get the result DataArray
        result_da: xr.DataArray = func(*args, **kwargs)

        # Clear unwanted inherited attributes (temperature, winds etc.)
        result_da.attrs.clear()

        # Determine the name provided to the function (default is 'name')
        name = kwargs.get('name') or result_da.name

        if name in _BUDGET_META:
            std_name, long_name, units = _BUDGET_META[name]
            result_da.attrs.update(
                {
                    "standard_name": std_name,
                    "long_name": long_name,
                    "units": units
                }
            )
        return result_da

    # Preserve original function metadata for documentation tools
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__dict__.update(func.__dict__)

    return wrapper


# --------------------------------------------------------------------------------------------------
# Energy budget terms
# --------------------------------------------------------------------------------------------------
@budget_metadata
def kinetic_energy_spectra(u: xr.DataArray, v: xr.DataArray, norm: str | None = None,
                           name="hke") -> xr.DataArray:
    """Horizontal kinetic energy per unit mass spectrum: ½(|Û|² + |V̂|²)."""
    hke = 0.5 * (scalar_spectrum(u, norm) + scalar_spectrum(v, norm))
    return hke.rename(name)


@budget_metadata
def divergent_kinetic_energy_spectra(divergence: xr.DataArray, dx: float, dy: float,
                                     norm: str | None = None, name="dke") -> xr.DataArray:
    """Divergent kinetic energy per unit mass spectrum: k² DKE = ½ |∂|²."""
    dke = scalar_spectrum(divergence, norm=norm)

    # Scale by k²
    nx, ny = 2 * (dke['kx'].size - 1), dke['ky'].size
    kh = horizontal_wavenumber_magnitude(nx, ny, 2.0 * np.pi * dx, 2.0 * np.pi * dy)

    dke = 0.5 * dke / (kh ** 2)

    return dke.rename(name)


@budget_metadata
def rotational_kinetic_energy_spectra(vorticity: xr.DataArray, dx: float, dy: float,
                                      norm: str | None = None, name="rke") -> xr.DataArray:
    """Rotational kinetic energy per unit mass spectrum: k² DKE = ½ |∂|²."""
    rke = scalar_spectrum(vorticity, norm=norm)

    # Scale by k²
    nx, ny = 2 * (rke['kx'].size - 1), rke['ky'].size
    kh = horizontal_wavenumber_magnitude(nx, ny, 2.0 * np.pi * dx, 2.0 * np.pi * dy)

    rke = 0.5 * rke / (kh ** 2)

    return rke.rename(name)


@budget_metadata
def nonlinear_hke_transfer_flux(
        u: xr.DataArray,
        v: xr.DataArray,
        w: xr.DataArray,
        divergence: Union[xr.DataArray | None] = None,
        norm: str | None = None,
        name="pi_nke"
) -> xr.DataArray:
    """Nonlinear transfer term for horizontal kinetic energy (HKE), vectorized.

    Uses the compact form:
        T = -⟨U, A⟩ + ⟨∂z U, w U⟩
    where U = (u, v),
          A = (u ∂x u + v ∂y u + ½ div·u + ½ w ∂z u,
               u ∂x v + v ∂y v + ½ div·v + ½ w ∂z v)
    """
    # Horizontal & vertical derivatives
    dzu = u.differentiate("z", edge_order=2)
    dzv = v.differentiate("z", edge_order=2)

    # Divergence (compute if absent)
    if divergence is None:
        divergence = horizontal_divergence(u, v)

    # Stack vectors
    wind = stack_vector(u, v, name="wind")  # (comp=2, z, y, x, …)
    wind_shear = stack_vector(dzu, dzv, name="shear")

    # Advection-like vector A and transport vector wU
    adv_u = horizontal_advection(u, u, v) + 0.5 * divergence * u
    adv_v = horizontal_advection(v, u, v) + 0.5 * divergence * v

    advection = - stack_vector(adv_u, adv_v, name="advection") + 0.5 * w * wind_shear

    # Spectral vector inner products (sum over components)
    t_adv = vector_cross_spectrum(wind, advection, norm=norm)  # -⟨U, A⟩
    t_shear = vector_cross_spectrum(wind_shear, w * wind, norm=norm)  # ⟨∂z U, w U⟩

    pi_nke = t_adv + t_shear

    return pi_nke.rename(name)


@budget_metadata
def nonlinear_hke_transfer_invariant(u: xr.DataArray, v: xr.DataArray, w: xr.DataArray,
                                     divergence: xr.DataArray, vorticity: xr.DataArray,
                                     norm: str | None = None, name="pi_nke") -> xr.DataArray:
    """Invariant-form nonlinear KE transfer (Augier & Lindborg 2013, Eq. A2).

    Works in the current workflow using xarray stacking and spectral primitives:
    - Build horizontal vectors with a ``comp={u,v}`` dimension
    - Use physical-space gradients and shears
    - Convert to spectral space via component-wise cross spectra and sum
    """

    # Physical-space kinetic energy and its horizontal gradient (vector)
    hke_phys = 0.5 * (u ** 2 + v ** 2)

    grad_hke = stack_vector(*horizontal_gradient(hke_phys), name="grad_hke")

    # Wind vector and vertical shear vector
    wind = stack_vector(u, v, name="wind")
    wind_shear = stack_vector(u.differentiate('z', edge_order=2),
                              v.differentiate('z', edge_order=2),
                              name="wind_shear")

    # Divergence and vorticity. Compute if any is absent for consistency.
    if divergence is None or vorticity is None:
        divergence = horizontal_divergence(u, v)
        vorticity = relative_vorticity(u, v)

    # Rotational form advection vector
    advection = grad_hke + vorticity * rotate_vector(wind)
    advection += (divergence * wind + w * wind_shear) / 2.0

    # Nonlinear spectral transfer
    adv_transfer = - vector_cross_spectrum(wind, advection, norm)
    vertical_transport = vector_cross_spectrum(wind_shear, w * wind, norm)

    pi_nke = adv_transfer + vertical_transport

    return pi_nke.rename(name)


@budget_metadata
def coriolis_linear_transfer(u: xr.DataArray, v: xr.DataArray,
                             norm: str | None = None, name="pi_lke") -> xr.DataArray:
    """Coriolis linear transfer term for horizontal kinetic energy (HKE), vectorized."""

    lat_dim = get_spatial_dims(u)[0]
    lat = u[lat_dim]

    if _is_geographic(lat, "lat"):
        # Convert to radians if units are detected as degrees
        lat_rad = np.deg2rad(lat) if _coord_is_degrees(lat) else lat
        # Coriolis parameter f = 2 Ω sin(φ)
        fc = 2 * Omega * np.sin(lat_rad)
    else:
        # Cartesian f-plane
        print("Assuming no rotation (f-plane Coriolis parameter at the Equator)")
        fc = 0.0

    # Coriolis linear transfer
    wind = stack_vector(u, v, name="wind")

    # Linear transfer term: -⟨U, f·(k̂ × U)⟩
    pi_lke = - vector_cross_spectrum(wind, fc * rotate_vector(wind), norm=norm)

    return pi_lke.rename(name)


@budget_metadata
def nonlinear_vke_transfer(
        u: xr.DataArray,
        v: xr.DataArray,
        w: xr.DataArray,
        divergence: Union[xr.DataArray | None] = None,
        norm: str | None = None,
        name="pi_vke"
) -> xr.DataArray:
    """Nonlinear transfer term for vertical kinetic energy (VKE), vectorized.

    Uses the compact form:
        T = -⟨w, Aw⟩ + ⟨∂z w, w^2⟩
    where U = (u, v),
          A = (u ∂x w + v ∂y w + ½ div·w + ½ w ∂z w) / 2
    """

    # Horizontal & vertical derivatives
    dzw = w.differentiate("z", edge_order=2)

    # Divergence (compute if absent)
    if divergence is None:
        divergence = horizontal_divergence(u, v)

    # Advection-like vector A and transport vector wU
    advection_w = horizontal_advection(w, u, v) + 0.5 * (divergence * w) + 0.5 * (w * dzw)

    # Spectral vector inner products (sum over components)
    t_adv = - scalar_cross_spectrum(w, advection_w, norm=norm)  # -⟨w, Aw⟩
    t_shear = scalar_cross_spectrum(dzw, w * w, norm=norm)  # ⟨∂z w, w^2⟩

    pi_nke = t_adv + t_shear

    return pi_nke.rename(name)


@budget_metadata
def turbulent_hke_flux(u: xr.DataArray, v: xr.DataArray, w: xr.DataArray,
                       norm: str | None, name="vf_hke") -> xr.DataArray:
    """Vertical flux of HKE: -½⟨u, w u⟩ - ½⟨v, w v⟩ in spectral space."""

    wind = stack_vector(u, v, name="wind")

    vf_hke = -0.5 * vector_cross_spectrum(wind, w * wind, norm)

    return vf_hke.rename(name)


@budget_metadata
def turbulent_vke_flux(w: xr.DataArray, norm: str | None, name="vf_vke") -> xr.DataArray:
    """Vertical flux of HKE: -½⟨u, w u⟩ - ½⟨v, w v⟩ in spectral space."""

    vf_vke = -0.5 * scalar_cross_spectrum(w, w * w, norm)
    return vf_vke.rename(name)


@budget_metadata
def pressure_flux(theta: xr.DataArray, w: xr.DataArray, exner: xr.DataArray,
                  norm: str | None, name="vf_pres") -> xr.DataArray:
    """Vertical pressure work flux term: -cp·θ·⟨w, exner⟩."""
    p_flux = -cp * domain_mean(theta) * scalar_cross_spectrum(w, exner, norm)

    return p_flux.rename(name)


@budget_metadata
def conversion_ape_to_dke(theta: xr.DataArray, w: xr.DataArray, exner: xr.DataArray,
                          norm: str | None, name="cad") -> xr.DataArray:
    """APE to DKE conversion term: cp·θ·⟨w, ∂z exner⟩."""
    dz_exner = exner.differentiate('z', edge_order=2)

    cad = cp * domain_mean(theta) * scalar_cross_spectrum(w, dz_exner, norm)

    return cad.rename(name)


@budget_metadata
def divergence_hke(u: xr.DataArray, v: xr.DataArray, w: xr.DataArray,
                   divergence: xr.DataArray, norm: str | None, name="div_hke") -> xr.DataArray:
    """Horizontal divergence contribution to the HKE budget."""
    if divergence is None:
        divergence = horizontal_divergence(u, v)

    wind = stack_vector(u, v, name="wind")
    divergence_3d = divergence + w.differentiate("z", edge_order=2)

    div_hke = 0.5 * vector_cross_spectrum(wind, wind * divergence_3d, norm)

    return div_hke.rename(name)


@budget_metadata
def nonconservative_adiabatic(u: xr.DataArray, v: xr.DataArray, w: xr.DataArray,
                              rho: xr.DataArray, vf_hke: xr.DataArray = None,
                              norm: str | None = None, name="j_hke") -> xr.DataArray:
    """Nonconservative adiabatic contribution to the HKE budget."""

    # Vertical flux of HKE (compute if absent)
    if vf_hke is None:
        vf_hke = turbulent_hke_flux(u, v, w, norm=norm)

    # Calculate the spatially averaged density at each height (z)
    rho_mean = domain_mean(rho)

    # Calculate the vertical derivative of the log of the mean density: ∂z( ln(rho_mean) )
    ddz_ln_rho = np.log(rho_mean).differentiate("z", edge_order=2)

    # Compute the budget term: - vf_hke * [∂z( ln(rho_mean) )]
    anc_hke = - vf_hke * ddz_ln_rho

    return anc_hke.rename(name)


@budget_metadata
def vertical_gradient_flux(da: xr.DataArray, vertical_dim: str = 'z', name="vfd_generic") -> (
        xr.DataArray):
    """Generic vertical gradient flux term: ⟨∂z da, w·da⟩."""
    dza = da.differentiate(vertical_dim, edge_order=2)

    return dza.rename(name)


def compute_budget(ds: xr.Dataset, cfg) -> xr.Dataset:
    """Compute the spectral non‑hydrostatic energy budget.

    Produces 1‑D isotropic spectra (k) for each budget term and, optionally,
    cumulative integrals toward low wavenumbers (default True via
    ``cfg.compute.cumulative``).

    Expected variables in ``cfg.variables``:
      - u, v, w  (required)
      - theta    (preferred). If absent, compute from ``pressure`` and ``temperature``.
      - pressure, temperature (optional; used if theta missing)
      - div (optional). If missing, computed as du/dx + dv/dy and added to the pipeline.
      - vorticity (optional). If missing, computed as dv/dx − du/dy for future use.

    Grid:
      - ``cfg.input.dims`` must list **[z, lat, lon]** (lon‑lat) or **[z, y, x]** (Cartesian). The
        vertical dimension name is taken directly from this list (no inference).
    """

    # --- dims & spacing ---
    # Expect cfg.input.dims as [z, y, x] or [z, lat, lon]
    space_dims = ("z",) + get_spatial_dims(ds)
    print(f"Resolved spatial dimensions {space_dims}")

    # After open_dataset(), variable names are normalized to logical names.
    u = ds["u"]
    v = ds["v"]
    w = ds["w"]

    # Validate that provided dims exist on w
    missing = [d for d in space_dims if d not in w.dims]
    if missing:
        raise ValueError(f"Configured dims {space_dims} not all found in 'w' dims {tuple(w.dims)}")

    # dx, dy infer if not set
    if cfg.compute.dx is None or cfg.compute.dy is None:
        dx, dy = infer_grid_resolution(ds)
        print(f"Estimated resolution: dx = {dx:.4f} m, dy = {dy:.4f} m")
    else:
        dx, dy = cfg.compute.dx, cfg.compute.dy
        print(f"Specified resolution: dx = {dx:.4f} m, dy = {dy:.4f} m")

    # --- Thermodynamics ---
    theta = ds.get("theta")
    pressure = ds.get("pressure")
    temperature = ds.get("temperature")
    rho = ds.get("density", None)

    if theta is None and pressure is not None:
        if temperature is None:
            raise ValueError("Provide either 'theta' or both 'pressure' and 'temperature'.")
        # compute potential temperature
        theta = potential_temperature(pressure, temperature)

    if rho is None:
        # Assuming density function takes pressure and temperature inputs
        rho = density(pressure, temperature)

    divergence = ds.get("divergence", None)
    vorticity = ds.get("vorticity", None)

    # Divergence and vorticity. Compute if any is absent for consistency.
    if divergence is None or vorticity is None:
        divergence = horizontal_divergence(u, v)
        vorticity = relative_vorticity(u, v)

    # ----------------------------------------------------------------------------------------------
    # Spectral energy budget terms
    # ----------------------------------------------------------------------------------------------
    # --- accumulation (optional; defaults True) ---
    cumulative = getattr(cfg.compute, "cumulative", True)
    norm = getattr(cfg.compute, "norm", None)

    # --- spectra: 2D HKE, RKE, and DKE and isotropic 1D ---
    hke_2d = kinetic_energy_spectra(u, v, norm=cfg.compute.norm, name="hke")
    hke_1d = isotropize(hke_2d, dx, dy, cumulative=False)  # non-cumulative HKE spectra

    # wavenumber kappa² = k² + l² [m**-2]
    # RKE and DKE spectra
    rke_2d = rotational_kinetic_energy_spectra(vorticity, dx, dy, norm=cfg.compute.norm)
    rke_1d = isotropize(rke_2d, dx, dy, cumulative=False)

    dke_2d = divergent_kinetic_energy_spectra(divergence, dx, dy, norm=cfg.compute.norm)
    dke_1d = isotropize(dke_2d, dx, dy, cumulative=False)

    # ----- Calculate nonlinear spectral transfer → π(HKE) -----
    transfer_mode = getattr(cfg.compute, "transfer_form", "flux")  # "invariant" | "flux"

    # --- NONLINEAR SPECTRAL TRANSFER (pi_nke) ---
    if transfer_mode == "invariant":
        pi_nke_2d = nonlinear_hke_transfer_invariant(u, v, w, divergence, vorticity,
                                                     norm=norm, name="pi_nke")
    else:
        pi_nke_2d = nonlinear_hke_transfer_flux(u, v, w, divergence,
                                                norm=norm, name="pi_nke")

    pi_nke = isotropize(pi_nke_2d, dx, dy, cumulative=cumulative)

    pi_lke_2d = coriolis_linear_transfer(u, v, norm=norm, name="pi_lke")
    pi_lke = isotropize(pi_lke_2d, dx, dy, cumulative=cumulative)

    # --- VERTICAL HKE FLUX (vf_hke) and its DIVERGENCE (vfd_dke) ---
    vf_hke_2d = turbulent_hke_flux(u, v, w, norm=norm, name="vf_hke")
    vf_hke = isotropize(vf_hke_2d, dx, dy, cumulative=cumulative)

    # Calculate divergence of the 2D flux, then isotropize
    vfd_dke_2d = vertical_gradient_flux(vf_hke_2d, name="vfd_dke")
    vfd_dke = isotropize(vfd_dke_2d, dx, dy, cumulative=cumulative)

    # --- ADIABATIC NON-CONSERVATIVE (j_hke) ---
    # Note: nonconservative_adiabatic uses vf_hke_2d (or calculates it if not passed)
    j_hke_2d = nonconservative_adiabatic(u, v, w, rho, vf_hke=vf_hke_2d,
                                         norm=norm, name="j_hke")
    j_hke = isotropize(j_hke_2d, dx, dy, cumulative=cumulative)

    # --- PRESSURE WORK & CONVERSION (vf_pres, vfd_pres, cad) ---
    # Assuming Exner function and domain_mean are available
    exner = exner_function(pressure) - exner_function(domain_mean(pressure))

    # Pressure Flux (vf_pres)
    vf_pres_2d = pressure_flux(theta, w, exner, norm=norm, name="vf_pres")
    vf_pres = isotropize(vf_pres_2d, dx, dy, cumulative=cumulative)

    # Pressure Flux Divergence (vfd_pres)
    vfd_pres_2d = vertical_gradient_flux(vf_pres_2d, name="vfd_pres")
    vfd_pres = isotropize(vfd_pres_2d, dx, dy, cumulative=cumulative)

    # Conversion (cad)
    cad_2d = conversion_ape_to_dke(theta, w, exner, norm=norm, name="cad")
    cad = isotropize(cad_2d, dx, dy, cumulative=cumulative)

    # --- HORIZONTAL KE DIVERGENCE (div_hke) ---
    div_hke_2d = divergence_hke(u, v, w, divergence, norm=norm, name="div_hke")
    div_hke = isotropize(div_hke_2d, dx, dy, cumulative=cumulative)

    # --- ASSEMBLE ---
    fluxes = [hke_1d, rke_1d, dke_1d,
              pi_nke, pi_lke, vfd_dke, j_hke, vf_hke,
              vf_pres, vfd_pres, cad, div_hke]

    # Filter out None and assemble into Dataset
    fluxes = xr.Dataset({da.name: da for da in fluxes if da is not None})

    # wavenumber coord attrs (unchanged, but placed after assembly)
    fluxes.wavenumber.attrs.update({'standard_name': 'wavenumber',
                                    'long_name': 'horizontal wavenumber',
                                    'axis': 'X', 'units': 'rad / m'})

    return fluxes.transpose(..., "wavenumber")
