from typing import Union, Callable, Any

import numpy as np
import xarray as xr

from .cf_coords import get_spatial_dims, infer_grid_resolution, _is_geographic, _coord_is_degrees
from .constants import cp, Omega, epsilon, g
from .memory_manager import ensure_optimal_chunking
from .numeric_tools import domain_mean, stack_vector, rotate_vector
from .numeric_tools import horizontal_advection, horizontal_gradient, skew_symmetric_advection
from .numeric_tools import mark_periodic
from .grid_utils import resolve_periodic_axes
from .numeric_tools import horizontal_divergence, relative_vorticity
from .spectral_primitives import horizontal_wavenumber_magnitude, isotropize
from .spectral_primitives import scalar_spectrum, scalar_cross_spectrum, vector_cross_spectrum
from .thermodynamics import hydrostatic_exner_function
from .thermodynamics import potential_temperature, exner_function, density, lorenz_parameter
from .vertical import (is_lorenz_staggered, interp_full_to_half, diff_full_to_half,
                       half_to_full_mean, diff_half_to_full)

# --------------------------------------------------------------------------------------------------
# Global spectral options
# --------------------------------------------------------------------------------------------------
# --- Global Metadata Configuration ---
_BUDGET_UNITS = "watt / kilogram"

_BUDGET_META = {
    "hke": ("horizontal_kinetic_energy_spectrum",
            "Spectrum of horizontal kinetic energy per unit mass", "m**2 / s**2"),
    "rke": ("rotational_kinetic_energy_spectrum",
            "Spectrum of rotational kinetic energy per unit mass", "m**2 / s**2"),
    "dke": ("divergent_kinetic_energy_spectrum",
            "Spectrum of divergent kinetic energy per unit mass", "m**2 / s**2"),
    "vke": ("vertical_kinetic_energy_spectrum",
            "Spectrum of vertical kinetic energy per unit mass", "m**2 / s**2"),
    "ape": ("available_potential_energy_spectrum",
            "Spectrum of available potential energy per unit mass", "m**2 / s**2"),
    "cad": ("conversion_ape_dke",
            "Conversion from available potential energy to kinetic energy", _BUDGET_UNITS),
    "caz": ("conversion_ape_vke",
            "Conversion from available potential energy to vertical kinetic energy", _BUDGET_UNITS),
    "pi_nke": ("nonlinear_hke_transfer",
               "Nonlinear spectral transfer of horizontal kinetic energy", _BUDGET_UNITS),
    "pi_lke": ("linear_hke_transfer",
               "Linear spectral transfer of horizontal kinetic energy due to Coriolis",
               _BUDGET_UNITS),
    "pi_vke": ("nonlinear_vke_transfer",
               "Nonlinear spectral transfer of vertical kinetic energy", _BUDGET_UNITS),
    "pi_ape": ("nonlinear_ape_transfer",
               "Nonlinear spectral transfer of available potential energy", _BUDGET_UNITS),
    "vfd_dke": ("vertical_dke_flux_divergence",
                "Vertical flux divergence of horizontal kinetic energy", _BUDGET_UNITS),
    "vfd_vke": ("vertical_vke_flux_divergence",
                "Vertical flux divergence of vertical kinetic energy", _BUDGET_UNITS),
    "vfd_ape": ("vertical_ape_flux_divergence",
                "Vertical flux divergence of available potential energy", _BUDGET_UNITS),
    "vf_hke": ("vertical_dke_flux",
               "Vertical flux of horizontal kinetic energy", _BUDGET_UNITS),
    "vf_vke": ("vertical_vke_flux",
               "Vertical flux of vertical kinetic energy", _BUDGET_UNITS),
    "vf_ape": ("vertical_ape_flux",
               "Vertical flux of available potential energy", _BUDGET_UNITS),
    "vfd_pres": ("pressure_flux_divergence",
                 "Vertical divergence of pressure work", _BUDGET_UNITS),
    "vf_pres": ("hke_pressure_vertical_flux",
                "Vertical flux of horizontal kinetic energy (pressure work)", _BUDGET_UNITS),
    "div_hke": ("hke_divergence",
                "Divergence contribution to HKE budget", _BUDGET_UNITS),
    "div_vke": ("vke_divergence",
                "Divergence contribution to VKE budget", _BUDGET_UNITS),
    "div_ape": ("ape_divergence",
                "Divergence contribution to APE budget", _BUDGET_UNITS),
    "j_hke": ("adiabatic_nonconservative",
              "Adiabatic nonconservative contribution to HKE budget", _BUDGET_UNITS),
    "j_vke": ("adiabatic_nonconservative_vke",
              "Adiabatic nonconservative contribution to VKE budget", _BUDGET_UNITS),
    "j_ape": ("adiabatic_nonconservative_ape",
              "Adiabatic nonconservative contribution to APE budget", _BUDGET_UNITS),
    "j_pres": ("adiabatic_nonconservative_pressure",
               "Reference-profile term of the vertical pressure flux", _BUDGET_UNITS),
    "div_pres": ("pressure_divergence_work",
                 "Pressure work by the anelastic-departure divergence", _BUDGET_UNITS),
    "np_hke": ("nonlinear_pressure_work_hke",
               "Horizontal pressure work by potential temperature perturbations", _BUDGET_UNITS),
    "np_vke": ("nonlinear_pressure_work_vke",
               "Vertical pressure work by potential temperature perturbations", _BUDGET_UNITS),
}


def _set_budget_metadata(da: xr.DataArray, name: str) -> xr.DataArray:
    """Replace inherited attributes (temperature, winds etc.) by the entry of _BUDGET_META."""
    da.attrs.clear()
    if name in _BUDGET_META:
        std_name, long_name, units = _BUDGET_META[name]
        da.attrs.update({"standard_name": std_name, "long_name": long_name, "units": units})
    return da


def budget_metadata(func: Callable) -> Callable:
    """
    Decorator that applies standard_name, long_name, and units attributes
    to the resulting DataArray based on the 'name' argument passed to the function.
    """

    def wrapper(*args: Any, **kwargs: Any) -> xr.DataArray:
        # Execute the wrapped function to get the result DataArray
        result_da: xr.DataArray = func(*args, **kwargs)

        # Determine the name provided to the function (default is 'name')
        name = kwargs.get('name') or result_da.name

        return _set_budget_metadata(result_da, name)

    # Preserve original function metadata for documentation tools
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__dict__.update(func.__dict__)

    return wrapper


# --------------------------------------------------------------------------------------------------
# Energy budget terms
# --------------------------------------------------------------------------------------------------
@budget_metadata
def kinetic_energy_spectra(u: xr.DataArray, v: xr.DataArray, name="hke") -> xr.DataArray:
    """Horizontal kinetic energy per unit mass spectrum: ½(|Û|² + |V̂|²)."""
    hke = 0.5 * (scalar_spectrum(u) + scalar_spectrum(v))

    return hke.rename(name)


@budget_metadata
def vertical_kinetic_energy_spectra(w: xr.DataArray, name="vke") -> xr.DataArray:
    """Vertical kinetic energy per unit mass spectrum: VKE = ½ |Ŵ|²."""
    vke = 0.5 * scalar_spectrum(w)
    return vke.rename(name)


@budget_metadata
def available_potential_energy_spectra(theta_prime: xr.DataArray, gamma: xr.DataArray,
                                       name="ape") -> xr.DataArray:
    """Available Potential Energy per unit mass spectrum: APE = ½ γ |θ'|²."""

    ape = 0.5 * gamma * scalar_spectrum(theta_prime)
    return ape.rename(name)


@budget_metadata
def divergent_kinetic_energy_spectra(divergence: xr.DataArray, dx: float, dy: float,
                                     name="dke") -> xr.DataArray:
    """Divergent kinetic energy per unit mass spectrum: DKE = ½ |∂|² / k²."""
    dke = scalar_spectrum(divergence)

    # recover original grid size
    nx, ny = 2 * (dke['kx'].size - 1), dke['ky'].size

    # Scale by 2 k²
    kh = horizontal_wavenumber_magnitude(nx, ny, dx, dy)

    dke = 0.5 * dke / kh.clip(min=epsilon) ** 2

    return dke.rename(name)


@budget_metadata
def rotational_kinetic_energy_spectra(vorticity: xr.DataArray, dx: float, dy: float,
                                      name="rke") -> xr.DataArray:
    """Rotational kinetic energy per unit mass spectrum: RKE = ½ |ζ|² / k²."""
    rke = scalar_spectrum(vorticity)

    # recover original grid size
    nx, ny = 2 * (rke['kx'].size - 1), rke['ky'].size

    # Scale by 2 k²
    kh = horizontal_wavenumber_magnitude(nx, ny, dx, dy)
    rke = 0.5 * rke / kh.clip(min=epsilon) ** 2

    return rke.rename(name)


@budget_metadata
def nonlinear_hke_transfer_flux(
        u: xr.DataArray,
        v: xr.DataArray,
        w: xr.DataArray,
        divergence: Union[xr.DataArray | None] = None,
        conservative: bool = False,
        name="pi_nke"
) -> xr.DataArray:
    """Nonlinear transfer term for horizontal kinetic energy (HKE), vectorized.

    Uses the compact form:
        T = -⟨U, A⟩ + ½⟨∂z U, w U⟩
    where U = (u, v),
          A = (u ∂x u + v ∂y u + ½ div·u + ½ w ∂z u,
               u ∂x v + v ∂y v + ½ div·v + ½ w ∂z v)

    The vertical part ½[⟨∂z U, w U⟩ - ⟨U, w ∂z U⟩] is antisymmetric, so T sums to zero over
    wavenumbers on a periodic domain, and T + ∂z F_h + Div_h equals the spectrum of the
    advective tendency -⟨U, (U·∇)U + w ∂z U⟩ exactly (Augier & Lindborg 2013).

    conservative=True uses the skew-symmetric form ½[(U·∇)U + ∇·(U U)] for the horizontal part
    (no divergence needed): equal on the continuum, but the sum over wavenumbers vanishes on
    the grid too (periodic axes, see numeric_tools.skew_symmetric_advection). The flux form
    relies on the product rule and leaves a discretization residual.
    """
    # Horizontal & vertical derivatives
    dzu = u.differentiate("z", edge_order=2)
    dzv = v.differentiate("z", edge_order=2)

    # Stack vectors
    wind = stack_vector(u, v, name="wind")  # (comp=2, z, y, x, …)
    wind_shear = stack_vector(dzu, dzv, name="shear")

    # Advection-like vector A and transport vector wU
    if conservative:
        adv_u = skew_symmetric_advection(u, u, v)
        adv_v = skew_symmetric_advection(v, u, v)
    else:
        if divergence is None:
            divergence = horizontal_divergence(u, v)
        adv_u = horizontal_advection(u, u, v) + 0.5 * divergence * u
        adv_v = horizontal_advection(v, u, v) + 0.5 * divergence * v

    advection = - stack_vector(adv_u, adv_v, name="advection") - 0.5 * w * wind_shear

    # Spectral vector inner products (sum over components)
    t_adv = vector_cross_spectrum(wind, advection)  # -⟨U, A⟩
    t_shear = 0.5 * vector_cross_spectrum(wind_shear, w * wind)  # ½⟨∂z U, w U⟩

    pi_nke = t_adv + t_shear

    return pi_nke.rename(name)


@budget_metadata
def nonlinear_hke_transfer_invariant(u: xr.DataArray, v: xr.DataArray, w: xr.DataArray,
                                     divergence: xr.DataArray, vorticity: xr.DataArray,
                                     name="pi_nke") -> xr.DataArray:
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

    # Nonlinear spectral transfer (antisymmetric vertical part: see nonlinear_hke_transfer_flux)
    adv_transfer = - vector_cross_spectrum(wind, advection)
    vertical_transport = 0.5 * vector_cross_spectrum(wind_shear, w * wind)

    pi_nke = adv_transfer + vertical_transport

    return pi_nke.rename(name)


@budget_metadata
def coriolis_linear_transfer(u: xr.DataArray, v: xr.DataArray, name="pi_lke") -> xr.DataArray:
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
    pi_lke = - vector_cross_spectrum(wind, fc * rotate_vector(wind))

    return pi_lke.rename(name)


@budget_metadata
def nonlinear_vke_transfer(
        u: xr.DataArray,
        v: xr.DataArray,
        w: xr.DataArray,
        divergence: Union[xr.DataArray | None] = None,
        conservative: bool = False,
        name="pi_vke"
) -> xr.DataArray:
    """Nonlinear transfer term for vertical kinetic energy (VKE), vectorized.

    Uses the compact form:
        T = -⟨w, Aw⟩ + ½⟨∂z w, w^2⟩
    where U = (u, v),
          Aw = u ∂x w + v ∂y w + ½ div·w + ½ w ∂z w
    (antisymmetric vertical part, as in nonlinear_hke_transfer_flux; conservative=True: the
    skew-symmetric horizontal part ½[U·∇w + ∇·(U w)], as there).
    """

    # Horizontal & vertical derivatives
    dzw = w.differentiate("z", edge_order=2)

    # Advection-like vector A and transport vector wU
    if conservative:
        advection_w = skew_symmetric_advection(w, u, v) + 0.5 * (w * dzw)
    else:
        if divergence is None:
            divergence = horizontal_divergence(u, v)
        advection_w = horizontal_advection(w, u, v) + 0.5 * (divergence * w) + 0.5 * (w * dzw)

    # Spectral vector inner products (sum over components)
    t_adv = - scalar_cross_spectrum(w, advection_w)  # -⟨w, Aw⟩
    t_shear = 0.5 * scalar_cross_spectrum(dzw, w * w)  # ½⟨∂z w, w^2⟩

    pi_vke = t_adv + t_shear

    return pi_vke.rename(name)


@budget_metadata
def nonlinear_ape_transfer(
        u: xr.DataArray,
        v: xr.DataArray,
        w: xr.DataArray,
        theta_prime: xr.DataArray,
        gamma: xr.DataArray,
        divergence: Union[xr.DataArray | None] = None,
        conservative: bool = False,
        name="pi_ape"
) -> xr.DataArray:
    """Nonlinear transfer term for available potential energy (APE), vectorized.

    Uses the compact form:
        T = γ (-⟨theta, At⟩ + ½⟨∂z theta, w.theta⟩)
    where U = (u, v),
          At = (u ∂x theta + v ∂y theta + ½ div·theta + ½ w ∂z theta)
    (antisymmetric vertical part, as in nonlinear_hke_transfer_flux; conservative=True: the
    skew-symmetric horizontal part ½[U·∇θ + ∇·(U θ)], as there).
    """

    # Horizontal & vertical derivatives
    dztheta = theta_prime.differentiate("z", edge_order=2)

    # Advection-like vector A and transport vector wU
    if conservative:
        advection_theta = skew_symmetric_advection(theta_prime, u, v) + 0.5 * (w * dztheta)
    else:
        if divergence is None:
            divergence = horizontal_divergence(u, v)
        advection_theta = horizontal_advection(theta_prime, u, v) + 0.5 * (
                divergence * theta_prime) + 0.5 * (w * dztheta)

    # Spectral vector inner products (sum over components)
    t_adv = - scalar_cross_spectrum(theta_prime, advection_theta)  # -⟨theta, At⟩
    t_shear = 0.5 * scalar_cross_spectrum(dztheta, w * theta_prime)  # ½⟨∂z theta, w.theta⟩

    pi_ape = gamma * (t_adv + t_shear)

    return pi_ape.rename(name)


@budget_metadata
def turbulent_hke_flux(u: xr.DataArray, v: xr.DataArray, w: xr.DataArray,
                       name="vf_hke") -> xr.DataArray:
    """Vertical flux of HKE: -½⟨u, w u⟩ - ½⟨v, w v⟩ in spectral space."""

    wind = stack_vector(u, v, name="wind")

    vf_hke = -0.5 * vector_cross_spectrum(wind, w * wind)

    return vf_hke.rename(name)


@budget_metadata
def turbulent_vke_flux(w: xr.DataArray, name="vf_vke") -> xr.DataArray:
    """Vertical flux of VKE: -½⟨w, w w⟩ in spectral space."""

    vf_vke = -0.5 * scalar_cross_spectrum(w, w * w)
    return vf_vke.rename(name)


@budget_metadata
def turbulent_ape_flux(w: xr.DataArray, theta_prime: xr.DataArray, gamma: xr.DataArray,
                       name="vf_ape") -> xr.DataArray:
    """Vertical flux of APE: -½ γ ⟨theta, w theta⟩ in spectral space."""

    vf_ape = -0.5 * gamma * scalar_cross_spectrum(theta_prime, w * theta_prime)
    return vf_ape.rename(name)


@budget_metadata
def pressure_flux(theta_mean: xr.DataArray, w: xr.DataArray, exner: xr.DataArray,
                  name="vf_pres") -> xr.DataArray:
    """Vertical pressure work flux term: -cp·θ_mean·⟨w, exner⟩."""
    p_flux = -cp * theta_mean * scalar_cross_spectrum(w, exner)

    return p_flux.rename(name)


@budget_metadata
def conversion_ape_to_dke(theta_mean: xr.DataArray, w: xr.DataArray,
                          exner: xr.DataArray, name="cad") -> xr.DataArray:
    """APE to DKE conversion term: cp·θ_mean·⟨w, ∂z exner⟩."""
    dz_exner = exner.differentiate('z', edge_order=2)

    cad = cp * theta_mean * scalar_cross_spectrum(w, dz_exner)

    return cad.rename(name)


@budget_metadata
def conversion_ape_to_vke(theta_mean: xr.DataArray, theta_prime: xr.DataArray, w: xr.DataArray,
                          exner: xr.DataArray, name="caz") -> xr.DataArray:
    """APE to VKE conversion term: (g / θ_bar) * ⟨w, θ'⟩ - cp * θ_bar * ⟨w, ∂z π'⟩ - g * ⟨w, q_t⟩"""

    dz_exner = exner.differentiate('z', edge_order=2)

    theta_transport = (g / theta_mean) * scalar_cross_spectrum(w, theta_prime)
    exner_transport = -cp * theta_mean * scalar_cross_spectrum(w, dz_exner)

    caz = theta_transport + exner_transport

    return caz.rename(name)


@budget_metadata
def conversion_ape(theta_mean: xr.DataArray, theta_prime: xr.DataArray, w: xr.DataArray,
                   name="ca") -> xr.DataArray:
    """APE  conversion term: (g / θ_bar) * ⟨w, θ'⟩"""

    ca = (g / theta_mean) * scalar_cross_spectrum(w, theta_prime)

    return ca.rename(name)


# --------------------------------------------------------------------------------------------------
# Compressibility (Div) and reference-profile (J) terms
#
# Per unit mass, the advective tendency of X splits exactly into
#     -(X, u·∇X + w ∂z X)_k = t_X + ∂z F_X + J_X + Div_X,
# with the anelastic-departure divergence D = ∇·v + w ∂z ln ρ̄ (zero for anelastic flow) and
#     Div_X = ½ (X, X D)_k,   J_X = F_X ∂z ln ρ̄   (J_A = F_A ∂z ln(ρ̄/γ) for APE).
# Then ∂z F_X + J_X = ρ̄⁻¹ ∂z(ρ̄ F_X) for HKE and VKE (γ ρ̄⁻¹ ∂z(ρ̄ F_A / γ) for APE), so the
# density-weighted layer integrals of the HKE and VKE flux terms reduce to boundary fluxes ρ̄ F.
# --------------------------------------------------------------------------------------------------
def _ddz_log_rho_mean(rho: xr.DataArray) -> xr.DataArray:
    """∂z ln ρ̄ with ρ̄ the horizontal domain mean of density."""
    return np.log(domain_mean(rho)).differentiate("z", edge_order=2)


def anelastic_divergence(u: xr.DataArray, v: xr.DataArray, w: xr.DataArray,
                         divergence: xr.DataArray = None,
                         rho: xr.DataArray = None,
                         dzw: xr.DataArray = None) -> xr.DataArray:
    """D = ∇·v + w ∂z ln ρ̄ = ρ̄⁻¹ ∇·(ρ̄ v). Without rho, returns the full 3-D divergence ∇·v.

    ``dzw`` is ∂w/∂z on full levels; on Lorenz-staggered grids pass the compact difference of the
    native half-level w (vertical.diff_half_to_full). Otherwise it is the centred difference of w.
    """
    if divergence is None:
        divergence = horizontal_divergence(u, v)

    if dzw is None:
        dzw = w.differentiate("z", edge_order=2)

    div_3d = divergence + dzw

    if rho is not None:
        div_3d = div_3d + w * _ddz_log_rho_mean(rho)

    return div_3d


@budget_metadata
def divergence_hke(u: xr.DataArray, v: xr.DataArray, w: xr.DataArray,
                   divergence: xr.DataArray, rho: xr.DataArray = None,
                   dzw: xr.DataArray = None, name="div_hke") -> xr.DataArray:
    """Compressibility term of the HKE budget: ½(U, U D)_k.

    With ``rho``, D is the anelastic-departure divergence ρ̄⁻¹∇·(ρ̄v), consistent with
    J_h = F_h ∂z ln ρ̄. Without ``rho``, D = ∇·v and J_h must be omitted.
    """
    wind = stack_vector(u, v, name="wind")
    div_d = anelastic_divergence(u, v, w, divergence, rho, dzw)

    div_hke = 0.5 * vector_cross_spectrum(wind, wind * div_d)

    return div_hke.rename(name)


@budget_metadata
def divergence_vke(u: xr.DataArray, v: xr.DataArray, w: xr.DataArray,
                   divergence: xr.DataArray, rho: xr.DataArray = None,
                   dzw: xr.DataArray = None, name="div_vke") -> xr.DataArray:
    """Compressibility term of the VKE budget: ½(w, w D)_k (see divergence_hke)."""
    div_d = anelastic_divergence(u, v, w, divergence, rho, dzw)

    div_vke = 0.5 * scalar_cross_spectrum(w, w * div_d)

    return div_vke.rename(name)


@budget_metadata
def divergence_ape(u: xr.DataArray, v: xr.DataArray, w: xr.DataArray, theta_prime: xr.DataArray,
                   gamma: xr.DataArray, divergence: xr.DataArray, rho: xr.DataArray = None,
                   dzw: xr.DataArray = None, name="div_ape") -> xr.DataArray:
    """Compressibility term of the APE budget: ½γ(θ', θ' D)_k (see divergence_hke)."""
    div_d = anelastic_divergence(u, v, w, divergence, rho, dzw)

    div_ape = 0.5 * gamma * scalar_cross_spectrum(theta_prime, theta_prime * div_d)

    return div_ape.rename(name)


@budget_metadata
def nonconservative_adiabatic_hke(u: xr.DataArray, v: xr.DataArray, w: xr.DataArray,
                                  rho: xr.DataArray, vf_hke: xr.DataArray = None,
                                  name="j_hke") -> xr.DataArray:
    """Reference-density term of the HKE budget: J_h = F_h ∂z ln ρ̄.

    Together with ∂z F_h it forms the density-weighted flux divergence ρ̄⁻¹∂z(ρ̄ F_h). Use with
    divergence_hke(..., rho=rho); with the full-divergence Div_h, J_h would double-count.
    """
    if vf_hke is None:
        vf_hke = turbulent_hke_flux(u, v, w)

    anc_hke = vf_hke * _ddz_log_rho_mean(rho)

    return anc_hke.rename(name)


@budget_metadata
def nonconservative_adiabatic_vke(w: xr.DataArray, rho: xr.DataArray, vf_vke: xr.DataArray = None,
                                  name="j_vke") -> xr.DataArray:
    """Reference-density term of the VKE budget: J_z = F_z ∂z ln ρ̄ (see the HKE version)."""
    if vf_vke is None:
        vf_vke = turbulent_vke_flux(w)

    anc_vke = vf_vke * _ddz_log_rho_mean(rho)

    return anc_vke.rename(name)


@budget_metadata
def nonconservative_adiabatic_ape(theta_prime: xr.DataArray, gamma: xr.DataArray, rho: xr.DataArray,
                                  vf_ape: xr.DataArray = None, name="j_ape") -> xr.DataArray:
    """Reference-profile term of the APE budget: J_A = F_A ∂z ln(ρ̄/γ).

    The γ(z) part arises because E_A = ½γ(θ', θ')_k with a height-dependent γ.
    """
    if vf_ape is None:
        raise ValueError("nonconservative_adiabatic_ape needs vf_ape (turbulent_ape_flux).")

    ddz_ln_rho_over_gamma = _ddz_log_rho_mean(rho) - np.log(gamma).differentiate("z", edge_order=2)

    anc_ape = vf_ape * ddz_ln_rho_over_gamma

    return anc_ape.rename(name)


# --------------------------------------------------------------------------------------------------
# Pressure work
#
# With π' the Exner deviation from the hydrostatic reference built on θ̄(z), the exact horizontal
# pressure work on HKE, -c_p (U, θ ∇h π')_k, splits exactly into
#     ∂z F_p + J_p + C_{A→h} + Div_p + N_h,
#     F_p = -c_p θ̄ (w, π')_k,  J_p = F_p ∂z ln(ρ̄/θ̄),  C_{A→h} = c_p θ̄ (w, ∂z π')_k,
#     Div_p = c_p θ̄ (π', D)_k,  N_h = -c_p (U, θ' ∇h π')_k.
# For VKE, (w, -c_p θ ∂z π - g)_k = C_{A→z} + N_z with N_z = -c_p (w, θ' ∂z π')_k.
# --------------------------------------------------------------------------------------------------
@budget_metadata
def pressure_divergence_work(u: xr.DataArray, v: xr.DataArray, w: xr.DataArray,
                             divergence: xr.DataArray, rho: xr.DataArray,
                             theta_mean: xr.DataArray, exner: xr.DataArray,
                             dzw: xr.DataArray = None, name="div_pres") -> xr.DataArray:
    """Pressure work by the anelastic-departure divergence: c_p θ̄ (π', D)_k."""
    div_d = anelastic_divergence(u, v, w, divergence, rho, dzw)

    div_pres = cp * theta_mean * scalar_cross_spectrum(exner, div_d)

    return div_pres.rename(name)


@budget_metadata
def nonconservative_adiabatic_pressure(vf_pres: xr.DataArray, rho: xr.DataArray,
                                       theta_mean: xr.DataArray, name="j_pres") -> xr.DataArray:
    """Reference-profile term of the pressure flux: J_p = F_p ∂z ln(ρ̄/θ̄)."""
    ddz = _ddz_log_rho_mean(rho) - np.log(theta_mean).differentiate("z", edge_order=2)

    j_pres = vf_pres * ddz

    return j_pres.rename(name)


@budget_metadata
def nonlinear_pressure_work_hke(u: xr.DataArray, v: xr.DataArray, theta_prime: xr.DataArray,
                                exner: xr.DataArray, name="np_hke") -> xr.DataArray:
    """Horizontal pressure work by θ' (neglected by the θ̄-based split): -c_p (U, θ' ∇h π')_k."""
    grad_x, grad_y = horizontal_gradient(exner)

    wind = stack_vector(u, v, name="wind")
    force = stack_vector(theta_prime * grad_x, theta_prime * grad_y, name="force")

    np_hke = -cp * vector_cross_spectrum(wind, force)

    return np_hke.rename(name)


@budget_metadata
def nonlinear_pressure_work_vke(w: xr.DataArray, theta_prime: xr.DataArray,
                                exner: xr.DataArray, name="np_vke") -> xr.DataArray:
    """Vertical pressure work by θ' (not in C_{A→z}): -c_p (w, θ' ∂z π')_k."""
    dz_exner = exner.differentiate("z", edge_order=2)

    np_vke = -cp * scalar_cross_spectrum(w, theta_prime * dz_exner)

    return np_vke.rename(name)


@budget_metadata
def vertical_gradient_flux(da: xr.DataArray, vertical_dim: str = 'z', name="vfd_generic") -> (
        xr.DataArray):
    """Generic vertical gradient flux term: ⟨∂z da, w·da⟩."""
    dza = da.differentiate(vertical_dim, edge_order=2)

    return dza.rename(name)


def staggered_vertical_terms(u: xr.DataArray, v: xr.DataArray, w_half: xr.DataArray,
                             theta_mean: xr.DataArray, theta_prime: xr.DataArray,
                             gamma: xr.DataArray, exner: xr.DataArray) -> dict:
    """
    Vertical fluxes and pressure-work conversions on a Lorenz-staggered grid (2-D spectra on
    full levels), using the native half-level w.

    Fluxes are formed at half levels, where w lives, with full-level fields interpolated there:
        F_h = -½(U, wU),  F_z = -½(w, w²),  F_A = -½γ(θ', wθ'),  F_p = -c_p θ̄ (w, π')
    Their divergences are compact differences to full levels, so layer sums telescope exactly;
    the fluxes themselves are reported on full levels as the mean of the bounding interfaces.
    The conversions use the compact vertical gradient of π' at half levels (co-located with w):
        C_A→h = c_p θ̄ (w, ∂z π'),  C_A→z = (g/θ̄)(w, θ') - c_p θ̄ (w, ∂z π'),
        N_z = -c_p (w, θ' ∂z π'),
    averaged to full levels. Against analytic waves at 5 levels per vertical wavelength this
    reduces the error of the conversions from ~38% (w interpolated, centred differences) to ~5%.
    """
    z_full, z_half = u["z"], w_half["z_half"]
    up = lambda a: interp_full_to_half(a, z_half)

    u_h, v_h = up(u), up(v)
    theta_mean_h, theta_prime_h = up(theta_mean), up(theta_prime)
    gamma_h, exner_h = up(gamma), up(exner)
    dz_exner_h = diff_full_to_half(exner, z_half)

    flux_half = {
        "hke": turbulent_hke_flux(u_h, v_h, w_half),
        "vke": turbulent_vke_flux(w_half),
        "ape": turbulent_ape_flux(w_half, theta_prime_h, gamma_h),
        "pres": pressure_flux(theta_mean_h, w_half, exner_h),
    }
    w_dpi = scalar_cross_spectrum(w_half, dz_exner_h)
    conv_half = {
        "cad": cp * theta_mean_h * w_dpi,
        "caz": (g / theta_mean_h) * scalar_cross_spectrum(w_half, theta_prime_h)
               - cp * theta_mean_h * w_dpi,
        "np_vke": -cp * scalar_cross_spectrum(w_half, theta_prime_h * dz_exner_h),
    }

    out = {}
    for key, flux in flux_half.items():
        out[f"vf_{key}"] = half_to_full_mean(flux, z_full)
        out[f"vfd_{key}"] = diff_half_to_full(flux, z_full)
    for key, conv in conv_half.items():
        out[key] = half_to_full_mean(conv, z_full)
    return out


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

    y_dim, x_dim = get_spatial_dims(ds)
    space_dims = ("z", y_dim, x_dim)

    print(f"[budget] Resolved spatial dimensions {space_dims}")

    # Rechunk spatial dims switch (default False)
    rechunk_spatial = getattr(cfg.compute, "rechunk_spatial", False)

    # ----------------------------------------------------------------------------
    # Ensure 'optimal' chunks along non-spatial dimensions for high parallelism
    # ----------------------------------------------------------------------------
    ds = ensure_optimal_chunking(ds, spatial_dims=(y_dim, x_dim), vertical_dim="z",
                                 # Use up to 90% of available memory
                                 memory_threshold_ratio=0.9,
                                 # Stencil order for finite difference formulas
                                 deriv_edge_order=2,
                                 # Switch for spatial rechunk: small impact on the performance
                                 rechunk_spatial=rechunk_spatial)

    # Periodic horizontal axes (detected from the data unless given): wrap-around derivatives,
    # no taper and no detrending there; repeated end points dropped
    ds, periodic, report = resolve_periodic_axes(ds, getattr(cfg.compute, "periodic_axes", None))
    for d, (ratio, status) in report.items():
        print(f"[budget] {d}: {status} (seam/neighbour difference {ratio:.2f})")
    ds = mark_periodic(ds.copy(), periodic)

    # Get wind field, variable names are normalized to logical names.
    u = ds["u"]
    v = ds["v"]
    w = ds["w"]

    # Validate that provided dims exist on w
    missing = [d for d in space_dims if d not in w.dims]
    if missing:
        raise ValueError(f"Configured dims {space_dims} not all found in 'w' dims {tuple(w.dims)}")

    # Infer grid resolution if not provided (provided for legacy reasons, should always be inferred)
    if cfg.compute.dx is None or cfg.compute.dy is None:
        dx, dy = infer_grid_resolution(ds)
        print(f"[budget] Estimated grid resolution: dx = {dx:.4f} m, dy = {dy:.4f} m")
    else:
        dx, dy = cfg.compute.dx, cfg.compute.dy
        print(f"[budget] Specified grid resolution: dx = {dx:.4f} m, dy = {dy:.4f} m")

    # ----------------------------------------------------------------------------
    # --- Thermodynamics computations ---
    # ----------------------------------------------------------------------------
    # pressure is guaranteed present by open_dataset's mode validation
    pressure = ds["pressure"]
    rho = ds.get("density")
    theta = ds.get("theta")
    exner = exner_function(pressure)

    if theta is None:
        # temperature must be present (validated upstream)
        theta = potential_temperature(pressure, ds["temperature"])

    if rho is None:
        # density() needs temperature; recover it from theta if not directly available
        rho = density(pressure, ds["temperature"] if "temperature" in ds else theta * exner)

    divergence = ds.get("divergence", None)
    vorticity = ds.get("vorticity", None)

    # Divergence and vorticity. Compute if any is absent for numerical consistency.
    if divergence is None or vorticity is None:
        divergence = horizontal_divergence(u, v)
        vorticity = relative_vorticity(u, v)

    # Split into mean and fluctuations
    theta_mean = domain_mean(theta)
    theta_prime = theta - theta_mean
    gamma = lorenz_parameter(theta_mean)

    # Lorenz-staggered input (native half-level w): use the grid's own vertical operators for
    # dw/dz, the vertical fluxes and the pressure-work conversions (see budget.vertical)
    w_half = ds.get("w_half")
    staggered = w_half is not None and is_lorenz_staggered(u["z"], w_half["z_half"])
    if w_half is not None and not staggered:
        print("[budget] w_half present but not Lorenz-staggered with z: using collocated w")
    dzw = diff_half_to_full(w_half, u["z"]) if staggered else None
    print(
        f"[budget] Vertical discretization: {'Lorenz-staggered (native half-level w)' if staggered else 'collocated (centred differences)'}")

    # ----------------------------------------------------------------------------------------------
    # Spectral energy budget terms
    # ----------------------------------------------------------------------------------------------
    # --- accumulation (optional; defaults True) ---
    cumulative = getattr(cfg.compute, "cumulative", True)

    # --- spectra: 2D HKE, RKE, and DKE and isotropic 1D ---
    hke_2d = kinetic_energy_spectra(u, v, name="hke")
    hke_1d = isotropize(hke_2d, dx, dy, cumulative=False)  # non-cumulative HKE spectra

    # wavenumber kappa² = k² + l² [m**-2]
    # RKE and DKE spectra
    rke_2d = rotational_kinetic_energy_spectra(vorticity, dx, dy)
    rke_1d = isotropize(rke_2d, dx, dy, cumulative=False)

    dke_2d = divergent_kinetic_energy_spectra(divergence, dx, dy)
    dke_1d = isotropize(dke_2d, dx, dy, cumulative=False)

    # --- spectra: 2D VKE and isotropic 1D ---
    vke_2d = vertical_kinetic_energy_spectra(w, name="vke")
    vke_1d = isotropize(vke_2d, dx, dy, cumulative=False)  # non-cumulative VKE spectra

    # --- spectra: 2D APE and isotropic 1D ---
    ape_2d = available_potential_energy_spectra(theta_prime, gamma, name="ape")
    ape_1d = isotropize(ape_2d, dx, dy, cumulative=False)  # non-cumulative APE spectra

    # ----- Calculate nonlinear spectral transfer → π(HKE) -----
    # "conservative" (default; skew-symmetric, sums to zero on periodic axes) | "flux" | "invariant"
    transfer_mode = getattr(cfg.compute, "transfer_form", None) or "conservative"
    if transfer_mode not in ("flux", "conservative", "invariant"):
        raise ValueError(f"transfer_form {transfer_mode!r}: expected flux, conservative or "
                         "invariant")
    conservative = transfer_mode == "conservative"

    # --- NONLINEAR SPECTRAL TRANSFER (pi_nke, pi_vke, pi_ape) ---
    if transfer_mode == "invariant":
        pi_nke_2d = nonlinear_hke_transfer_invariant(u, v, w, divergence, vorticity, name="pi_nke")
    else:
        pi_nke_2d = nonlinear_hke_transfer_flux(u, v, w, divergence, conservative=conservative,
                                                name="pi_nke")

    pi_nke = isotropize(pi_nke_2d, dx, dy, cumulative=cumulative)

    pi_lke_2d = coriolis_linear_transfer(u, v, name="pi_lke")
    pi_lke = isotropize(pi_lke_2d, dx, dy, cumulative=cumulative)

    pi_vke_2d = nonlinear_vke_transfer(u, v, w, divergence, conservative=conservative,
                                       name="pi_vke")
    pi_vke = isotropize(pi_vke_2d, dx, dy, cumulative=cumulative)

    pi_ape_2d = nonlinear_ape_transfer(u, v, w, theta_prime, gamma, divergence,
                                       conservative=conservative, name="pi_ape")
    pi_ape = isotropize(pi_ape_2d, dx, dy, cumulative=cumulative)

    # --- PRESSURE PERTURBATION (Exner deviation from the hydrostatic reference) ---
    exner_surface = domain_mean(exner.sel(z=exner.z.min()))
    exner_prime = exner - hydrostatic_exner_function(theta_mean, pi_sfc=exner_surface)

    # --- VERTICAL FLUXES (vf_*) and their DIVERGENCES (vfd_*) ---
    if staggered:
        stag = staggered_vertical_terms(u, v, w_half, theta_mean, theta_prime, gamma, exner_prime)
        vf_hke_2d, vfd_dke_2d = stag["vf_hke"], stag["vfd_hke"]
        vf_vke_2d, vfd_vke_2d = stag["vf_vke"], stag["vfd_vke"]
        vf_ape_2d, vfd_ape_2d = stag["vf_ape"], stag["vfd_ape"]
    else:
        vf_hke_2d = turbulent_hke_flux(u, v, w, name="vf_hke")
        vfd_dke_2d = vertical_gradient_flux(vf_hke_2d, name="vfd_dke")
        vf_vke_2d = turbulent_vke_flux(w, name="vf_vke")
        vfd_vke_2d = vertical_gradient_flux(vf_vke_2d, name="vfd_vke")
        vf_ape_2d = turbulent_ape_flux(w, theta_prime, gamma, name="vf_ape")
        vfd_ape_2d = vertical_gradient_flux(vf_ape_2d, name="vfd_ape")

    vf_hke = isotropize(vf_hke_2d.rename("vf_hke"), dx, dy, cumulative=cumulative)
    vfd_dke = isotropize(vfd_dke_2d.rename("vfd_dke"), dx, dy, cumulative=cumulative)
    vf_vke = isotropize(vf_vke_2d.rename("vf_vke"), dx, dy, cumulative=cumulative)
    vfd_vke = isotropize(vfd_vke_2d.rename("vfd_vke"), dx, dy, cumulative=cumulative)
    vf_ape = isotropize(vf_ape_2d.rename("vf_ape"), dx, dy, cumulative=cumulative)
    vfd_ape = isotropize(vfd_ape_2d.rename("vfd_ape"), dx, dy, cumulative=cumulative)

    # --- ADIABATIC NON-CONSERVATIVE (j_hke, j_vke, j_ape) ---
    # Note: nonconservative_adiabatic uses vf_hke_2d (or calculates it if not passed)
    j_hke_2d = nonconservative_adiabatic_hke(u, v, w, rho, vf_hke=vf_hke_2d, name="j_hke")
    j_hke = isotropize(j_hke_2d, dx, dy, cumulative=cumulative)

    # Note: nonconservative_adiabatic uses vf_vke_2d (or calculates it if not passed)
    j_vke_2d = nonconservative_adiabatic_vke(w, rho, vf_vke=vf_vke_2d, name="j_vke")
    j_vke = isotropize(j_vke_2d, dx, dy, cumulative=cumulative)

    # Note: nonconservative_adiabatic uses vf_ape_2d (or calculates it if not passed)
    j_ape_2d = nonconservative_adiabatic_ape(theta_prime, gamma, rho, vf_ape=vf_ape_2d,
                                             name="j_ape")
    j_ape = isotropize(j_ape_2d, dx, dy, cumulative=cumulative)

    # --- PRESSURE WORK & CONVERSION (vf_pres, vfd_pres, cad) ---
    if staggered:
        vf_pres_2d, vfd_pres_2d = stag["vf_pres"], stag["vfd_pres"]
    else:
        vf_pres_2d = pressure_flux(theta_mean, w, exner_prime, name="vf_pres")
        vfd_pres_2d = vertical_gradient_flux(vf_pres_2d, name="vfd_pres")
    vf_pres = isotropize(vf_pres_2d.rename("vf_pres"), dx, dy, cumulative=cumulative)
    vfd_pres = isotropize(vfd_pres_2d.rename("vfd_pres"), dx, dy, cumulative=cumulative)

    # Reference-profile term of the pressure flux (j_pres): ∂z F_p + J_p = θ̄ ρ̄⁻¹ ∂z(ρ̄ F_p / θ̄)
    j_pres_2d = nonconservative_adiabatic_pressure(vf_pres_2d, rho, theta_mean, name="j_pres")
    j_pres = isotropize(j_pres_2d, dx, dy, cumulative=cumulative)

    # Conversion (cad, caz)
    if staggered:
        cad_2d, caz_2d = stag["cad"], stag["caz"]
    else:
        cad_2d = conversion_ape_to_dke(theta_mean, w, exner_prime, name="cad")
        caz_2d = conversion_ape_to_vke(theta_mean, theta_prime, w, exner_prime, name="caz")
    cad = isotropize(cad_2d.rename("cad"), dx, dy, cumulative=cumulative)
    caz = isotropize(caz_2d.rename("caz"), dx, dy, cumulative=cumulative)

    # Pressure work by the anelastic-departure divergence (div_pres) and by θ' (np_hke, np_vke):
    # with these, ∂z F_p + J_p + C_{A→h} + Div_p + N_h is the exact horizontal pressure work.
    div_pres_2d = pressure_divergence_work(u, v, w, divergence, rho, theta_mean, exner_prime,
                                           dzw=dzw, name="div_pres")
    div_pres = isotropize(div_pres_2d, dx, dy, cumulative=cumulative)

    np_hke_2d = nonlinear_pressure_work_hke(u, v, theta_prime, exner_prime, name="np_hke")
    np_hke = isotropize(np_hke_2d, dx, dy, cumulative=cumulative)

    if staggered:
        np_vke_2d = stag["np_vke"].rename("np_vke")
    else:
        np_vke_2d = nonlinear_pressure_work_vke(w, theta_prime, exner_prime, name="np_vke")
    np_vke = isotropize(np_vke_2d, dx, dy, cumulative=cumulative)

    # --- COMPRESSIBILITY TERMS: D = ρ̄⁻¹∇·(ρ̄v), consistent with the J terms above ---
    div_hke_2d = divergence_hke(u, v, w, divergence, rho=rho, dzw=dzw, name="div_hke")
    div_hke = isotropize(div_hke_2d, dx, dy, cumulative=cumulative)

    div_vke_2d = divergence_vke(u, v, w, divergence, rho=rho, dzw=dzw, name="div_vke")
    div_vke = isotropize(div_vke_2d, dx, dy, cumulative=cumulative)

    div_ape_2d = divergence_ape(u, v, w, theta_prime, gamma, divergence, rho=rho, dzw=dzw,
                                name="div_ape")
    div_ape = isotropize(div_ape_2d, dx, dy, cumulative=cumulative)

    # --- ASSEMBLE ---
    fluxes = [hke_1d, rke_1d, dke_1d, vke_1d, ape_1d,
              pi_nke, pi_lke, pi_vke, pi_ape, vfd_dke, vfd_vke, vfd_ape,
              j_hke, j_vke, j_ape, vf_hke, vf_vke, vf_ape, cad, caz,
              vf_pres, vfd_pres, j_pres, div_pres, np_hke, np_vke,
              div_hke, div_vke, div_ape]

    # Filter out None and assemble into Dataset. Metadata by output name: terms built outside the
    # decorated functions (staggered vertical terms, flux divergences) would otherwise keep the
    # attributes of their inputs.
    fluxes = xr.Dataset({da.name: _set_budget_metadata(da, da.name)
                         for da in fluxes if da is not None})

    # wavenumber coord attrs (unchanged, but placed after assembly)
    fluxes.wavenumber.attrs.update({'standard_name': 'wavenumber',
                                    'long_name': 'horizontal wavenumber',
                                    'axis': 'X', 'units': 'rad / m'})

    return fluxes.transpose(..., "wavenumber")
