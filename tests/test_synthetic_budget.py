"""
Regression tests on synthetic periodic fields for the spectral transfers and the
Duchon-Robert inter-scale transfer.

Run with pytest, or directly:  python tests/test_synthetic_budget.py

Spectral checks use periodic fields with tapering and detrending switched off, where the exact
identities hold up to finite-difference error:
  * every nonlinear transfer sums to zero over wavenumbers (conservation);
  * t_h + dF_h/dz + Div_h equals the spectrum of -(u, (u.grad)u + w du/dz) (closure);
  * the flux and invariant forms of t_h agree.
Physical-space checks compare D_ell with a brute-force lattice evaluation of
  D_ell = 1/4 sum_s grad G_ell(s) . du |du|^2 dA,   sum_s G_ell dA = 1,
and test zero transfer for a steady Euler flow and positive (forward) transfer across shocks.
"""
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import budget.spectral_primitives as sp  # noqa: E402
from budget.budget import (nonlinear_hke_transfer_flux, nonlinear_hke_transfer_invariant,  # noqa: E402
                           nonlinear_vke_transfer, nonlinear_ape_transfer, turbulent_hke_flux,
                           vertical_gradient_flux, divergence_hke)
from budget.inter_scale_transfers import inter_scale_kinetic_energy_transfer  # noqa: E402
from budget.numeric_tools import (horizontal_divergence, relative_vorticity,  # noqa: E402
                                  horizontal_advection, stack_vector)
from budget.spectral_primitives import isotropize, vector_cross_spectrum  # noqa: E402


# ------------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------------
class _NoTaper:
    """Context manager: periodic-domain spectral setting (no taper, no detrending)."""

    def __enter__(self):
        self.saved = (sp.SPECTRAL_CFG["taper"]["enabled"], sp.SPECTRAL_CFG["detrend"]["type"])
        sp.SPECTRAL_CFG["taper"]["enabled"] = False
        sp.SPECTRAL_CFG["detrend"]["type"] = None
        return self

    def __exit__(self, *exc):
        sp.SPECTRAL_CFG["taper"]["enabled"], sp.SPECTRAL_CFG["detrend"]["type"] = self.saved


def _random_3d_fields(seed=3, n=40, nz=33, L=400e3, H=40e3, w_amp=0.05, m_max=4.0):
    """Smooth periodic (in x, y) fields with vertical structure; w deliberately not small."""
    rng = np.random.default_rng(seed)
    dx = L / n
    x = np.arange(n) * dx
    z = np.linspace(0, H, nz)
    Z, Y, X = np.meshgrid(z, x, x, indexing="ij")

    def field(amp=1.0):
        f = np.zeros_like(X)
        for _ in range(10):
            kx, ky = rng.integers(1, 5, 2)
            m = rng.uniform(1, m_max) * 2 * np.pi / H
            f += rng.normal() * np.cos(2 * np.pi * (kx * X + ky * Y) / L + m * Z + rng.uniform(0, 6.3))
        return amp * f

    co = {"z": z, "y": x, "x": x}
    da = lambda a: xr.DataArray(a, dims=("z", "y", "x"), coords=co)
    return da(field()), da(field()), da(field(w_amp)), da(field(0.5)), dx


def _shell_sum(da, dx):
    """Sum of a 2-D spectrum over all wavenumbers (including k = 0)."""
    return isotropize(da, dx, dx, cumulative=False, include_dc=True).sum("wavenumber")


def _interior(da):
    """Drop levels next to the top/bottom where vertical derivatives are one-sided."""
    return da.isel(z=slice(3, -3))


def _rel(a, scale):
    return float(np.abs(a).mean()) / scale


# ------------------------------------------------------------------------------------------------
# Spectral transfers
# ------------------------------------------------------------------------------------------------
def test_hke_transfer_conserves_and_closes():
    # n = 80: the invariant form relies on the discrete identity (u.grad)u = grad K + zeta k x u,
    # which holds only to second order (13% pointwise error at n = 40, 3% at n = 80).
    u, v, w, _, dx = _random_3d_fields(n=80)
    with _NoTaper():
        div, vor = horizontal_divergence(u, v), relative_vorticity(u, v)
        wind = stack_vector(u, v)
        adv = stack_vector(horizontal_advection(u, u, v) + w * u.differentiate("z"),
                           horizontal_advection(v, u, v) + w * v.differentiate("z"))
        exact = _interior(-_shell_sum(vector_cross_spectrum(wind, adv), dx))
        scale = float(np.abs(exact).mean())

        df = _interior(_shell_sum(vertical_gradient_flux(turbulent_hke_flux(u, v, w)), dx))
        dv = _interior(_shell_sum(divergence_hke(u, v, w, div), dx))

        for name, t2d in [("flux", nonlinear_hke_transfer_flux(u, v, w, div)),
                          ("invariant", nonlinear_hke_transfer_invariant(u, v, w, div, vor))]:
            t = _interior(_shell_sum(t2d, dx))
            conservation = _rel(t, scale)
            closure = _rel(t + df + dv - exact, scale)
            print(f"  t_h {name:9s}: sum_k t_h = {conservation:.3f}, closure error = {closure:.3f}"
                  f"  (relative to the advective tendency)")
            assert conservation < 0.04, f"{name} form does not conserve HKE ({conservation:.3f})"
            assert closure < 0.03, f"{name} form does not close the HKE budget ({closure:.3f})"


def test_flux_and_invariant_forms_agree():
    u, v, w, _, dx = _random_3d_fields(seed=5, n=80)
    with _NoTaper():
        div, vor = horizontal_divergence(u, v), relative_vorticity(u, v)
        a = _interior(isotropize(nonlinear_hke_transfer_flux(u, v, w, div), dx, dx))
        b = _interior(isotropize(nonlinear_hke_transfer_invariant(u, v, w, div, vor), dx, dx))
        diff = float(np.abs(a - b).max() / np.abs(a).max())
        print(f"  max |flux - invariant| / max |flux| = {diff:.3f}")
        assert diff < 0.1


def test_vke_and_ape_transfers_conserve():
    # w of the same order as u, so the vertical part of the transfer matters
    u, v, w, th, dx = _random_3d_fields(seed=7, n=80, w_amp=1.0)
    with _NoTaper():
        div = horizontal_divergence(u, v)
        gamma = xr.ones_like(th.z, dtype=float)
        for name, t2d, x in [("VKE", nonlinear_vke_transfer(u, v, w, div), w),
                             ("APE", nonlinear_ape_transfer(u, v, w, th, gamma, div), th)]:
            t = _interior(_shell_sum(t2d, dx))
            # scale: level-wise domain mean of the vertical advection term X w dX/dz, which is
            # what a non-antisymmetric vertical part leaves behind in sum_k t
            vert = _interior((x * w * x.differentiate("z")).mean(("y", "x")))
            scale = float(np.abs(vert).mean())
            conservation = _rel(t, scale)
            print(f"  t_{name}: sum_k t = {conservation:.3f} (relative to |<X w dX/dz>|)")
            assert conservation < 0.1, f"{name} transfer does not conserve ({conservation:.3f})"


# ------------------------------------------------------------------------------------------------
# Duchon-Robert inter-scale transfer
# ------------------------------------------------------------------------------------------------
def _mollifier(r, ell):
    q = (r / (2 * ell)) ** 2
    g = np.zeros_like(r, dtype=float)
    m = q < 1
    g[m] = np.exp(-1 / (1 - q[m]))
    return g


def _brute_force_dell(u, v, d, ell):
    """1/4 sum over all lattice shifts of grad G . du |du|^2 dA (periodic, mean over domain)."""
    m = int(np.ceil(2 * ell / d))
    sx, sy = np.meshgrid(np.arange(-m, m + 1), np.arange(-m, m + 1))
    r = np.hypot(sx, sy) * d
    g = _mollifier(r, ell)
    norm = g.sum() * d * d
    q = (r / (2 * ell)) ** 2
    dg = np.zeros_like(r)
    ok = (q < 1) & (r > 0)
    dg[ok] = -g[ok] * r[ok] / (2 * ell ** 2) / (1 - q[ok]) ** 2
    acc = 0.0
    for i, j in zip(*np.nonzero(dg)):
        du = np.roll(u, (-sy[i, j], -sx[i, j]), (0, 1)) - u
        dv = np.roll(v, (-sy[i, j], -sx[i, j]), (0, 1)) - v
        proj = (du * sx[i, j] + dv * sy[i, j]) * d / r[i, j]
        acc += dg[i, j] * float((proj * (du ** 2 + dv ** 2)).mean())
    return 0.25 * acc * d * d / norm


def _as_dataset(u, v, x):
    z = np.array([0.0, 1.0, 2.0])
    da = lambda a: xr.DataArray(np.repeat(a[None], 3, 0), dims=("z", "y", "x"),
                                coords={"z": z, "y": x, "x": x})
    return xr.Dataset({"u": da(u), "v": da(v)})


def _dell(u, v, x, scales):
    out = inter_scale_kinetic_energy_transfer(
        _as_dataset(u, v, x), scales=np.asarray(scales), x_coord_boundary="periodic",
        y_coord_boundary="periodic", allow_rechunk=False)
    return out["energy_transfer"].isel(z=1).mean(("x", "y")).values


def _synthetic_2d(seed=1, n=96, L=96e3):
    rng = np.random.default_rng(seed)
    d = L / n
    x = np.arange(n) * d
    X, Y = np.meshgrid(x, x, indexing="xy")
    u = np.zeros((n, n))
    v = np.zeros((n, n))
    for kx in range(1, 5):
        for ky in range(0, 5):
            a, b, c, e = rng.normal(size=4)
            ph = 2 * np.pi * (kx * X + ky * Y) / L
            u += (a * np.cos(ph) + b * np.sin(ph)) / (kx ** 2 + ky ** 2)
            v += (c * np.cos(ph) + e * np.sin(ph)) / (kx ** 2 + ky ** 2)
    u += 0.5 * np.tanh(4 * np.sin(2 * np.pi * X / L))  # steep fronts -> nonzero transfer
    return u, v, x, d


def test_dell_matches_lattice_sum():
    u, v, x, d = _synthetic_2d()
    factors = np.array([1.5, 2.0, 3.0, 6.0, 12.0])
    code = _dell(u, v, x, factors * d)
    ref = np.array([_brute_force_dell(u, v, d, f * d) for f in factors])
    ratio = code / ref
    for f, r in zip(factors, ratio):
        print(f"  ell = {f:4.1f} dx : D_ell(sbudget) / D_ell(lattice) = {r:.4f}")
    assert np.all(np.abs(ratio - 1) < 0.01), ratio


def test_dell_zero_for_steady_euler_flow():
    """Taylor-Green vortex: steady Euler solution, no inter-scale transfer on average."""
    n, L = 64, 2 * np.pi * 1e4
    d = L / n
    x = np.arange(n) * d
    X, Y = np.meshgrid(x, x, indexing="xy")
    k = 2 * np.pi / L
    u = np.sin(k * X) * np.cos(k * Y)
    v = -np.cos(k * X) * np.sin(k * Y)
    val = _dell(u, v, x, np.array([2.0, 4.0, 8.0]) * d)
    scale = float(np.abs(_dell(*_synthetic_2d(n=64, L=L)[:2], x, np.array([4.0]) * d)).max())
    print(f"  Taylor-Green: max |<D_ell>| = {np.abs(val).max():.2e} (shock field: {scale:.2e})")
    assert np.abs(val).max() < 1e-6 * max(scale, 1.0)


def test_dell_positive_across_shocks():
    """Sawtooth (Burgers-shock) profile: negative velocity jumps -> forward (positive) transfer."""
    n, L = 96, 96e3
    d = L / n
    x = np.arange(n) * d
    X, _ = np.meshgrid(x, x, indexing="xy")
    u = ((2 * np.pi * 3 * X / L) % (2 * np.pi)) - np.pi
    val = _dell(u, np.zeros_like(u), x, np.array([1.5, 3.0, 6.0]) * d)
    print(f"  sawtooth: <D_ell> = {val}")
    assert np.all(val > 0)


def _brute_force_dell_aniso(u, v, dx, dy, ell):
    """Brute-force lattice D_ell on a grid with dx != dy (arrays indexed [y, x])."""
    mx, my = int(np.ceil(2 * ell / dx)), int(np.ceil(2 * ell / dy))
    sx, sy = np.meshgrid(np.arange(-mx, mx + 1), np.arange(-my, my + 1))
    rx, ry = sx * dx, sy * dy
    r = np.hypot(rx, ry)
    g = _mollifier(r, ell)
    norm = g.sum() * dx * dy
    q = (r / (2 * ell)) ** 2
    dg = np.zeros_like(r)
    ok = (q < 1) & (r > 0)
    dg[ok] = -g[ok] * r[ok] / (2 * ell ** 2) / (1 - q[ok]) ** 2
    acc = 0.0
    for i, j in zip(*np.nonzero(dg)):
        du = np.roll(u, (-sy[i, j], -sx[i, j]), (0, 1)) - u
        dv = np.roll(v, (-sy[i, j], -sx[i, j]), (0, 1)) - v
        proj = (du * rx[i, j] + dv * ry[i, j]) / r[i, j]
        acc += dg[i, j] * float((proj * (du ** 2 + dv ** 2)).mean())
    return 0.25 * acc * dx * dy / norm


def _smooth_2d(n, L=96e3, seed=11):
    """Analytic smooth periodic field (resolution independent)."""
    rng = np.random.default_rng(seed)
    x = np.arange(n) * L / n
    X, Y = np.meshgrid(x, x, indexing="xy")
    u = np.zeros((n, n))
    v = np.zeros((n, n))
    for kx, ky in [(1, 0), (1, 1), (2, 1), (0, 2), (3, 1), (2, 3)]:
        a, b, c, e = rng.normal(size=4)
        ph = 2 * np.pi * (kx * X + ky * Y) / L
        u += a * np.cos(ph) + b * np.sin(ph)
        v += c * np.cos(ph) + e * np.sin(ph)
    return u, v, x, L / n


def test_lattice_bins_partition_shifts():
    """Every shift up to r_max lies in exactly one bin; small-r bins hold a single distance."""
    from budget.grid_utils import scale_increments
    n, d = 64, 1000.0
    x = xr.DataArray(np.arange(n) * d, dims="x", name="x", attrs={"units": "m"})
    inc = scale_increments(x, x.rename("y").swap_dims(x="y"))
    mask = inc["mask"].values
    per_shift = mask.sum(axis=0)
    dist = inc["distance_grid"].values
    r_max = float(inc["r"].max()) + 0.5 * d
    inside = (dist > 0) & (dist <= float(inc["r"].max()))
    assert per_shift.max() == 1, "a shift is assigned to more than one bin"
    assert np.all(per_shift[inside] == 1), "a shift inside r_max is not assigned to any bin"
    assert np.all(per_shift[dist == 0] == 0), "the zero shift must not be in any bin"
    assert np.isclose(float(inc["bin_area"].sum()), mask.sum() * d * d)
    for k, rk in enumerate(inc["r"].values):
        dk = dist[mask[k]]
        tol = 1e-9 * rk  # bin centres are means of distances: allow round-off
        assert dk.min() - tol <= rk <= dk.max() + tol and dk.max() <= r_max
        if rk < 8 * d - 1e-6 and dk.max() < 8 * d:
            assert np.allclose(dk, dk[0]), f"bin at r = {rk:.1f} mixes distances below 8 dx"


def test_lattice_kernel_normalization():
    """With support inside the exact-distance range, W equals 1/4 G'(|s|) / sum_s G(|s|) dA."""
    from budget.grid_utils import scale_increments
    from budget.inter_scale_transfers import lattice_kernel_weights
    n, d = 64, 1000.0
    x = xr.DataArray(np.arange(n) * d, dims="x", name="x", attrs={"units": "m"})
    inc = scale_increments(x, x.rename("y").swap_dims(x="y"))
    scales = xr.DataArray(np.array([1.5, 2.5, 3.5]) * d, dims="scale", name="scale")
    for p in (1.0, 2.0):
        w = lattice_kernel_weights(inc["r"], inc["bin_area"], scales, float(inc["cell_area"]), p=p)
        dist = inc["distance_grid"].values
        for ell in scales.values:
            q = (dist / (2 * ell)) ** 2
            g = np.where(q < 1, np.exp(-1 / (1 - np.minimum(q, 1 - 1e-12) ** p)), 0.0)
            norm = g.sum() * d * d
            r = inc["r"].values
            qr = (r / (2 * ell)) ** 2
            gr = np.where(qr < 1, np.exp(-1 / (1 - np.minimum(qr, 1 - 1e-12) ** p)), 0.0)
            dgr = np.where(qr < 1, -gr * 2 * p * (r / (2 * ell)) ** (2 * p - 1) / (2 * ell)
                           / (1 - np.minimum(qr, 1 - 1e-12) ** p) ** 2, 0.0)
            expect = 0.25 * dgr / norm
            got = w.sel(scale=ell).values
            assert np.allclose(got, expect, rtol=1e-6, atol=1e-30), f"p = {p}, ell = {ell}"


def test_dell_anisotropic_grid():
    """dx != dy: the lattice quadrature must use the true shift geometry and cell area."""
    rng = np.random.default_rng(4)
    nx, ny, dx, dy = 72, 48, 1000.0, 1500.0
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y, indexing="xy")
    Lx, Ly = nx * dx, ny * dy
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    for kx, ky in [(1, 1), (2, 1), (3, 2), (1, 3)]:
        a, b, c, e = rng.normal(size=4)
        ph = 2 * np.pi * (kx * X / Lx + ky * Y / Ly)
        u += a * np.cos(ph) + b * np.sin(ph)
        v += c * np.cos(ph) + e * np.sin(ph)
    u += 0.5 * np.tanh(4 * np.sin(2 * np.pi * X / Lx))
    z = np.array([0.0, 1.0, 2.0])
    da = lambda a: xr.DataArray(np.repeat(a[None], 3, 0), dims=("z", "y", "x"),
                                coords={"z": z, "y": y, "x": x})
    scales = np.array([2.0, 4.0, 8.0]) * 1500.0
    out = inter_scale_kinetic_energy_transfer(
        xr.Dataset({"u": da(u), "v": da(v)}), scales=scales, x_coord_boundary="periodic",
        y_coord_boundary="periodic", allow_rechunk=False)
    code = out["energy_transfer"].isel(z=1).mean(("x", "y")).values
    ref = np.array([_brute_force_dell_aniso(u, v, dx, dy, ell) for ell in scales])
    print(f"  anisotropic grid: D_ell(sbudget) / D_ell(lattice) = {np.round(code / ref, 4)}")
    assert np.all(np.abs(code / ref - 1) < 0.01)


def test_dell_converges_with_resolution():
    """Same analytic field at two resolutions: D_ell at fixed physical ell must agree.

    ell >= 4 dx on the coarse grid (dx = 2 km); at ell = 3 dx the estimator itself still carries
    ~3% discretization error relative to the continuous integral.
    """
    ell = np.array([8e3, 12e3])
    vals = []
    for n in (48, 96):
        u, v, x, _ = _smooth_2d(n)
        vals.append(_dell(u, v, x, ell))
    rel = np.abs(vals[0] / vals[1] - 1)
    print(f"  D_ell(n=48) / D_ell(n=96) - 1 = {np.round(vals[0] / vals[1] - 1, 4)}")
    assert np.all(rel < 0.03)


def test_dell_rotation_invariance():
    """Rotating the flow by 90 degrees maps the lattice onto itself: the mean must not change."""
    u, v, x, d = _synthetic_2d(n=64, L=64e3)
    a = _dell(u, v, x, np.array([2.0, 5.0]) * d)
    # np.rot90 on [y, x] arrays maps the old point (x, y) to (y, -x), a clockwise rotation;
    # the velocity must be rotated the same way: (u, v) -> (v, -u)
    ur, vr = np.rot90(v), -np.rot90(u)
    b = _dell(ur, vr, x, np.array([2.0, 5.0]) * d)
    print(f"  rotation: {a} vs {b}")
    assert np.allclose(a, b, rtol=1e-4)


def test_spectrum_parseval_and_cumulative_convention():
    """No taper: sum_k E_h (with k = 0) = domain mean of (u^2 + v^2)/2; cumulative runs high -> low k."""
    from budget.budget import kinetic_energy_spectra
    from budget.spectral_primitives import accumulate
    u, v, _, _, dx = _random_3d_fields(n=40)
    with _NoTaper():
        e2d = kinetic_energy_spectra(u, v)
        total = isotropize(e2d, dx, dx, cumulative=False, include_dc=True).sum("wavenumber")
        # bins stop at the Nyquist radius; the field has no energy beyond it (k <= 4 * 2 pi / L)
        expect = (0.5 * (u ** 2 + v ** 2)).mean(("y", "x"))
        assert np.allclose(total, expect, rtol=1e-6)
        e1d = isotropize(e2d, dx, dx, cumulative=False)
        cum = accumulate(e1d)
        manual = e1d.isel(wavenumber=slice(None, None, -1)).cumsum("wavenumber") \
            .isel(wavenumber=slice(None, None, -1))
        assert np.allclose(cum, manual)
        assert np.allclose(cum.isel(wavenumber=0), e1d.sum("wavenumber"))


# ------------------------------------------------------------------------------------------------
# Density-weighted closure of the advective and pressure terms (reference profiles rho(z), theta(z))
# ------------------------------------------------------------------------------------------------
def _stratified_case(seed=13, n=64):
    """Synthetic fields plus reference profiles of an isothermal-like atmosphere."""
    from budget.thermodynamics import lorenz_parameter
    # 97 levels over 40 km and vertical wavelengths >= 20 km: vertical finite-difference error is
    # small, so the tests isolate the algebra of the split
    kw = dict(n=n, nz=97, m_max=2.0)
    u, v, w, th, dx = _random_3d_fields(seed=seed, w_amp=0.5, **kw)
    z = u.z
    rho = xr.ones_like(u) * np.exp(-z / 7000.0)                    # rho_bar(z) after domain mean
    theta_mean = 250.0 * np.exp(0.02 ** 2 * z / 9.81)              # N = 0.02 s^-1
    theta_prime = 2.0 * th
    gamma = lorenz_parameter(theta_mean)
    exner_prime = 1e-3 * _random_3d_fields(seed=seed + 1, **kw)[0]
    return u, v, w, theta_prime, theta_mean, gamma, rho, exner_prime, dx


def _closure(lhs_terms, exact, dx):
    lhs = sum(_interior(_shell_sum(t, dx)) for t in lhs_terms)
    ex = _interior(_shell_sum(exact, dx))
    return _rel(lhs - ex, float(np.abs(ex).mean()))


def test_density_weighted_closure_hke_vke_ape():
    """t + dF/dz + J + Div reproduces the advective tendency for HKE, VKE and APE."""
    from budget.budget import (turbulent_vke_flux, turbulent_ape_flux, divergence_vke,
                               divergence_ape, nonconservative_adiabatic_hke,
                               nonconservative_adiabatic_vke, nonconservative_adiabatic_ape)
    from budget.spectral_primitives import scalar_cross_spectrum
    u, v, w, th, th_mean, gamma, rho, _, dx = _stratified_case()
    with _NoTaper():
        div = horizontal_divergence(u, v)
        wind = stack_vector(u, v)

        f_h = turbulent_hke_flux(u, v, w)
        adv_h = stack_vector(horizontal_advection(u, u, v) + w * u.differentiate("z"),
                             horizontal_advection(v, u, v) + w * v.differentiate("z"))
        err_h = _closure([nonlinear_hke_transfer_flux(u, v, w, div), vertical_gradient_flux(f_h),
                          nonconservative_adiabatic_hke(u, v, w, rho, vf_hke=f_h),
                          divergence_hke(u, v, w, div, rho=rho)],
                         -vector_cross_spectrum(wind, adv_h), dx)

        f_z = turbulent_vke_flux(w)
        err_z = _closure([nonlinear_vke_transfer(u, v, w, div), vertical_gradient_flux(f_z),
                          nonconservative_adiabatic_vke(w, rho, vf_vke=f_z),
                          divergence_vke(u, v, w, div, rho=rho)],
                         -scalar_cross_spectrum(w, horizontal_advection(w, u, v)
                                                + w * w.differentiate("z")), dx)

        f_a = turbulent_ape_flux(w, th, gamma)
        err_a = _closure([nonlinear_ape_transfer(u, v, w, th, gamma, div),
                          vertical_gradient_flux(f_a),
                          nonconservative_adiabatic_ape(th, gamma, rho, vf_ape=f_a),
                          divergence_ape(u, v, w, th, gamma, div, rho=rho)],
                         -gamma * scalar_cross_spectrum(th, horizontal_advection(th, u, v)
                                                        + w * th.differentiate("z")), dx)
    print(f"  closure error: HKE {err_h:.4f}, VKE {err_z:.4f}, APE {err_a:.4f}")
    assert max(err_h, err_z, err_a) < 0.02


def test_pressure_work_closure():
    """dF_p/dz + J_p + C_{A->h} + Div_p + N_h = -c_p (u, theta grad_h pi')  (exact pressure work)."""
    from budget.budget import (pressure_flux, conversion_ape_to_dke, pressure_divergence_work,
                               nonconservative_adiabatic_pressure, nonlinear_pressure_work_hke)
    from budget.constants import cp
    from budget.numeric_tools import horizontal_gradient
    u, v, w, th, th_mean, _, rho, exner, dx = _stratified_case(seed=17)
    with _NoTaper():
        div = horizontal_divergence(u, v)
        f_p = pressure_flux(th_mean, w, exner)
        gx, gy = horizontal_gradient(exner)
        theta = th_mean + th
        exact = -cp * vector_cross_spectrum(stack_vector(u, v), stack_vector(theta * gx, theta * gy))
        err = _closure([vertical_gradient_flux(f_p),
                        nonconservative_adiabatic_pressure(f_p, rho, th_mean),
                        conversion_ape_to_dke(th_mean, w, exner),
                        pressure_divergence_work(u, v, w, div, rho, th_mean, exner),
                        nonlinear_pressure_work_hke(u, v, th, exner)], exact, dx)
    print(f"  pressure-work closure error: {err:.4f}")
    assert err < 0.03


def test_vertical_pressure_buoyancy_closure():
    """C_{A->z} + N_z = (w, -c_p theta dpi/dz - g)_k with pi = hydrostatic reference + pi'."""
    from budget.budget import conversion_ape_to_vke, nonlinear_pressure_work_vke
    from budget.constants import cp, g
    from budget.spectral_primitives import scalar_cross_spectrum
    from budget.thermodynamics import hydrostatic_exner_function
    u, v, w, th, th_mean, _, _, exner_prime, dx = _stratified_case(seed=23)
    with _NoTaper():
        exner_ref = hydrostatic_exner_function(th_mean, pi_sfc=xr.DataArray(1.0))
        theta = th_mean + th
        dz_pi = (exner_ref + exner_prime).differentiate("z", edge_order=2)
        exact = scalar_cross_spectrum(w, -cp * theta * dz_pi - g)
        err = _closure([conversion_ape_to_vke(th_mean, th, w, exner_prime),
                        nonlinear_pressure_work_vke(w, th, exner_prime)], exact, dx)
    print(f"  vertical pressure/buoyancy closure error: {err:.4f}")
    assert err < 0.01


def test_layer_integral_gives_boundary_flux():
    """int rho_bar (dF/dz + J) dz = [rho_bar F] between layer bounds (density-weighted flux form)."""
    from budget.budget import nonconservative_adiabatic_hke
    u, v, w, *_, rho, _, dx = _stratified_case(seed=19)
    with _NoTaper():
        f_h = turbulent_hke_flux(u, v, w)
        src = isotropize(vertical_gradient_flux(f_h)
                         + nonconservative_adiabatic_hke(u, v, w, rho, vf_hke=f_h), dx, dx)
        flux = isotropize(f_h, dx, dx)
        rho_bar = rho.mean(("y", "x"))
        lay = slice(5, 25)
        integral = (rho_bar * src).isel(z=lay).integrate("z")
        bounds = (rho_bar * flux).isel(z=lay)
        jump = bounds.isel(z=-1) - bounds.isel(z=0)
        err = float(np.abs(integral - jump).max() / np.abs(jump).max())
    print(f"  layer integral vs boundary-flux difference: {err:.4f}")
    assert err < 0.02


_NP_MODE = {"periodic": "wrap", "reflect": "reflect", "edge": "edge", "symmetric": "symmetric"}


def _brute_force_dell_padded(fields, dx, dy, ell, x_mode, y_mode):
    """Exact lattice D_ell(x) on a non-periodic plane stack: fields padded with numpy modes."""
    m_x, m_y = int(np.ceil(2 * ell / dx)), int(np.ceil(2 * ell / dy))
    pad = {k: np.pad(np.pad(a, ((0, 0), (m_y, m_y), (0, 0)), mode=_NP_MODE[y_mode]),
                     ((0, 0), (0, 0), (m_x, m_x)), mode=_NP_MODE[x_mode]) for k, a in fields.items()}
    nz, ny, nx = fields["u"].shape
    sx, sy = np.meshgrid(np.arange(-m_x, m_x + 1), np.arange(-m_y, m_y + 1))
    r = np.hypot(sx * dx, sy * dy)
    g = _mollifier(r, ell)
    norm = g.sum() * dx * dy
    q = (r / (2 * ell)) ** 2
    dg = np.zeros_like(r)
    ok = (q < 1) & (r > 0)
    dg[ok] = -g[ok] * r[ok] / (2 * ell ** 2) / (1 - q[ok]) ** 2
    acc = np.zeros((nz, ny, nx))
    for i, j in zip(*np.nonzero(dg)):
        ys = slice(m_y + sy[i, j], m_y + sy[i, j] + ny)
        xs = slice(m_x + sx[i, j], m_x + sx[i, j] + nx)
        du = {k: pad[k][:, ys, xs] - fields[k] for k in fields}
        proj = (du["u"] * sx[i, j] * dx + du["v"] * sy[i, j] * dy) / r[i, j]
        acc += dg[i, j] * proj * sum(v ** 2 for v in du.values())
    return 0.25 * acc * dx * dy / norm


def _nonperiodic_case():
    rng = np.random.default_rng(29)
    ny, nx, d = 40, 56, 1000.0
    x, y, z = np.arange(nx) * d, np.arange(ny) * d, np.array([0.0, 1.0, 2.0])
    fields = {k: rng.normal(size=(3, ny, nx)) for k in ("u", "v", "w")}
    ds = xr.Dataset({k: (("z", "y", "x"), a.astype(np.float32)) for k, a in fields.items()},
                    coords={"z": z, "y": y, "x": x})
    fields = {k: ds[k].values.astype(np.float64) for k in fields}
    return ds, fields, d, np.array([1.5, 3.0, 6.0]) * d


def test_fft_path_matches_exact_lattice_nonperiodic():
    """FFT path == exact lattice sum with reflect, periodic/reflect and edge/symmetric boundaries."""
    ds, fields, d, scales = _nonperiodic_case()
    for xb, yb in [("reflect", "reflect"), ("periodic", "reflect"), ("edge", "symmetric")]:
        out = inter_scale_kinetic_energy_transfer(
            ds, scales=scales, x_coord_boundary=xb, y_coord_boundary=yb,
            allow_rechunk=False)["energy_transfer"].transpose("scale", "z", "y", "x").values
        for k, ell in enumerate(scales):
            ref = _brute_force_dell_padded(fields, d, d, ell, xb, yb)
            err = float(np.abs(out[k] - ref).max() / np.abs(ref).max())
            assert err < 1e-5, (xb, yb, ell, err)
        print(f"  boundaries x={xb:8s} y={yb:9s}: FFT path matches the exact lattice sum (<1e-5)")


def test_shift_fallback_close_to_exact():
    """r-bin streaming path (fallback): exact inside 8 dx, domain means within 1% beyond."""
    import budget.inter_scale_transfers as ist
    ds, fields, d, scales = _nonperiodic_case()
    saved = set(ist._FFT_PAD_MODES)
    try:
        ist._FFT_PAD_MODES.clear()  # force the shift path
        out = inter_scale_kinetic_energy_transfer(
            ds, scales=scales, x_coord_boundary="reflect", y_coord_boundary="reflect",
            allow_rechunk=False)["energy_transfer"].transpose("scale", "z", "y", "x").values
    finally:
        ist._FFT_PAD_MODES.update(saved)
    for k, ell in enumerate(scales):
        ref = _brute_force_dell_padded(fields, d, d, ell, "reflect", "reflect")
        point = float(np.abs(out[k] - ref).max() / np.abs(ref).max())
        mean = float(np.abs(out[k].mean((1, 2)) - ref.mean((1, 2))).max()
                     / np.abs(ref).mean())
        print(f"  shift path, ell = {ell / d:.1f} dx: pointwise error {point:.1e}, "
              f"domain-mean error {mean:.1e}")
        # exact inside 8 dx; beyond, the binned kernel is approximate pointwise on rough fields
        assert mean < 0.01, (ell, mean)
        if 2 * ell < 8 * d:
            assert point < 1e-5, (ell, point)


# ------------------------------------------------------------------------------------------------
# Divergence-corrected transfer D_ell^c = D_ell - C_ell (horizontally divergent flow)
# ------------------------------------------------------------------------------------------------
def _brute_force_exchange_padded(fields, div, dx, dy, ell, x_mode, y_mode):
    """-C_ell = 1/8 sum_s G(s) [(d(x) + d(x+s)) |du|^2 - d[d] d[|u|^2]] dA / sum G dA, padded."""
    m_x, m_y = int(np.ceil(2 * ell / dx)), int(np.ceil(2 * ell / dy))
    pad = lambda a: np.pad(np.pad(a, ((0, 0), (m_y, m_y), (0, 0)), mode=_NP_MODE[y_mode]),
                           ((0, 0), (0, 0), (m_x, m_x)), mode=_NP_MODE[x_mode])
    allf = dict(fields, div=div)
    padded = {k: pad(a) for k, a in allf.items()}
    nz, ny, nx = div.shape
    sx, sy = np.meshgrid(np.arange(-m_x, m_x + 1), np.arange(-m_y, m_y + 1))
    g = _mollifier(np.hypot(sx * dx, sy * dy), ell)
    norm = g.sum() * dx * dy
    k0 = sum(a ** 2 for a in fields.values())
    acc = np.zeros((nz, ny, nx))
    for i, j in zip(*np.nonzero(g)):
        ys = slice(m_y + sy[i, j], m_y + sy[i, j] + ny)
        xs = slice(m_x + sx[i, j], m_x + sx[i, j] + nx)
        sh = {k: padded[k][:, ys, xs] for k in allf}
        dq = sum((sh[k] - fields[k]) ** 2 for k in fields)
        dk = sum(sh[k] ** 2 for k in fields) - k0
        acc += g[i, j] * ((div + sh["div"]) * dq - (sh["div"] - div) * dk)
    return 0.125 * acc * dx * dy / norm


def test_divergence_correction_matches_increment_form():
    """FFT C_ell == the increment form of the derivation, any boundary; D_ell itself unchanged."""
    ds, fields, d, scales = _nonperiodic_case()
    div_np = np.random.default_rng(5).normal(size=fields["u"].shape)
    div = xr.DataArray(div_np.astype(np.float32), dims=("z", "y", "x"), coords=ds["u"].coords)
    div_np = div.values.astype(np.float64)
    for xb, yb in [("reflect", "reflect"), ("periodic", "reflect"), ("edge", "symmetric")]:
        kw = dict(scales=scales, x_coord_boundary=xb, y_coord_boundary=yb, allow_rechunk=False)
        out = inter_scale_kinetic_energy_transfer(ds, divergence_correction=True, divergence=div,
                                                  **kw).transpose("scale", "z", "y", "x")
        plain = inter_scale_kinetic_energy_transfer(ds, **kw).transpose("scale", "z", "y", "x")
        assert np.array_equal(out["energy_transfer"].values, plain["energy_transfer"].values)
        minus_c = (out["energy_transfer_corrected"] - out["energy_transfer"]).values
        for k, ell in enumerate(scales):
            ref = _brute_force_exchange_padded(fields, div_np, d, d, ell, xb, yb)
            err = float(np.abs(minus_c[k] - ref).max() / np.abs(ref).max())
            assert err < 1e-4, (xb, yb, ell, err)
        print(f"  boundaries x={xb:8s} y={yb:9s}: C_ell matches the increment form (<1e-4), "
              f"D_ell unchanged")
    zero = inter_scale_kinetic_energy_transfer(ds, divergence_correction=True,
                                               divergence=xr.zeros_like(div), scales=scales,
                                               allow_rechunk=False)
    assert np.array_equal(zero["energy_transfer_corrected"].values,
                          zero["energy_transfer"].values), "C_ell must vanish for div = 0"
    print("  div = 0: D_ell^c == D_ell exactly")


def _spectral_divergent_case(n=64, L=64e3, seed=17):
    """Periodic trigonometric (u, v), both rotational and divergent, with exact derivatives."""
    rng = np.random.default_rng(seed)
    d = L / n
    x = np.arange(n) * d
    X, Y = np.meshgrid(x, x, indexing="xy")
    u = np.zeros((n, n))
    v = np.zeros((n, n))
    for _ in range(14):
        kx, ky = rng.integers(-4, 5, 2)
        ph = 2 * np.pi * (kx * X + ky * Y) / L + rng.uniform(0, 2 * np.pi)
        u += rng.normal() * np.cos(ph)
        v += rng.normal() * np.cos(ph + rng.uniform(0, 2 * np.pi))
    k1 = 2 * np.pi * np.fft.fftfreq(n, d)
    KX, KY = np.meshgrid(k1, k1, indexing="xy")
    ddx = lambda f: np.real(np.fft.ifft2(1j * KX * np.fft.fft2(f)))
    ddy = lambda f: np.real(np.fft.ifft2(1j * KY * np.fft.fft2(f)))
    return u, v, x, d, ddx, ddy


def test_divergence_corrected_dell_closes_local_budget():
    """Exact local identity -1/2 [ubar.N + u.Nbar] = -div J - D_ell^c for divergent flow.

    N = (U.grad)U + 1/2 div(U) U (skew-symmetric, the flux-form Pi_h), J the Duchon-Robert flux,
    bar the lattice filter. Derivatives are spectral (exact for these fields); the remaining
    error is the lattice quadrature of grad G in D_ell (decreases with ell / dx). Plain D_ell
    must fail the same identity.
    """
    u, v, x, d, ddx, ddy = _spectral_divergent_case()
    div = ddx(u) + ddy(v)
    assert np.abs(div).max() > 0.5 * np.abs(ddx(u)).max()          # strongly divergent
    ell = 8 * d
    ds = _as_dataset(u, v, x)
    div_da = xr.DataArray(np.repeat(div[None], 3, 0), dims=("z", "y", "x"), coords=ds["u"].coords)
    out = inter_scale_kinetic_energy_transfer(
        ds, scales=np.array([ell]), x_coord_boundary="periodic", y_coord_boundary="periodic",
        allow_rechunk=False, divergence_correction=True, divergence=div_da)
    dell = out["energy_transfer"].isel(z=1, scale=0).transpose("y", "x").values.astype(float)
    dc = out["energy_transfer_corrected"].isel(z=1, scale=0).transpose("y", "x").values
    dc = dc.astype(float)

    m = int(np.ceil(2 * ell / d))
    sx, sy = np.meshgrid(np.arange(-m, m + 1), np.arange(-m, m + 1))
    g = _mollifier(np.hypot(sx, sy) * d, ell)
    g = g / g.sum()
    bar = lambda f: sum(g[i, j] * np.roll(f, (-sy[i, j], -sx[i, j]), (0, 1))
                        for i, j in zip(*np.nonzero(g)))
    ub, vb = bar(u), bar(v)
    k = u ** 2 + v ** 2
    nu = u * ddx(u) + v * ddy(u) + 0.5 * div * u
    nv = u * ddx(v) + v * ddy(v) + 0.5 * div * v
    s_f = -0.5 * (ub * nu + vb * nv + u * bar(nu) + v * bar(nv))
    jx = 0.5 * u * (u * ub + v * vb) + 0.25 * (bar(u * k) - u * bar(k))
    jy = 0.5 * v * (u * ub + v * vb) + 0.25 * (bar(v * k) - v * bar(k))
    div_j = ddx(jx) + ddy(jy)

    scale = np.abs(dell).max()
    res_c = float(np.abs(s_f + div_j + dc).max() / scale)
    res_d = float(np.abs(s_f + div_j + dell).max() / scale)
    mean_c = abs(dc.mean() + s_f.mean()) / abs(s_f.mean())
    print(f"  ell = 8 dx: local residual with D_ell^c {res_c:.1e}, with D_ell {res_d:.2f};"
          f" domain mean: |<D^c> + <S>| / |<S>| = {mean_c:.1e}")
    assert res_c < 2e-3, res_c
    assert res_d > 0.1, res_d                     # the test discriminates
    assert mean_c < 2e-3, mean_c


def _spectral_3d_case(n=64, L=64e3, H=8e3, u0=6.0, seed=23):
    """Periodic (x, y) trigonometric u, v, w with vertical structure, exact derivatives at z = 0.

    u and v include a uniform background u0 (the flux-form transfer depends on it when the flow
    is divergent). Returns 2-D fields at z = 0 and the exact derivative operators.
    """
    rng = np.random.default_rng(seed)
    d = L / n
    x = np.arange(n) * d
    X, Y = np.meshgrid(x, x, indexing="xy")
    f = {k: np.zeros((n, n)) for k in ("u", "v", "w", "dzu", "dzv")}
    for _ in range(14):
        kx, ky = rng.integers(-4, 5, 2)
        m = 2 * np.pi / H * rng.uniform(0.5, 2.0)
        ph = 2 * np.pi * (kx * X + ky * Y) / L
        for c, amp in (("u", 1.0), ("v", 1.0), ("w", 0.6)):
            a, phase = rng.normal() * amp, rng.uniform(0, 2 * np.pi)
            f[c] += a * np.cos(ph + phase)                 # value at z = 0 of cos(ph + m z + phase)
            if c != "w":
                f[f"dz{c}"] += -a * m * np.sin(ph + phase)
    f["u"] += u0
    f["v"] += 0.5 * u0
    k1 = 2 * np.pi * np.fft.fftfreq(n, d)
    KX, KY = np.meshgrid(k1, k1, indexing="xy")
    ddx = lambda g: np.real(np.fft.ifft2(1j * KX * np.fft.fft2(g)))
    ddy = lambda g: np.real(np.fft.ifft2(1j * KY * np.fft.fft2(g)))
    return f, x, d, ddx, ddy


def _lattice_filter_response(n, d, ell):
    """G^(k) of the normalized lattice mollifier on the periodic n x n grid (real, symmetric)."""
    m = int(np.ceil(2 * ell / d))
    kernel = np.zeros((n, n))
    for i in range(-m, m + 1):
        for j in range(-m, m + 1):
            kernel[j % n, i % n] += _mollifier(np.array([np.hypot(i, j) * d]), ell)[0]
    return np.real(np.fft.fft2(kernel / kernel.sum()))


def test_corrected_transfer_equals_kernel_weighted_spectral_transfer():
    """<D_ell - C_ell + V_ell> == sum_k T(k) [1 - G^(k)]: the spectral counterpart, computed
    independently in Fourier space with exact derivatives.

    T(k) is the flux-form HKE transfer of sbudget's spectral budget,
        T = -<U, (U.grad)U + 1/2 div U + 1/2 w dz U>_k + 1/2 <dz U, w U>_k,
    with divergent flow, vertical advection and a uniform background flow. The physical-space
    side uses increments of (u, v) only. The remaining error is the lattice quadrature of grad G
    in D_ell, which decreases with ell / dx. D_ell alone and D_ell - C_ell (no vertical term)
    must both miss the spectral value.
    """
    f, x, d, ddx, ddy = _spectral_3d_case()
    n = x.size
    u, v, w, dzu, dzv = (f[k] for k in ("u", "v", "w", "dzu", "dzv"))
    div = ddx(u) + ddy(v)

    # spectral side (Fourier space, exact derivatives)
    hat = lambda g: np.fft.fft2(g) / n ** 2
    cross = lambda a, b: np.real(np.conj(hat(a)) * hat(b))            # sums to mean(a b)
    au = u * ddx(u) + v * ddy(u) + 0.5 * div * u + 0.5 * w * dzu
    av = u * ddx(v) + v * ddy(v) + 0.5 * div * v + 0.5 * w * dzv
    t2d = -(cross(u, au) + cross(v, av)) + 0.5 * (cross(dzu, w * u) + cross(dzv, w * v))
    assert abs(t2d.sum()) < 1e-10 * np.abs(t2d).sum()                # conservative

    da = lambda g: xr.DataArray(np.repeat(g[None], 3, 0), dims=("z", "y", "x"),
                                coords={"z": [-1.0, 0.0, 1.0], "y": x, "x": x})
    ds = xr.Dataset({"u": da(u), "v": da(v), "w": da(w)})
    print("  ell/dx   spectral      D*        D*/spec-1   (D - C)/spec   D/spec")
    for factor in (5.0, 8.0, 12.0):
        ell = factor * d
        spec = float((t2d * (1.0 - _lattice_filter_response(n, d, ell))).sum())
        kw = dict(scales=np.array([ell]), x_coord_boundary="periodic",
                  y_coord_boundary="periodic", allow_rechunk=False,
                  increment_components=("u", "v"), divergence=da(div))
        full = inter_scale_kinetic_energy_transfer(ds, divergence_correction=True,
                                                   vertical_advection=True, dzu=da(dzu),
                                                   dzv=da(dzv), **kw).isel(z=1, scale=0)
        hor = inter_scale_kinetic_energy_transfer(ds, divergence_correction=True,
                                                  **kw).isel(z=1, scale=0)
        d_star = float(full["energy_transfer_corrected"].mean())
        d_hor = float(hor["energy_transfer_corrected"].mean())
        d_plain = float(full["energy_transfer"].mean())
        err = abs(d_star / spec - 1)
        print(f"  {factor:5.1f}  {spec: .4e}  {d_star: .4e}  {err:9.1e}   {d_hor / spec:9.3f}"
              f"   {d_plain / spec:7.3f}")
        assert err < (5e-3 if factor < 8 else 1e-3), (factor, err)
        assert abs(d_hor / spec - 1) > 0.05 and abs(d_plain / spec - 1) > 0.05


def test_perturbation_frame():
    """perturbation=True: transfer of the perturbation about the level-mean wind.

    D_ell is unchanged (Galilean invariant); D_ell* equals the kernel-weighted spectral transfer
    of the perturbation fields and differs from the ground-frame value, as the flux-form
    spectral transfer does (the 1/2 div c term couples the mean flow c to every wavenumber).
    """
    f, x, d, ddx, ddy = _spectral_3d_case()
    n = x.size
    u, v, w, dzu, dzv = (f[k] for k in ("u", "v", "w", "dzu", "dzv"))
    up, vp = u - u.mean(), v - v.mean()
    div = ddx(u) + ddy(v)
    hat = lambda g: np.fft.fft2(g) / n ** 2
    cross = lambda a, b: np.real(np.conj(hat(a)) * hat(b))

    def spectral(uu, vv, ell):
        au = uu * ddx(uu) + vv * ddy(uu) + 0.5 * div * uu + 0.5 * w * dzu
        av = uu * ddx(vv) + vv * ddy(vv) + 0.5 * div * vv + 0.5 * w * dzv
        t = -(cross(uu, au) + cross(vv, av)) + 0.5 * (cross(dzu, w * uu) + cross(dzv, w * vv))
        return float((t * (1.0 - _lattice_filter_response(n, d, ell))).sum())

    da = lambda g: xr.DataArray(np.repeat(g[None], 3, 0), dims=("z", "y", "x"),
                                coords={"z": [-1.0, 0.0, 1.0], "y": x, "x": x})
    ds = xr.Dataset({"u": da(u), "v": da(v), "w": da(w)})
    ell = 8 * d
    kw = dict(scales=np.array([ell]), x_coord_boundary="periodic", y_coord_boundary="periodic",
              allow_rechunk=False, increment_components=("u", "v"), divergence=da(div),
              divergence_correction=True, vertical_advection=True, dzu=da(dzu), dzv=da(dzv))
    ground = inter_scale_kinetic_energy_transfer(ds, **kw).isel(z=1, scale=0)
    pert = inter_scale_kinetic_energy_transfer(ds, perturbation=True, **kw).isel(z=1, scale=0)
    assert np.allclose(pert["energy_transfer"], ground["energy_transfer"], rtol=0,
                       atol=1e-5 * float(np.abs(ground["energy_transfer"]).max()))
    s_g, s_p = spectral(u, v, ell), spectral(up, vp, ell)
    d_g = float(ground["energy_transfer_corrected"].mean())
    d_p = float(pert["energy_transfer_corrected"].mean())
    print(f"  ground frame: D* = {d_g:.4e}, spectral {s_g:.4e};  perturbation: D* = {d_p:.4e}, "
          f"spectral {s_p:.4e}  (frames differ by {abs(s_g / s_p - 1):.0%})")
    assert abs(d_p / s_p - 1) < 1e-3 and abs(d_g / s_g - 1) < 1e-3
    assert abs(s_g / s_p - 1) > 0.05


def _lattice_filter_at(kx, ky, d, ell):
    """G^(k) = sum_s G(s) cos(k.s) / sum_s G(s) of the lattice mollifier at one wavevector."""
    m = int(np.ceil(2 * ell / d))
    sx, sy = np.meshgrid(np.arange(-m, m + 1), np.arange(-m, m + 1))
    g = _mollifier(np.hypot(sx, sy) * d, ell)
    return float((g * np.cos(kx * sx * d + ky * sy * d)).sum() / g.sum())


def _planes(fields, x, nz=3):
    """Stack 2-D periodic fields into (z, y, x) DataArrays with nz identical levels."""
    co = {"z": np.arange(nz, dtype=float), "y": x, "x": x}
    return {k: xr.DataArray(np.repeat(a[None], nz, 0), dims=("z", "y", "x"), coords=co)
            for k, a in fields.items()}


def test_corrected_transfer_closed_form_triad():
    """Closed form for the dilatational exchange: a divergent triad along x.

    u = c + A cos(kx) + B cos(2kx + b), v = w = 0. The flux-form transfer is
    t(k) = -t(2k) = 3/8 k A^2 B sin b (c drops out in 1-D), so
        <D_ell - C_ell> = 3/8 k A^2 B sin b [G^(2k) - G^(k)],
    with G^ the lattice filter. The remaining error is the lattice quadrature of grad G in D_ell,
    which falls steeply with ell / dx. D_ell alone must miss the closed form.
    """
    n, L = 64, 64e3
    d = L / n
    x = np.arange(n) * d
    X = np.meshgrid(x, x, indexing="xy")[0]
    k, A, B, b, c = 2 * np.pi / L, 1.3, 0.8, 0.9, 4.0
    th = k * X
    u = c + A * np.cos(th) + B * np.cos(2 * th + b)
    div = -k * A * np.sin(th) - 2 * k * B * np.sin(2 * th + b)
    f = _planes({"u": u, "v": np.zeros_like(u), "div": div}, x)
    ds = xr.Dataset({"u": f["u"], "v": f["v"]})
    print("  ell/dx   closed form   <D*>/closed-1   <D>/closed")
    for factor, tol in ((6.0, 5e-3), (8.0, 1e-3), (12.0, 1e-4), (16.0, 1e-5)):
        ell = factor * d
        exact = 3 / 8 * k * A ** 2 * B * np.sin(b) * (_lattice_filter_at(2 * k, 0, d, ell)
                                                     - _lattice_filter_at(k, 0, d, ell))
        out = inter_scale_kinetic_energy_transfer(
            ds, scales=np.array([ell]), x_coord_boundary="periodic", y_coord_boundary="periodic",
            allow_rechunk=False, divergence_correction=True, divergence=f["div"]).isel(z=1, scale=0)
        d_star = float(out["energy_transfer_corrected"].mean())
        d_plain = float(out["energy_transfer"].mean())
        err = abs(d_star / exact - 1)
        print(f"  {factor:5.1f}  {exact: .4e}  {err:13.1e}   {d_plain / exact:9.3f}")
        assert err < tol, (factor, err)
        assert abs(d_plain / exact - 1) > 0.05


def _boussinesq_wave(damping, n=64, L=64e3, N=0.02, amp=1.0):
    """Steady internal gravity wave in a uniform wind c, exact linear Boussinesq solution.

    Rayleigh friction and Newtonian cooling at the rate `damping` give
    m^2 = N^2 k^2 / (k.c - i r)^2 - k^2; the root with Im m >= 0 decays upwards. The horizontal
    velocity is along k (no rotation) and w = -|k| u^ / m from continuity. Returns the fields at
    z = 0 with exact derivatives, the wavevector and 1/2 c . dz<u'w>, the closed-form prefactor.
    """
    d = L / n
    x = np.arange(n) * d
    X, Y = np.meshgrid(x, x, indexing="xy")
    kv = 2 * np.pi / L * np.array([2.0, 1.0])
    kh = float(np.hypot(*kv))
    khat = kv / kh
    c = np.array([20.0, 5.0])
    m = np.sqrt(N ** 2 * kh ** 2 / (kv @ c - 1j * damping) ** 2 - kh ** 2 + 0j)
    if m.imag < 0 or (m.imag == 0 and m.real < 0):
        m = -m
    phase = np.exp(1j * (kv[0] * X + kv[1] * Y))
    up = np.real(amp * phase)                       # wave velocity along k
    dz_up = np.real(1j * m * amp * phase)           # exact dz, including the decay
    fields = {"u": c[0] + khat[0] * up, "v": c[1] + khat[1] * up,
              "w": np.real(-kh * amp / m * phase), "dzu": khat[0] * dz_up,
              "dzv": khat[1] * dz_up, "div": np.real(1j * kh * amp * phase)}
    dz_stress = m.imag * kh * amp ** 2 * m.real / abs(m) ** 2      # dz<u'w> along k at z = 0
    return fields, x, d, kv, 0.5 * float(c @ khat) * dz_stress


def test_corrected_transfer_closed_form_boussinesq_wave():
    """Closed form for the vertical exchange: a steady Boussinesq gravity wave in a uniform wind.

    One horizontal wavenumber, so <D_ell> = <C_ell> = 0, while the flux-form transfer exchanges
    energy between the wave and the mean wind c:
        <D*_ell> = <V_ell> = 1/2 [1 - G^(k)] c . <w dz u'> = 1/2 [1 - G^(k)] c . dz<u'w>
    (Boussinesq continuity). A conservative wave has a constant stress (Eliassen and Palm 1961)
    and <D*> = 0, although C_ell and V_ell are not zero pointwise; a damped wave decays with
    height, its stress converges, and the mean wind loses energy to it (<D*> > 0). In the
    perturbation frame both vanish.
    """
    damping = 0.2 * abs(np.array([2.0, 1.0]) @ np.array([20.0, 5.0])) * 2 * np.pi / 64e3
    for label, r in (("conservative", 0.0), ("damped", damping)):
        fields, x, d, kv, unit = _boussinesq_wave(r)
        f = _planes(fields, x)
        ds = xr.Dataset({k: f[k] for k in ("u", "v", "w")})
        kw = dict(x_coord_boundary="periodic", y_coord_boundary="periodic", allow_rechunk=False,
                  increment_components=("u", "v"), divergence_correction=True,
                  vertical_advection=True, divergence=f["div"], dzu=f["dzu"], dzv=f["dzv"])
        w_dzu = float((f["w"] * (f["dzu"] * kv[0] + f["dzv"] * kv[1])).isel(z=1).mean())
        assert abs(w_dzu) < 1e-12 if r == 0 else w_dzu > 0              # <w dz u'> along k
        for factor in (6.0, 10.0):
            ell = factor * d
            exact = unit * (1 - _lattice_filter_at(*kv, d, ell))
            ground = inter_scale_kinetic_energy_transfer(ds, scales=np.array([ell]),
                                                         **kw).isel(z=1, scale=0)
            pert = inter_scale_kinetic_energy_transfer(ds, scales=np.array([ell]),
                                                       perturbation=True, **kw).isel(z=1, scale=0)
            d_star = float(ground["energy_transfer_corrected"].mean())
            d_plain = float(ground["energy_transfer"].mean())
            d_pert = float(pert["energy_transfer_corrected"].mean())
            corr_max = float(np.abs(ground["energy_transfer_corrected"]
                                    - ground["energy_transfer"]).max())
            scale = abs(exact) if r else corr_max
            print(f"  {label:12s} ell = {factor:.0f} dx: closed form {exact: .4e}, <D*> {d_star: .4e};"
                  f" <D> {d_plain: .1e}, perturbation <D*> {d_pert: .1e}, max |V - C| {corr_max:.1e}")
            assert abs(d_star - exact) < 1e-4 * scale, (label, factor, d_star, exact)
            assert abs(d_plain) < 1e-6 * scale and abs(d_pert) < 1e-6 * scale
            assert corr_max > 0.1 * scale                    # the corrections are not zero pointwise
        if r:
            assert exact > 0                                 # mean wind loses energy to the wave


def test_correction_divergence_source():
    """The correction uses the divergence the spectral budget uses: an explicit `divergence`,
    else the dataset's divergence when it also holds the vorticity, else centred differences."""
    ds, fields, d, scales = _nonperiodic_case()
    rng = np.random.default_rng(41)
    given = xr.DataArray(rng.normal(size=fields["u"].shape).astype(np.float32),
                         dims=("z", "y", "x"), coords=ds["u"].coords)
    kw = dict(scales=scales[:1], allow_rechunk=False, divergence_correction=True)
    corrected = lambda data, **extra: inter_scale_kinetic_energy_transfer(
        data, **kw, **extra)["energy_transfer_corrected"].values

    with_both = ds.assign(divergence=given, vorticity=xr.zeros_like(given))
    with_div = ds.assign(divergence=given)
    finite_diff = horizontal_divergence(ds["u"], ds["v"])
    assert np.array_equal(corrected(with_both), corrected(ds, divergence=given))
    assert np.array_equal(corrected(with_div), corrected(ds, divergence=finite_diff))
    assert np.array_equal(corrected(with_both, divergence=finite_diff),
                          corrected(ds, divergence=finite_diff))
    assert not np.allclose(corrected(with_both), corrected(with_div))      # discriminates
    print("  divergence + vorticity in the dataset: dataset divergence; divergence alone: "
          "centred differences; explicit divergence overrides both")


def test_correction_error_paths():
    """Invalid requests fail with a clear error before any computation."""
    from budget.inter_scale_transfers import transfer_correction
    ds, fields, d, scales = _nonperiodic_case()
    kw = dict(scales=scales[:1], allow_rechunk=False)
    cases = [
        ("w missing", ValueError,
         lambda: inter_scale_kinetic_energy_transfer(ds[["u", "v"]], vertical_advection=True, **kw)),
        ("u only", ValueError,
         lambda: inter_scale_kinetic_energy_transfer(ds, increment_components=("u",), **kw)),
        ("w requested, absent", ValueError,
         lambda: inter_scale_kinetic_energy_transfer(ds[["u", "v"]],
                                                     increment_components=("u", "v", "w"), **kw)),
        ("2 levels, no shear", ValueError,
         lambda: inter_scale_kinetic_energy_transfer(ds.isel(z=slice(0, 2)),
                                                     vertical_advection=True, **kw)),
        ("constant boundary", NotImplementedError,
         lambda: inter_scale_kinetic_energy_transfer(ds, divergence_correction=True,
                                                     x_coord_boundary="constant", **kw)),
        ("no term requested", ValueError,
         lambda: transfer_correction(ds, {}, xr.DataArray(scales, dims="scale"), "x", "y",
                                     ["u", "v"], divergence=False, vertical=False)),
    ]
    for label, error, call in cases:
        try:
            call()
        except error as e:
            print(f"  {label:20s}: {error.__name__}: {e}")
        else:
            raise AssertionError(f"{label}: no {error.__name__}")
    # without the correction, the shift path still handles the constant boundary
    inter_scale_kinetic_energy_transfer(ds, x_coord_boundary="constant", **kw)


def test_correction_geographic_grid():
    """On a lon-lat grid the correction uses the lattice filter of D_ell and the spherical
    divergence: finite, zero for non-divergent input, and non-zero otherwise."""
    lon = xr.DataArray(np.arange(0.0, 16.0, 0.25), dims="lon", name="lon",
                       attrs={"units": "degrees_east", "standard_name": "longitude"})
    lat = xr.DataArray(np.arange(30.0, 42.0, 0.25), dims="lat", name="lat",
                       attrs={"units": "degrees_north", "standard_name": "latitude"})
    rng = np.random.default_rng(13)
    shape = (3, lat.size, lon.size)
    LON, LAT = np.meshgrid(np.deg2rad(lon.values), np.deg2rad(lat.values))
    wave = lambda: sum(rng.normal() * np.cos(kx * LON * 20 + ky * LAT * 20 + rng.uniform(0, 6))
                       for kx, ky in rng.integers(1, 4, (6, 2)))
    co = {"z": [0.0, 1.0, 2.0], "lat": lat, "lon": lon}
    da = lambda a: xr.DataArray(np.broadcast_to(a, shape).astype(np.float32),
                                dims=("z", "lat", "lon"), coords=co)
    ds = xr.Dataset({"u": da(5 + wave()), "v": da(wave())})
    kw = dict(scales=np.array([80e3, 150e3]), allow_rechunk=False, divergence_correction=True)
    out = inter_scale_kinetic_energy_transfer(ds, **kw)
    zero = inter_scale_kinetic_energy_transfer(ds, divergence=xr.zeros_like(ds["u"]), **kw)
    corr = (out["energy_transfer_corrected"] - out["energy_transfer"]).values
    assert np.isfinite(out["energy_transfer_corrected"].values).all()
    assert np.array_equal(zero["energy_transfer_corrected"].values, zero["energy_transfer"].values)
    ratio = np.abs(corr).max() / np.abs(out["energy_transfer"].values).max()
    print(f"  lon-lat grid: finite; max |C_ell| / max |D_ell| = {ratio:.2f}; C_ell = 0 for div = 0")
    assert ratio > 1e-3


def test_correction_independent_of_chunking():
    """Blocks of one plane (chunked in time and z, with and without automatic rechunking) give
    bitwise the same D_ell and D*_ell as one block."""
    ds, fields, d, scales = _nonperiodic_case()
    rng = np.random.default_rng(7)
    ds = xr.concat([ds, ds + rng.normal(size=1).astype(np.float32)], "time")
    ds = xr.concat([ds, ds.assign_coords(z=ds.z + 3)], "z").assign_coords(time=[0.0, 1.0])
    kw = dict(scales=scales, divergence_correction=True, vertical_advection=True,
              increment_components=("u", "v"))
    whole = inter_scale_kinetic_energy_transfer(ds.load(), allow_rechunk=False, **kw).compute()
    for label, data, rechunk in (("time, z chunks", ds.chunk({"time": 1, "z": 1}), False),
                                 ("automatic rechunk", ds.chunk({"time": 1, "z": 2}), True)):
        out = inter_scale_kinetic_energy_transfer(data, allow_rechunk=rechunk, **kw).compute()
        for key in ("energy_transfer", "energy_transfer_corrected"):
            assert np.array_equal(out[key].transpose(*whole[key].dims).values,
                                  whole[key].values), (label, key)
        print(f"  {label}: identical to one block")


def _cli_case(tmp):
    """Small netCDF file with a stretched vertical grid and a sheared mean wind."""
    n, nz, d = 32, 9, 2000.0
    x = np.arange(n) * d
    z = 1000.0 * np.arange(1, nz + 1) ** 1.5
    rng = np.random.default_rng(19)
    X, Y = np.meshgrid(x, x, indexing="xy")
    L = n * d

    def field(amp, mean):
        out = np.zeros((nz, n, n))
        for _ in range(8):
            kx, ky = rng.integers(-3, 4, 2)
            m = 2 * np.pi / rng.uniform(5e3, 20e3)
            ph = 2 * np.pi * (kx * X + ky * Y) / L + rng.uniform(0, 6.3)
            out += rng.normal() * np.cos(ph[None] + m * z[:, None, None])
        return (amp * out + mean * (z / z[-1])[:, None, None])[None].astype(np.float32)

    dims = ("time", "height", "y", "x")
    ds = xr.Dataset({"u": (dims, field(2.0, 15.0)), "v": (dims, field(2.0, -6.0)),
                     "w": (dims, field(0.3, 0.0))},
                    coords={"time": [0.0], "height": ("height", z, {"units": "m"}),
                            "y": ("y", x, {"units": "m"}), "x": ("x", x, {"units": "m"})})
    for name in ("u", "v", "w"):
        ds[name].attrs["units"] = "m s-1"
    path = tmp / "stretched.nc"
    ds.to_netcdf(path)
    return path, ds.rename(height="z"), z


def test_compute_cli_corrected_transfer(tmp_path=None):
    """budget compute with the corrected transfer, end to end: YAML and command-line options
    reach the computation, the level mean is removed, and dz u is taken on the full column
    before the levels are selected."""
    import tempfile
    from pathlib import Path
    import yaml
    from budget.client import main
    tmp = Path(tmp_path or tempfile.mkdtemp())
    path, ds, z = _cli_case(tmp)
    levels, scales = [float(z[2]), float(z[4]), float(z[6])], [5000.0, 9000.0]
    config = {"input": {"path": str(path), "dims": ["height", "y", "x"]},
              "output": {"path": str(tmp / "out.nc"), "store": "netcdf"},
              "compute": {"mode": "scale_transfer", "scales": scales, "levels": levels,
                          "client": False, "divergence_correction": True},
              "variables": {"u": "u", "v": "v", "w": "w"}}
    with open(tmp / "config.yaml", "w") as fh:
        yaml.safe_dump(config, fh)
    main(["compute", str(tmp / "config.yaml"), "--no-client", "--vertical-advection",
          "--perturbation", "--increment-components", "u,v"])
    out = xr.open_dataset(tmp / "out.nc")

    kw = dict(scales=np.array(scales), divergence_correction=True, vertical_advection=True,
              increment_components=("u", "v"), allow_rechunk=False)
    full = inter_scale_kinetic_energy_transfer(ds, perturbation=True, **kw).sel(z=levels)
    selected = inter_scale_kinetic_energy_transfer(ds.sel(z=levels), perturbation=True, **kw)
    ground = inter_scale_kinetic_energy_transfer(ds, **kw).sel(z=levels)

    key = "energy_transfer_corrected"
    got = out[key].transpose(*full[key].dims).values
    size = float(np.abs(full[key]).max())
    err = float(np.abs(got - full[key].values).max()) / size
    off_levels = float(np.abs(got - selected[key].values).max()) / size
    off_frame = float(np.abs(got - ground[key].values).max()) / size
    print(f"  CLI vs full-column perturbation transfer: {err:.1e}; vs shear on selected levels: "
          f"{off_levels:.2f}; vs ground frame: {off_frame:.2f}")
    assert err < 1e-5, err
    assert off_levels > 1e-2 and off_frame > 1e-2        # both choices are visible
    assert np.allclose(out["energy_transfer"].transpose(*full[key].dims).values,
                       full["energy_transfer"].values, rtol=0, atol=1e-6 * size)


# ------------------------------------------------------------------------------------------------
# Lorenz-staggered vertical grid (w on half levels)
# ------------------------------------------------------------------------------------------------
def _lorenz_case(levels_per_lambda=8, n=24, lam=4000.0, depth=24000.0, seed=31):
    """Analytic vertical waves (growing as exp(z/2H)) on a uniform Lorenz-staggered grid."""
    rng = np.random.default_rng(seed)
    L = 240e3
    x = np.arange(n) * L / n
    X, Y = np.meshgrid(x, x, indexing="xy")
    m, H = 2 * np.pi / lam, 7000.0
    modes = [(1, 0), (1, 1), (2, 1), (0, 2)]
    pars = {k: (rng.normal(size=len(modes)), rng.uniform(0, 2 * np.pi, len(modes)))
            for k in ("u", "v", "w", "p", "t")}
    scale = {"u": 10.0, "v": 10.0, "w": 0.5, "p": 1e-3, "t": 2.0}

    def f(name, z, deriv=False):
        z = np.asarray(z)[:, None, None]
        out = np.zeros((z.shape[0], n, n))
        for (kx, ky), a, ph in zip(modes, *pars[name]):
            arg = 2 * np.pi * (kx * X + ky * Y) / L - m * z + ph
            out += a * (np.cos(arg) / H + m * np.sin(arg)) if deriv else a * np.cos(arg)
        return scale[name] * np.exp(z / H) * out

    dz = lam / levels_per_lambda
    zh = np.arange(0.0, depth + 0.5 * dz, dz)
    zf = 0.5 * (zh[1:] + zh[:-1])
    full = lambda a: xr.DataArray(a, dims=("z", "y", "x"), coords={"z": zf, "y": x, "x": x})
    half = lambda a: xr.DataArray(a, dims=("z_half", "y", "x"),
                                  coords={"z_half": zh, "y": x, "x": x})
    return f, full, half, zf, zh, x[1] - x[0]


def _collocated(w_half_da, zf):
    """Linear interpolation of a half-level field to full levels (the collocated path)."""
    w = w_half_da.interp(z_half=zf).rename(z_half="z")
    return w.drop_vars("z", errors="ignore").assign_coords(z=zf)


def test_vertical_operators_exact_for_linear_profiles():
    from budget.vertical import (is_lorenz_staggered, half_to_full_mean, diff_half_to_full,
                                 interp_full_to_half, diff_full_to_half)
    zh = np.cumsum(np.r_[0.0, 300.0 * 1.03 ** np.arange(40)])   # stretched interfaces
    zf = 0.5 * (zh[1:] + zh[:-1])
    a, b = 2.0e-3, -5.0
    lin_h = xr.DataArray(a * zh + b, dims="z_half", coords={"z_half": zh})
    lin_f = xr.DataArray(a * zf + b, dims="z", coords={"z": zf})
    z_full = lin_f["z"]
    assert is_lorenz_staggered(z_full, lin_h["z_half"])
    assert not is_lorenz_staggered(z_full, xr.DataArray(zf, dims="z_half", name="z_half"))
    assert np.allclose(half_to_full_mean(lin_h, z_full), lin_f)
    assert np.allclose(diff_half_to_full(lin_h, z_full), a)
    assert np.allclose(interp_full_to_half(lin_f, lin_h["z_half"]), lin_h)   # incl. ends
    assert np.allclose(diff_full_to_half(lin_f, lin_h["z_half"]), a)
    # descending orientation (ICON stores levels top-down)
    lin_h_d, lin_f_d = lin_h[::-1], lin_f[::-1]
    assert is_lorenz_staggered(lin_f_d["z"], lin_h_d["z_half"])
    assert np.allclose(diff_half_to_full(lin_h_d, lin_f_d["z"]), a)
    assert np.allclose(interp_full_to_half(lin_f_d, lin_h_d["z_half"]), lin_h_d)


def test_staggered_conversions_and_dwdz_more_accurate():
    """At 5 levels per vertical wavelength the native staggered operators are far more accurate
    for dw/dz and the pressure-work conversion than interpolating w and centred differences."""
    from budget.budget import staggered_vertical_terms, conversion_ape_to_dke
    from budget.constants import cp
    from budget.spectral_primitives import scalar_cross_spectrum
    from budget.thermodynamics import lorenz_parameter
    from budget.vertical import diff_half_to_full
    f, full, half, zf, zh, dx = _lorenz_case(levels_per_lambda=5)
    u, v, th, pp = (full(f(k, zf)) for k in ("u", "v", "t", "p"))
    w_half = half(f("w", zh))
    w_col = _collocated(w_half, zf)
    th_mean = xr.DataArray(250.0 * np.exp(0.02 ** 2 * zf / 9.81), dims="z", coords={"z": zf})
    gamma = lorenz_parameter(th_mean)
    inner = slice(3, zf.size - 3)
    err = lambda a, b: float(np.linalg.norm((a - b)[inner]) / np.linalg.norm(b[inner]))
    with _NoTaper():
        dzw_true = f("w", zf, True)
        e_dzw_stag = err(diff_half_to_full(w_half, u["z"]).values, dzw_true)
        e_dzw_col = err(w_col.differentiate("z", edge_order=2).values, dzw_true)

        iso = lambda s: isotropize(s, dx, dx, cumulative=False).values
        cad_true = iso(cp * th_mean * scalar_cross_spectrum(full(f("w", zf)),
                                                            full(f("p", zf, True))))
        stag = staggered_vertical_terms(u, v, w_half, th_mean, th, gamma, pp)
        e_cad_stag = err(iso(stag["cad"]), cad_true)
        e_cad_col = err(iso(conversion_ape_to_dke(th_mean, w_col, pp)), cad_true)
    print(f"  dw/dz error: staggered {e_dzw_stag:.3f}, collocated {e_dzw_col:.3f}; "
          f"C_A->h error: staggered {e_cad_stag:.3f}, collocated {e_cad_col:.3f}")
    assert e_dzw_stag < 0.08 and e_cad_stag < 0.08
    assert e_dzw_col > 4 * e_dzw_stag and e_cad_col > 4 * e_cad_stag


def test_staggered_flux_divergence_telescopes():
    """Layer sums of the staggered flux divergences equal the interface flux difference."""
    from budget.budget import staggered_vertical_terms, turbulent_hke_flux, pressure_flux
    from budget.thermodynamics import lorenz_parameter
    from budget.vertical import interp_full_to_half
    f, full, half, zf, zh, dx = _lorenz_case(levels_per_lambda=6)
    u, v, th, pp = (full(f(k, zf)) for k in ("u", "v", "t", "p"))
    w_half = half(f("w", zh))
    th_mean = xr.DataArray(250.0 * np.exp(0.02 ** 2 * zf / 9.81), dims="z", coords={"z": zf})
    gamma = lorenz_parameter(th_mean)
    zh_da = w_half["z_half"]
    with _NoTaper():
        stag = staggered_vertical_terms(u, v, w_half, th_mean, th, gamma, pp)
        f_h = turbulent_hke_flux(interp_full_to_half(u, zh_da), interp_full_to_half(v, zh_da),
                                 w_half)
        f_p = pressure_flux(interp_full_to_half(th_mean, zh_da), w_half,
                            interp_full_to_half(pp, zh_da))
        dz = xr.DataArray(np.diff(zh), dims="z", coords={"z": zf})
        a, b = 4, zf.size - 4
        for name, flux in (("vfd_hke", f_h), ("vfd_pres", f_p)):
            layer = (stag[name] * dz).isel(z=slice(a, b)).sum("z")
            jump = flux.isel(z_half=b) - flux.isel(z_half=a)
            err = float(np.abs(layer - jump).max() / np.abs(jump).max())
            print(f"  {name}: layer sum vs interface flux difference {err:.1e}")
            assert err < 1e-10


def test_compute_budget_with_half_level_w():
    """compute_budget detects a Lorenz-staggered w_half and returns finite terms."""
    from types import SimpleNamespace
    from budget.budget import compute_budget
    n = 16
    f, full, half, zf, zh, dx = _lorenz_case(levels_per_lambda=8, n=n, depth=12000.0)
    theta = full(250.0 * np.exp(0.02 ** 2 * zf / 9.81)[:, None, None] + f("t", zf))
    pres = full(np.broadcast_to((1e5 * np.exp(-zf / 7000.0))[:, None, None],
                                (zf.size, n, n)).copy())
    ds = xr.Dataset({"u": full(f("u", zf)), "v": full(f("v", zf)), "theta": theta,
                     "pressure": pres, "density": pres / (287.0 * 240.0)})
    ds["w_half"] = half(f("w", zh))
    ds["w"] = _collocated(ds["w_half"], zf)
    cfg = SimpleNamespace(compute=SimpleNamespace(dx=dx, dy=dx, cumulative=True,
                                                  transfer_form="flux", rechunk_spatial=False))
    out = compute_budget(ds, cfg).compute()
    assert all(bool(np.isfinite(out[k]).all()) for k in out.data_vars)
    for k in ("vf_hke", "vfd_dke", "vf_pres", "vfd_pres", "cad", "caz", "np_vke", "div_hke"):
        assert k in out


def test_staggered_outputs_have_zarr_compatible_chunks():
    """Shifted half-level slices must not leave irregular dask chunks along the vertical.
    On Levante, 501 half levels chunked (251, 250) gave vf_vke chunks (250, 1, 249) on the 500
    full levels, which zarr refuses."""
    import tempfile
    from budget.vertical import (half_to_full_mean, diff_half_to_full, interp_full_to_half,
                                 diff_full_to_half)
    nf = 20
    zh = np.linspace(0.0, 10000.0, nf + 1)
    zf = 0.5 * (zh[1:] + zh[:-1])
    rng = np.random.default_rng(3)
    h = xr.DataArray(rng.standard_normal((nf + 1, 4)), dims=("z_half", "x"),
                     coords={"z_half": zh}).chunk({"z_half": 11})          # (11, 10)
    fl = xr.DataArray(rng.standard_normal((nf, 4)), dims=("z", "x"),
                      coords={"z": zf}).chunk({"z": 10})                   # (10, 10)
    outs = {"mean": half_to_full_mean(h, fl["z"]), "dh": diff_half_to_full(h, fl["z"]),
            "interp": interp_full_to_half(fl, h["z_half"]), "df": diff_full_to_half(fl, h["z_half"])}
    for name, da in outs.items():
        sizes = da.chunks[0]
        assert len(set(sizes[:-1])) <= 1 and sizes[-1] <= sizes[0], (name, sizes)
    assert np.allclose(outs["mean"], 0.5 * (h.values[1:] + h.values[:-1]))
    with tempfile.TemporaryDirectory() as tmp:
        xr.Dataset({k: v.rename({v.dims[0]: f"lev_{k}"}).drop_vars(v.dims[0], errors="ignore")
                    for k, v in outs.items()}).to_zarr(f"{tmp}/out.zarr", mode="w", consolidated=False)


def test_write_guard_makes_any_chunks_zarr_safe():
    """The Levante div_hke pattern (z chunks 11, ..., 9, 2, 11, ..., 5) is rechunked before the
    zarr write; values are unchanged and uniform variables are left alone."""
    import tempfile
    from budget.io_utils import zarr_safe_chunks
    rng = np.random.default_rng(5)
    zc = (11,) * 22 + (9, 2) + (11,) * 23 + (5,)
    data = rng.standard_normal((2, sum(zc), 3))
    bad = xr.DataArray(data, dims=("time", "z", "k")).chunk({"time": 2, "z": zc, "k": 3})
    good = xr.DataArray(data, dims=("time", "z", "k")).chunk({"time": 1, "z": 50})
    ds = xr.Dataset({"div_hke": bad, "pi_nke": good})
    out = zarr_safe_chunks(ds)
    sizes = out["div_hke"].chunks[1]
    assert len(set(sizes[:-1])) == 1 and sizes[-1] <= sizes[0], sizes
    assert out["pi_nke"].chunks == good.chunks
    with tempfile.TemporaryDirectory() as tmp:
        out.to_zarr(f"{tmp}/o.zarr", mode="w", consolidated=False)
        assert np.array_equal(xr.open_zarr(f"{tmp}/o.zarr", consolidated=False)["div_hke"].values,
                              data)


def test_reader_keeps_native_half_level_w(tmp_path=None):
    """ICON-like file (full and half levels as separate dims, stored top-down, w on half levels):
    open_dataset keeps w_half on z_half for the spectral budget, and it is Lorenz-staggered."""
    import tempfile
    from pathlib import Path
    from budget.config import Config, IOConfig, OutputConfig, ComputeConfig
    from budget.io_utils import open_dataset
    from budget.vertical import is_lorenz_staggered
    tmp = Path(tmp_path or tempfile.mkdtemp())
    n, k = 8, 6
    zh = np.linspace(12000.0, 0.0, k + 1)                 # interfaces, top-down
    zf = 0.5 * (zh[1:] + zh[:-1])
    x = np.arange(n) * 1000.0
    rng = np.random.default_rng(3)
    full = lambda: (("time", "height", "y", "x"), rng.normal(size=(1, k, n, n)).astype("float32"))
    ds = xr.Dataset(
        {"u": full(), "v": full(), "temp": full(),
         "pres": (("time", "height", "y", "x"),
                  np.broadcast_to((1e5 * np.exp(-zf / 7000.0))[None, :, None, None],
                                  (1, k, n, n)).astype("float32")),
         "w": (("time", "height_2", "y", "x"), rng.normal(size=(1, k + 1, n, n)).astype("float32"))},
        coords={"time": [0.0], "height": ("height", zf, {"units": "m", "standard_name": "height"}),
                "height_2": ("height_2", zh, {"units": "m", "standard_name": "height"}),
                "y": ("y", x, {"units": "m"}), "x": ("x", x, {"units": "m"})})
    for name, units in (("u", "m s-1"), ("v", "m s-1"), ("w", "m s-1"), ("temp", "K"),
                        ("pres", "Pa")):
        ds[name].attrs["units"] = units
    path = tmp / "icon_like.nc"
    ds.to_netcdf(path)
    cfg = Config(input=IOConfig(path=str(path), dims=["height", "y", "x"]),
                 output=OutputConfig(path=str(tmp / "out.nc")),
                 compute=ComputeConfig(mode="spectral_budget"),
                 variables={"u": "u", "v": "v", "w": "w", "temperature": "temp",
                            "pressure": "pres"})
    out = open_dataset(cfg)
    assert "w_half" in out and out["w_half"].sizes["z_half"] == k + 1
    assert np.allclose(out["w_half"]["z_half"], zh)
    assert np.allclose(out["w_half"].values, ds["w"].values)            # native values kept
    assert is_lorenz_staggered(out["z"], out["w_half"]["z_half"])
    cfg.compute.mode = "scale_transfer"
    assert "w_half" not in open_dataset(cfg)                            # only for the budget


def test_geometry_recipe_matches_direct_geometry():
    """The per-worker geometry rebuilt from a recipe equals the geometry built directly."""
    from budget.grid_utils import scale_increments
    from budget.inter_scale_transfers import (_geometry_spec, _increments_from_spec,
                                              _resolve_increments_data)
    x = xr.DataArray(np.arange(48) * 900.0, dims="x", name="x", attrs={"units": "m"})
    y = xr.DataArray(np.arange(36) * 2500.0, dims="y", name="y", attrs={"units": "m"})
    kw = dict(scales=[3000.0, 8000.0], x_coord_boundary="reflect", y_coord_boundary="reflect")
    direct = _resolve_increments_data(scale_increments(x, y, **kw))
    rebuilt = _increments_from_spec(_geometry_spec(x, y, dict(kw, verbose=True, p=1))["geometry_spec"])
    assert np.array_equal(direct["r"], rebuilt["r"])
    for key in ("sy", "sx", "dist", "angle"):
        assert np.array_equal(direct["lattice"][key], rebuilt["lattice"][key])
    for r in direct["r"]:
        for key in ("nx_values", "ny_values", "angles", "weights"):
            assert np.array_equal(direct[r][key], rebuilt[r][key])


def test_cartesian_grid_is_not_global():
    """A Cartesian x in metres must not be detected as global longitude (forced periodic)."""
    from budget.grid_utils import infer_boundary_conditions
    x = xr.DataArray(np.arange(-1500e3, 1500e3, 4900.0), dims="x", name="x", attrs={"units": "m"})
    lon = xr.DataArray(np.arange(0, 360, 0.5), dims="lon", name="lon",
                       attrs={"units": "degrees_east", "standard_name": "longitude"})
    assert infer_boundary_conditions(x)[0] == "reflect"
    assert infer_boundary_conditions(lon)[0] == "periodic"


if __name__ == "__main__":
    tests = [test_hke_transfer_conserves_and_closes, test_flux_and_invariant_forms_agree,
             test_vke_and_ape_transfers_conserve, test_dell_matches_lattice_sum,
             test_dell_zero_for_steady_euler_flow, test_dell_positive_across_shocks,
             test_lattice_bins_partition_shifts, test_lattice_kernel_normalization,
             test_dell_anisotropic_grid, test_dell_converges_with_resolution,
             test_dell_rotation_invariance, test_spectrum_parseval_and_cumulative_convention,
             test_density_weighted_closure_hke_vke_ape, test_pressure_work_closure,
             test_vertical_pressure_buoyancy_closure, test_layer_integral_gives_boundary_flux,
             test_fft_path_matches_exact_lattice_nonperiodic, test_shift_fallback_close_to_exact,
             test_divergence_correction_matches_increment_form,
             test_divergence_corrected_dell_closes_local_budget,
             test_corrected_transfer_equals_kernel_weighted_spectral_transfer,
             test_perturbation_frame,
             test_corrected_transfer_closed_form_triad,
             test_corrected_transfer_closed_form_boussinesq_wave,
             test_correction_divergence_source, test_correction_error_paths,
             test_correction_geographic_grid,
             test_correction_independent_of_chunking,
             test_compute_cli_corrected_transfer,
             test_vertical_operators_exact_for_linear_profiles,
             test_staggered_conversions_and_dwdz_more_accurate,
             test_staggered_flux_divergence_telescopes, test_compute_budget_with_half_level_w,
             test_staggered_outputs_have_zarr_compatible_chunks,
             test_write_guard_makes_any_chunks_zarr_safe,
             test_reader_keeps_native_half_level_w, test_geometry_recipe_matches_direct_geometry,
             test_cartesian_grid_is_not_global]
    failed = 0
    for t in tests:
        print(f"{t.__name__}")
        try:
            t()
            print("  PASSED")
        except AssertionError as e:
            failed += 1
            print(f"  FAILED: {e}")
    sys.exit(1 if failed else 0)
