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


def test_fast_path_matches_roll_path_nonperiodic():
    """Pad-once fast path == original per-shift pad/roll path, for reflect and mixed boundaries."""
    import budget.inter_scale_transfers as ist
    rng = np.random.default_rng(29)
    ny, nx, d = 40, 56, 1000.0
    x = np.arange(nx) * d
    y = np.arange(ny) * d
    z = np.array([0.0, 1.0, 2.0])
    fields = {k: rng.normal(size=(3, ny, nx)).astype(np.float32) for k in ("u", "v", "w")}
    ds = xr.Dataset({k: (("z", "y", "x"), a) for k, a in fields.items()},
                    coords={"z": z, "y": y, "x": x})
    scales = np.array([1.5, 3.0, 6.0]) * d
    saved = dict(ist._NP_PAD_MODES)
    try:
        for xb, yb in [("reflect", "reflect"), ("periodic", "reflect"), ("edge", "symmetric")]:
            kw = dict(scales=scales, x_coord_boundary=xb, y_coord_boundary=yb, allow_rechunk=False)
            fast = inter_scale_kinetic_energy_transfer(ds, **kw)["energy_transfer"].values
            ist._NP_PAD_MODES.clear()          # force the original pad/roll implementation
            slow = inter_scale_kinetic_energy_transfer(ds, **kw)["energy_transfer"].values
            ist._NP_PAD_MODES.update(saved)
            err = float(np.abs(fast - slow).max() / np.abs(slow).max())
            print(f"  boundaries x={xb:8s} y={yb:9s}: max |fast - roll| / max |roll| = {err:.1e}")
            assert err < 1e-5, (xb, yb, err)
    finally:
        ist._NP_PAD_MODES.clear()
        ist._NP_PAD_MODES.update(saved)


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
             test_fast_path_matches_roll_path_nonperiodic, test_cartesian_grid_is_not_global]
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
