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


class _Spectral:
    """Context manager: explicit taper on/off and detrend type (restored on exit)."""

    def __init__(self, taper, detrend):
        self.taper, self.detrend = taper, detrend

    def __enter__(self):
        self.saved = (sp.SPECTRAL_CFG["taper"]["enabled"], sp.SPECTRAL_CFG["detrend"]["type"])
        sp.SPECTRAL_CFG["taper"]["enabled"] = self.taper
        sp.SPECTRAL_CFG["detrend"]["type"] = self.detrend
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
# Conservative (skew-symmetric) transfers and periodic axes
# ------------------------------------------------------------------------------------------------
def _periodic(*arrays, dims=("y", "x")):
    """Copies of the arrays with the horizontal coordinates marked periodic."""
    from budget.numeric_tools import mark_periodic
    return [mark_periodic(a.copy(), dims) for a in arrays]


def _all_transfers(u, v, w, th, conservative):
    gamma = xr.ones_like(th.z, dtype=float)
    div = None if conservative else horizontal_divergence(u, v)
    return {"HKE": nonlinear_hke_transfer_flux(u, v, w, div, conservative=conservative),
            "VKE": nonlinear_vke_transfer(u, v, w, div, conservative=conservative),
            "APE": nonlinear_ape_transfer(u, v, w, th, gamma, div, conservative=conservative)}


def test_conservative_transfers_sum_to_zero_on_periodic_axes():
    """The skew-symmetric transfers sum to zero over wavenumbers to round-off on a periodic
    domain, with a DPSS taper and linear detrend switched on (skipped along periodic axes); the
    flux form leaves the discretization residual of the product rule (here > 1e-3)."""
    u, v, w, th, dx = _random_3d_fields(seed=7, n=40, w_amp=1.0)
    u, v, w, th = _periodic(u, v, w, th)
    with _Spectral(True, "linear"):
        cons, flux = _all_transfers(u, v, w, th, True), _all_transfers(u, v, w, th, False)
    for name in cons:
        scale = float(np.abs(isotropize(flux[name], dx, dx, cumulative=True)).max())
        t_c = float(np.abs(_shell_sum(cons[name], dx)).max()) / scale
        t_f = float(np.abs(_shell_sum(flux[name], dx)).max()) / scale
        print(f"  {name}: max_z |sum_k T| / max|Pi|: conservative {t_c:.1e}, flux {t_f:.1e}")
        assert t_c < 1e-12, f"{name} conservative form does not sum to zero ({t_c:.1e})"
        assert t_f > 1e-3, f"{name}: flux-form residual unexpectedly small ({t_f:.1e})"


def test_periodic_derivative_float32_coordinates():
    """Model output often stores x and y in single precision: x = -700..700 km every 2.5 km has
    differences scattering by ~4e-5 of the spacing. The periodic derivative accepts that grid
    (spacing from the end points, 2500 m to 1e-9), gives the float64-coordinate result, and the
    conservative transfers still sum to zero; a grid stretched by 0.1% is still rejected."""
    from budget.numeric_tools import _coordinate_derivative, mark_periodic, uniform_spacing
    # cell centres offset by 1 cm, rounded to float32 (as written by the model)
    x32 = (np.linspace(-700e3, 700e3, 561) + 0.01).astype(np.float32)[:-1]
    assert np.ptp(np.diff(x32.astype(float))) / 2500.0 > 1e-5          # the scatter is real
    assert abs(uniform_spacing(xr.DataArray(x32, dims="x")) - 2500.0) < 2500.0 * 1e-9
    x64 = np.linspace(-700e3, 700e3, 561)[:-1] + 0.01
    values = np.sin(2 * np.pi * 3 * x64 / 1.4e6)                       # the same samples,
    f = lambda x: xr.DataArray(values, dims="x", coords={"x": x})       # two coordinate labels
    d32 = _coordinate_derivative(mark_periodic(f(x32).to_dataset(name="a"), ["x"])["a"], "x")
    d64 = _coordinate_derivative(mark_periodic(f(x64).to_dataset(name="a"), ["x"])["a"], "x")
    err = float(np.abs(d32.values - d64.values).max() / np.abs(d64.values).max())
    print(f"  float32 vs float64 coordinates: max relative difference {err:.1e}")
    assert err < 1e-6

    stretched = 2500.0 * np.cumsum(np.linspace(1.0, 1.001, 560))
    assert uniform_spacing(xr.DataArray(stretched, dims="x")) is None
    try:
        _coordinate_derivative(mark_periodic(f(stretched).to_dataset(name="a"), ["x"])["a"], "x")
    except ValueError:
        pass
    else:
        raise AssertionError("a stretched periodic grid was accepted")

    u, v, w, th, dx = _random_3d_fields(seed=7, n=40, w_amp=1.0)
    as32 = lambda a: a.assign_coords(x=a.x.astype(np.float32) - 7.0e5, y=a.y.astype(np.float32) - 7.0e5)
    u, v, w, th = _periodic(*(as32(a) for a in (u, v, w, th)))
    with _Spectral(True, "linear"):
        cons, flux = _all_transfers(u, v, w, th, True), _all_transfers(u, v, w, th, False)
    for name in cons:
        scale = float(np.abs(isotropize(flux[name], dx, dx, cumulative=True)).max())
        t_c = float(np.abs(_shell_sum(cons[name], dx)).max()) / scale
        assert t_c < 1e-12, f"{name}: float32 coordinates, sum_k T = {t_c:.1e} max|Pi|"


def test_conservative_and_flux_forms_converge():
    """Both forms discretize the same transfer: their difference is second order in dx."""
    diffs = []
    for n in (40, 80):
        u, v, w, th, dx = _random_3d_fields(seed=5, n=n, w_amp=1.0)
        u, v, w, th = _periodic(u, v, w, th)
        with _NoTaper():
            a = _interior(isotropize(nonlinear_hke_transfer_flux(u, v, w, conservative=True),
                                     dx, dx))
            b = _interior(isotropize(nonlinear_hke_transfer_flux(u, v, w), dx, dx))
        diffs.append(float(np.abs(a - b).max() / np.abs(b).max()))
    print(f"  max |conservative - flux| / max |flux|: {diffs[0]:.1e} (n = 40), "
          f"{diffs[1]:.1e} (n = 80), ratio {diffs[0] / diffs[1]:.1f}")
    assert diffs[1] < 0.02 and diffs[0] / diffs[1] > 3.0


def test_periodic_axes_skip_taper_and_linear_detrend():
    """Along a periodic axis the taper is 1, the derivative wraps around (exact for a resolved
    sine) and a linear trend is not removed; along the other axis both still apply."""
    from budget.numeric_tools import first_derivative
    n, L = 32, 320e3
    x = np.arange(n) * L / n
    X = np.broadcast_to(x, (n, n))
    da = xr.DataArray(np.sin(2 * np.pi * X / L), dims=("y", "x"), coords={"y": x, "x": x})
    (per_x,) = _periodic(da, dims=("x",))
    with _Spectral(True, "linear"):
        w, _ = sp._taper_2d(per_x)
    assert np.allclose(w.isel(y=n // 2), w.isel(y=n // 2, x=0))           # flat along x
    assert not np.allclose(w.isel(x=0), w.isel(x=0, y=n // 2))             # tapered along y
    exact = 2 * np.pi / L * np.cos(2 * np.pi * X / L) * np.sin(np.pi / n * 2) / (2 * np.pi / n)
    assert np.allclose(first_derivative(per_x, "x"), exact, atol=1e-12)    # centred, wrapped
    assert not np.allclose(first_derivative(da, "x"), exact, atol=1e-8)    # one-sided at ends
    ramp = da.copy(data=np.add.outer(np.arange(n) * 1.0, np.arange(n) * 2.0))
    (ramp_x,) = _periodic(ramp, dims=("x",))
    left = sp.detrend(ramp_x, detrend_type="linear")
    assert np.allclose(left.isel(y=0) - left.isel(y=0).mean(), ramp.isel(y=0) - ramp.isel(y=0).mean())
    assert np.allclose(left.isel(x=0).diff("y"), 0.0, atol=1e-9)           # y trend removed


def test_compute_cli_conservative_periodic(tmp_path=None):
    """budget compute --transfer-form conservative --periodic-axes x,y end to end (YAML + flags,
    netCDF in and out): the cumulative transfers vanish at the smallest wavenumber."""
    import tempfile
    from pathlib import Path
    import yaml
    from budget.client import main
    tmp = Path(tmp_path or tempfile.mkdtemp())
    u, v, w, th, dx = _random_3d_fields(seed=11, n=32, w_amp=1.0)
    zf = u.z.values
    theta = 250.0 * np.exp(0.02 ** 2 * zf / 9.81)[:, None, None] + th
    pres = xr.ones_like(u) * (1e5 * np.exp(-zf / 7000.0))[:, None, None]
    ds = xr.Dataset({"u": 10 * u, "v": 10 * v, "w": w, "theta": theta, "pres": pres})
    ds = ds.expand_dims(time=[0.0]).rename(z="height").astype("float64")
    ds["height"].attrs["units"] = "m"
    for k, units in (("u", "m s-1"), ("v", "m s-1"), ("w", "m s-1"), ("theta", "K"),
                     ("pres", "Pa")):
        ds[k].attrs["units"] = units
    ds.to_netcdf(tmp / "periodic.nc")
    config = {"input": {"path": str(tmp / "periodic.nc"), "dims": ["height", "y", "x"]},
              "output": {"path": str(tmp / "out.nc"), "store": "netcdf"},
              "compute": {"mode": "spectral_budget", "client": False, "dx": dx, "dy": dx},
              "variables": {"u": "u", "v": "v", "w": "w", "theta": "theta", "pressure": "pres"}}
    with open(tmp / "config.yaml", "w") as fh:
        yaml.safe_dump(config, fh)
    main(["compute", str(tmp / "config.yaml"), "--no-client", "--transfer-form", "conservative",
          "--periodic-axes", "x,y"])
    out = xr.open_dataset(tmp / "out.nc")
    for k in ("pi_nke", "pi_vke", "pi_ape"):
        pi = _interior(out[k].sortby("z"))
        err = float(np.abs(pi.isel(wavenumber=0)).max() / np.abs(pi).max())
        print(f"  CLI {k}: max_z |Pi(k_min)| / max|Pi| = {err:.1e}")
        assert err < 1e-3, (k, err)     # float32 storage: theta' ~ 1 K on 250 K


def test_spatial_dims_from_units_and_axes():
    """Horizontal dimensions are found from units and CF axes, not names: metric x, y with
    long_name "longitude"/"latitude" are Cartesian, degree x, y are geographic, and lat/lon may
    come in either order."""
    from budget.cf_coords import get_spatial_dims, is_geographic_grid
    n = np.arange(4.0)

    def case(dims, attrs, values=(n * 5000.0, n * 5000.0)):
        coords = {d: (d, v, a) for d, v, a in zip(dims, values, attrs)}
        coords["height"] = ("height", [1.0, 2.0], {"units": "m", "axis": "Z"})
        da = xr.DataArray(np.zeros((2, 4, 4)), dims=("height",) + dims, coords=coords)
        return da, get_spatial_dims(da)

    metric = ({"long_name": "latitude", "units": "m", "axis": "Y"},
              {"long_name": "longitude", "units": "m", "axis": "X"})
    da, dims = case(("y", "x"), metric)
    assert dims == ("y", "x") and not is_geographic_grid(da.x, da.y)
    da, dims = case(("x", "y"), metric[::-1])                                 # transposed
    assert dims == ("y", "x")
    deg = ({"standard_name": "latitude", "units": "degrees_north"},
           {"standard_name": "longitude", "units": "degrees_east"})
    da, dims = case(("y", "x"), deg, values=(n * 10.0, n * 30.0))
    assert dims == ("y", "x") and is_geographic_grid(da.x, da.y)            # was ("y", "y")
    da, dims = case(("lon", "lat"), deg[::-1], values=(n * 30.0, n * 10.0))
    assert dims == ("lat", "lon")
    da, dims = case(("rlat", "rlon"), ({"axis": "Y", "units": "m"}, {"axis": "X", "units": "m"}))
    assert dims == ("rlat", "rlon")                                          # any names, by axis
    print("  metric x/y named lon/lat, degree x/y, lat/lon, axis-only: dimensions resolved")


def test_errico_detrend():
    """Errico detrending removes the end-to-end jump along non-periodic axes only and the mean
    (restored at k = 0), leaves periodic content alone, and the cross spectrum sums to
    <a' b'> + mean(a) mean(b)."""
    n, L = 32, 320e3
    x = np.arange(n) * L / n
    Y, X = np.meshgrid(x, x, indexing="ij")
    periodic = np.cos(2 * np.pi * X / L) * np.sin(4 * np.pi * Y / L)
    da = xr.DataArray(periodic + 3e-5 * Y + 1e-5 * X + 7.0, dims=("y", "x"),
                      coords={"y": x, "x": x})
    (da,) = _periodic(da, dims=("x",))                       # x periodic: only y is detrended
    out = sp.detrend(da, detrend_type="errico")
    assert np.allclose(out.isel(y=0), out.isel(y=-1))                     # no jump in y
    assert not np.allclose(out.isel(x=0), out.isel(x=-1))                 # x left alone
    assert abs(float(out.mean())) < 1e-12                                  # mean removed
    # a purely periodic field in y (sampled without the end point) keeps a small jump term only
    resid = out - (periodic + 1e-5 * X + 7.0 - float(da.mean()))
    assert float(np.abs(resid - resid.mean("y")).max()) < 0.2
    saved = (sp.SPECTRAL_CFG["taper"]["enabled"], sp.SPECTRAL_CFG["detrend"]["type"])
    try:
        sp.SPECTRAL_CFG["taper"]["enabled"], sp.SPECTRAL_CFG["detrend"]["type"] = False, "errico"
        got = float(sp.scalar_cross_spectrum(da, 2 * da).sum(("ky", "kx")))
        assert np.isclose(got, float((out * 2 * out).mean() + 2 * da.mean() ** 2), rtol=1e-12)
    finally:
        sp.SPECTRAL_CFG["taper"]["enabled"], sp.SPECTRAL_CFG["detrend"]["type"] = saved
    print("  jump in y and mean removed, x unchanged, Parseval with the mean restored")


def test_periodicity_detected_from_the_seam():
    """periodic_axes auto: periodic axes (seam ~ neighbour difference) are found, a repeated end
    point is dropped, a cut axis is bounded; explicit lists and 'none' override; and with the
    defaults the transfers of undeclared periodic data equal those of declared ones."""
    from types import SimpleNamespace
    from budget.budget import compute_budget
    from budget.grid_utils import resolve_periodic_axes
    u, v, w, th, dx = _random_3d_fields(seed=11, n=40, w_amp=1.0)
    ds = xr.Dataset({"u": 10 * u, "v": 10 * v})
    _, per, rep = resolve_periodic_axes(ds)
    assert per == ["x", "y"], rep
    dup = xr.concat([ds, ds.isel(x=[0]).assign_coords(x=[float(ds.x[-1]) + dx])], "x")
    out, per, rep = resolve_periodic_axes(dup)
    assert per == ["x", "y"] and out.sizes["x"] == 40 and "dropped" in rep["x"][1], rep
    cut, per, rep = resolve_periodic_axes(ds.isel(y=slice(5, 30)))
    assert per == ["x"] and rep["y"][1] == "bounded", rep
    assert resolve_periodic_axes(ds, ["y"])[1] == ["y"]
    assert resolve_periodic_axes(ds, "none")[1] == [] and resolve_periodic_axes(ds, [])[1] == []

    zf = u.z.values
    theta = 250.0 * np.exp(0.02 ** 2 * zf / 9.81)[:, None, None] + th
    pres = xr.ones_like(u) * (1e5 * np.exp(-zf / 7000.0))[:, None, None]
    full = xr.Dataset({"u": 10 * u + 40.0, "v": 10 * v, "w": w, "theta": theta,
                       "pressure": pres, "density": pres / (287.0 * 240.0)})
    pis = {}
    for axes in (None, ["x", "y"]):
        cfg = SimpleNamespace(compute=SimpleNamespace(dx=dx, dy=dx, cumulative=True,
                                                      periodic_axes=axes, rechunk_spatial=False))
        pis[str(axes)] = compute_budget(full, cfg)["pi_nke"].compute()
    a, b = pis["None"], pis["['x', 'y']"]
    err = float(np.abs(a - b).max() / np.abs(b).max())
    print(f"  seam ratios {rep}; defaults, undeclared vs declared periodic: {err:.1e}")
    assert err < 1e-12


def test_scale_transfer_detects_periodic_axes():
    """scale_transfer resolves periodic axes as the spectral budget does: a repeated end column
    is dropped and x wraps (equal to the trimmed data declared periodic), a cut y axis is padded
    by reflection, and explicit boundary keywords take precedence."""
    u, v, w, _, dx = _random_3d_fields(seed=13, n=32, nz=5)
    ds = xr.Dataset({"u": 10 * u, "v": 10 * v, "w": w})
    kw = dict(scales=np.array([4 * dx]), allow_rechunk=False, divergence_correction=True)
    run = lambda data, **extra: inter_scale_kinetic_energy_transfer(data, **kw, **extra)

    dup = xr.concat([ds, ds.isel(x=[0]).assign_coords(x=[float(ds.x[-1]) + dx])], "x")
    auto = run(dup)
    declared = run(ds, x_coord_boundary="periodic", y_coord_boundary="periodic")
    assert auto.sizes["x"] == 32
    for k in ("energy_transfer", "energy_transfer_generalized"):
        assert np.allclose(auto[k].values, declared[k].values, rtol=0, atol=1e-12), k

    cut = ds.isel(y=slice(4, 28))
    auto = run(cut)
    explicit = run(cut, x_coord_boundary="periodic", y_coord_boundary="reflect")
    assert np.array_equal(auto["energy_transfer"].values, explicit["energy_transfer"].values)
    forced = run(cut, x_coord_boundary="reflect", y_coord_boundary="reflect")
    assert not np.allclose(forced["energy_transfer"].values, auto["energy_transfer"].values)
    print("  repeated end column dropped and x wrapped; cut y reflected; explicit keywords win")


def test_cross_spectrum_parseval_with_mean_and_no_detrend():
    """Sum over all modes of the cross spectrum equals the domain mean of the product, with a
    non-zero mean and detrending off (the mean was once restored at k = 0 a second time)."""
    u, v, _, _, dx = _random_3d_fields(seed=2, n=24, nz=3)
    a, b = u + 40.0, v * u + 3.0
    saved = (sp.SPECTRAL_CFG["taper"]["enabled"], sp.SPECTRAL_CFG["detrend"]["type"])
    try:
        for detrend in (None, "constant", "linear"):
            sp.SPECTRAL_CFG["taper"]["enabled"], sp.SPECTRAL_CFG["detrend"]["type"] = False, detrend
            got = sp.scalar_cross_spectrum(a, b).sum(("ky", "kx"))
            err = float(np.abs(got - (a * b).mean(("y", "x"))).max() / np.abs(a * b).mean())
            print(f"  detrend {str(detrend):8s}: |sum_k - <a b>| / <|a b|> = {err:.1e}")
            assert err < 1e-12 or detrend == "linear"
    finally:
        sp.SPECTRAL_CFG["taper"]["enabled"], sp.SPECTRAL_CFG["detrend"]["type"] = saved


def test_compute_budget_conservative_periodic_axes():
    """End to end: with transfer_form conservative and periodic_axes [x, y], the cumulative
    transfers of compute_budget vanish at the smallest wavenumber (no DC, no corner modes) to
    the energy beyond the Nyquist circle; the flux form with the previous default spectral
    settings (DPSS taper, linear detrend, no periodic axes) does not."""
    from types import SimpleNamespace
    from budget.budget import compute_budget
    from budget.grid_utils import periodic_dims
    u, v, w, th, dx = _random_3d_fields(seed=11, n=40, w_amp=1.0)
    zf = u.z.values
    theta = 250.0 * np.exp(0.02 ** 2 * zf / 9.81)[:, None, None] + th
    pres = xr.ones_like(u) * (1e5 * np.exp(-zf / 7000.0))[:, None, None]
    ds = xr.Dataset({"u": 10 * u, "v": 10 * v, "w": w, "theta": theta, "pressure": pres,
                     "density": pres / (287.0 * 240.0)})
    k_min = {}
    for form, axes in (("conservative", ["x", "y"]), ("flux", [])):
        cfg = SimpleNamespace(compute=SimpleNamespace(dx=dx, dy=dx, cumulative=True,
                                                      transfer_form=form, periodic_axes=axes,
                                                      rechunk_spatial=False))
        with _Spectral(True, "linear"):
            out = compute_budget(ds, cfg).compute()
        k_min[form] = {k: float(np.abs(_interior(out[k].isel(wavenumber=0)) /
                                       np.abs(_interior(out[k])).max()).max())
                       for k in ("pi_nke", "pi_vke", "pi_ape")}
        print(f"  {form:12s}: max_z |Pi(k_min)| / max|Pi| " +
              ", ".join(f"{k} {e:.1e}" for k, e in k_min[form].items()))
    assert all(e < 1e-6 for e in k_min["conservative"].values())
    assert all(e > 1e-2 for e in k_min["flux"].values())
    assert "periodic" not in ds.x.attrs                                     # input untouched
    assert periodic_dims(ds, ["y"]) == ["y"]
    glob = ds.rename(x="lon", y="lat").assign_coords(lon=np.linspace(0, 351, 40),
                                                      lat=np.linspace(-60, 60, 40))
    glob.lon.attrs.update(units="degrees_east", standard_name="longitude")
    glob.lat.attrs.update(units="degrees_north", standard_name="latitude")
    assert periodic_dims(glob) == ["lon"]                                   # global longitude


# ------------------------------------------------------------------------------------------------
# Lateral fluxes: interior transfer + lateral flux divergence
# ------------------------------------------------------------------------------------------------
def _lateral_case(periodic, seed=7):
    """Random fields with a 30 m/s mean wind (large lateral fluxes), optionally marked periodic."""
    from budget.numeric_tools import mark_periodic
    u, v, w, th, dx = _random_3d_fields(seed=seed, n=40, w_amp=1.0)
    u = u + 30.0
    exner = 1e-3 * _random_3d_fields(seed=seed + 1, n=40)[0]
    fields = [u, v, w, th, exner]
    if periodic:
        fields = [mark_periodic(a.copy(), ("y", "x")) for a in fields]
    return (*fields, dx)


def _lateral_terms(u, v, w, th, exner, conservative=True):
    """Full transfers and lateral fluxes (2-D spectra) for HKE, VKE, APE and the pressure work."""
    from budget.budget import (lateral_hke_flux, lateral_vke_flux, lateral_ape_flux,
                               lateral_pressure_flux)
    gamma = xr.ones_like(th.z, dtype=float)
    theta_mean = xr.ones_like(th.z, dtype=float) * 250.0
    total = {"HKE": nonlinear_hke_transfer_flux(u, v, w, conservative=conservative),
             "VKE": nonlinear_vke_transfer(u, v, w, conservative=conservative),
             "APE": nonlinear_ape_transfer(u, v, w, th, gamma, conservative=conservative)}
    lateral = {"HKE": lateral_hke_flux(u, v), "VKE": lateral_vke_flux(u, v, w),
               "APE": lateral_ape_flux(u, v, th, gamma),
               "pres": lateral_pressure_flux(u, v, theta_mean, exner)}
    return total, lateral


def test_lateral_fluxes_vanish_on_periodic_axes():
    """Along periodic axes the lateral flux spectra are zero at every wavenumber (the wrap-around
    centred difference is skew-adjoint at each k), with or without the Errico detrend, so the
    interior transfer equals the full transfer."""
    u, v, w, th, exner, dx = _lateral_case(periodic=True)
    for detrend in (None, "errico"):
        with _Spectral(False, detrend):
            total, lateral = _lateral_terms(u, v, w, th, exner)
            for name, lf in lateral.items():
                ref = total.get(name, lateral["pres"] * 0 + total["HKE"])
                scale = float(np.abs(isotropize(ref, dx, dx, cumulative=True)).max())
                err = float(np.abs(isotropize(lf, dx, dx, cumulative=False,
                                              include_dc=True)).max()) / scale
                print(f"  detrend {str(detrend):6s} {name:4s}: max_k |L(k)| / max |Pi| = {err:.1e}")
                assert err < 1e-12, (detrend, name, err)


def test_interior_transfer_sums_to_zero_on_bounded_axes():
    """On axes treated as bounded (one-sided differences at the edges), the interior transfer
    pi - L sums to zero over all wavenumbers (k = 0 included) to round-off, for the conservative
    and the flux form, while the full transfer keeps the lateral import; pi = (pi - L) + L at
    every k. Without detrending: Parseval's sum_k (a, b)_k = <a b> carries the pointwise
    cancellation grad X . U X = X U . grad X over to the spectra. The flux form keeps its
    product-rule residual, (X/2)[div(U X) - U.grad X - X div U], here < 5%."""
    u, v, w, th, exner, dx = _lateral_case(periodic=False)
    for conservative in (True, False):
        with _Spectral(False, None):
            total, lateral = _lateral_terms(u, v, w, th, exner, conservative)
        for name in ("HKE", "VKE", "APE"):
            scale = float(np.abs(isotropize(total[name], dx, dx, cumulative=True)).max())
            interior = total[name] - lateral[name]
            s_int = float(np.abs(_shell_sum(interior, dx)).max()) / scale
            s_tot = float(np.abs(_shell_sum(total[name], dx)).max()) / scale
            print(f"  {'conservative' if conservative else 'flux':12s} {name}: "
                  f"max_z |sum_k (pi - L)| = {s_int:.1e}, |sum_k pi| = {s_tot:.1e} (of max|Pi|)")
            assert s_int < (1e-12 if conservative else 5e-2), (conservative, name, s_int)
            assert s_tot > 1e-3, (conservative, name, s_tot)


def _nonperiodic_2d(n, L=500e3):
    """Analytic, non-periodic fields on [0, L]^2 (both end points on the grid), one level."""
    x = np.linspace(0.0, L, n)
    X, Y = np.meshgrid(x, x)
    a, b = 2 * np.pi / (1.3 * L), 2 * np.pi / (1.7 * L)
    f = {"u": lambda X, Y: 20.0 + 5.0 * np.sin(a * X + 0.4) * np.cos(b * Y + 0.2),
         "v": lambda X, Y: 3.0 * np.cos(a * X - 0.3) * np.sin(b * Y + 0.5) + 1e-5 * X,
         "p": lambda X, Y: 1e-3 * np.cos(a * X + b * Y + 0.1)}
    da = lambda q: xr.DataArray(q[None], dims=("z", "y", "x"),
                                coords={"z": [0.0], "y": x, "x": x})
    return {k: da(g(X, Y)) for k, g in f.items()}, f, x[1] - x[0]


def _boundary_flux(flux_x, flux_y, L=500e3, m=4001):
    """(1/A) of the outward flux of (flux_x, flux_y) through the edges of [0, L]^2 (trapezoid)."""
    s = np.linspace(0.0, L, m)
    trap = lambda q: float(np.trapz(q, s))
    out = (trap(flux_x(L, s)) - trap(flux_x(0.0, s)) + trap(flux_y(s, L)) - trap(flux_y(s, 0.0)))
    return out / L ** 2


def test_lateral_fluxes_converge_to_boundary_fluxes():
    """Summed over wavenumbers, L_h and L_p equal minus the advective HKE flux U |U|^2/2 and the
    pressure flux pi' U through the edges of a non-periodic domain, divided by the area: the
    grid sum is a domain mean of a discrete divergence, so it converges at first order (end-point
    weights) to the boundary integral."""
    from budget.budget import lateral_hke_flux, lateral_pressure_flux
    from budget.constants import cp
    _, f, _ = _nonperiodic_2d(3)
    ke = lambda X, Y: 0.5 * (f["u"](X, Y) ** 2 + f["v"](X, Y) ** 2)
    exact_h = -_boundary_flux(lambda X, Y: f["u"](X, Y) * ke(X, Y),
                              lambda X, Y: f["v"](X, Y) * ke(X, Y))
    exact_p = -_boundary_flux(lambda X, Y: f["p"](X, Y) * f["u"](X, Y),
                              lambda X, Y: f["p"](X, Y) * f["v"](X, Y))
    errors = []
    for n in (41, 81, 161):
        q, _, dx = _nonperiodic_2d(n)
        with _Spectral(False, None):
            lh = float(_shell_sum(lateral_hke_flux(q["u"], q["v"]), dx).squeeze())
            lp = float(_shell_sum(lateral_pressure_flux(q["u"], q["v"], 1.0 / cp, q["p"]),
                                  dx).squeeze())
        errors.append((abs(lh / exact_h - 1), abs(lp / exact_p - 1)))
        print(f"  n = {n:3d}: sum_k L_h / boundary flux - 1 = {lh / exact_h - 1:+.4f}, "
              f"L_p: {lp / exact_p - 1:+.4f}")
    for i in range(2):
        e = [err[i] for err in errors]
        assert e[0] / e[1] > 1.7 and e[1] / e[2] > 1.7, e         # first-order convergence
        assert e[2] < 0.05, e


def test_pressure_work_closes_on_bounded_axes():
    """On a non-periodic domain the vertical-flux decomposition of the horizontal pressure work
    misses the lateral pressure flux; with L_p (lf_pres) it closes at every wavenumber."""
    from budget.budget import (pressure_flux, conversion_ape_to_dke, pressure_divergence_work,
                               nonconservative_adiabatic_pressure, nonlinear_pressure_work_hke,
                               lateral_pressure_flux)
    from budget.constants import cp
    from budget.numeric_tools import horizontal_gradient
    u, v, w, th, th_mean, _, rho, exner, dx = _stratified_case(seed=17)
    ramp = xr.DataArray(np.linspace(0.0, 1.0, u.sizes["x"]), dims="x", coords={"x": u.x})
    u, exner = u + 3.0 * ramp, exner + 2e-3 * ramp                    # not periodic in x
    with _NoTaper():
        div = horizontal_divergence(u, v)
        f_p = pressure_flux(th_mean, w, exner)
        gx, gy = horizontal_gradient(exner)
        exact = -cp * vector_cross_spectrum(stack_vector(u, v),
                                            stack_vector((th_mean + th) * gx, (th_mean + th) * gy))
        terms = [vertical_gradient_flux(f_p), nonconservative_adiabatic_pressure(f_p, rho, th_mean),
                 conversion_ape_to_dke(th_mean, w, exner),
                 pressure_divergence_work(u, v, w, div, rho, th_mean, exner),
                 nonlinear_pressure_work_hke(u, v, th, exner)]
        lateral = lateral_pressure_flux(u, v, th_mean, exner)
        iso = lambda a: _interior(isotropize(a, dx, dx, cumulative=False, include_dc=True))
        ex = iso(exact)
        scale = float(np.abs(ex).max())
        without = float(np.abs(sum(iso(t) for t in terms) - ex).max()) / scale
        with_lf = float(np.abs(sum(iso(t) for t in terms) + iso(lateral) - ex).max()) / scale
    print(f"  per-k closure error of the pressure work: without L_p {without:.3f}, "
          f"with L_p {with_lf:.4f}")
    assert without > 0.05 and with_lf < 0.01, (without, with_lf)


def test_compute_budget_lateral_split():
    """compute_budget writes lf_hke, lf_vke, lf_ape, lf_pres with metadata. Periodic axes: all
    vanish and the transfers are unchanged. Bounded axes without detrending: pi_* + lf_* is the
    full transfer at every k."""
    from types import SimpleNamespace
    from budget.budget import compute_budget, _BUDGET_META
    from budget.spectral_primitives import isotropize as iso
    u, v, w, th, dx = _random_3d_fields(seed=11, n=40, w_amp=1.0)
    zf = u.z.values
    theta = 250.0 * np.exp(0.02 ** 2 * zf / 9.81)[:, None, None] + th
    pres = xr.ones_like(u) * (1e5 * np.exp(-zf / 7000.0))[:, None, None]
    ds = xr.Dataset({"u": 10 * u + 20.0, "v": 10 * v, "w": w, "theta": theta, "pressure": pres,
                     "density": pres / (287.0 * 240.0)})
    lf_names = ("lf_hke", "lf_vke", "lf_ape", "lf_pres")
    out = {}
    for axes in (["x", "y"], []):
        cfg = SimpleNamespace(compute=SimpleNamespace(dx=dx, dy=dx, cumulative=True,
                                                      transfer_form="conservative",
                                                      periodic_axes=axes, rechunk_spatial=False))
        with _Spectral(False, None):
            out[str(axes)] = compute_budget(ds, cfg).compute()
    per, bnd = out["['x', 'y']"], out["[]"]
    for k in lf_names:
        assert per[k].attrs["standard_name"] == _BUDGET_META[k][0]
        r = float(np.abs(per[k]).max() / np.abs(bnd[k]).max())
        print(f"  {k}: max |periodic| / max |bounded| = {r:.1e}")
        assert r < 1e-10, (k, r)
    with _Spectral(False, None):
        full = iso(nonlinear_hke_transfer_flux(ds.u, ds.v, ds.w, conservative=True), dx, dx,
                   cumulative=True)
    err = float(np.abs(bnd.pi_nke + bnd.lf_hke - full).max() / np.abs(full).max())
    print(f"  bounded: max |pi_nke + lf_hke - full transfer| / max |full| = {err:.1e}")
    assert err < 1e-12


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


def test_isotropize_conserves_variance_on_any_grid():
    """Summed over all bins (DC and the corners beyond the Nyquist circle included), the
    isotropic spectrum equals the 2-D sum, i.e. the domain mean: on even and odd grids, square
    and not. The last bin edge used to sit below the resolution of the largest |k|, which dropped
    the corner mode (and, with the Nyquist cut-off, the Nyquist column): 1e-6 of white noise, but
    several % of a transfer on interpolated model output, whose 2-dx checkerboard sits there."""
    rng = np.random.default_rng(0)
    dx = 600.0
    with _NoTaper():
        for ny, nx in ((40, 40), (41, 40), (40, 41), (41, 41), (301, 751), (300, 751), (301, 750)):
            co = {"y": np.arange(ny) * dx, "x": np.arange(nx) * dx}
            a = xr.DataArray(rng.normal(size=(ny, nx)) + 3.0, dims=("y", "x"), coords=co)
            checker = xr.DataArray(np.outer((-1.0) ** np.arange(ny), (-1.0) ** np.arange(nx)),
                                   dims=("y", "x"), coords=co)
            for f in (a, a + 5.0 * checker):
                s2d = sp.scalar_spectrum(f)
                total = float(isotropize(s2d, dx, dx, nyquist=False, include_dc=True).sum())
                expect = float((f ** 2).mean())
                assert abs(total / expect - 1) < 1e-12, (ny, nx, total / expect - 1)
        print("  variance conserved to 1e-12 on 40/41 x 40/41 and 300/301 x 750/751 grids")


def test_isotropize_odd_grid_wavenumbers():
    """On an odd grid the rFFT length nx // 2 + 1 equals that of nx - 1; the spectra carry the
    physical nx, so a pure mode lands in its own bin: on a 7 x 7 grid, cos(2 pi 2 x / L) has
    |k| = 2 * 2 pi / L (reading nx as 6 put it at 2.33 x 2 pi / L)."""
    n, dx = 7, 1000.0
    x = np.arange(n) * dx
    L = n * dx
    f = xr.DataArray(np.cos(2 * np.pi * 2 * x / L)[None, :].repeat(n, 0), dims=("y", "x"),
                     coords={"y": x, "x": x})
    with _NoTaper():
        e = isotropize(sp.scalar_spectrum(f), dx, dx, nyquist=False, include_dc=True)
    peak = float(e.wavenumber[int(e.argmax())])
    print(f"  pure mode at 2 x 2 pi / L: peak bin at {peak * L / (2 * np.pi):.3f} x 2 pi / L")
    assert abs(peak * L / (2 * np.pi) - 2.0) < 1e-9
    assert "nx_grid" not in e.coords


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
    """FFT path == exact lattice sum with reflect, periodic/reflect and edge/symmetric boundaries,
    for increments of (u, v, w) and for the default (u, v)."""
    ds, fields, d, scales = _nonperiodic_case()
    horizontal = {k: fields[k] for k in ("u", "v")}
    for xb, yb in [("reflect", "reflect"), ("periodic", "reflect"), ("edge", "symmetric")]:
        for label, components, ref_fields in (("u, v, w", ("u", "v", "w"), fields),
                                              ("default u, v", None, horizontal)):
            out = inter_scale_kinetic_energy_transfer(
                ds, scales=scales, x_coord_boundary=xb, y_coord_boundary=yb,
                allow_rechunk=False, increment_components=components)["energy_transfer"]
            out = out.transpose("scale", "z", "y", "x").values
            for k, ell in enumerate(scales):
                ref = _brute_force_dell_padded(ref_fields, d, d, ell, xb, yb)
                err = float(np.abs(out[k] - ref).max() / np.abs(ref).max())
                assert err < 1e-5, (xb, yb, label, ell, err)
        print(f"  boundaries x={xb:8s} y={yb:9s}: FFT path matches the exact lattice sum (<1e-5) "
              f"for (u, v, w) and the default (u, v)")


def test_shift_fallback_close_to_exact():
    """r-bin streaming path (fallback): exact inside 8 dx, domain means within 1% beyond."""
    import budget.inter_scale_transfers as ist
    ds, fields, d, scales = _nonperiodic_case()
    saved = set(ist._FFT_PAD_MODES)
    try:
        ist._FFT_PAD_MODES.clear()  # force the shift path
        out = inter_scale_kinetic_energy_transfer(
            ds, scales=scales, x_coord_boundary="reflect", y_coord_boundary="reflect",
            allow_rechunk=False, increment_components=("u", "v", "w"))["energy_transfer"]
        out = out.transpose("scale", "z", "y", "x").values
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
# Generalized transfer D_ell - C_ell (horizontally divergent flow)
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
        kw = dict(scales=scales, x_coord_boundary=xb, y_coord_boundary=yb, allow_rechunk=False,
                  increment_components=("u", "v", "w"))
        out = inter_scale_kinetic_energy_transfer(ds, divergence_correction=True, divergence=div,
                                                  **kw).transpose("scale", "z", "y", "x")
        plain = inter_scale_kinetic_energy_transfer(ds, **kw).transpose("scale", "z", "y", "x")
        assert np.array_equal(out["energy_transfer"].values, plain["energy_transfer"].values)
        minus_c = (out["energy_transfer_generalized"] - out["energy_transfer"]).values
        for k, ell in enumerate(scales):
            ref = _brute_force_exchange_padded(fields, div_np, d, d, ell, xb, yb)
            err = float(np.abs(minus_c[k] - ref).max() / np.abs(ref).max())
            assert err < 1e-4, (xb, yb, ell, err)
        print(f"  boundaries x={xb:8s} y={yb:9s}: C_ell matches the increment form (<1e-4), "
              f"D_ell unchanged")
    zero = inter_scale_kinetic_energy_transfer(ds, divergence_correction=True,
                                               divergence=xr.zeros_like(div), scales=scales,
                                               allow_rechunk=False)
    assert np.array_equal(zero["energy_transfer_generalized"].values,
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


def test_generalized_dell_closes_local_budget():
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
    dc = out["energy_transfer_generalized"].isel(z=1, scale=0).transpose("y", "x").values
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
    f = {k: np.zeros((n, n)) for k in ("u", "v", "w", "dzu", "dzv", "dzw")}
    for _ in range(14):
        kx, ky = rng.integers(-4, 5, 2)
        m = 2 * np.pi / H * rng.uniform(0.5, 2.0)
        ph = 2 * np.pi * (kx * X + ky * Y) / L
        for c, amp in (("u", 1.0), ("v", 1.0), ("w", 0.6)):
            a, phase = rng.normal() * amp, rng.uniform(0, 2 * np.pi)
            f[c] += a * np.cos(ph + phase)                 # value at z = 0 of cos(ph + m z + phase)
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


def test_generalized_transfer_equals_kernel_weighted_spectral_transfer():
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
        d_star = float(full["energy_transfer_generalized"].mean())
        d_hor = float(hor["energy_transfer_generalized"].mean())
        d_plain = float(full["energy_transfer"].mean())
        err = abs(d_star / spec - 1)
        print(f"  {factor:5.1f}  {spec: .4e}  {d_star: .4e}  {err:9.1e}   {d_hor / spec:9.3f}"
              f"   {d_plain / spec:7.3f}")
        assert err < (5e-3 if factor < 8 else 1e-3), (factor, err)
        assert abs(d_hor / spec - 1) > 0.05 and abs(d_plain / spec - 1) > 0.05


def test_generalized_transfer_with_w_equals_hke_plus_vke():
    """With increments of (u, v, w), <D*_ell> == sum_k [T_h(k) + T_w(k)] [1 - G^(k)].

    T_w is the flux-form VKE transfer of the spectral budget,
        T_w = -<w, U.grad w + 1/2 div w + 1/2 w dz w>_k + 1/2 <dz w, w w>_k,
    the HKE form with w advected, so C_ell and V_ell sum over u, v and w. D_ell itself sees w only
    through |du|^2 (horizontal filter). Same fields and independent Fourier-space evaluation as
    test_generalized_transfer_equals_kernel_weighted_spectral_transfer.
    """
    f, x, d, ddx, ddy = _spectral_3d_case()
    n = x.size
    u, v, w, dzu, dzv, dzw = (f[k] for k in ("u", "v", "w", "dzu", "dzv", "dzw"))
    div = ddx(u) + ddy(v)

    hat = lambda g: np.fft.fft2(g) / n ** 2
    cross = lambda a, b: np.real(np.conj(hat(a)) * hat(b))
    au = u * ddx(u) + v * ddy(u) + 0.5 * div * u + 0.5 * w * dzu
    av = u * ddx(v) + v * ddy(v) + 0.5 * div * v + 0.5 * w * dzv
    aw = u * ddx(w) + v * ddy(w) + 0.5 * div * w + 0.5 * w * dzw
    t_h = -(cross(u, au) + cross(v, av)) + 0.5 * (cross(dzu, w * u) + cross(dzv, w * v))
    t_w = -cross(w, aw) + 0.5 * cross(dzw, w * w)
    assert abs(t_w.sum()) < 1e-10 * np.abs(t_w).sum()                  # conservative

    da = lambda g: xr.DataArray(np.repeat(g[None], 3, 0), dims=("z", "y", "x"),
                                coords={"z": [-1.0, 0.0, 1.0], "y": x, "x": x})
    ds = xr.Dataset({"u": da(u), "v": da(v), "w": da(w)})
    def generalized_mean(ell, shear_w):
        out = inter_scale_kinetic_energy_transfer(
            ds, scales=np.array([ell]), x_coord_boundary="periodic", y_coord_boundary="periodic",
            allow_rechunk=False, increment_components=("u", "v", "w"), divergence=da(div),
            divergence_correction=True, vertical_advection=True, dzu=da(dzu), dzv=da(dzv),
            dzw=da(shear_w)).isel(z=1, scale=0)
        return float(out["energy_transfer_generalized"].mean())

    print("  ell/dx   spectral (h+w)   D*/spec-1   without w in V_ell   T_h only / (h+w)")
    for factor, tol in ((5.0, 5e-3), (8.0, 1e-3), (12.0, 1e-3)):
        ell = factor * d
        weight = 1.0 - _lattice_filter_response(n, d, ell)
        spec, spec_h = float(((t_h + t_w) * weight).sum()), float((t_h * weight).sum())
        err = abs(generalized_mean(ell, dzw) / spec - 1)
        err_no_vw = abs(generalized_mean(ell, np.zeros_like(dzw)) / spec - 1)   # V_ell over u, v
        print(f"  {factor:5.1f}  {spec: .4e}  {err:12.1e}   {err_no_vw:16.1e}   "
              f"{spec_h / spec:12.3f}")
        assert err < tol, (factor, err)
        assert err_no_vw > 10 * err and abs(spec_h / spec - 1) > 10 * err   # discriminates


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
    d_g = float(ground["energy_transfer_generalized"].mean())
    d_p = float(pert["energy_transfer_generalized"].mean())
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


def test_generalized_transfer_closed_form_triad():
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
        d_star = float(out["energy_transfer_generalized"].mean())
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


def test_generalized_transfer_closed_form_boussinesq_wave():
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
            d_star = float(ground["energy_transfer_generalized"].mean())
            d_plain = float(ground["energy_transfer"].mean())
            d_pert = float(pert["energy_transfer_generalized"].mean())
            corr_max = float(np.abs(ground["energy_transfer_generalized"]
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
    kw = dict(scales=scales[:1], allow_rechunk=False, divergence_correction=True,
              periodic_axes="none")
    generalized = lambda data, **extra: inter_scale_kinetic_energy_transfer(
        data, **kw, **extra)["energy_transfer_generalized"].values

    with_both = ds.assign(divergence=given, vorticity=xr.zeros_like(given))
    with_div = ds.assign(divergence=given)
    finite_diff = horizontal_divergence(ds["u"], ds["v"])
    assert np.array_equal(generalized(with_both), generalized(ds, divergence=given))
    assert np.array_equal(generalized(with_div), generalized(ds, divergence=finite_diff))
    assert np.array_equal(generalized(with_both, divergence=finite_diff),
                          generalized(ds, divergence=finite_diff))
    assert not np.allclose(generalized(with_both), generalized(with_div))      # discriminates
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
    corr = (out["energy_transfer_generalized"] - out["energy_transfer"]).values
    assert np.isfinite(out["energy_transfer_generalized"].values).all()
    assert np.array_equal(zero["energy_transfer_generalized"].values, zero["energy_transfer"].values)
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
        for key in ("energy_transfer", "energy_transfer_generalized"):
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


def test_compute_cli_generalized_transfer(tmp_path=None):
    """budget compute with the generalized transfer, end to end: YAML and command-line options
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

    key = "energy_transfer_generalized"
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
    """compute_budget detects a Lorenz-staggered w_half and returns finite terms; every output
    carries its own metadata (staggered and collocated), not the attributes of its inputs."""
    from types import SimpleNamespace
    from budget.budget import compute_budget, _BUDGET_META
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
    ds["theta"].attrs.update(standard_name="air_potential_temperature", units="K")
    ds["w_half"].attrs.update(standard_name="upward_air_velocity", units="m s-1")
    ds["w"].attrs.update(standard_name="upward_air_velocity", units="m s-1")
    out = compute_budget(ds, cfg).compute()
    assert all(bool(np.isfinite(out[k]).all()) for k in out.data_vars)
    for k in ("vf_hke", "vfd_dke", "vf_pres", "vfd_pres", "cad", "caz", "np_vke", "div_hke"):
        assert k in out
    collocated = compute_budget(ds.drop_vars("w_half"), cfg)
    for label, budget in (("staggered", out), ("collocated", collocated)):
        for k, da in budget.data_vars.items():
            want = dict(zip(("standard_name", "long_name", "units"), _BUDGET_META[k]))
            assert da.attrs == want, (label, k, da.attrs)
    print(f"  metadata of all {len(out.data_vars)} outputs from the table (staggered, collocated)")


def _vke_wave(dz, n, lam_z=20e3, z0=60e3, depth=24e3, lam_x=60e3, U=40.0, W0=2.0, H=7e3,
              theta0=300.0, N=0.02, harmonic=0.5, beta=0.7):
    """Gravity wave whose VKE budget terms are known exactly.

    theta_bar = theta0 exp(a z) (N constant), pi_bar exactly hydrostatic, uniform wind U. w and
    Phi = c_p theta_bar pi' grow as exp(z / 2H) and carry a second harmonic (so that the cubic
    terms have a non-zero mean); theta' = (theta_bar / g)(c_p theta_bar dz pi' + U dx w), so that
    the linear w equation holds and B + P = (w, U dx w) = 0 while B and P are large.
    Returns the full and half levels, x, and a function of z giving the fields and their exact
    derivatives.
    """
    from budget.constants import cp, g
    a = N ** 2 / g
    zh = np.arange(z0 - depth / 2, z0 + depth / 2 + 0.5 * dz, dz)
    zf = 0.5 * (zh[1:] + zh[:-1])
    x = np.arange(n) * 2 * lam_x / n
    k, m = 2 * np.pi / lam_x, 2 * np.pi / lam_z
    F0 = U * (m / k) * W0

    def fields(z):
        Z, Y, X = np.meshgrid(z, x, x, indexing="ij")
        ph = k * X + m * Z
        grow = np.exp((Z - z0) / (2 * H))
        cos_ = np.cos(ph) + harmonic * np.cos(2 * ph + beta)
        sin_ = np.sin(ph) + 2 * harmonic * np.sin(2 * ph + beta)
        th_bar = theta0 * np.exp(a * Z)
        w = W0 * grow * cos_
        dxw = -k * W0 * grow * sin_
        phi = F0 * grow * cos_
        dz_pi = (F0 * grow * (cos_ / (2 * H) - m * sin_) - a * phi) / (cp * th_bar)
        return dict(w=w, dzw=W0 * grow * (cos_ / (2 * H) - m * sin_), dxw=dxw,
                    u=U - (m / k) * w, th_bar=th_bar, pi_bar=1.0 + g / (cp * theta0 * a)
                    * (np.exp(-a * Z) - 1.0), pi_p=phi / (cp * th_bar), dz_pi_p=dz_pi,
                    th_p=th_bar / g * (cp * th_bar * dz_pi + U * dxw))

    return zf, zh, x, fields


def test_vke_budget_synthetic_gravity_wave():
    """The VKE terms of compute_budget converge to their exact values at second order.

    VKE equation (dry): d/dt (w, w)/2 = (w, -u.grad w - w dz w - c_p theta dz pi - g)_k, split as
    advection pi_vke + vfd_vke + j_vke + div_vke = -(w, u.grad w + w dz w)_k,
    caz = B + P = (g/theta_bar)(w, theta') - c_p theta_bar (w, dz pi'), np_vke = -c_p (w, theta' dz pi').
    In the test wave B + P = 0 exactly while B and P are large, as in mountain waves, so the
    error of caz is the discretization error of a near-cancellation. Halving dz and dx must
    reduce every error by about 4, for native half-level w (staggered) and interpolated w.
    """
    from types import SimpleNamespace
    from budget.budget import compute_budget
    from budget.constants import cp, g, Rd

    def exact(zf, x, fields):
        f = fields(zf)
        pert = lambda q: q - q.mean((1, 2), keepdims=True)
        mean = lambda q: q.mean((1, 2))
        w, th_bar = pert(f["w"]), f["th_bar"][:, 0, 0]
        return dict(adv=-mean(w * pert(f["u"] * f["dxw"] + f["w"] * f["dzw"])),
                    P=-cp * th_bar * mean(w * pert(f["dz_pi_p"])),
                    BP=g / th_bar * mean(w * pert(f["th_p"])) - cp * th_bar * mean(w * pert(f["dz_pi_p"])),
                    N=-cp * mean(w * pert(f["th_p"] * f["dz_pi_p"])))

    def computed(zf, zh, x, fields, staggered):
        f, fh = fields(zf), fields(zh)
        da = lambda a: xr.DataArray(a, dims=("z", "y", "x"), coords={"z": zf, "y": x, "x": x})
        theta, exner = f["th_bar"] + f["th_p"], f["pi_bar"] + f["pi_p"]
        pres = 1e5 * exner ** (cp / Rd)
        w_half = xr.DataArray(fh["w"], dims=("z_half", "y", "x"),
                              coords={"z_half": zh, "y": x, "x": x})
        ds = xr.Dataset({"u": da(f["u"]), "v": da(np.zeros_like(f["u"])),
                         "w": _collocated(w_half, zf), "theta": da(theta), "pressure": da(pres),
                         "density": da(pres / (Rd * theta * exner))})
        if staggered:
            ds["w_half"] = w_half
        cfg = SimpleNamespace(compute=SimpleNamespace(dx=x[1] - x[0], dy=x[1] - x[0],
                                                      cumulative=True, transfer_form="flux",
                                                      rechunk_spatial=False))
        with _NoTaper():
            out = compute_budget(ds, cfg).compute()
        tot = {k: out[k].isel(wavenumber=0).values for k in
               ("pi_vke", "vfd_vke", "j_vke", "div_vke", "caz", "np_vke")}
        return dict(adv=tot["pi_vke"] + tot["vfd_vke"] + tot["j_vke"] + tot["div_vke"],
                    BP=tot["caz"], N=tot["np_vke"])

    errors = {}
    for dz, n in ((500.0, 96), (250.0, 192)):
        zf, zh, x, fields = _vke_wave(dz, n)
        ex = exact(zf, x, fields)
        inner = (zf > 54e3) & (zf < 66e3)
        p_scale = np.abs(ex["P"][inner]).mean()
        for staggered in (True, False):
            got = computed(zf, zh, x, fields, staggered)
            for key in ("adv", "N", "BP"):
                scale = p_scale if key == "BP" else np.abs(ex[key][inner]).mean()
                errors[dz, staggered, key] = float(np.abs(got[key] - ex[key])[inner].mean() / scale)
    for staggered in (True, False):
        label = "staggered " if staggered else "collocated"
        coarse = [errors[500.0, staggered, k] for k in ("adv", "N", "BP")]
        fine = [errors[250.0, staggered, k] for k in ("adv", "N", "BP")]
        print(f"  {label}: errors at dz = 500 / 250 m (advection, N, B + P relative to |P|): "
              + ", ".join(f"{c:.4f} / {f:.4f}" for c, f in zip(coarse, fine)))
        for c, f in zip(coarse, fine):
            assert c / f > 3.0, (label, c, f)                  # second-order convergence
            assert f < 0.015, (label, f)


def test_staggered_outputs_have_zarr_compatible_chunks():
    """Shifted half-level slices must not leave irregular dask chunks along the vertical.
    501 half levels chunked (251, 250) gave vf_vke chunks (250, 1, 249) on the 500 full levels,
    which zarr refuses."""
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
    """An irregular div_hke pattern (z chunks 11, ..., 9, 2, 11, ..., 5) is rechunked before the
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
             test_generalized_dell_closes_local_budget,
             test_generalized_transfer_equals_kernel_weighted_spectral_transfer,
             test_generalized_transfer_with_w_equals_hke_plus_vke,
             test_perturbation_frame,
             test_generalized_transfer_closed_form_triad,
             test_generalized_transfer_closed_form_boussinesq_wave,
             test_correction_divergence_source, test_correction_error_paths,
             test_correction_geographic_grid,
             test_correction_independent_of_chunking,
             test_compute_cli_generalized_transfer,
             test_vertical_operators_exact_for_linear_profiles,
             test_staggered_conversions_and_dwdz_more_accurate,
             test_staggered_flux_divergence_telescopes, test_compute_budget_with_half_level_w,
             test_vke_budget_synthetic_gravity_wave,
             test_staggered_outputs_have_zarr_compatible_chunks,
             test_write_guard_makes_any_chunks_zarr_safe,
             test_reader_keeps_native_half_level_w, test_geometry_recipe_matches_direct_geometry,
             test_cartesian_grid_is_not_global,
             test_periodic_derivative_float32_coordinates,
             test_lateral_fluxes_vanish_on_periodic_axes,
             test_interior_transfer_sums_to_zero_on_bounded_axes,
             test_lateral_fluxes_converge_to_boundary_fluxes,
             test_pressure_work_closes_on_bounded_axes,
             test_compute_budget_lateral_split,
             test_isotropize_conserves_variance_on_any_grid,
             test_isotropize_odd_grid_wavenumbers,
             test_conservative_transfers_sum_to_zero_on_periodic_axes,
             test_conservative_and_flux_forms_converge,
             test_periodic_axes_skip_taper_and_linear_detrend,
             test_compute_budget_conservative_periodic_axes,
             test_cross_spectrum_parseval_with_mean_and_no_detrend,
             test_compute_cli_conservative_periodic,
             test_spatial_dims_from_units_and_axes,
             test_errico_detrend,
             test_periodicity_detected_from_the_seam,
             test_scale_transfer_detects_periodic_axes]
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
