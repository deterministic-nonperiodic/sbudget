"""
Vertical discretization study for the spectral budget on a Lorenz-staggered grid.

Model grid (as in ICON): u, v, theta', pi' on full levels z_f (layer centres), w on half levels
z_h (interfaces), z_f[k] = (z_h[k] + z_h[k+1]) / 2. Synthetic fields with analytic vertical
derivatives: vertical waves of wavenumber m with amplitude growing as exp(z / 2H), several
horizontal modes, periodic in x, y (no taper). Errors are relative L2 norms over interior levels
and all wavenumber shells, against the analytic truth.

Methods
  current  : w linearly interpolated to full levels, centred 2nd-order differences on full levels
             (xarray differentiate, edge_order=2), vertical derivatives of spectra (dF/dz).
  gridpoint: as current, but dF/dz written with the product rule on grid-point fields:
             dF_h/dz = -1/2 [(dU, wU) + (U, w dU) + (U, U dw)]   (closes t + dF + Div exactly)
  C-type   : native operators of the staggered grid. dw/dz compact at full levels; dpi'/dz and
             dU/dz compact at half levels; fluxes formed at half levels with the native w and
             full-level fields interpolated to half levels; flux divergence = compact difference
             of half-level fluxes to full levels (telescopes exactly).
"""
import sys
import numpy as np
import xarray as xr

sys.path.insert(0, "/home/deterministic-nonperiodic/Projects/sbudget/src")
import budget.spectral_primitives as sp
sp.SPECTRAL_CFG["taper"]["enabled"] = False
sp.SPECTRAL_CFG["detrend"]["type"] = None
from budget.spectral_primitives import scalar_cross_spectrum, isotropize

CP, G = 1004.64, 9.80665
N_H, L_H = 32, 320e3                  # horizontal grid
H_RHO = 7000.0                        # amplitude growth scale (2H)
LAMBDA_Z = 4000.0                     # vertical wavelength
DEPTH = 24000.0
rng = np.random.default_rng(0)
MODES = [(1, 0), (1, 1), (2, 1), (0, 2), (3, 1)]
PHASES = {k: rng.uniform(0, 2 * np.pi, len(MODES)) for k in ("u", "v", "w", "p", "t")}
AMPS = {k: rng.normal(size=len(MODES)) for k in ("u", "v", "w", "p", "t")}
SCALE = {"u": 10.0, "v": 10.0, "w": 0.5, "p": 1e-3, "t": 2.0}
dx = L_H / N_H
xh = np.arange(N_H) * dx
X, Y = np.meshgrid(xh, xh, indexing="xy")
M = 2 * np.pi / LAMBDA_Z


def field(name, z, deriv=False):
    """A(z) sum_n a_n cos(k_n.x - m z + phi_n); deriv=True returns the analytic d/dz."""
    z = np.asarray(z)[:, None, None]
    amp = SCALE[name] * np.exp(z / H_RHO)
    out = np.zeros((z.shape[0], N_H, N_H))
    for (kx, ky), a, ph in zip(MODES, AMPS[name], PHASES[name]):
        arg = 2 * np.pi * (kx * X + ky * Y) / L_H - M * z + ph
        out += a * (np.cos(arg) / H_RHO + M * np.sin(arg)) if deriv else a * np.cos(arg)
    return amp * out


def theta_bar(z):
    return 250.0 * np.exp(0.02 ** 2 * np.asarray(z) / G)


def grid(n_per_lambda, stretch):
    """Half levels with mean spacing LAMBDA_Z / n and geometric stretching factor per level."""
    dz0 = LAMBDA_Z / n_per_lambda
    if stretch == 1.0:
        dzs = np.full(int(DEPTH / dz0), dz0)
    else:
        n = int(np.log(1 + DEPTH * (stretch - 1) / dz0) / np.log(stretch))
        dzs = dz0 * stretch ** np.arange(n)
        dzs *= DEPTH / dzs.sum()
    zh = np.concatenate([[0.0], np.cumsum(dzs)])
    return zh, 0.5 * (zh[1:] + zh[:-1])


def da(a, z):
    return xr.DataArray(a, dims=("z", "y", "x"), coords={"z": z, "y": xh, "x": xh})


def cs(a, b, z):
    """Isotropic cross-spectrum (z, k) of two (z, y, x) arrays."""
    s = isotropize(scalar_cross_spectrum(da(a, z), da(b, z)), dx, dx, cumulative=False)
    return s.values


def dz_centred(a, z):
    return da(a, z).differentiate("z", edge_order=2).values


def dz_spec_centred(s, z):
    return xr.DataArray(s, dims=("z", "k"), coords={"z": z}).differentiate(
        "z", edge_order=2).values


def h2f(a):   # half -> full (midpoint)
    return 0.5 * (a[1:] + a[:-1])


def f2h(a, zf, zh):   # full -> interior half levels (linear)
    wgt = ((zh[1:-1] - zf[:-1]) / (zf[1:] - zf[:-1]))[:, None, None] if a.ndim == 3 else \
        ((zh[1:-1] - zf[:-1]) / (zf[1:] - zf[:-1]))[:, None]
    return a[:-1] + wgt * (a[1:] - a[:-1])


def dz_h2f(a, zh):    # compact derivative of a half-level field to full levels
    d = (zh[1:] - zh[:-1])
    return (a[1:] - a[:-1]) / (d[:, None, None] if a.ndim == 3 else d[:, None])


def dz_f2h(a, zf):    # compact derivative of a full-level field to interior half levels
    d = (zf[1:] - zf[:-1])
    return (a[1:] - a[:-1]) / (d[:, None, None] if a.ndim == 3 else d[:, None])


def rel(est, tru, sl):
    return float(np.linalg.norm(est[sl] - tru[sl]) / np.linalg.norm(tru[sl]))


def run(n_per_lambda, stretch):
    zh, zf = grid(n_per_lambda, stretch)
    K = zf.size
    sl = slice(3, K - 3)                                  # interior full levels
    u, v, pp = (field(n, zf) for n in ("u", "v", "p"))
    du, dv, dpp = (field(n, zf, True) for n in ("u", "v", "p"))
    w_h = field("w", zh)                                   # native half-level w
    w_t, dw_t = field("w", zf), field("w", zf, True)       # truth at full levels
    th_f, th_h = theta_bar(zf)[:, None], theta_bar(zh)[:, None]
    res = {}

    # 1. dw/dz at full levels
    w_i = h2f(w_h)
    res["dw/dz: interp+centred"] = rel(dz_centred(w_i, zf), dw_t, sl)
    res["dw/dz: C-type compact"] = rel(dz_h2f(w_h, zh), dw_t, sl)

    # 2. C_A->h = cp theta_bar (w, dpi'/dz) at full levels
    c_true = CP * th_f * cs(w_t, dpp, zf)
    c_cur = CP * th_f * cs(w_i, dz_centred(pp, zf), zf)
    c_half = CP * theta_bar(zh[1:-1])[:, None] * cs(w_h[1:-1], dz_f2h(pp, zf), zh[1:-1])
    c_ctype = np.full_like(c_true, np.nan)
    c_ctype[1:-1] = h2f(c_half)                            # half -> full average
    c_true_half = CP * theta_bar(zh[1:-1])[:, None] * cs(field("w", zh[1:-1]),
                                                         field("p", zh[1:-1], True), zh[1:-1])
    res["C_A->h: current"] = rel(c_cur, c_true, sl)
    res["C_A->h: C-type (avg to full)"] = rel(c_ctype, c_true, sl)
    res["C_A->h: C-type (native half)"] = rel(c_half, c_true_half, slice(3, K - 4))

    # 3. HKE vertical flux divergence dF_h/dz, F_h = -1/2 (U, wU)
    def F_h(uu, vv, ww, z):
        return -0.5 * (cs(uu, ww * uu, z) + cs(vv, ww * vv, z))
    dF_true = -0.5 * sum(cs(dq, w_t * q, zf) + cs(q, dw_t * q, zf) + cs(q, w_t * dq, zf)
                         for q, dq in ((u, du), (v, dv)))
    dF_cur = dz_spec_centred(F_h(u, v, w_i, zf), zf)
    du_c, dv_c, dw_c = dz_centred(u, zf), dz_centred(v, zf), dz_centred(w_i, zf)
    dF_gp = -0.5 * sum(cs(dq, w_i * q, zf) + cs(q, dw_c * q, zf) + cs(q, w_i * dq, zf)
                       for q, dq in ((u, du_c), (v, dv_c)))
    dw_k = dz_h2f(w_h, zh)
    dF_gp_c = -0.5 * sum(cs(dq, w_i * q, zf) + cs(q, dw_k * q, zf) + cs(q, w_i * dq, zf)
                         for q, dq in ((u, du_c), (v, dv_c)))
    u_h, v_h = f2h(u, zf, zh), f2h(v, zf, zh)
    F_half = F_h(u_h, v_h, w_h[1:-1], zh[1:-1])
    dF_c = np.full_like(dF_true, np.nan)
    dF_c[1:-1] = dz_h2f(F_half, zh[1:-1])
    res["dF_h/dz: current (spectral d/dz)"] = rel(dF_cur, dF_true, sl)
    res["dF_h/dz: grid-point, centred dw"] = rel(dF_gp, dF_true, sl)
    res["dF_h/dz: grid-point, compact dw"] = rel(dF_gp_c, dF_true, sl)
    res["dF_h/dz: C-type flux form"] = rel(dF_c, dF_true, sl)

    # 4. pressure flux divergence dF_p/dz, F_p = -cp theta_bar (w, pi')
    dFp_true = -CP * (th_f * (cs(dw_t, pp, zf) + cs(w_t, dpp, zf))
                      + (theta_bar(zf) * 0.02 ** 2 / G)[:, None] * cs(w_t, pp, zf))
    dFp_cur = dz_spec_centred(-CP * th_f * cs(w_i, pp, zf), zf)
    Fp_half = -CP * theta_bar(zh[1:-1])[:, None] * cs(w_h[1:-1], f2h(pp, zf, zh), zh[1:-1])
    dFp_c = np.full_like(dFp_true, np.nan)
    dFp_c[1:-1] = dz_h2f(Fp_half, zh[1:-1])
    res["dF_p/dz: current (spectral d/dz)"] = rel(dFp_cur, dFp_true, sl)
    res["dF_p/dz: C-type flux form"] = rel(dFp_c, dFp_true, sl)

    # 5. vertical part of the HKE transfer, 1/2[(dU, wU) - (U, w dU)]
    tv_true = 0.5 * sum(cs(dq, w_t * q, zf) - cs(q, w_t * dq, zf) for q, dq in ((u, du), (v, dv)))
    tv_cur = 0.5 * sum(cs(dq, w_i * q, zf) - cs(q, w_i * dq, zf) for q, dq in ((u, du_c), (v, dv_c)))
    du_h, dv_h = dz_f2h(u, zf), dz_f2h(v, zf)
    tv_half = 0.5 * sum(cs(dq, w_h[1:-1] * q, zh[1:-1]) - cs(q, w_h[1:-1] * dq, zh[1:-1])
                        for q, dq in ((u_h, du_h), (v_h, dv_h)))
    tv_c = np.full_like(tv_true, np.nan)
    tv_c[1:-1] = h2f(tv_half)
    res["t_vert: current"] = rel(tv_cur, tv_true, sl)
    res["t_vert: C-type (half, avg to full)"] = rel(tv_c, tv_true, sl)

    # Properties: telescoping of layer sums, and discrete closure t + dF/dz + Div = -(U, w dU)
    dzf = (zh[1:] - zh[:-1])[:, None]
    a, b = 3, K - 3
    F_cur = F_h(u, v, w_i, zf)
    tele = {
        "current": (np.sum(dzf[a:b] * dF_cur[a:b], 0), 0.5 * (F_cur[b] + F_cur[b - 1])
                    - 0.5 * (F_cur[a] + F_cur[a - 1])),
        "C-type": (np.sum(dzf[a:b] * dF_c[a:b], 0), F_half[b - 1] - F_half[a - 1]),
    }
    for k, (lhs, rhs) in tele.items():
        res[f"telescoping: {k}"] = float(np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs))
    div_c = 0.5 * sum(cs(q, q * dw_c, zf) for q in (u, v))
    exact_c = -sum(cs(q, w_i * dq, zf) for q, dq in ((u, du_c), (v, dv_c)))
    for k, dF in (("current", dF_cur), ("grid-point", dF_gp)):
        res[f"closure: {k}"] = rel(tv_cur + dF + div_c, exact_c, sl)
    return res


if __name__ == "__main__":
    for stretch in (1.0, 1.02):
        table = {}
        cols = (4, 5, 6, 8, 12, 16, 32)
        for n in cols:
            for key, val in run(n, stretch).items():
                table.setdefault(key, []).append(val)
        print(f"\n=== {'uniform' if stretch == 1.0 else f'stretched x{stretch}/level'} grid; "
              f"columns: levels per vertical wavelength {cols}")
        for key, vals in table.items():
            print(f"  {key:38s} " + " ".join(f"{v:8.1e}" for v in vals))
