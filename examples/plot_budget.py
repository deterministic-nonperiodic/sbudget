"""Plot a spectral budget written by `sbudget compute` (mode spectral_budget).

One row per energy (HKE, APE), one column per layer. Each panel shows cumulative terms,
X(k) = sum over k' >= k of the term at k': the net effect of all scales smaller than 2 pi / k.
See "Reading the output" in the README for the meaning of each curve.

What the script does, and why:
  - time mean first (the terms are linear in the co-spectra, so averaging them is exact);
  - layer integrals: with a density profile (--density), the integral of rho_bar times each term
    over the layer, in W m-2; the vertical flux terms are drawn as d_z F + J, whose density-
    weighted integral is the net flux through the top and bottom of the layer (exactly for HKE;
    for the pressure and APE fluxes up to the variation of theta_bar or gamma). Without a density
    profile: the layer mean per unit mass, in W kg-1;
  - k = 0: the cumulative output stops at the lowest wavenumber and leaves out the domain-mean
    mode. Interior transfers (pi_*) and lateral fluxes (lf_*) get one more point at the left edge
    (x), their value including k = 0 (mf_*, lf0_*). The interior point is zero up to the modes
    beyond the Nyquist circle; the lateral point is the flux through the side boundaries.
    The lateral pressure flux gets no k = 0 point: lf0_pres depends on the reference of pi';
  - on doubly periodic domains the lateral terms are zero and are left out.
It also prints, per layer, the totals at the lowest wavenumber and two checks: the interior
closure |Pi(k_min) + T(0)| / max |Pi|, and the lateral boundary flux with k = 0.

Requires matplotlib (not a dependency of sbudget).

usage:
  python plot_budget.py budget.nc --layers 20e3,50e3 50e3,80e3 -o budget.png
  python plot_budget.py budget.nc --layers 20e3,50e3 --density input.nc --density-var rho \\
      --density-zdim height
  python plot_budget.py budget.nc --not-cumulative ...      # older output, cumulative: false
Outputs record cumulative, nyquist_cutoff and include_k0 as global attributes; with include_k0
the k = 0 points come from the k = 0 bin, otherwise from mf_* and lf0_*.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# (variable(s), label, colour, k = 0 variable(s)); a list of variables is summed, a name with a
# leading "-" is subtracted
HKE_TERMS = [
    (["pi_nke"], r"$\Pi_h$ interior transfer", "tab:blue", ["mf_hke"]),
    (["lf_hke"], r"$\nabla_h\cdot F_h$ lateral HKE flux", "tab:purple", ["lf0_hke"]),
    (["lf_pres"], r"$\nabla_h\cdot F_p$ lateral pressure flux", "tab:orange", None),
    (["cad"], r"$C_{A\to h}$ conversion from APE", "tab:red", None),
    (["vfd_dke", "j_hke"], r"$\partial_z F_{h\uparrow} + J_h$ vertical HKE flux", "tab:green", None),
    (["vfd_pres", "j_pres"], r"$\partial_z F_{p\uparrow} + J_p$ vertical pressure flux", "tab:cyan",
     None),
]
APE_TERMS = [
    (["pi_ape"], r"$\Pi_A$ interior transfer", "tab:blue", ["mf_ape"]),
    (["lf_ape"], r"$\nabla_h\cdot F_A$ lateral APE flux", "tab:purple", ["lf0_ape"]),
    (["-cad", "-caz"], r"$-C_A$ conversion to kinetic energy", "tab:red", None),
    (["vfd_ape", "j_ape"], r"$\partial_z F_{A\uparrow} + J_A$ vertical APE flux", "tab:green", None),
]
LATERAL = {"lf_hke", "lf_pres", "lf_ape"}


def combine(ds, names):
    """Sum of the listed variables ("-name" subtracts); None if any is missing."""
    out = 0
    for n in names:
        sign, var = (-1, n[1:]) if n.startswith("-") else (1, n)
        if var not in ds:
            return None
        out = out + sign * ds[var]
    return out


def accumulate(ds):
    """Cumulative sum from the highest wavenumber down, for outputs written with cumulative: false
    (k = 0 terms and energy spectra are left as they are)."""
    out = ds.copy()
    for v in ds.data_vars:
        if "wavenumber" in ds[v].dims and v not in ("hke", "rke", "dke", "vke", "ape"):
            rev = ds[v].isel(wavenumber=slice(None, None, -1))
            out[v] = rev.cumsum("wavenumber").isel(wavenumber=slice(None, None, -1))
    return out


def density_profile(path, var, zdim, z):
    """rho_bar(z) on the budget levels: var averaged over every dimension except zdim."""
    rho = xr.open_dataset(path)[var]
    zdim = zdim or (rho.dims[0] if rho.ndim == 1 else "z")
    if zdim not in rho.dims:
        raise ValueError(f"{var} has no dimension {zdim!r}; set --density-zdim ({rho.dims})")
    rho = rho.mean([d for d in rho.dims if d != zdim]).rename({zdim: "z"}).sortby("z")
    return rho.interp(z=z)


def layer(da, zb, zt, rho):
    """Density-weighted integral over [zb, zt] (W m-2), or layer mean per unit mass (W kg-1)."""
    if rho is not None:
        return (rho * da).sel(z=slice(zb, zt)).integrate("z")
    return da.sel(z=slice(zb, zt)).integrate("z") / (zt - zb)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("budget", help="output of sbudget compute (spectral_budget)")
    p.add_argument("--layers", nargs="+", required=True,
                   help="layers as zb,zt in metres, e.g. 20e3,50e3 50e3,80e3")
    p.add_argument("--density", help="file with the density (any dims; averaged except z)")
    p.add_argument("--density-var", default="rho", help="density variable name (default rho)")
    p.add_argument("--density-zdim", default=None, help="vertical dimension of the density")
    p.add_argument("--not-cumulative", action="store_true",
                   help="the output was written with cumulative: false")
    p.add_argument("-o", "--output", default="budget.png", help="figure file")
    args = p.parse_args(argv)

    b = xr.open_dataset(args.budget)
    b = b.mean("time") if "time" in b.dims else b
    b = b.sortby("z")
    # outputs record whether they are cumulative (sbudget writes the attribute); --not-cumulative
    # is for older files
    if args.not_cumulative or b.attrs.get("cumulative", 1) == 0:
        b = accumulate(b)
    # with include_k0 the first bin is k = 0: its cumulative value is the domain total
    b0 = None
    if float(b.wavenumber[0]) == 0.0:
        b0, b = b.isel(wavenumber=0), b.isel(wavenumber=slice(1, None))
    rho = density_profile(args.density, args.density_var, args.density_zdim, b.z) \
        if args.density else None
    units, units_tex = ("W m-2", r"W m$^{-2}$") if rho is not None else \
        ("W kg-1", r"W kg$^{-1}$")
    layers = [tuple(float(x) for x in s.split(",")) for s in args.layers]

    k = b.wavenumber * 1e3                                           # rad/km
    k0 = 0.5 * float(k[0])                                           # where k = 0 is drawn
    pi_scale = max(float(np.abs(b[v]).max()) for v in ("pi_nke", "pi_ape"))
    present = [v for v in LATERAL if v in b]
    periodic = bool(present) and all(float(np.abs(b[v]).max()) <= 1e-10 * pi_scale
                                     for v in present)
    if periodic:
        print("lateral terms are zero: doubly periodic domain, lateral curves left out")

    rows = [("HKE", HKE_TERMS), ("APE", APE_TERMS)]
    fig, axes = plt.subplots(len(rows), len(layers), figsize=(5.5 * len(layers), 8.5),
                             layout="constrained", squeeze=False)
    for r, (energy, terms) in enumerate(rows):
        for c, (zb, zt) in enumerate(layers):
            ax = axes[r, c]
            print(f"{energy}, {zb / 1e3:g}-{zt / 1e3:g} km, totals at k_min ({units}):")
            for names, label, color, at0 in terms:
                if periodic and names[0] in LATERAL:
                    continue
                da = combine(b, names)
                if da is None:
                    continue
                f = layer(da, zb, zt, rho)
                ax.semilogx(k, f, color=color, lw=2, label=label)
                total = float(f.isel(wavenumber=0))
                expr = " ".join(n if i == 0 else (f"- {n[1:]}" if n[0] == "-" else f"+ {n}")
                                for i, n in enumerate(names))
                msg = f"  {expr:22s} {total:+.3e}"
                k0_value = None
                if at0 is not None and b0 is not None:
                    k0_value = float(layer(combine(b0, names), zb, zt, rho))
                elif at0 is not None and combine(b, at0) is not None:
                    k0_value = total + float(layer(combine(b, at0), zb, zt, rho))
                if k0_value is not None:
                    with_k0 = k0_value
                    ax.plot([k0, float(k[0])], [with_k0, total], color=color, lw=2)
                    ax.plot(k0, with_k0, marker="x", color=color, ms=9, mew=2.5, zorder=5)
                    msg += f"   with k = 0: {with_k0:+.3e}"
                    if names[0].startswith("pi_"):
                        scale = float(np.abs(f).max())
                        msg += f"  (closure {abs(with_k0) / scale:.1e} of max |Pi|)"
                    else:
                        msg += "  (flux through the side boundaries)"
                print(msg)
            ax.axhline(0, color="0.4", lw=0.8)
            ax.set_xlim(0.7 * k0, None)
            ax.text(k0, 0.02, "k = 0", transform=ax.get_xaxis_transform(), ha="center",
                    fontsize=8, color="0.3")
            ax.set_title(f"{energy}, {zb / 1e3:g}–{zt / 1e3:g} km")
            ax.set_xlabel("Wavenumber (rad/km)")
            ax.grid(True, which="both", ls=":", color="0.85")
            sec = ax.secondary_xaxis("top", functions=(lambda x: 2 * np.pi / np.maximum(x, 1e-12),
                                                       lambda x: 2 * np.pi / np.maximum(x, 1e-12)))
            sec.set_xlabel("Wavelength (km)")
            if c == 0:
                ax.set_ylabel(f"Cumulative flux ({units_tex})")
        axes[r, 0].legend(fontsize=8, loc="best")
    fig.savefig(args.output, dpi=120)
    print("written", args.output)


if __name__ == "__main__":
    main()
