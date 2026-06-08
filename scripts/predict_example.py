#!/usr/bin/env python
"""Load a (precomputed) emulator and predict C_ell for one cosmology.

    # uses a Planck-ish fiducial point, filtered to the params this emulator needs
    python scripts/predict_example.py --emu-dir emulators/gmvspafg_post5sig

    # override any input parameter(s), and optionally save a D_ell plot
    python scripts/predict_example.py --emu-dir emulators/gmvspafg_post5sig \
        --params "logA=3.00,ns=0.97" --plot dell.png

Loading prints the emulator's stored held-out precision (if recorded at train
time); the script then predicts every observable the emulator provides at the
chosen point and prints the spectra (and any derived scalars).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cambemul  # noqa: E402

# Planck 2018-ish defaults; only the params the emulator actually needs are used.
FIDUCIAL = {
    "theta_MC_100": 1.04109, "cosmomc_theta": 1.04109e-2,
    "H0": 67.36, "logA": 3.044, "As": 2.1e-9,
    "ns": 0.9649, "ombh2": 0.02237, "omch2": 0.1200, "tau": 0.0544,
    "mnu": 0.06,
}
_SPECTRA = ("tt", "ee", "te", "bb", "tt_unlensed", "ee_unlensed",
            "te_unlensed", "bb_unlensed", "pp")


def _dell(key, ell, cl):
    """l(l+1)/2pi C_l for CMB (muK^2); [l(l+1)]^2/2pi C_l^phiphi for pp."""
    if key.startswith("pp"):
        return (ell * (ell + 1.0)) ** 2 / (2.0 * np.pi) * cl
    return ell * (ell + 1.0) / (2.0 * np.pi) * cl


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emu-dir", default="emulators/gmvspafg_post5sig",
                    help="emulator directory (or single emu_*.npz file)")
    ap.add_argument("--params", default="",
                    help="overrides as 'name=val,name=val' (else Planck fiducial)")
    ap.add_argument("--plot", default=None, help="save a D_ell plot to this PNG")
    args = ap.parse_args()

    # load (prints the stored precision table)
    e = cambemul.loademul(args.emu_dir)

    # build the input point for exactly the params this emulator consumes
    overrides = {}
    for kv in args.params.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            overrides[k.strip()] = float(v)
    pars, missing = {}, []
    for nm in e.param_names:
        if nm in overrides:
            pars[nm] = overrides[nm]
        elif nm in FIDUCIAL:
            pars[nm] = FIDUCIAL[nm]
        else:
            missing.append(nm)
    if missing:
        sys.exit(f"no fiducial default for {missing}; pass e.g. "
                 f"--params '{missing[0]}=<value>'")

    print("\ninput point:")
    for nm in e.param_names:
        tag = "  (override)" if nm in overrides else ""
        print(f"  {nm:14s} = {pars[nm]:.6g}{tag}")

    out = e.predict(pars)
    ell = out.get("ell")
    spec_keys = [k for k in _SPECTRA if k in out]

    if ell is not None and spec_keys:
        sample = [int(x) for x in (30, 220, 800, 1500, 3000)
                  if ell[0] <= x <= ell[-1]]
        if not sample:   # tiny ell range: just spread a few across it
            sample = sorted({int(v) for v in np.linspace(ell[0], ell[-1], 4)})
        print(f"\npredicted spectra  (ell {int(ell[0])}..{int(ell[-1])}, "
              f"{len(ell)} multipoles):")
        for k in spec_keys:
            cl = np.asarray(out[k])
            dl = _dell(k, ell, cl)
            unit = "[l(l+1)]^2 C_l^pp/2pi" if k.startswith("pp") \
                else "D_l=l(l+1)C_l/2pi [muK^2]"
            pts = "   ".join(f"l={x}: {dl[np.searchsorted(ell, x)]:.4g}"
                             for x in sample)
            print(f"  {k:13s} shape={cl.shape}   {unit}")
            print(f"      {pts}")

    if "Pk" in out:
        print(f"\nPk grid: k {out['k'][0]:.3g}..{out['k'][-1]:.3g} h/Mpc, "
              f"z={list(np.atleast_1d(out['z']))}, shape={np.asarray(out['Pk']).shape}")

    if e.derived_names:
        print("\nderived:")
        for nm in e.derived_names:
            print(f"  {nm:10s} = {float(out[nm]):.6g}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            sys.exit("matplotlib not installed; `pip install matplotlib`")
        keys = [k for k in spec_keys if ell is not None]
        fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.2),
                                 squeeze=False)
        for ax, k in zip(axes[0], keys):
            ax.plot(ell, _dell(k, ell, np.asarray(out[k])))
            ax.set_xlabel(r"$\ell$")
            ax.set_title(k)
            if not k.startswith("te"):
                ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"\nsaved plot -> {args.plot}")


if __name__ == "__main__":
    main()
