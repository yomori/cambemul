#!/usr/bin/env python
"""(f): test emulator accuracy against a held-out CAMB set.

    python scripts/test_emulator.py --emu-dir emulators --test training/lcdm_test.h5 \
        --obs TT,EE,TE,PP,Pk

Per observable, reports the distribution over the test set of the per-bin
fractional error |pred-true|/|true| (median/68%/95%/max) plus a sign-safe metric
(error / per-bin RMS of truth) meaningful for TE which crosses zero.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cambemul.dataset import load_store, read_obs, read_param_names  # noqa: E402
from cambemul.emulator import load, predict  # noqa: E402


def _summ(name, e):
    p = np.percentile(e, [50, 68, 95, 100])
    print(f"  {name:>10s}  median={p[0]:.2e}  68%={p[1]:.2e}  95%={p[2]:.2e}  max={p[3]:.2e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emu-dir", default="emulators")
    ap.add_argument("--test", required=True, help="held-out .h5/.npz (make_training format)")
    ap.add_argument("--obs", default=None)
    ap.add_argument("--save-npz", default=None)
    args = ap.parse_args()

    store = load_store(args.test)
    Xtest = store["params"]
    test_obs = read_obs(store)
    test_names = read_param_names(store)
    want = [o.strip() for o in args.obs.split(",")] if args.obs else test_obs

    saved: dict = {}
    print(f"Test set: {args.test}  N={Xtest.shape[0]}")
    for o in want:
        emu_path = os.path.join(args.emu_dir, f"emu_{o}.npz")
        if not os.path.exists(emu_path):
            print(f"\n[skip {o}] no emulator at {emu_path}")
            continue
        if o not in test_obs:
            print(f"\n[skip {o}] not in test set")
            continue

        params, meta, extra = load(emu_path)
        emu_names = [str(n) for n in extra["param_names"]]
        cols = [test_names.index(n) for n in emu_names]   # align column order
        pred = predict(params, meta, Xtest[:, cols])
        true = np.asarray(store[o])

        eps = 1e-30
        frac = np.abs(pred - true) / (np.abs(true) + eps)
        rms = np.sqrt(np.mean(true ** 2, axis=0)) + eps
        rel = np.abs(pred - true) / rms

        print(f"\n=== {o}  (D={true.shape[1]}) ===")
        _summ("frac", frac)
        _summ("vs-RMS", rel)
        if o == "TE":
            print("   (TE crosses zero; use the vs-RMS metric, not fractional.)")
        if args.save_npz:
            saved[f"{o}_frac"], saved[f"{o}_relrms"] = frac, rel

    if args.save_npz and saved:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_npz)) or ".", exist_ok=True)
        np.savez(args.save_npz, **saved)
        print(f"\nSaved per-bin errors -> {args.save_npz}")


if __name__ == "__main__":
    main()
