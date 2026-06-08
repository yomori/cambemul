#!/usr/bin/env python
"""(e): train a JAX emulator per observable and save it.

    # --train accepts a single file OR a shard directory (auto-concatenated):
    python scripts/train_emulator.py --train training/lcdm_shards --obs TT,EE,TE,PP,Pk \
        --out-dir emulators --arch resnet --pca 256 --width 512 --depth 4 --epochs 1000

Architectures (--arch): mlp | resnet | cnn.  --pca K emulates the top-K PCA
coefficients of the (transformed) spectrum instead of the full vector (big
accuracy + speed win at scale; set 0 to emulate the full spectrum). One portable
emulator file per observable: emulators/emu_<OBS>.npz.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cambemul.dataset import load_store, read_obs, read_param_names  # noqa: E402
from cambemul.emulator import (  # noqa: E402
    accuracy_report, fit, predict, save, target_transform,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True,
                    help="shard directory (auto-concatenated), or a .npz/.h5 file")
    ap.add_argument("--obs", default=None, help="comma list (default: all in file)")
    ap.add_argument("--out-dir", default="emulators")
    ap.add_argument("--arch", default="mlp", choices=["mlp", "resnet", "cnn"])
    ap.add_argument("--pca", type=int, default=0, help="emulate top-K PCA coeffs (0=off)")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1,
                    help="fraction held out to measure the precision that gets "
                         "stored in (and printed on load of) each emulator")
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    store = load_store(args.train)
    X = store["params"]
    param_names = read_param_names(store)
    file_obs = read_obs(store)
    obs = [o.strip() for o in args.obs.split(",")] if args.obs else file_obs
    os.makedirs(args.out_dir, exist_ok=True)

    # Hold out a test split (never seen by fit) to record honest precision.
    rng = np.random.default_rng(args.seed + 12345)
    Ntot = X.shape[0]
    perm = rng.permutation(Ntot)
    nte = max(1, int(args.test_frac * Ntot))
    te_idx, tr_idx = perm[:nte], perm[nte:]
    Xtr, Xte = X[tr_idx], X[te_idx]

    print(f"Training set: {args.train}")
    print(f"  N={Ntot}  ({len(tr_idx)} train / {len(te_idx)} held-out)  "
          f"params={param_names}")
    print(f"  obs in file: {file_obs} -> training: {obs}")
    print(f"  arch={args.arch}  pca={args.pca}  width={args.width}  depth={args.depth}")

    for o in obs:
        if o not in file_obs:
            print(f"  [skip {o}] not in training file")
            continue
        Y = store[o]
        transform = target_transform(o)
        print(f"\n=== {o}  (transform={transform}, D={Y.shape[1]}) ===")
        params, meta, best_val = fit(
            Xtr, Y[tr_idx], transform=transform, arch=args.arch, pca=args.pca,
            width=args.width, depth=args.depth, epochs=args.epochs, lr=args.lr,
            batch=args.batch, val_frac=args.val_frac, patience=args.patience,
            seed=args.seed,
        )
        ell_grid = None if o == "Pk" else store["ell"]
        acc = accuracy_report(o, Y[te_idx], predict(params, meta, Xte), ell=ell_grid)
        extra = dict(obs=o, param_names=param_names, lmin=store["lmin"],
                     lmax=store["lmax"], accuracy_json=json.dumps(acc))
        if o == "Pk":
            extra.update(kh=store["kh"], z=store["z"], Pk_shape=store["Pk_shape"])
        else:
            extra.update(ell=store["ell"])
        out = os.path.join(args.out_dir, f"emu_{o}.npz")
        save(out, params, meta, extra=extra)
        print(f"  best val_mse={best_val:.4e}  |  held-out {acc['metric']}: "
              f"median={acc['median'] * 100:.2f}% 95%={acc['p95'] * 100:.2f}% "
              f"->  {out}")

    # Derived scalars (e.g. sigma8, H0): one small emulator, linear transform.
    if "derived" in store and "derived_names" in store:
        dnames = [str(x) for x in store["derived_names"]]
        Y = store["derived"]
        print(f"\n=== derived {dnames}  (transform=linear, D={Y.shape[1]}) ===")
        params, meta, best_val = fit(
            Xtr, Y[tr_idx], transform="linear", arch=args.arch, pca=0,
            width=args.width, depth=args.depth, epochs=args.epochs, lr=args.lr,
            batch=args.batch, val_frac=args.val_frac, patience=args.patience,
            seed=args.seed,
        )
        dpred, dtrue = predict(params, meta, Xte), Y[te_idx]
        dparams = []
        for j, nm in enumerate(dnames):
            e = np.abs(dpred[:, j] - dtrue[:, j]) / (np.abs(dtrue[:, j]) + 1e-300)
            q = np.percentile(e, [50, 95, 100])
            dparams.append([nm, float(q[0]), float(q[1]), float(q[2])])
        acc = dict(metric="derived", n_test=int(len(te_idx)), params=dparams)
        out = os.path.join(args.out_dir, "emu_derived.npz")
        save(out, params, meta, extra=dict(obs="derived", param_names=param_names,
                                           derived_names=np.array(dnames),
                                           accuracy_json=json.dumps(acc)))
        print("  derived precision: " + ", ".join(
            f"{nm} {med * 100:.2f}%" for nm, med, *_ in dparams) + f"  ->  {out}")


if __name__ == "__main__":
    main()
