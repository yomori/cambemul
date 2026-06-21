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
from cambemul import warp as warp_mod  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True,
                    help="shard directory (auto-concatenated), or a .npz/.h5 file")
    ap.add_argument("--obs", default=None, help="comma list (default: all in file)")
    ap.add_argument("--out-dir", default="emulators")
    ap.add_argument("--arch", default="mlp", choices=["mlp", "resnet", "cnn"])
    ap.add_argument("--pca", type=int, default=None,
                    help="emulate top-K PCA coeffs for ALL obs (overrides the tuned "
                         "per-obs map; 0=full spectrum). Default: tuned map.")
    ap.add_argument("--pca-map", default=None,
                    help="per-obs rank override, e.g. 'PP=8,TE=16'")
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
    ap.add_argument("--no-warp", action="store_true",
                    help="disable the acoustic-scale warp for EE/TE/uEE/uTE "
                         "(on by default; needs CAMB for theta*)")
    args = ap.parse_args()

    # Tuned per-observable PCA rank: low-dimensional PP wants ~8; the CMB spectra
    # ~16 (see the rank study). --pca overrides all; --pca-map overrides per-obs.
    TUNED_RANK = {"PP": 8}
    pca_override = {}
    if args.pca_map:
        for kv in args.pca_map.split(","):
            k, v = kv.split("=")
            pca_override[k.strip()] = int(v)

    def rank_for(o):
        if o in pca_override:
            return pca_override[o]
        if args.pca is not None:
            return args.pca
        return TUNED_RANK.get(o, 16)

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
    rank_str = "  ".join(f"{o}:{rank_for(o)}" for o in obs if o in file_obs)
    print(f"  arch={args.arch}  width={args.width}  depth={args.depth}  ranks: {rank_str}")

    fitkw = dict(arch=args.arch, width=args.width, depth=args.depth, epochs=args.epochs,
                 lr=args.lr, batch=args.batch, val_frac=args.val_frac,
                 patience=args.patience, seed=args.seed)

    # Acoustic-scale warp for EE/TE/uEE/uTE (default on). Needs theta* per
    # cosmology (CAMB) + an embedded theta* emulator the Emulator uses at predict.
    warped_obs = ([o for o in obs if o in file_obs and warp_mod.uses_warp(o)]
                  if not args.no_warp else [])
    theta_tr = theta_te = None
    if warped_obs:
        cache = (os.path.join(args.train, "theta_100.npy") if os.path.isdir(args.train)
                 else os.path.splitext(args.train)[0] + ".theta_100.npy")
        if os.path.exists(cache) and len(np.load(cache)) == Ntot:
            theta_all = np.load(cache)
            print(f"  warp: theta* loaded from cache ({cache})")
        else:
            print("  warp: computing theta* via CAMB "
                  "(run with OMP_NUM_THREADS=1 NPROC=<ncores> for speed) ...")
            theta_all = warp_mod.compute_theta(X)
            np.save(cache, theta_all)
        theta_tr, theta_te = theta_all[tr_idx], theta_all[te_idx]
        pt, mt, _ = fit(Xtr, theta_tr[:, None], transform="linear",
                        **{**fitkw, "epochs": max(args.epochs, 600), "width": 128, "depth": 3})
        terr = np.abs(predict(pt, mt, Xte).ravel() - theta_te) / theta_te
        save(os.path.join(args.out_dir, "emu_theta.npz"), pt, mt,
             extra=dict(obs="theta", param_names=param_names,
                        accuracy_json=json.dumps(dict(
                            metric="theta_frac", median=float(np.median(terr)),
                            p95=float(np.percentile(terr, 95))))))
        print(f"  theta* emulator: median frac err={np.median(terr) * 100:.4f}%"
              f"  ->  emu_theta.npz   (warped: {warped_obs})")

    for o in obs:
        if o not in file_obs:
            print(f"  [skip {o}] not in training file")
            continue
        Y = store[o]
        rank = rank_for(o)
        out = os.path.join(args.out_dir, f"emu_{o}.npz")

        if o in warped_obs:
            print(f"\n=== {o}  (theta-warp[{warp_mod._outer(o)}], pca={rank}, "
                  f"D={Y.shape[1]}) ===")
            mem = warp_mod.train_warped(Xtr, Y[tr_idx], theta_tr, store["ell"], o,
                                        rank=rank, **fitkw)
            acc = accuracy_report(o, Y[te_idx],
                                  warp_mod.predict_warped(mem, Xte, theta_te),
                                  ell=store["ell"])
            warp_mod.save_warped(out, mem, extra=dict(
                param_names=param_names, lmin=store["lmin"], lmax=store["lmax"],
                accuracy_json=json.dumps(acc)))
            print(f"  held-out {acc['metric']}: median={acc['median'] * 100:.2f}% "
                  f"95%={acc['p95'] * 100:.2f}%  ->  {out}")
            continue

        transform = target_transform(o)
        print(f"\n=== {o}  (transform={transform}, pca={rank}, D={Y.shape[1]}) ===")
        params, meta, best_val = fit(
            Xtr, Y[tr_idx], transform=transform, pca=rank, **fitkw,
        )
        ell_grid = None if o == "Pk" else store["ell"]
        acc = accuracy_report(o, Y[te_idx], predict(params, meta, Xte), ell=ell_grid)
        extra = dict(obs=o, param_names=param_names, lmin=store["lmin"],
                     lmax=store["lmax"], accuracy_json=json.dumps(acc))
        if o == "Pk":
            extra.update(kh=store["kh"], z=store["z"], Pk_shape=store["Pk_shape"])
        else:
            extra.update(ell=store["ell"])
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
