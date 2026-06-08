#!/usr/bin/env python
"""Held-out accuracy benchmark for a cambemul shard dataset.

Trains emulators on the first N train shards and evaluates on the last N test
shards (disjoint, deterministic), then prints per-observable accuracy:
fractional error |pred-true|/|true| per (test cosmology, ell) -- summarized as
percentiles, ell-band medians, and the fraction of bins within 0.1% / 1%.
TE (sign-changing) uses error / per-ell RMS instead of fractional.

    python scripts/bench_accuracy.py --train-dir training/lcdm_shards \
        --ntrain 160 --ntest 16 --pca 512 --arch resnet --epochs 400
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cambemul.emulator import fit, predict, target_transform  # noqa: E402
from cambemul.theory import LINEAR_OBS  # noqa: E402


def load_block(shards, key):
    return np.concatenate(
        [np.load(s, allow_pickle=True)[key] for s in shards], axis=0)


def load_all(shards, keys):
    """Single pass over the shards: open each .npz ONCE and collect all keys.

    Much faster than calling load_block per key (which re-opens every shard for
    each observable) when the shards live on a metadata-slow filesystem.
    """
    acc = {k: [] for k in keys}
    for s in shards:
        z = np.load(s, allow_pickle=True)
        for k in keys:
            acc[k].append(z[k])
    return {k: np.concatenate(v, axis=0) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dir", default="training/lcdm_shards")
    ap.add_argument("--ntrain", type=int, default=160, help="# train shards")
    ap.add_argument("--ntest", type=int, default=16, help="# held-out test shards")
    ap.add_argument("--obs", default=None, help="comma list (default: all in file)")
    ap.add_argument("--pca", type=int, default=512)
    ap.add_argument("--arch", default="resnet")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(args.train_dir, "shard*.npz")))
    if len(shards) < args.ntrain + args.ntest:
        sys.exit(f"only {len(shards)} shards; need {args.ntrain}+{args.ntest}")
    tr, te = shards[:args.ntrain], shards[-args.ntest:]

    z0 = np.load(tr[0], allow_pickle=True)
    file_obs = [str(o) for o in z0["obs"]]
    obs = [o.strip() for o in args.obs.split(",")] if args.obs else file_obs
    ell = np.asarray(z0["ell"])
    pnames = [str(x) for x in z0["param_names"]]

    has_derived = "derived" in z0.files
    keys = (["params"] + [o for o in obs if o in file_obs]
            + (["derived"] if has_derived else []))
    t_load = time.time()
    TR = load_all(tr, keys)
    TE = load_all(te, keys)
    print(f"  loaded {len(tr)}+{len(te)} shards (single pass) "
          f"in {time.time() - t_load:.0f}s", flush=True)
    Xtr, Xte = TR["params"], TE["params"]
    Ntr, Nte = Xtr.shape[0], Xte.shape[0]

    lmax = int(ell[-1])
    edges = [e for e in (2, 30, 300, 2000) if e < lmax] + [lmax + 1]
    bands = list(zip(edges[:-1], edges[1:]))
    _k = lambda v: f"{v // 1000}k" if v >= 1000 else str(v)
    band_lbl = [f"{_k(lo)}-{_k(hi - 1)}" for lo, hi in bands]

    print("=" * 78)
    print("cambemul accuracy benchmark")
    print(f"  dataset   : {args.train_dir}  ({len(shards)} shards)")
    print(f"  train/test: {Ntr} / {Nte} cosmologies  ({args.ntrain}/{args.ntest} shards)")
    print(f"  emulator  : arch={args.arch} pca={args.pca} width={args.width} "
          f"depth={args.depth} epochs={args.epochs}")
    print(f"  ell range : {ell[0]}..{ell[-1]}  ({len(ell)} multipoles)")
    print("=" * 78)

    def summarize(o, err, metric):
        p = np.percentile(err, [50, 68, 95, 99, 100])
        w01 = 100.0 * np.mean(err < 1e-3)
        w1 = 100.0 * np.mean(err < 1e-2)
        bandmed = [np.median(err[:, (ell >= lo) & (ell < hi)]) for lo, hi in bands]
        print(f"\n  [{o}]  metric={metric}")
        print(f"    overall: median={p[0]:.2e}  68%={p[1]:.2e}  95%={p[2]:.2e}  "
              f"99%={p[3]:.2e}  max={p[4]:.2e}")
        print(f"    within : {w01:.1f}% < 0.1%   {w1:.1f}% < 1%")
        print("    median by ell-band: " +
              "  ".join(f"{lbl}={bm:.2e}" for lbl, bm in zip(band_lbl, bandmed)))

    for o in obs:
        if o not in file_obs:
            print(f"\n  [{o}] not in dataset; skip")
            continue
        Ytr, Yte = TR[o], TE[o]
        t0 = time.time()
        params, meta, val = fit(
            Xtr, Ytr, transform=target_transform(o), arch=args.arch, pca=args.pca,
            width=args.width, depth=args.depth, epochs=args.epochs,
            batch=args.batch, seed=args.seed, verbose=False)
        pred = predict(params, meta, Xte)
        dt = time.time() - t0
        if o in LINEAR_OBS:
            rms = np.sqrt(np.mean(Yte ** 2, axis=0)) + 1e-300
            err = np.abs(pred - Yte) / rms
            metric = "err/RMS"
        else:
            err = np.abs(pred - Yte) / (np.abs(Yte) + 1e-300)
            metric = "fractional"
        summarize(o, err, metric)
        print(f"    (val_mse={val:.2e}, train+predict {dt:.0f}s)")
        del Ytr, Yte, pred, err

    if has_derived:
        dn = [str(x) for x in z0["derived_names"]]
        Ytr, Yte = TR["derived"], TE["derived"]
        params, meta, val = fit(
            Xtr, Ytr, transform="linear", arch=args.arch, pca=0,
            width=args.width, depth=args.depth, epochs=args.epochs,
            batch=args.batch, seed=args.seed, verbose=False)
        pred = predict(params, meta, Xte)
        print("\n  [derived]  fractional error")
        for j, nm in enumerate(dn):
            e = np.abs(pred[:, j] - Yte[:, j]) / (np.abs(Yte[:, j]) + 1e-300)
            q = np.percentile(e, [50, 95, 100])
            print(f"    {nm:8s} median={q[0]:.2e}  95%={q[1]:.2e}  max={q[2]:.2e}")

    print("\n" + "=" * 78)
    print("done.")


if __name__ == "__main__":
    main()
