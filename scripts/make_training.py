#!/usr/bin/env python
"""(a)-(d): cobaya yaml + --obs -> sample the prior, run CAMB, save a training set.

Single file (small runs)::

    python scripts/make_training.py --yaml examples/lcdm.yaml --obs TT,EE,TE,PP,Pk \
        --n 2000 --lmax 3000 --kmax 10 --out training/lcdm.npz

Sharded (large runs, one SLURM array task per shard)::

    python scripts/make_training.py --yaml examples/lcdm.yaml --obs TT,EE,TE,PP,Pk \
        --n 500000 --nshard 128 --shard $SLURM_ARRAY_TASK_ID \
        --shard-dir training/lcdm_shards --lmax 3000 --kmax 10
    # then train straight off the shard dir (no concat step):
    python scripts/train_emulator.py --train training/lcdm_shards --obs TT,EE,TE,PP,Pk ...

All shards draw the SAME deterministic Latin-hypercube set (fixed --seed) and each
computes a contiguous slice, so no coordination is needed.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cambemul.priors import sample_prior, to_camb_params  # noqa: E402
from cambemul.theory import CMB_OBS, run_camb  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--obs", required=True, help="comma list from TT,EE,TE,PP,Pk")
    ap.add_argument("--n", type=int, default=2000, help="TOTAL training points")
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0, help="this shard index [0,nshard)")
    ap.add_argument("--shard-dir", default=None, help="dir for shard*.npz (sharded mode)")
    ap.add_argument("--out", default=None, help="output path (single-file mode)")
    ap.add_argument("--lmax", type=int, default=3000)
    ap.add_argument("--lmin", type=int, default=2)
    ap.add_argument("--kmax", type=float, default=10.0)
    ap.add_argument("--zpk", default="0.0")
    ap.add_argument("--linear", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    obs = [o.strip() for o in args.obs.split(",") if o.strip()]
    z_pk = [float(z) for z in args.zpk.split(",") if z.strip()]

    # Deterministic full LHS draw; this task computes one contiguous slice.
    names, samples_all, fixed = sample_prior(args.yaml, args.n, seed=args.seed)
    idx = np.array_split(np.arange(args.n), args.nshard)[args.shard]
    samples = samples_all[idx]

    if args.nshard > 1:
        shard_dir = args.shard_dir or (
            (args.out.rsplit(".", 1)[0] if args.out else "training/shards") + "_shards")
        os.makedirs(shard_dir, exist_ok=True)
        out_path = os.path.join(shard_dir, f"shard{args.shard:04d}.npz")
    else:
        out_path = args.out or "training/train.npz"

    print(f"[shard {args.shard}/{args.nshard}] {len(idx)} of {args.n} points")
    print(f"  varied={names}  fixed={fixed}  obs={obs}  -> {out_path}")

    rows = {o: [] for o in obs}
    grids: dict = {}
    good = []
    try:
        from tqdm import tqdm
        it = tqdm(range(len(idx)))
    except ImportError:
        it = range(len(idx))

    for i in it:
        p = to_camb_params(names, samples[i], fixed)
        try:
            res = run_camb(p, obs, lmax=args.lmax, kmax=args.kmax,
                           z_pk=z_pk, nonlinear=not args.linear)
        except Exception as e:
            print(f"  [skip local {i}] CAMB failed: {e}")
            continue
        for o in obs:
            if o in CMB_OBS or o == "PP":
                rows[o].append(res[o][args.lmin:args.lmax + 1])
            elif o == "Pk":
                rows[o].append(res["Pk"].reshape(-1))
        if not grids:
            if any(o in CMB_OBS or o == "PP" for o in obs):
                grids["ell"] = res["ell"][args.lmin:args.lmax + 1]
            if "Pk" in obs:
                grids["kh"] = res["kh"]
                grids["z"] = res["z"]
                grids["Pk_shape"] = np.array(res["Pk"].shape)
        good.append(i)

    if not good:
        sys.exit("All CAMB evaluations failed; nothing to save.")

    out = dict(params=samples[good], param_names=np.array(names),
               obs=np.array(obs), lmin=args.lmin, lmax=args.lmax,
               kmax=args.kmax, nonlinear=not args.linear)
    out.update(grids)
    for o in obs:
        out[o] = np.asarray(rows[o])
        print(f"  {o:3s}: {out[o].shape}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez(out_path, **out)
    print(f"Saved {len(good)}/{len(idx)} points -> {out_path}")


if __name__ == "__main__":
    main()
