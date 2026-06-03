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
from cambemul.priors import (  # noqa: E402
    build_camb_inputs,
    build_camb_inputs_h0,
    iid_box,
    invert_linear_lambda,
    lhs_box,
    parse_cosmo_yaml,
)
from cambemul.theory import LINEAR_OBS, run_camb  # noqa: E402


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
    ap.add_argument("--derived", default="",
                    help="comma list of derived scalars to also emulate, "
                         "e.g. 'sigma8,H0' (sigma8 forces the matter transfer)")
    ap.add_argument("--box", choices=["prior", "h0"], default="prior",
                    help="prior: sample the yaml prior directly. "
                         "h0: for theta-parameterized yamls, sample H0 instead of "
                         "cosmomc_theta (no theta->H0 solve, zero rejections) and "
                         "store CAMB's derived theta as the emulator input.")
    ap.add_argument("--h0-range", default=None,
                    help="'lo,hi' for the H0 route (default: theory.camb "
                         "theta_H0_range, else 40,100)")
    ap.add_argument("--require-valid", action="store_true",
                    help="top up with extra draws until this shard has its full "
                         "quota of VALID points (replaces CAMB failures / NaNs)")
    ap.add_argument("--maxtries-factor", type=float, default=5.0,
                    help="cap top-up attempts at this multiple of the quota")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    obs = [o.strip() for o in args.obs.split(",") if o.strip()]
    z_pk = [float(z) for z in args.zpk.split(",") if z.strip()]
    derived_names = [d.strip() for d in args.derived.split(",") if d.strip()]

    # Parse the cosmological sampling space (excludes likelihood nuisances).
    parsed = parse_cosmo_yaml(args.yaml)
    extra_args = parsed["extra_args"]
    emu_names = parsed["sampled"]          # emulator input coordinates (chain order)
    if not emu_names:
        sys.exit(f"No sampled cosmological parameters found in {args.yaml}")

    h0_mode = args.box == "h0" and parsed["theta_param"] is not None
    if args.box == "h0" and not h0_mode:
        print("  [note] --box h0 ignored: yaml is not theta-parameterized "
              "(no cosmomc_theta lambda); sampling the prior directly.")

    # Build the DRIVING sampling specs (what we actually LHS-sample).
    if h0_mode:
        tp = parsed["theta_param"]
        theta_sampled = tp["sampled"]
        if args.h0_range:
            lo, hi = (float(x) for x in args.h0_range.split(","))
        else:
            rr = extra_args.get("theta_H0_range", [40.0, 100.0])
            lo, hi = float(rr[0]), float(rr[1])
        non_theta = [n for n in emu_names if n != theta_sampled]
        driving_specs = ([(n, parsed["priors"][n]) for n in non_theta]
                         + [("H0", {"min": lo, "max": hi})])
    else:
        driving_specs = [(n, parsed["priors"][n]) for n in emu_names]

    # Deterministic full LHS draw; this task computes one contiguous slice.
    driving_names, samples_all = lhs_box(driving_specs, args.n, seed=args.seed)
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
    print(f"  emulator inputs = {emu_names}")
    print(f"  sampling ({args.box}) = {driving_names}"
          + (f"  H0 in [{lo},{hi}]" if h0_mode else ""))
    print(f"  fixed inputs    = {parsed['fixed']}")
    print(f"  extra_args      = {extra_args}")
    print(f"  obs={obs}  derived={derived_names or None}  -> {out_path}")

    rows = {o: [] for o in obs}
    derived_rows = []
    param_rows = []
    grids: dict = {}
    stats = {"camb_fail": 0, "invalid": 0}

    def _slice(res, o):
        if o == "Pk":
            return res["Pk"].reshape(-1)
        return res[o][args.lmin:args.lmax + 1]    # any ell-spectrum (CMB/PP)

    def _is_valid(sample):
        # drop NaN/inf anywhere, and require positivity for log10 observables
        for o in obs:
            a = sample[o]
            if not np.all(np.isfinite(a)):
                return False
            if o not in LINEAR_OBS and np.any(a <= 0):
                return False
        return True

    def attempt(rowvals):
        nonlocal grids
        row = dict(zip(driving_names, rowvals))
        if h0_mode:
            p = build_camb_inputs_h0(parsed, row, row["H0"])
        else:
            p = build_camb_inputs(parsed, row)
        try:
            res = run_camb(p, obs, lmax=args.lmax, kmax=args.kmax,
                           z_pk=z_pk, nonlinear=not args.linear,
                           extra_args=extra_args, derived=derived_names)
        except Exception:
            stats["camb_fail"] += 1
            return False
        sample = {o: _slice(res, o) for o in obs}
        if not _is_valid(sample):
            stats["invalid"] += 1
            return False
        dvec = None
        if derived_names:
            dvec = np.array([res["derived"][nm] for nm in derived_names], float)
            if not np.all(np.isfinite(dvec)):
                stats["invalid"] += 1
                return False
        # emulator-input row in the chain's coordinate order
        if h0_mode:
            theta_val = invert_linear_lambda(tp["lam"], res["cosmomc_theta"])
            emu_row = np.array([theta_val if nm == theta_sampled else row[nm]
                                for nm in emu_names], float)
        else:
            emu_row = np.array([row[nm] for nm in emu_names], float)
        if not grids:
            if any(o != "Pk" for o in obs):
                grids["ell"] = res["ell"][args.lmin:args.lmax + 1]
            if "Pk" in obs:
                grids["kh"] = res["kh"]
                grids["z"] = res["z"]
                grids["Pk_shape"] = np.array(res["Pk"].shape)
        for o in obs:
            rows[o].append(sample[o])
        if derived_names:
            derived_rows.append(dvec)
        param_rows.append(emu_row)
        return True

    try:
        from tqdm import tqdm
        it = tqdm(range(len(idx)))
    except ImportError:
        it = range(len(idx))

    # primary (space-filling LHS) slice
    for i in it:
        attempt(samples[i])

    # optional top-up to reach the full quota of VALID points
    quota = len(idx)
    if args.require_valid and len(param_rows) < quota:
        rng = np.random.default_rng(args.seed * 1_000_003 + args.shard + 1)
        cap, tries = int(args.maxtries_factor * quota), 0
        print(f"  top-up: {len(param_rows)}/{quota} valid; drawing extras (cap {cap})")
        while len(param_rows) < quota and tries < cap:
            _, extra = iid_box(driving_specs,
                               max(1, (quota - len(param_rows)) * 2), rng)
            for rowvals in extra:
                tries += 1
                attempt(rowvals)
                if len(param_rows) >= quota or tries >= cap:
                    break
        if len(param_rows) < quota:
            print(f"  WARNING: only {len(param_rows)}/{quota} valid after "
                  f"{tries} extra tries (hit cap)")

    if not param_rows:
        sys.exit("No valid CAMB evaluations; nothing to save.")

    print(f"  kept {len(param_rows)} valid  |  camb_fail={stats['camb_fail']}  "
          f"invalid(NaN/inf/<=0)={stats['invalid']}")

    out = dict(params=np.asarray(param_rows), param_names=np.array(emu_names),
               obs=np.array(obs), lmin=args.lmin, lmax=args.lmax,
               kmax=args.kmax, nonlinear=not args.linear, box=args.box)
    out.update(grids)
    for o in obs:
        out[o] = np.asarray(rows[o])
        print(f"  {o:3s}: {out[o].shape}")
    if derived_names:
        out["derived"] = np.asarray(derived_rows)
        out["derived_names"] = np.array(derived_names)
        print(f"  derived {derived_names}: {out['derived'].shape}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez(out_path, **out)
    print(f"Saved {len(param_rows)} points -> {out_path}")


if __name__ == "__main__":
    main()
