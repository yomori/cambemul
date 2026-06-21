"""Acoustic-scale (theta*) warping for the oscillatory CMB spectra — internal.

The acoustic peaks sit at l_n ~ n*pi/theta*, so they slide in l as theta* changes
with cosmology. Emulating in the rescaled coordinate x = l*theta* phase-aligns the
oscillations across cosmologies -> the target becomes smooth and low-dimensional,
which is what lets TE/EE/uTE/uEE reach ~0.1-0.5% instead of a few %.

This module is used automatically by train_emulator / the Emulator class; it is
NOT part of the public API. A warped member bundles two inner emulators (a warp
emulator on the aligned x-grid + a plain emulator for the heavily-damped high-l
tail, which falls outside the common x-grid) and is hybrid-stitched per cosmology.
At predict time the query cosmology's theta* comes from a tiny embedded
theta*-emulator, so the user just calls .predict() as usual.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.interpolate import CubicSpline

from .emulator import emu_from_state, emu_state, fit, predict, target_transform
from .theory import LINEAR_OBS

# Spectra that empirically benefit from the warp (E-mode & cross spectra). The
# temperature autos (TT/uTT) are already at their interpolation-error floor with
# the plain log10 emulator, and PP has no acoustic oscillations.
WARP_OBS = frozenset({"EE", "TE", "uEE", "uTE", "lEE", "lTE"})


def uses_warp(obs: str) -> bool:
    return obs in WARP_OBS


# --------------------------------------------------------------------------- #
# theta* (100*cosmomc_theta) via CAMB — used at TRAIN time to warp the data.
# --------------------------------------------------------------------------- #
def _theta_one(row):
    import camb
    p = camb.CAMBparams()
    p.set_cosmology(H0=float(row[0]), ombh2=float(row[1]), omch2=float(row[2]),
                    mnu=0.06, nnu=3.044, num_massive_neutrinos=1,
                    bbn_predictor="PArthENoPE_880.2_standard.dat")
    return 100.0 * camb.get_background(p).cosmomc_theta()


def compute_theta(X, nproc=None):
    """100*theta* for each cosmology (rows = [H0, ombh2, omch2, ...]). Set
    OMP_NUM_THREADS=1 before calling so the process Pool doesn't oversubscribe."""
    from multiprocessing import Pool
    nproc = nproc or int(os.environ.get("NPROC", "16"))
    with Pool(nproc) as pool:
        return np.array(pool.map(_theta_one, list(np.asarray(X, float)), chunksize=8))


# --------------------------------------------------------------------------- #
# warp / un-warp on a common x = l*theta grid
# --------------------------------------------------------------------------- #
def _grid(ell, theta):
    # SAFE intersection so the forward warp never extrapolates.
    return np.linspace(ell[0] * theta.max(), ell[-1] * theta.min(), len(ell))


def _warp(Y, ell, theta, xg):
    out = np.empty((len(Y), len(xg)))
    for i in range(len(Y)):
        out[i] = CubicSpline(ell, Y[i])(xg / theta[i])
    return out


def _outer(obs):
    """Outer transform applied before warping: identity for sign-changing
    TE/uTE, log10 for the positive auto-spectra."""
    return "linear" if obs in LINEAR_OBS else "log10"


def _fwd(T, kind):
    return np.asarray(T, float) if kind == "linear" else np.log10(np.clip(T, 1e-300, None))


def _inv(T, kind):
    return T if kind == "linear" else 10.0 ** T


# --------------------------------------------------------------------------- #
# train / predict a warped member
# --------------------------------------------------------------------------- #
def train_warped(Xtr, Ytr, theta_tr, ell, obs, rank=16, **fitkw):
    """Returns a warped-member dict (two inner emulators + warp metadata)."""
    ell = np.asarray(ell, float)
    outer = _outer(obs)
    xg = _grid(ell, np.asarray(theta_tr, float))
    Tw = _warp(_fwd(Ytr, outer), ell, np.asarray(theta_tr, float), xg)
    pw, mw, _ = fit(Xtr, Tw, transform="whiten", pca=rank, **fitkw)       # warp emu (x-grid)
    pp, mp, _ = fit(Xtr, Ytr, transform=target_transform(obs), pca=rank, **fitkw)  # plain (tail)
    return dict(obs=obs, outer=outer, ell=ell, xg=xg,
                warp=(pw, mw), plain=(pp, mp))


def predict_warped(member, X, theta):
    """Predict raw C_l for cosmologies X given their 100*theta* (one per row)."""
    X = np.atleast_2d(np.asarray(X, float))
    theta = np.atleast_1d(np.asarray(theta, float))
    ell, xg, outer = member["ell"], member["xg"], member["outer"]
    Tw = predict(*member["warp"], X)              # warp(outer(C_l)) on the x-grid
    plain = predict(*member["plain"], X)          # raw C_l on native l (tail fallback)
    out = np.empty((len(X), len(ell)))
    for i in range(len(X)):
        cs = CubicSpline(xg, Tw[i])
        xq = ell * theta[i]
        valid = (xq >= xg[0]) & (xq <= xg[-1])
        warped = _inv(cs(np.clip(xq, xg[0], xg[-1])), outer)
        out[i] = np.where(valid, warped, plain[i])
    return out


# --------------------------------------------------------------------------- #
# serialization: one emu_<obs>.npz holding both inner emulators + warp metadata
# --------------------------------------------------------------------------- #
def save_warped(path, member, extra=None):
    d = {"warp_member": np.array(True), "obs": np.array(member["obs"]),
         "outer": np.array(member["outer"]), "ell": member["ell"], "xg": member["xg"]}
    for pre, emu in (("w__", member["warp"]), ("p__", member["plain"])):
        d.update({pre + k: v for k, v in emu_state(*emu).items()})
    if extra:
        d.update(extra)
    np.savez(path, **d)


def is_warped_file(z):
    return "warp_member" in z.files


def load_warped(z):
    """z = an open np.load(...) handle for a warped emu_<obs>.npz."""
    def getter(pre):
        return lambda k: z[pre + k] if (pre + k) in z.files else None
    member = dict(obs=str(z["obs"]), outer=str(z["outer"]),
                  ell=z["ell"], xg=z["xg"],
                  warp=emu_from_state(getter("w__")),
                  plain=emu_from_state(getter("p__")))
    reserved = {"warp_member", "obs", "outer", "ell", "xg"}
    extra = {k: z[k] for k in z.files
             if k not in reserved and not k.startswith(("w__", "p__"))}
    return member, extra
