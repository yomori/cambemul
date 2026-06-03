"""Run CAMB for one cosmology and return the requested observables.

Supported observables (``--obs`` tokens):

* ``TT``, ``EE``, ``TE`` : lensed scalar CMB power spectra (C_ell, muK^2).
* ``lTT``, ``lEE``, ``lTE`` : explicitly LENSED (same as bare TT/EE/TE).
* ``uTT``, ``uEE``, ``uTE`` : UNLENSED scalar CMB power spectra.
* ``PP``                 : CMB lensing potential spectrum C_ell^{phiphi}.
* ``Pk``                 : matter power spectrum P(k, z) in (Mpc/h)^3.

The cosmology dict uses CAMB-native names (ombh2, omch2, ns, tau, mnu, As, and
either H0 or cosmomc_theta). ``extra_args`` (e.g. from a cobaya
``theory.camb.extra_args`` block) are forwarded to ``camb.set_params`` so the
training spectra match the chain's theory settings exactly.

All spectra are raw C_ell (not D_ell). CMB arrays are indexed by multipole from
ell=0; CAMB may return beyond ``lmax`` (lensing margin) -- slice as needed.
"""
from __future__ import annotations

import numpy as np

# column index of each base spectrum in CAMB's lensed/unlensed_scalar arrays
_CMB_COL = {"TT": 0, "EE": 1, "BB": 2, "TE": 3}
CMB_OBS = ("TT", "EE", "TE")
ALL_OBS = ("TT", "EE", "TE", "PP", "Pk")
# Observables that can change sign -> emulated linearly (and only finite-checked).
# Everything else is positive and emulated in log10 (so it must be > 0).
LINEAR_OBS = ("TE", "lTE", "uTE")


def parse_obs(token: str) -> dict:
    """Classify an --obs token.

    Returns one of:
      {'kind':'cmb', 'base':'TT'|'EE'|'TE'|'BB', 'lensed':bool, 'token':...}
      {'kind':'pp', 'token':'PP'}
      {'kind':'pk', 'token':'Pk'}
    Bare TT/EE/TE are LENSED; prefix 'l'/'u' forces lensed/unlensed.
    """
    if token == "PP":
        return {"kind": "pp", "token": token}
    if token == "Pk":
        return {"kind": "pk", "token": token}
    lensed, base = True, token
    if token[:1] == "l" and token[1:] in _CMB_COL:
        base = token[1:]
    elif token[:1] == "u" and token[1:] in _CMB_COL:
        lensed, base = False, token[1:]
    if base in _CMB_COL:
        return {"kind": "cmb", "base": base, "lensed": lensed, "token": token}
    raise ValueError(f"unknown observable token {token!r}")


def is_spectrum(token: str) -> bool:
    """True for ell-indexed spectra (CMB or PP); False for Pk."""
    return token != "Pk"


def _get_derived(results, pars, names):
    """Extract scalar derived parameters from a CAMB result.

    Supports 'H0', 'sigma8', 'omegam' explicitly; any other name is looked up in
    results.get_derived_params() (e.g. 'rdrag', 'age', 'zstar', ...).
    """
    out = {}
    dp = None
    for nm in names:
        if nm == "H0":
            out[nm] = float(pars.H0)
        elif nm == "sigma8":
            out[nm] = float(results.get_sigma8_0())
        elif nm in ("omegam", "Omega_m", "omega_m"):
            h = pars.H0 / 100.0
            out[nm] = float((pars.ombh2 + pars.omch2
                             + getattr(pars, "omnuh2", 0.0)) / h ** 2)
        else:
            if dp is None:
                dp = results.get_derived_params()
            if nm not in dp:
                raise ValueError(f"unknown derived parameter {nm!r}")
            out[nm] = float(dp[nm])
    return out


def run_camb(params: dict, obs, lmax: int = 3000, kmax: float = 10.0,
             z_pk=(0.0,), nonlinear: bool = True, extra_args=None,
             derived=None, npoints_k: int = 300) -> dict:
    """Compute the requested observables for one CAMB-native parameter dict.

    params     : cosmology (ombh2, omch2, ns, tau, mnu, As|logA, H0|cosmomc_theta)
    obs        : iterable of {'TT','EE','TE','PP','Pk'}
    extra_args : optional dict forwarded to camb.set_params (accuracy/nonlinear)
    derived    : optional iterable of derived scalars to return under out['derived']
                 (e.g. ['sigma8','H0']). 'sigma8' forces the matter transfer.
    """
    import camb
    from camb import model

    obs = list(obs)
    parsed = [parse_obs(o) for o in obs]   # validates tokens

    derived = list(derived or [])
    cmb = [p for p in parsed if p["kind"] == "cmb"]
    want_lensed_cmb = any(p["lensed"] for p in cmb)
    want_unlensed_cmb = any(not p["lensed"] for p in cmb)
    want_pp = any(p["kind"] == "pp" for p in parsed)
    want_pk = any(p["kind"] == "pk" for p in parsed)
    want_cls = bool(cmb) or want_pp
    want_lens = want_lensed_cmb or want_pp   # lensing needed (CAMB DoLensing)
    need_sigma8 = "sigma8" in derived
    need_transfer = want_pk or need_sigma8

    merged = dict(params)
    if "logA" in merged and "As" not in merged:
        merged["As"] = 1e-10 * np.exp(merged.pop("logA"))

    ea = dict(extra_args or {})
    if "theta_H0_range" in ea:
        ea["theta_H0_range"] = tuple(ea["theta_H0_range"])
    merged.update(ea)
    merged.setdefault("lmax", lmax)
    if want_lens:
        merged.setdefault("lens_potential_accuracy", 1)

    pars = camb.set_params(**merged)
    pars.WantCls = want_cls
    pars.DoLensing = want_lens
    if "nonlinear" not in ea:
        pars.NonLinear = model.NonLinear_both if nonlinear else model.NonLinear_none
    if need_transfer:
        # ensure z=0 is present so get_sigma8_0() works
        zs = list(z_pk) if want_pk else [0.0]
        if need_sigma8 and 0.0 not in zs:
            zs = sorted(set(zs) | {0.0})
        kw = {}
        if "k_per_logint" in ea:
            kw["k_per_logint"] = ea["k_per_logint"]
        pars.set_matter_power(redshifts=zs, kmax=(kmax if want_pk else 2.0), **kw)

    results = camb.get_results(pars)
    out: dict = {}
    try:  # derived theta (useful for the H0 sampling route)
        out["cosmomc_theta"] = float(results.cosmomc_theta())
    except Exception:
        pass

    if want_cls:
        powers = results.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=True)
        lensed = powers.get("lensed_scalar")        # (n,4) cols [TT,EE,BB,TE]
        unlensed = powers.get("unlensed_scalar")
        ref = lensed if lensed is not None else unlensed
        out["ell"] = np.arange(ref.shape[0])
        for p in cmb:
            arr = lensed if p["lensed"] else unlensed
            out[p["token"]] = arr[:, _CMB_COL[p["base"]]]
        if want_pp:
            out["PP"] = powers["lens_potential"][:, 0]   # col 0 = phiphi

    if want_pk:
        kh, z, pk = results.get_matter_power_spectrum(
            minkh=1e-4, maxkh=kmax, npoints=npoints_k
        )
        out["kh"] = np.asarray(kh)
        out["z"] = np.asarray(z)
        out["Pk"] = np.asarray(pk)           # (nz, nk)

    if derived:
        out["derived"] = _get_derived(results, pars, derived)

    return out
