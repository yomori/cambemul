"""Read a cobaya YAML and draw training points from the COSMOLOGICAL prior only.

We sample exactly the parameters CAMB consumes -- not the likelihood nuisances.
For a real cobaya ``chain.updated.yaml`` the authoritative list is written under
``theory.camb.input_params`` (e.g. ``[ns, As, mnu, ombh2, omch2, cosmomc_theta,
tau]``). Some of those CAMB inputs are not sampled directly but computed from a
sampled parameter via a ``value`` lambda, e.g.::

    cosmomc_theta: {value: 'lambda theta_MC_100: 1e-2*theta_MC_100'}
    As:            {value: 'lambda logA: 1e-10*np.exp(logA)'}

so the *sampled* cosmological parameters here are
``theta_MC_100, logA, ns, ombh2, omch2, tau`` (``mnu`` fixed), while nuisances
like ``A_planck, Tcal, ...`` -- which feed the likelihood, not CAMB -- are
excluded automatically because no CAMB input depends on them.

For a hand-written YAML with no ``theory`` block we fall back to selecting varied
parameters whose names are CAMB-native.
"""
from __future__ import annotations

import math

import numpy as np
import yaml
from scipy.stats import norm, qmc

# CAMB-native cosmological input names (used for the no-theory-block fallback).
DEFAULT_CAMB_INPUTS = {
    "ombh2", "omch2", "H0", "cosmomc_theta", "theta_MC_100", "thetastar",
    "tau", "ns", "As", "logA", "mnu", "nnu", "omk", "omegak", "w", "wa",
    "YHe", "yhe", "nrun", "nrunrun", "r",
}


# --------------------------------------------------------------------------- #
# YAML loading (tolerant of cobaya's python/* tags in the likelihood block)
# --------------------------------------------------------------------------- #
class _TolerantLoader(yaml.SafeLoader):
    pass


_TolerantLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/", lambda loader, suffix, node: None
)


def load_info(yaml_path: str) -> dict:
    """Load a cobaya YAML, ignoring unresolved python/* object tags."""
    with open(yaml_path) as f:
        return yaml.load(f, Loader=_TolerantLoader)


# --------------------------------------------------------------------------- #
# Param-spec classification (hand-written yamls)
# --------------------------------------------------------------------------- #
def classify(spec) -> str:
    if not isinstance(spec, dict):
        return "fixed"
    if "prior" in spec:
        return "varied"
    if "value" in spec:
        return "fixed"
    if "derived" in spec:
        return "derived"
    if set(spec) <= {"latex"}:
        return "derived"
    return "fixed"


def _lambda_args(s: str):
    """['a','b'] from 'lambda a, b: expr'."""
    head = s.split(":", 1)[0].replace("lambda", "", 1)
    return [a.strip() for a in head.split(",") if a.strip()]


# --------------------------------------------------------------------------- #
# Cosmology-only parsing
# --------------------------------------------------------------------------- #
def parse_cosmo_yaml(yaml_path: str) -> dict:
    """Identify the sampled COSMOLOGICAL parameters CAMB expects.

    Returns a dict with:
      sampled     : ordered list of free param names to Latin-hypercube sample
      priors      : {name: prior-spec}
      direct      : CAMB input names that are themselves sampled (pass-through)
      lambdas     : [(camb_input, lambda_str, [arg_names])]  computed inputs
      fixed       : {camb_input: value}                      fixed cosmo inputs
      extra_args  : theory.camb.extra_args (accuracy settings) or {}
      camb_inputs : the theory.camb.input_params list (empty if no theory block)
    """
    info = load_info(yaml_path)
    params = info.get("params", {}) or {}
    camb_cfg = (info.get("theory") or {}).get("camb") or {}
    camb_inputs = list(camb_cfg.get("input_params") or [])
    extra_args = dict(camb_cfg.get("extra_args") or {})

    direct, lambdas, fixed, priors = [], [], {}, {}

    if camb_inputs:
        for ip in camb_inputs:
            spec = params.get(ip)
            if isinstance(spec, dict) and "prior" in spec:
                direct.append(ip)
                priors[ip] = spec["prior"]
            elif isinstance(spec, dict) and "value" in spec:
                val = spec["value"]
                if isinstance(val, str) and val.strip().startswith("lambda"):
                    args = _lambda_args(val)
                    lambdas.append((ip, val, args))
                    for a in args:
                        aspec = params.get(a)
                        if isinstance(aspec, dict) and "prior" in aspec:
                            priors[a] = aspec["prior"]
                else:
                    fixed[ip] = val
            elif spec is not None and not isinstance(spec, dict):
                fixed[ip] = spec
            # else: CAMB-computed / derived input -> nothing to sample
    else:
        # no theory block: keep varied params with CAMB-native names
        for k, v in params.items():
            if classify(v) == "varied" and k in DEFAULT_CAMB_INPUTS:
                direct.append(k)
                priors[k] = v["prior"]
            elif classify(v) == "fixed" and k in DEFAULT_CAMB_INPUTS:
                fixed[k] = v if not isinstance(v, dict) else v.get("value")

    # Detect a theta-parameterized cosmology: a CAMB input cosmomc_theta computed
    # from a single sampled param via a (linear) lambda. Enables the H0 route.
    theta_param = None
    _THETA = {"cosmomc_theta", "cosmomc_theta100", "thetastar", "theta"}
    for camb_name, lam, largs in lambdas:
        if camb_name in _THETA and len(largs) == 1:
            theta_param = {"camb": camb_name, "sampled": largs[0], "lam": lam}
            break

    # sampled list in params-block order, for reproducibility
    sampled = [k for k in params if k in priors]
    return dict(sampled=sampled, priors=priors, direct=direct, lambdas=lambdas,
                fixed=fixed, extra_args=extra_args, camb_inputs=camb_inputs,
                theta_param=theta_param)


def _sample_prior_1d(prior: dict, u):
    if prior.get("dist") in ("norm", "normal", "Normal"):
        return norm.ppf(u, loc=prior["loc"], scale=prior["scale"])
    lo, hi = float(prior["min"]), float(prior["max"])
    return lo + (hi - lo) * u


def lhs_box(specs, n: int, seed: int = 0):
    """Latin-hypercube sample an ordered list of (name, prior) specs."""
    names = [nm for nm, _ in specs]
    lhs = qmc.LatinHypercube(d=len(names), seed=seed).random(n)
    out = np.empty((n, len(names)))
    for j, (_, prior) in enumerate(specs):
        out[:, j] = _sample_prior_1d(prior, lhs[:, j])
    return names, out


def iid_box(specs, n: int, rng):
    """i.i.d. (non-LHS) draws for an ordered list of (name, prior) specs."""
    names = [nm for nm, _ in specs]
    u = rng.random((n, len(names)))
    out = np.empty((n, len(names)))
    for j, (_, prior) in enumerate(specs):
        out[:, j] = _sample_prior_1d(prior, u[:, j])
    return names, out


def sample_cosmo(yaml_path: str, n: int, seed: int = 0):
    """Latin-hypercube sample the cosmological free parameters of a yaml.

    Returns (sampled_names, samples[n, k], parsed).
    """
    parsed = parse_cosmo_yaml(yaml_path)
    names = parsed["sampled"]
    if not names:
        raise ValueError(f"No sampled cosmological parameters found in {yaml_path}")
    specs = [(nm, parsed["priors"][nm]) for nm in names]
    names, samples = lhs_box(specs, n, seed=seed)
    return names, samples, parsed


def draw_from_priors(parsed: dict, names, n: int, rng) -> np.ndarray:
    """Draw ``n`` plain (non-LHS) samples from the priors with a given RNG.

    Used to top-up failed CAMB evaluations: the primary set is a space-filling
    Latin hypercube; these backfill draws are i.i.d. inverse-CDF samples so they
    can be generated on demand without resizing the LHS design.
    """
    u = rng.random((n, len(names)))
    out = np.empty((n, len(names)))
    for j, nm in enumerate(names):
        out[:, j] = _sample_prior_1d(parsed["priors"][nm], u[:, j])
    return out


def build_camb_inputs(parsed: dict, row: dict) -> dict:
    """Construct the CAMB input dict for one sampled point.

    ``row`` maps sampled-param name -> value. Applies the ``value`` lambdas
    (e.g. cosmomc_theta from theta_MC_100, As from logA), passes through the
    directly-sampled inputs, and adds the fixed inputs.
    """
    p = {k: (float(v) if not isinstance(v, str) else v)
         for k, v in parsed["fixed"].items()}
    for nm in parsed["direct"]:
        p[nm] = float(row[nm])
    for camb_name, lam, args in parsed["lambdas"]:
        f = eval(lam, {"np": np, "math": math, "__builtins__": {}})  # noqa: S307
        p[camb_name] = float(f(*[row[a] for a in args]))
    if "logA" in p and "As" not in p:  # fallback convenience
        p["As"] = 1e-10 * np.exp(p.pop("logA"))
    return p


def _eval_lambda(lam: str):
    return eval(lam, {"np": np, "math": math, "__builtins__": {}})  # noqa: S307


def invert_linear_lambda(lam: str, y):
    """Invert a 1-arg LINEAR lambda y = a*x + b, returning x for given y.

    Used to recover the sampled theta param (e.g. theta_MC_100) from CAMB's
    derived cosmomc_theta, since cosmomc_theta = 1e-2 * theta_MC_100 etc.
    """
    f = _eval_lambda(lam)
    b = float(f(0.0))
    a = float(f(1.0)) - b
    return (y - b) / a


def build_camb_inputs_h0(parsed: dict, row: dict, H0: float) -> dict:
    """Build CAMB inputs for the H0 route: pass H0 directly (no theta solve).

    ``row`` maps the non-theta sampled params -> values (plus they may include
    'H0', which is ignored here in favor of the explicit ``H0`` arg). The theta
    lambda is skipped; every other lambda (e.g. As<-logA) and the fixed inputs
    are applied as usual.
    """
    theta_camb = (parsed.get("theta_param") or {}).get("camb")
    p = {k: (float(v) if not isinstance(v, str) else v)
         for k, v in parsed["fixed"].items()}
    for nm in parsed["direct"]:
        p[nm] = float(row[nm])
    for camb_name, lam, args in parsed["lambdas"]:
        if camb_name == theta_camb:
            continue  # using H0 instead of cosmomc_theta
        p[camb_name] = float(_eval_lambda(lam)(*[row[a] for a in args]))
    if "logA" in p and "As" not in p:
        p["As"] = 1e-10 * np.exp(p.pop("logA"))
    p["H0"] = float(H0)
    return p
