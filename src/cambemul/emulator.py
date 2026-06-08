"""JAX emulator (flax + optax) with selectable architectures and a PCA front-end.

One emulator maps standardized cosmological parameters -> a (transformed,
optionally PCA-compressed, standardized) observable. Recipe (CosmoPower-style):

* standardize inputs (per-parameter mean/std),
* transform the target: ``log10`` for positive spectra (TT/EE/PP/Pk),
  ``linear`` for sign-changing TE,
* optional PCA on the transformed target (emulate ~N_pca coefficients instead of
  the full length-D spectrum -- big accuracy + speed win at scale),
* standardize the target (per-output, or per-PCA-coeff),
* train the chosen backbone with Adam + cosine LR decay + early stopping.

Backbones (``arch``):
  'mlp'    -- dense GELU stack (baseline; CosmoPower-style).
  'resnet' -- residual dense blocks (LayerNorm -> Dense -> GELU -> Dense + skip);
              trains deeper nets stably, usually a modest accuracy gain. Low risk.
  'cnn'    -- dense head to length-D, then 1-D conv residual refinement (SAME
              padding) that exploits locality/smoothness along ell. Experimental;
              most useful when emulating the FULL spectrum (no PCA).

Everything (weights, config, normalization, transform, PCA basis, grid) is saved
to one ``.npz`` so an emulator is a single portable file.
"""
from __future__ import annotations

import json

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from .theory import LINEAR_OBS, parse_obs  # single source of truth

_T_CMB = 2.7255  # K; for cobaya unit conversions


def _out_key(token: str) -> str:
    """Map an obs token to its predict()-dict key (cobaya lowercase).

    Lensed CMB -> 'tt'/'ee'/'te'/'bb'; unlensed -> same + '_unlensed'; PP -> 'pp'.
    """
    p = parse_obs(token)
    if p["kind"] == "pp":
        return "pp"
    base = p["base"].lower()
    return base if p["lensed"] else base + "_unlensed"
_CMB_UNIT_FACTORS = {
    "1": 1.0, "muK2": _T_CMB * 1e6, "K2": _T_CMB,
    "FIRASmuK2": 2.7255e6, "FIRASK2": 2.7255,
}


def target_transform(obs: str) -> str:
    return "linear" if obs in LINEAR_OBS else "log10"


def fwd_transform(y, kind):
    return np.log10(y) if kind == "log10" else np.asarray(y, float)


def inv_transform(t, kind):
    return 10.0 ** t if kind == "log10" else t


def accuracy_report(obs, y_true, y_pred, ell=None):
    """Held-out precision of an emulator: per-(sample, bin) error percentiles.

    Positive spectra use fractional error |pred-true|/|true|; sign-changing TE
    uses |pred-true| / per-bin RMS. Returns a JSON-friendly dict (median, 68/95/
    99/max percentiles, fraction of bins within 0.1%/1%, and per-ell-band medians).
    """
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    if obs in LINEAR_OBS:
        err = np.abs(yp - yt) / (np.sqrt(np.mean(yt ** 2, axis=0)) + 1e-300)
        metric = "err/RMS"
    else:
        err = np.abs(yp - yt) / (np.abs(yt) + 1e-300)
        metric = "fractional"
    p = np.percentile(err, [50, 68, 95, 99, 100])
    rep = dict(metric=metric, n_test=int(yt.shape[0]),
               median=float(p[0]), p68=float(p[1]), p95=float(p[2]),
               p99=float(p[3]), max=float(p[4]),
               within_0p1pct=float(100.0 * np.mean(err < 1e-3)),
               within_1pct=float(100.0 * np.mean(err < 1e-2)))
    if ell is not None and err.ndim == 2:
        ell = np.asarray(ell)
        lmax = int(ell[-1])
        edges = [e for e in (2, 30, 300, 2000) if e < lmax] + [lmax + 1]
        _k = lambda v: f"{v // 1000}k" if v >= 1000 else str(v)
        rep["bands"] = [[f"{_k(lo)}-{_k(hi - 1)}",
                         float(np.median(err[:, (ell >= lo) & (ell < hi)]))]
                        for lo, hi in zip(edges[:-1], edges[1:])]
    return rep


# --------------------------------------------------------------------------- #
# Backbones
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    out_dim: int
    width: int = 256
    depth: int = 4

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.gelu(nn.Dense(self.width)(x))
        return nn.Dense(self.out_dim)(x)


class ResNetMLP(nn.Module):
    out_dim: int
    width: int = 256
    depth: int = 4

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.width)(x)
        for _ in range(self.depth):
            h = nn.LayerNorm()(x)
            h = nn.gelu(nn.Dense(self.width)(h))
            h = nn.Dense(self.width)(h)
            x = x + h
        x = nn.LayerNorm()(x)
        return nn.Dense(self.out_dim)(x)


class CNNDecoder(nn.Module):
    """Dense head -> (out_dim,1) -> 1-D conv residual refinement along ell."""
    out_dim: int
    width: int = 256
    depth: int = 4
    channels: int = 32
    conv_blocks: int = 3
    ksize: int = 5

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.gelu(nn.Dense(self.width)(x))
        coarse = nn.Dense(self.out_dim)(x)            # (B, D)
        h = coarse[..., None]                          # (B, D, 1)
        h = nn.Conv(self.channels, (self.ksize,), padding="SAME")(h)
        for _ in range(self.conv_blocks):
            r = nn.gelu(nn.Conv(self.channels, (self.ksize,), padding="SAME")(h))
            r = nn.Conv(self.channels, (self.ksize,), padding="SAME")(r)
            h = nn.gelu(h + r)
        h = nn.Conv(1, (self.ksize,), padding="SAME")(h)   # (B, D, 1)
        return coarse + h[..., 0]                       # residual on coarse pred


def build_model(config: dict):
    arch = config["arch"]
    common = dict(out_dim=int(config["out_dim"]),
                  width=int(config["width"]), depth=int(config["depth"]))
    if arch == "mlp":
        return MLP(**common)
    if arch == "resnet":
        return ResNetMLP(**common)
    if arch == "cnn":
        return CNNDecoder(channels=int(config.get("channels", 32)),
                          conv_blocks=int(config.get("conv_blocks", 3)),
                          ksize=int(config.get("ksize", 5)), **common)
    raise ValueError(f"unknown arch {arch!r}; choose mlp|resnet|cnn")


# --------------------------------------------------------------------------- #
# Fit
# --------------------------------------------------------------------------- #
def fit(
    X, Y, transform="log10", *, arch="mlp", width=256, depth=4, pca=0,
    epochs=500, lr=1e-3, batch=256, val_frac=0.1, patience=50, seed=0,
    channels=32, conv_blocks=3, ksize=5, verbose=True,
):
    """Fit an emulator on raw (X, Y). Returns (params, meta, best_val)."""
    X = np.asarray(X, np.float32)
    Y = np.asarray(Y, float)
    N, P = X.shape
    D = Y.shape[1]

    # ---- input normalization ----
    x_mean, x_std = X.mean(0), X.std(0)
    x_std[x_std == 0] = 1.0
    Xn = ((X - x_mean) / x_std).astype(np.float32)

    # ---- target transform + (optional) PCA + standardization ----
    T = fwd_transform(Y, transform).astype(np.float64)
    pca_basis = pca_mean = None
    if pca and pca < D:
        pca_mean = T.mean(0)                                   # (D,)
        Tc = T - pca_mean
        cov = (Tc.T @ Tc) / N                                  # (D, D)
        w, V = np.linalg.eigh(cov)
        pca_basis = V[:, ::-1][:, :pca]                        # (D, k) top-k
        coeffs = Tc @ pca_basis                                # (N, k)
        t_mean, t_std = coeffs.mean(0), coeffs.std(0)
        t_std[t_std == 0] = 1.0
        target = (coeffs - t_mean) / t_std
        out_dim = pca
    else:
        t_mean, t_std = T.mean(0), T.std(0)
        t_std[t_std == 0] = 1.0
        target = (T - t_mean) / t_std
        out_dim = D
    target = target.astype(np.float32)

    # ---- split ----
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    nval = max(1, int(val_frac * N))
    vi, ti = perm[:nval], perm[nval:]
    Xtr, Ytr = jnp.asarray(Xn[ti]), jnp.asarray(target[ti])
    Xv, Yv = jnp.asarray(Xn[vi]), jnp.asarray(target[vi])

    # ---- model + optimizer (cosine schedule) ----
    config = dict(arch=arch, out_dim=int(out_dim), width=int(width),
                  depth=int(depth), n_in=int(P),
                  channels=int(channels), conv_blocks=int(conv_blocks),
                  ksize=int(ksize))
    model = build_model(config)
    params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, P), jnp.float32))

    ntr = Xtr.shape[0]
    steps_per_epoch = max(1, ntr // batch)
    sched = optax.cosine_decay_schedule(lr, decay_steps=epochs * steps_per_epoch)
    opt = optax.adam(sched)
    opt_state = opt.init(params)

    @jax.jit
    def loss_fn(params, xb, yb):
        return jnp.mean((model.apply(params, xb) - yb) ** 2)

    @jax.jit
    def step(params, opt_state, xb, yb):
        loss, grad = jax.value_and_grad(loss_fn)(params, xb, yb)
        upd, opt_state = opt.update(grad, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, loss

    best = serialization.to_state_dict(params)
    best_val, since = np.inf, 0
    for ep in range(epochs):
        idx = rng.permutation(ntr)
        for s in range(0, ntr - batch + 1, batch):
            bi = idx[s : s + batch]
            params, opt_state, _ = step(params, opt_state, Xtr[bi], Ytr[bi])
        vloss = float(loss_fn(params, Xv, Yv))
        if vloss < best_val - 1e-9:
            best_val, since = vloss, 0
            best = serialization.to_state_dict(params)
        else:
            since += 1
        if verbose and (ep % 25 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:4d}  val_mse={vloss:.4e}  best={best_val:.4e}")
        if since >= patience:
            if verbose:
                print(f"  early stop at epoch {ep} (no val improvement in {patience})")
            break

    params = serialization.from_state_dict(model.init(
        jax.random.PRNGKey(seed), jnp.zeros((1, P), jnp.float32)), best)

    meta = dict(
        config=config, transform=transform,
        x_mean=x_mean, x_std=x_std, t_mean=t_mean, t_std=t_std,
        pca_basis=(pca_basis if pca_basis is not None else np.zeros((0, 0))),
        pca_mean=(pca_mean if pca_mean is not None else np.zeros((0,))),
    )
    return params, meta, best_val


# --------------------------------------------------------------------------- #
# Predict
# --------------------------------------------------------------------------- #
def predict(params, meta, X):
    X = np.atleast_2d(np.asarray(X, np.float32))
    xn = ((X - meta["x_mean"]) / meta["x_std"]).astype(np.float32)
    config = meta["config"]
    if isinstance(config, np.ndarray):
        config = json.loads(str(config))
    model = build_model(config)
    out = np.asarray(model.apply(params, jnp.asarray(xn)))      # (N, out_dim)

    pca_basis = np.asarray(meta["pca_basis"])
    if pca_basis.size > 0:
        coeffs = out * meta["t_std"] + meta["t_mean"]
        T = coeffs @ pca_basis.T + meta["pca_mean"]
    else:
        T = out * meta["t_std"] + meta["t_mean"]
    return inv_transform(T, str(meta["transform"]))


# --------------------------------------------------------------------------- #
# Persistence: one .npz file
# --------------------------------------------------------------------------- #
def save(path, params, meta, extra=None):
    d = {}
    d["flax_bytes"] = np.frombuffer(serialization.to_bytes(params), dtype=np.uint8)
    d["config_json"] = json.dumps(meta["config"])
    for k in ("transform", "x_mean", "x_std", "t_mean", "t_std",
              "pca_basis", "pca_mean"):
        d[k] = meta[k]
    if extra:
        d.update(extra)
    np.savez(path, **d)


def load(path):
    z = np.load(path, allow_pickle=True)
    config = json.loads(str(z["config_json"]))
    model = build_model(config)
    template = model.init(jax.random.PRNGKey(0),
                          jnp.zeros((1, config["n_in"]), jnp.float32))
    params = serialization.from_bytes(template, z["flax_bytes"].tobytes())
    meta = dict(
        config=config, transform=str(z["transform"]),
        x_mean=z["x_mean"], x_std=z["x_std"], t_mean=z["t_mean"], t_std=z["t_std"],
        pca_basis=z["pca_basis"], pca_mean=z["pca_mean"],
    )
    skip = {"flax_bytes", "config_json", "transform", "x_mean", "x_std",
            "t_mean", "t_std", "pca_basis", "pca_mean"}
    extra = {k: z[k] for k in z.files if k not in skip}
    return params, meta, extra


# --------------------------------------------------------------------------- #
# High-level user API:  e = cambemul.loademul(path);  e.predict({...})
# --------------------------------------------------------------------------- #
def cobaya_cl(sp, ell_factor=False, units="FIRASmuK2"):
    """Apply cobaya's get_Cl ell_factor/units to a raw spectrum dict ``sp``.

    ``sp`` holds raw C_ell with cobaya keys ('ell','tt','ee','te','bb','pp'),
    tt/ee/te/bb in muK^2 (== FIRASmuK2 at T_cmb=2.7255), pp unitless.
    """
    if units not in _CMB_UNIT_FACTORS:
        raise ValueError(f"units {units!r}; choose {list(_CMB_UNIT_FACTORS)}")
    ell = sp["ell"]
    scale = (_CMB_UNIT_FACTORS[units] / 2.7255e6) ** 2
    lf = ell * (ell + 1) / (2.0 * np.pi)
    res = {"ell": ell}
    for key in ("tt", "ee", "te", "bb"):
        if key in sp:
            c = sp[key] * scale
            res[key] = c * lf if ell_factor else c
    if "pp" in sp:
        c = sp["pp"]
        res["pp"] = c * (lf ** 2) * (2.0 * np.pi) if ell_factor else c
    return res


class Emulator:
    """A bundle of per-observable emulators sharing one parameter space.

    >>> import cambemul
    >>> e = cambemul.loademul("emulators/")          # dir of emu_*.npz
    >>> e.param_names
    ['theta_MC_100', 'logA', 'ns', 'ombh2', 'omch2', 'tau']
    >>> out = e.predict({'theta_MC_100': 1.0411, 'logA': 3.05, 'ns': 0.965,
    ...                  'ombh2': 0.0224, 'omch2': 0.12, 'tau': 0.054})
    >>> out['TT'].shape, e.ell.shape                  # C_ell per multipole
    """

    def __init__(self, members):
        # members: {obs: (params, meta, extra)}
        self.members = dict(members)
        self.obs = list(self.members)

        names = None
        for o, (_, _, ex) in self.members.items():
            nm = [str(x) for x in ex["param_names"]]
            if names is None:
                names = nm
            elif nm != names:
                raise ValueError(
                    f"emulator '{o}' has param_names {nm} != {names}")
        self.param_names = names

        # grids + derived scalar names
        self.ell = self.kh = self.z = None
        self.derived_names = []
        for o, (_, _, ex) in self.members.items():
            if o == "Pk":
                self.kh = ex.get("kh")
                self.z = ex.get("z")
            elif o == "derived":
                self.derived_names = [str(x) for x in ex["derived_names"]]
            elif "ell" in ex:
                self.ell = ex["ell"]

        # held-out precision recorded at train time (optional)
        self.accuracy = {}
        for o, (_, _, ex) in self.members.items():
            aj = ex.get("accuracy_json")
            if aj is not None:
                try:
                    self.accuracy[o] = json.loads(str(aj))
                except Exception:
                    pass

    def print_precision(self):
        """Print the held-out precision stored with each emulator (if any)."""
        if not self.accuracy:
            print(f"cambemul.Emulator(obs={self.obs}) "
                  "[no stored precision: re-train to record it]")
            return
        grid = ""
        if self.ell is not None:
            grid = f"  nell={len(self.ell)} (ell {int(self.ell[0])}..{int(self.ell[-1])})"
        print(f"cambemul.Emulator loaded: obs={self.obs}")
        print(f"  params={self.param_names}{grid}")

        def _pct(x):
            return f"{x * 100:.2f}%"

        rows = []
        for o in self.obs:
            a = self.accuracy.get(o)
            if not a:
                continue
            if a.get("metric") == "derived":
                for nm, med, p95, mx in a["params"]:
                    rows.append([nm, "fractional", _pct(med), _pct(p95),
                                 _pct(mx), "-", a.get("n_test", "-")])
            else:
                rows.append([o, a["metric"], _pct(a["median"]), _pct(a["p95"]),
                             _pct(a["max"]), f"{a['within_1pct']:.0f}%",
                             a["n_test"]])
        headers = ["observable", "metric", "median", "95%", "max", "<1%", "n_test"]
        print("held-out precision:")
        try:
            from tabulate import tabulate
            print(tabulate(rows, headers=headers, tablefmt="github",
                           colalign=("left", "left", "right", "right",
                                     "right", "right", "right")))
        except ImportError:  # graceful fallback if tabulate isn't installed
            print("  " + " | ".join(headers))
            for r in rows:
                print("  " + " | ".join(str(c) for c in r))

    def __repr__(self):
        return (f"Emulator(obs={self.obs}, "
                f"params={self.param_names}, "
                f"nell={None if self.ell is None else len(self.ell)})")

    def _design(self, pars):
        """Build (X[N,P], scalar) from a dict {name: scalar or 1d-array}."""
        missing = [n for n in self.param_names if n not in pars]
        if missing:
            raise KeyError(f"missing parameter(s) {missing}; "
                           f"need {self.param_names}")
        cols, n, scalar = [], 1, True
        for nm in self.param_names:
            v = np.asarray(pars[nm], float)
            if v.ndim > 0:
                scalar = False
                n = max(n, v.shape[0])
            cols.append(v)
        X = np.empty((n, len(self.param_names)), float)
        for j, v in enumerate(cols):
            X[:, j] = v   # broadcasts scalar or length-n array
        return X, scalar

    def predict(self, pars):
        """Predict all observables for ``pars`` (a dict of cosmological params).

        Uses cobaya naming: returns ``{'ell', 'tt', 'ee', 'te', 'pp', ...}`` with
        raw C_ell (muK^2 for tt/ee/te/bb; unitless for pp), plus ``'Pk'`` (with
        ``'k'``, ``'z'``) if a matter-power emulator is present. ``pars`` values
        may be scalars (-> 1-D spectra) or equal-length arrays (-> leading batch
        axis). For cobaya's exact get_Cl semantics (ell_factor / units) use
        :meth:`get_Cl`.
        """
        X, scalar = self._design(pars)
        out = {}
        if self.ell is not None:
            out["ell"] = self.ell
        for o, (params, meta, ex) in self.members.items():
            Y = predict(params, meta, X)                 # (N, D)
            if o == "Pk":
                if "Pk_shape" in ex:
                    nz, nk = (int(s) for s in ex["Pk_shape"])
                    Y = Y.reshape(Y.shape[0], nz, nk)
                out["Pk"] = Y[0] if scalar else Y
                if self.kh is not None:
                    out["k"] = self.kh
                if self.z is not None:
                    out["z"] = self.z
            elif o == "derived":
                for j, nm in enumerate(self.derived_names):
                    out[nm] = float(Y[0, j]) if scalar else Y[:, j]
            else:
                out[_out_key(o)] = Y[0] if scalar else Y
        return out

    def get_derived(self, pars):
        """Return the emulated derived scalars, e.g. {'sigma8': ..., 'H0': ...}."""
        if not self.derived_names:
            return {}
        sp = self.predict(pars)
        return {nm: sp[nm] for nm in self.derived_names}

    def get_Cl(self, pars, ell_factor=False, units="FIRASmuK2"):
        """cobaya-style CMB spectra: mirrors ``provider.get_Cl``.

        Returns ``{'ell', 'tt', 'ee', 'te', 'bb', 'pp'}`` (those present).
        ``ell_factor=True`` multiplies tt/ee/te/bb by l(l+1)/2pi and pp by
        [l(l+1)]^2/2pi. ``units`` in {'1','muK2','K2','FIRASmuK2','FIRASK2'};
        pp is always unitless. (Stored C_ell are raw muK^2 == FIRASmuK2 at
        T_cmb=2.7255.)
        """
        return cobaya_cl(self.predict(pars), ell_factor=ell_factor, units=units)

    def get_unlensed_Cl(self, pars, ell_factor=False, units="FIRASmuK2"):
        """cobaya-style UNLENSED CMB spectra (requires uTT/uEE/uTE emulators)."""
        sp = self.predict(pars)
        tmp = {"ell": sp["ell"]}
        for k in ("tt", "ee", "te", "bb"):
            if k + "_unlensed" in sp:
                tmp[k] = sp[k + "_unlensed"]
        return cobaya_cl(tmp, ell_factor=ell_factor, units=units)

    def get_Pk_grid(self, pars):
        """(k, z, Pk) for the matter power emulator, cobaya-style."""
        if "Pk" not in self.members:
            raise KeyError("this emulator has no 'Pk' member")
        return self.kh, self.z, self.predict(pars)["Pk"]


def loademul(path, verbose=True):
    """Load an :class:`Emulator`.

    ``path`` may be a directory of ``emu_*.npz`` files (as written by
    ``train_emulator.py``) or a single ``emu_<OBS>.npz`` file. When ``verbose``
    (default), the emulator's stored held-out precision is printed on load.
    """
    import glob
    import os

    members = {}
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "emu_*.npz")))
        if not files:
            raise FileNotFoundError(f"no emu_*.npz files in {path}")
    else:
        files = [path]
    for f in files:
        params, meta, extra = load(f)
        members[str(extra["obs"])] = (params, meta, extra)
    emu = Emulator(members)
    if verbose:
        emu.print_precision()
    return emu
