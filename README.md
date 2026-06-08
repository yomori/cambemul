# cambemul

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Machine-learning emulator for CAMB cosmological power spectra.

`cambemul` builds a fast, differentiable surrogate for the spectra computed by
[CAMB](https://camb.info) — CMB `TT/EE/TE` (lensed or unlensed), the lensing
potential `PP`, and the matter power spectrum `Pk` — plus derived scalars like
`sigma8`/`H0`. It reads a cobaya YAML to define the parameter space, runs CAMB to
build a training set, trains a JAX network, and loads back as a drop-in
**cobaya theory**. Built on JAX (flax + optax), so it stays differentiable for
gradient-based inference.

## Installation

```bash
git clone https://github.com/yomori/cambemul.git
cd cambemul
pip install -e ".[dev]"
```

Requires `camb`, `cobaya`, `jax`, `flax`, `optax`, `numpy`, `scipy`, `pyyaml`.

## Precomputed emulators

Pretrained emulators are hosted on Google Drive, so you can skip the CAMB
generation + training steps and load one directly. Downloads use
[`gdown`](https://github.com/wkentaro/gdown) (install it with pip):

```bash
pip install gdown            # or:  pip install "cambemul[download]"

# download an emulator directory (a folder of emu_*.npz files) into emulators/
gdown --folder https://drive.google.com/drive/folders/<FOLDER_ID> -O emulators/<name>
```

Replace `<FOLDER_ID>` with the share id of the emulator you want (see the table
below), then load and use it — each emulator prints its held-out precision on
load:

```python
import cambemul
e = cambemul.loademul("emulators/<name>")        # prints per-observable precision
out = e.predict({"theta_MC_100": 1.0411, "logA": 3.05, "ns": 0.965,
                 "ombh2": 0.0224, "omch2": 0.109})
out["tt"], out["ee"], out["te"], out["pp"]
```

Or run the ready-made example, which loads an emulator and predicts the spectra
at a (Planck-ish) fiducial point — overridable, with an optional `D_ell` plot:

```bash
python scripts/predict_example.py --emu-dir emulators/<name>
python scripts/predict_example.py --emu-dir emulators/<name> \
    --params "logA=3.00,ns=0.97" --plot dell.png
```

| emulator | params | observables | Google Drive id |
|----------|--------|-------------|-----------------|
| _(add rows as you publish emulators)_ | | | `<FOLDER_ID>` |

## Usage

The workflow is three scripts plus a one-line load:

| stage | script | does |
|-------|--------|------|
| generate | `scripts/make_training.py` | read a cobaya YAML → sample its prior → run CAMB → save a training set |
| train | `scripts/train_emulator.py` | training set → one JAX emulator per observable |
| test | `scripts/test_emulator.py` | emulator + held-out set → accuracy report |
| use | `cambemul.loademul(dir)` | load and `.predict(params)` / `.get_Cl(...)` |

### Quick start

```bash
# 1. generate a training set + an independent test set (different --seed)
#    (lcdm.yaml samples H0 directly, so sigma8 is the meaningful derived output)
python scripts/make_training.py --yaml examples/lcdm.yaml --obs TT,EE,TE,PP,Pk \
    --n 2000 --lmax 3000 --kmax 10 --derived sigma8 --out training/lcdm.npz
python scripts/make_training.py --yaml examples/lcdm.yaml --obs TT,EE,TE,PP,Pk \
    --n 300  --lmax 3000 --kmax 10 --derived sigma8 --seed 1 --out training/lcdm_test.npz

# 2. train one emulator per observable (+ a small one for the derived scalars)
python scripts/train_emulator.py --train training/lcdm.npz --obs TT,EE,TE,PP,Pk \
    --out-dir emulators --arch resnet --pca 256 --width 512 --depth 4 --epochs 1000

# 3. accuracy report on the held-out set
python scripts/test_emulator.py --emu-dir emulators --test training/lcdm_test.npz
```

```python
# 4. use it
import cambemul
e = cambemul.loademul("emulators/")
cl = e.predict({"ombh2": 0.0224, "omch2": 0.12, "H0": 67.3,
                "logA": 3.05, "ns": 0.965, "tau": 0.054})
cl["tt"], cl["ee"], cl["te"], cl["pp"], cl["sigma8"]
```

### Observables (`--obs`)

A comma-separated list of:

| token | meaning |
|-------|---------|
| `TT` `EE` `TE` | lensed scalar CMB C_ℓ (μK²) — default |
| `lTT` `lEE` `lTE` | explicitly **lensed** (same as bare) |
| `uTT` `uEE` `uTE` | **unlensed** scalar CMB C_ℓ |
| `PP` | lensing potential C_ℓ^φφ (unitless) |
| `Pk` | matter power P(k, z) |

Add `--derived sigma8,H0` (or any of `sigma8,H0,omegam,rdrag,age,…`) to also emulate
derived scalars. `sigma8` triggers a cheap matter-transfer computation.

### Real cobaya chains: sample only cosmology

Point `--yaml` at a full `chain.updated.yaml` and the trainer samples **only the
parameters CAMB consumes**: it reads `theory.camb.input_params`, follows the
sampled→input `value` lambdas (`cosmomc_theta←theta_MC_100`, `As←logA`, …), and
**excludes likelihood nuisances** (calibrations, foregrounds). The chain's
`theory.camb.extra_args` (nonlinear, `lens_potential_accuracy`, halofit version,
…) are forwarded to CAMB so the training spectra match the chain's theory exactly.

**θ-parameterized chains → use `--box h0`.** Sampling `cosmomc_theta` directly
makes CAMB solve θ→H0 within `theta_H0_range`, which rejects nearly every draw
over a wide prior. `--box h0` instead samples `H0` (no solve, **zero rejections**),
runs CAMB, and stores CAMB's *derived* θ as the emulator input — so the emulator
still lives in the chain's θ coordinates. `--h0-range lo,hi` defaults to the
yaml's `theta_H0_range` (else `40,100`).

```bash
python scripts/make_training.py --yaml chain.updated.yaml --obs TT,EE,TE,PP \
    --box h0 --derived sigma8,H0 --require-valid \
    --n 500000 --nshard 128 --shard $SLURM_ARRAY_TASK_ID --shard-dir training/lens_shards
```

`--require-valid` tops up to the requested number of *valid* points (replacing
CAMB failures / NaNs); every sample is finite-checked and `log10` observables are
positivity-checked. Wide non-θ priors (ns, ωc, …) can still yield occasional
pathological cosmologies — those are filtered and backfilled.

### Large runs (sharded generation)

Generation dominates the cost, so it is sharded over a SLURM array. There is **no
concatenation step** — point `--train` at the shard directory and the trainer
concatenates on the fly, one observable at a time.

```bash
sbatch scripts/make_training.submit                 # array of NSHARD tasks
python scripts/train_emulator.py --train training/lens_shards --obs TT,EE,TE,PP \
    --out-dir emulators --arch resnet --pca 256
```

### Using a trained emulator

`predict` uses cobaya naming; pass scalars for one cosmology or equal-length
arrays for a batch (results gain a leading batch axis).

```python
e = cambemul.loademul("emulators/")
e.param_names            # input params (cobaya-native, from the yaml)
e.derived_names          # e.g. ['sigma8', 'H0']
e.ell, e.kh, e.z         # output grids

out = e.predict(pars)               # {'ell','tt','ee','te','pp', 'Pk','k','z', 'sigma8','H0', ...}
out["tt"]                           # raw C_ℓ in μK² (pp unitless); 'tt_unlensed' if trained
e.get_Cl(pars, ell_factor=True, units="FIRASmuK2")   # mirrors cobaya provider.get_Cl
e.get_unlensed_Cl(pars)             # if uTT/uEE/uTE were trained
e.get_derived(pars)                 # {'sigma8': ..., 'H0': ...}
k, z, Pk = e.get_Pk_grid(pars)
```

### Plugging into cobaya

`cambemul.cobaya_theory.CambEmul` is a cobaya `Theory`: it advertises the
emulator's parameters as inputs, provides `Cl` (and `unlensed_Cl`) via `get_Cl`,
and exposes the derived scalars. Drop it into a yaml in place of CAMB/CLASS:

```yaml
theory:
  cambemul.cobaya_theory.CambEmul:
    emulator_dir: /path/to/emulators
likelihood:
  your_cmb_likelihood: ...
params:
  theta_MC_100: {prior: {min: 0.5, max: 10}}
  logA:  {prior: {min: 1.61, max: 4}}
  ns:    {prior: {min: 0.2, max: 2}}
  ombh2: {prior: {min: 0.0, max: 0.1}}
  omch2: {prior: {min: 0.005, max: 0.99}}
  tau:   {prior: {min: 0.01, max: 0.1}}
  sigma8: {derived: true}      # reported in the chain
  H0:     {derived: true}      # genuine derived output for a θ-sampled chain
```

### Architectures (`--arch`) and PCA

| arch | what it is | when to use |
|------|-----------|-------------|
| `mlp` | dense GELU stack (CosmoPower-style baseline) | default |
| `resnet` | residual dense blocks + LayerNorm | recommended accuracy upgrade |
| `cnn` | dense head + 1-D conv refinement along ℓ | experimental; full-spectrum (no PCA) |

`--pca K` emulates the top-`K` PCA coefficients of the (log-)spectrum instead of
the full vector — the main accuracy + speed lever at scale. Output is always the
full per-ℓ spectrum; PCA only changes the regression target.

## Development

```bash
pip install -e ".[dev]"
pre-commit install
pytest
```
