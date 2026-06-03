# cambemul

Machine-learning emulator for CAMB cosmological power spectra.

## Overview

`cambemul` provides a fast, differentiable surrogate model that emulates the
cosmological power spectra produced by [CAMB](https://camb.info). The goal is to
evaluate spectra (e.g. CMB TT/TE/EE and matter power spectra) across cosmological
parameter space at a fraction of the cost of running the full Boltzmann solver,
while remaining accurate enough for inference pipelines.

## Stack

- numpy, scipy, pyyaml
- camb (training-data generation)
- jax + flax + optax (differentiable NN emulator)
- h5py / tqdm (large datasets, progress)

## Pipeline

1. `scripts/make_training.py` — read a cobaya YAML, Latin-hypercube sample the
   prior, run CAMB for each point, save a training set (single `.npz`, or sharded
   `.npz` for SLURM array generation).
2. `scripts/concat_shards.py` — concatenate shards into one chunked `.h5`.
3. `scripts/train_emulator.py` — train one emulator per observable; backbones
   `mlp` / `resnet` / `cnn`, optional PCA front-end.
4. `scripts/test_emulator.py` — fractional + sign-safe accuracy on a held-out set.

Target transform is `log10` for positive spectra (TT/EE/PP/Pk) and `linear` for
sign-changing TE. Inputs and (optionally PCA-compressed) targets are standardized.

## Requirements

- Python: >=3.10

## License

- MIT

## Authors

- yomori (eos.xfc@gmail.com)

## arXiv

- arXiv: XXXX.XXXXX (placeholder — update once available)

## Status

Working pipeline. Library under `src/cambemul/` (priors, theory, emulator,
dataset); CLI under `scripts/`. Verified end-to-end (sample -> CAMB -> train ->
test) for all observables and all three backbones in the `analysis4` env.
