# cambemul

Machine-learning emulator for CAMB cosmological power spectra.

## Overview

`cambemul` provides a fast, differentiable surrogate model that emulates the
cosmological power spectra produced by [CAMB](https://camb.info). The goal is to
evaluate spectra (e.g. CMB TT/TE/EE and matter power spectra) across cosmological
parameter space at a fraction of the cost of running the full Boltzmann solver,
while remaining accurate enough for inference pipelines.

## Stack

- numpy
- scipy
- jax (autodiff + GPU/TPU acceleration)

## Requirements

- Python: >=3.10

## License

- MIT

## Authors

- yomori (eos.xfc@gmail.com)

## arXiv

- arXiv: XXXX.XXXXX (placeholder — update once available)

## Status

Scaffold only. Core emulator implementation lives under `src/cambemul/`.
