"""Machine-learning emulator for CAMB cosmological power spectra.

Pipeline (one script each, under ``scripts/``):
  1. make_training.py  -- yaml + --obs -> sample prior, run CAMB, save training set
  2. train_emulator.py -- training set -> trained JAX emulator(s)
  3. test_emulator.py  -- emulator + held-out set -> accuracy report
"""

__version__ = "0.0.1"

from . import dataset, emulator, priors, theory
from .emulator import (
    Emulator,
    fit,
    load,
    loademul,
    predict,
    save,
    target_transform,
)
from .priors import (
    build_camb_inputs,
    build_camb_inputs_h0,
    invert_linear_lambda,
    parse_cosmo_yaml,
    sample_cosmo,
)
from .theory import run_camb

__all__ = [
    "priors",
    "theory",
    "emulator",
    "dataset",
    "parse_cosmo_yaml",
    "sample_cosmo",
    "build_camb_inputs",
    "build_camb_inputs_h0",
    "invert_linear_lambda",
    "run_camb",
    "fit",
    "predict",
    "save",
    "load",
    "loademul",
    "Emulator",
    "target_transform",
]
