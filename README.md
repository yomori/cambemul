# cambemul

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![codecov](https://codecov.io/gh/yomori/cambemul/branch/main/graph/badge.svg)](https://codecov.io/gh/yomori/cambemul)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Machine-learning emulator for CAMB cosmological power spectra.

`cambemul` provides a fast, differentiable surrogate for the cosmological power
spectra computed by [CAMB](https://camb.info), so you can evaluate spectra across
parameter space without invoking the full Boltzmann solver each time. It is built
on JAX for autodiff and accelerator support.

## Installation

```bash
pip install cambemul
```

Or from source:

```bash
git clone https://github.com/yomori/cambemul.git
cd cambemul
pip install -e ".[dev]"
```

## Usage

```python
import cambemul

print(cambemul.__version__)
```

## Development

```bash
pip install -e ".[dev]"
pre-commit install
pytest
```

## Citation

If you use this work, please cite:

```bibtex
@article{cambemul,
  title={Machine-learning emulator for CAMB cosmological power spectra},
  author={yomori},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

MIT. See [LICENSE](LICENSE) for details.
