"""Dataset I/O.

Training points are generated in independent shards (one SLURM array task each),
written as small ``.npz`` files in a shard directory. ``train_emulator.py`` reads
that directory directly — there is no separate concatenation step. A shard
directory behaves like a single store: row-valued arrays (``params`` and each
observable) are concatenated across shards *on access*, so only one observable is
ever held in memory at a time.

``load_store`` accepts:
  * a shard directory  -> :class:`_ShardStore`  (concatenate-on-access)
  * a ``.h5`` / ``.hdf5`` file -> :class:`_H5Store`
  * a ``.npz`` file     -> the numpy ``NpzFile`` handle

All three expose ``store[key]`` / ``key in store`` / ``store.files``.
"""
from __future__ import annotations

import glob
import os

import numpy as np


class _H5Store:
    """Dict-like read view over an HDF5 file (datasets + attrs)."""

    def __init__(self, path):
        import h5py

        self._f = h5py.File(path, "r")

    @property
    def files(self):
        return list(self._f.keys()) + list(self._f.attrs.keys())

    def __contains__(self, k):
        return k in self._f or k in self._f.attrs

    def __getitem__(self, k):
        if k in self._f:
            return self._f[k][...]
        return self._f.attrs[k]


class _ShardStore:
    """Treat a directory of ``shard*.npz`` files as one store.

    ``params`` and the per-observable arrays grow along the row axis and are
    concatenated across shards on access; everything else (names, obs list, ell/k
    grids, lmin/lmax, ...) is identical across shards and read from the first one.
    """

    def __init__(self, shard_dir, pattern="shard*.npz"):
        self._paths = sorted(glob.glob(os.path.join(shard_dir, pattern)))
        if not self._paths:
            raise FileNotFoundError(f"no shards matching {pattern} in {shard_dir}")
        self._first = np.load(self._paths[0], allow_pickle=True)
        obs = [str(o) for o in self._first["obs"]]
        # row-valued arrays (grow along sample axis) -> concatenate across shards
        self._row_keys = {"params", "derived", *obs}

    @property
    def files(self):
        return list(self._first.files)

    def __contains__(self, k):
        return k in self._first.files

    def __getitem__(self, k):
        if k not in self._first.files:
            raise KeyError(k)
        if k in self._row_keys:
            return np.concatenate(
                [np.load(p, allow_pickle=True)[k] for p in self._paths]
            )
        return self._first[k]


def load_store(path):
    """Open a training/test store from a shard dir, .h5, or .npz."""
    if os.path.isdir(path):
        return _ShardStore(path)
    if path.endswith((".h5", ".hdf5")):
        return _H5Store(path)
    return np.load(path, allow_pickle=True)


def read_param_names(store):
    return [str(n) for n in store["param_names"]]


def read_obs(store):
    return [str(o) for o in store["obs"]]
