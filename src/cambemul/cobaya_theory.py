"""A cobaya ``Theory`` that serves a trained cambemul emulator.

Drop it into a cobaya yaml in place of CAMB/CLASS::

    theory:
      cambemul.cobaya_theory.CambEmul:
        emulator_dir: /path/to/emulators      # dir of emu_*.npz (or a single file)

    likelihood:
      my_cmb_likelihood: ...                   # any likelihood that needs Cl

The emulator's parameter names (read from the original yaml, so already
cobaya-native, e.g. ``theta_MC_100, logA, ns, ombh2, omch2, tau``) become the
theory's input parameters, and it provides ``Cl`` via :meth:`get_Cl` with the
standard cobaya ``ell_factor`` / ``units`` semantics.

This module imports cobaya; the rest of cambemul does not, so importing
``cambemul`` stays lightweight.
"""
from __future__ import annotations

import numpy as np
from cobaya.theory import Theory

from .emulator import cobaya_cl, loademul


class CambEmul(Theory):
    # set in the yaml under the theory block
    emulator_dir: str = ""

    def initialize(self):
        if not self.emulator_dir:
            raise ValueError(
                "CambEmul needs 'emulator_dir' (a dir of emu_*.npz or a single file)")
        self.emu = loademul(self.emulator_dir)
        self._param_names = list(self.emu.param_names)
        self._derived_names = list(self.emu.derived_names)
        self._has_cl = any(k in self.emu.obs for k in ("TT", "EE", "TE", "BB", "PP"))

    # cosmological input parameters this theory consumes
    def get_requirements(self):
        return self._param_names

    # derived parameters this theory can provide (e.g. sigma8, H0)
    def get_can_provide_params(self):
        return self._derived_names

    # 'Cl' provision is auto-detected from the presence of get_Cl(); nothing else
    # is required from other components.
    def must_provide(self, **requirements):
        return None

    def calculate(self, state, want_derived=True, **params):
        pars = {k: params[k] for k in self._param_names}
        sp = self.emu.predict(pars)                       # cobaya-key raw spectra
        state["cambemul_raw"] = sp
        if want_derived and self._derived_names:
            state["derived"] = {nm: float(np.asarray(sp[nm]))
                                for nm in self._derived_names}
        return True

    def get_Cl(self, ell_factor=False, units="FIRASmuK2"):
        return cobaya_cl(self.current_state["cambemul_raw"],
                         ell_factor=ell_factor, units=units)

    def get_unlensed_Cl(self, ell_factor=False, units="FIRASmuK2"):
        sp = self.current_state["cambemul_raw"]
        tmp = {"ell": sp["ell"]}
        for k in ("tt", "ee", "te", "bb"):
            if k + "_unlensed" in sp:
                tmp[k] = sp[k + "_unlensed"]
        return cobaya_cl(tmp, ell_factor=ell_factor, units=units)
