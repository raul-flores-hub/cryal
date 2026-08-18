# Copyright 2026 Raúl Rodolfo Flores Mena
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ase_calculator.py — in-process relaxation with any ASE calculator.

Selected with `energyBackend = ase`. The calculator is named by dotted path in
the input file, so no calculator is a dependency of CrYAL:

    % BACKEND_ASE
    aseCalculator       = mace.calculators.mace_off
    aseCalculatorKwargs = model=medium device=cuda
    aseOptimizer        = FIRE
    aseFmax             = 0.05
    aseMaxSteps         = 500
    aseRelaxCell        = true

The dotted path may name a class (`ase.calculators.emt.EMT`) or a factory
function (`mace.calculators.mace_off`); it is called once, with the keyword
arguments given, and the resulting calculator is reused for the whole run —
loading a machine-learned potential per structure would dominate the cost.

Two differences from the LAMMPS backend are worth stating plainly:

  * This is a *local* relaxation (optimizer + cell filter), not the MD-NPT
    annealing protocol of the LAMMPS input script used in the article. It
    explores a smaller basin around each candidate.
  * Energies are only comparable within one backend and one potential. A
    database built with one must not be merged with a database built with
    another; on resume, keep the backend you started with.
"""

import importlib
import os

from ase.io import write

from . import register
from .base import EnergyBackend


#: optimizers accepted by `aseOptimizer`
_OPTIMIZERS = ("FIRE", "BFGS", "LBFGS", "BFGSLineSearch", "GPMin")


def _import_dotted(path: str):
    """
    Import 'package.module.attribute' and return the attribute.

    Raises ValueError with an actionable message on anything that goes wrong,
    because this is user input from INPUT.txt.
    """
    if not path or "." not in path:
        raise ValueError(
            f"aseCalculator must be a dotted path such as "
            f"'ase.calculators.emt.EMT', got '{path}'")
    module_path, _, attr = path.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ValueError(
            f"aseCalculator '{path}': cannot import '{module_path}' — {e}") from None
    try:
        return getattr(module, attr)
    except AttributeError:
        raise ValueError(
            f"aseCalculator '{path}': '{module_path}' has no '{attr}'") from None


@register
class AseCalculatorBackend(EnergyBackend):
    """Local relaxation in this process, with a user-named ASE calculator."""

    name = "ase"
    aliases = ("ase_calculator",)

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger)
        self._calc = None   # built on first use

    @classmethod
    def validate_config(cls, cfg):
        # Import now so a typo or a missing package fails in the first second
        # of the run, not after a hundred structures have been generated.
        # Building the calculator (loading model weights) still waits.
        _import_dotted(cfg.ase_calculator)
        if cfg.ase_optimizer not in _OPTIMIZERS:
            raise ValueError(
                f"aseOptimizer '{cfg.ase_optimizer}' not recognised — "
                f"choose one of: {', '.join(_OPTIMIZERS)}")

    def describe(self) -> str:
        cell = "cell+positions" if self.cfg.ase_relax_cell else "positions only"
        return (f"{self.name} — {self.cfg.ase_calculator}, "
                f"{self.cfg.ase_optimizer} to fmax={self.cfg.ase_fmax} eV/Å ({cell})")

    # -- calculator --------------------------------------------------------

    @property
    def calculator(self):
        """The shared calculator instance, built on first access."""
        if self._calc is None:
            factory = _import_dotted(self.cfg.ase_calculator)
            self._calc = factory(**self.cfg.ase_calculator_kwargs)
            if self.logger:
                self.logger.info(f"Calculator ready: {self.cfg.ase_calculator}")
        return self._calc

    # -- relaxation --------------------------------------------------------

    def relax(self, atoms, step_dir: str, logger=None):
        logger = logger or self.logger
        cfg = self.cfg

        from ase import optimize as ase_optimize
        from ase.filters import FrechetCellFilter

        work = atoms.copy()
        # Deliberately outside the try below: a calculator that cannot be built
        # is not a per-structure failure, and should stop the run instead of
        # being swallowed once per candidate.
        work.calc = self.calculator

        target = FrechetCellFilter(work) if cfg.ase_relax_cell else work
        optimizer = getattr(ase_optimize, cfg.ase_optimizer)

        try:
            opt = optimizer(target, logfile=os.path.join(step_dir, "relax.log"),
                            trajectory=None)
            converged = opt.run(fmax=cfg.ase_fmax, steps=cfg.ase_max_steps)
            energy = float(work.get_potential_energy())
        except Exception as e:
            if logger:
                logger.debug(f"  ASE: relaxation failed — {e}")
            return None, None

        if not converged:
            if logger:
                logger.debug(f"  ASE: not converged in {cfg.ase_max_steps} steps "
                             f"(fmax target {cfg.ase_fmax} eV/Å)")
            return None, None

        relaxed = work.copy()   # copy() drops the calculator
        try:
            write(os.path.join(step_dir, "minimized.cif"), relaxed)
            with open(os.path.join(step_dir, "energy.dat"), "w") as f:
                f.write(f"{energy:.6f}\n")
        except Exception as e:
            if logger:
                logger.debug(f"  ASE: could not write step output — {e}")

        return relaxed, energy
