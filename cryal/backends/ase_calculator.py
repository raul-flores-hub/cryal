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
  * `aseMaxSteps` is a per-calculator setting, not a per-project one. A
    structure that has not converged when the budget runs out is discarded,
    which is right — it was not evaluated at a minimum — but it looks exactly
    like a structure that lost on its merits. RelaxationBudget below is what
    makes the difference visible.
"""

import importlib
import logging
import os
import threading

import numpy as np
from ase.io import write

from . import register
from .base import EnergyBackend


#: optimizers accepted by `aseOptimizer`
_OPTIMIZERS = ("FIRE", "BFGS", "LBFGS", "BFGSLineSearch", "GPMin")


def _import_dotted(path: str):
    """
    Import 'package.module.attribute' and return the attribute.

    Raises ValueError with an actionable message on anything that goes wrong,
    because this is user input from cryal.in.
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


class RelaxationBudget:
    """Running tally of how close a cycle's relaxations came to aseMaxSteps.

    Discarding a structure that never converged is the right call: it was not
    evaluated at a minimum, so its energy does not belong in the database. It
    is also invisible — the cycle simply keeps fewer candidates, and the run
    looks unlucky rather than misconfigured.

    That matters because the budget does not carry between engines. From the
    same geometries MACE converged in about 25 steps where UMA needed 39 to
    582, so a value calibrated for one quietly throws structures away with the
    other.

    The number to watch is not the average but the largest: a cycle whose
    slowest converged relaxation already sits near the limit is one structure
    away from losing a real minimum.

    Counts arrive from several threads (the pool's local slots) and from other
    machines (merge), so they are taken under a lock.
    """

    #: report() stays quiet while the worst relaxation is below this fraction
    #: of the budget — a line printed every cycle stops being read.
    NEAR_LIMIT = 0.8

    def __init__(self, limit: int):
        self.limit = int(limit)
        self._lock = threading.Lock()
        self.n_converged = 0
        self.n_exhausted = 0
        self.max_steps_used = 0

    def record(self, steps: int, converged: bool):
        with self._lock:
            if converged:
                self.n_converged += 1
                self.max_steps_used = max(self.max_steps_used, int(steps))
            else:
                self.n_exhausted += 1

    def merge(self, other: dict):
        """Fold in a tally kept elsewhere — what a remote worker sent back."""
        with self._lock:
            self.n_converged += int(other.get("converged", 0))
            self.n_exhausted += int(other.get("exhausted", 0))
            self.max_steps_used = max(self.max_steps_used,
                                      int(other.get("max_steps_used", 0)))

    def as_dict(self) -> dict:
        with self._lock:
            return {"converged":     self.n_converged,
                    "exhausted":     self.n_exhausted,
                    "max_steps_used": self.max_steps_used}

    def reset(self):
        with self._lock:
            self.n_converged = 0
            self.n_exhausted = 0
            self.max_steps_used = 0

    def report(self):
        """One line for the end of a cycle as (log level, message), or None."""
        with self._lock:
            kept, lost, worst = (self.n_converged, self.n_exhausted,
                                 self.max_steps_used)
        if not kept + lost:
            return None
        if lost:
            tail = (f"; the {kept} that converged needed up to {worst} steps"
                    if kept else "")
            return (logging.WARNING,
                    f"  aseMaxSteps: {lost} of {kept + lost} relaxations spent "
                    f"the {self.limit}-step budget without converging and were "
                    f"discarded{tail} — raise aseMaxSteps if this cycle lost "
                    f"structures it should have kept")
        if worst >= self.NEAR_LIMIT * self.limit:
            return (logging.INFO,
                    f"  aseMaxSteps: all {kept} relaxations converged, the "
                    f"slowest in {worst} of {self.limit} steps — the budget is "
                    f"nearly spent")
        return None


def _reached_fmax(target):
    """The largest force left on the optimizer's degrees of freedom, or None.

    Read once the optimizer has stopped. ASE checks convergence right after
    the last step's force evaluation, so nothing has moved since and the
    calculator answers from its cache; this costs no extra work. Guarded all
    the same, because a number for the log must never turn a clean rejection
    into an exception.
    """
    try:
        forces = target.get_forces()
        return float(np.sqrt((forces ** 2).sum(axis=1).max()))
    except Exception:
        return None


@register
class AseCalculatorBackend(EnergyBackend):
    """Local relaxation in this process, with a user-named ASE calculator."""

    name = "ase"
    aliases = ("ase_calculator",)

    #: One calculator instance is shared by every relaxation (that is the
    #: point — reloading a model per candidate would dominate the run), and an
    #: ASE Calculator stores the atoms and results of the call in progress on
    #: itself. Two threads would hand each other the wrong energy. Distributing
    #: to other machines is unaffected: those are separate processes.
    thread_safe = False

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger)
        self._calc = None   # built on first use
        self._budget = RelaxationBudget(cfg.ase_max_steps)

    def relaxation_budget(self):
        return self._budget

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
            return self._reject(logger, f"relaxation failed — {e}")

        steps = opt.get_number_of_steps()
        self._budget.record(steps, converged)

        if not converged:
            # The fmax it reached is the number worth carrying: close to the
            # target means the budget was short, far from it means the
            # candidate was hopeless, and the run log cannot tell them apart
            # from the step count alone.
            reached = _reached_fmax(target)
            left = f"{reached:.3f}" if reached is not None else "unknown"
            return self._reject(
                logger,
                f"not converged: spent the aseMaxSteps budget of "
                f"{cfg.ase_max_steps} with fmax {left} eV/Å still above the "
                f"{cfg.ase_fmax} target")

        relaxed = work.copy()   # copy() drops the calculator
        try:
            write(os.path.join(step_dir, "minimized.cif"), relaxed)
            with open(os.path.join(step_dir, "energy.dat"), "w") as f:
                f.write(f"{energy:.6f}\n")
        except Exception as e:
            if logger:
                logger.debug(f"  {self.name}: could not write step output — {e}")

        return relaxed, energy
