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
uma.py — Meta's UMA as an energy backend.

UMA is reachable through the `ase` backend already: it is an ASE calculator
like any other. It gets a backend of its own for three reasons that the dotted
path cannot cover.

First, it cannot actually be named by a dotted path. UMA is built in two steps
-- fetch a predict unit, then wrap it for a task -- and the one-call form,
`FAIRChemCalculator.from_model_checkpoint`, is a classmethod. `aseCalculator`
resolves `module.attribute` through importlib, which tries to import
`fairchem.core.FAIRChemCalculator` as a module and fails. Every user would
otherwise write the same four-line factory.

Second, that factory would have to exist on every machine in a distributed run,
under the same importable name. A module in this package already does.

Third, and the reason this is not merely convenience: UMA carries a **task**,
and choosing it wrong is silent. Each head was trained on a different domain
and each has its own zero of energy -- for one periodic C50Au4 cell, `omat`
gives -466.55 eV and `oc20` -447.06 eV. Numbers from two heads are not
comparable, and nothing downstream would reveal the mix. `umaTask` is therefore
mandatory, with no default, and validated here against the installed fairchem.

Installation is *not* the ORCA-external route. See INSTALL.md: CrYAL drives UMA
in this process through ASE, and needs neither the Flask server nor the ORCA
external-tools client.

Configuration (INPUT.txt):
    energyBackend = uma

    % BACKEND_UMA
    umaModel  = uma-s-1p1
    umaTask   = omc          # mandatory: omc, omat, omol, oc20, odac...
    umaDevice = cuda
"""

import os

from . import register
from .ase_calculator import AseCalculatorBackend

#: Fallback when fairchem is not importable and we still want a useful message.
_KNOWN_TASKS = ("omc", "omat", "omol", "oc20", "odac", "oc25")


def _installed_tasks():
    """Task names this installation of fairchem knows, or the fallback list.

    Read from fairchem rather than hardcoded because the set grows: 2.13.0 and
    2.21.0 do not offer the same heads, and a hardcoded list would reject a
    valid task on a newer install.
    """
    try:
        from fairchem.core.units.mlip_unit.api.inference import UMATask
        return tuple(t.value for t in UMATask)
    except Exception:
        return _KNOWN_TASKS


def _available_models():
    try:
        from fairchem.core import pretrained_mlip
        return tuple(pretrained_mlip.available_models)
    except Exception:
        return ()


@register
class UmaBackend(AseCalculatorBackend):
    """In-process relaxation with UMA, through ASE."""

    name = "uma"
    aliases = ("fairchem",)

    #: One predict unit is shared by every relaxation, and it holds the state
    #: of the call in progress -- same reason the ase backend cannot run twice
    #: at once in one process. Distributing to other machines is unaffected.
    thread_safe = False

    # -- validation --------------------------------------------------------

    @classmethod
    def validate_config(cls, cfg):
        """
        Fail at load time, and on every remote worker at preflight, rather than
        at the first candidate. cryal.parallel runs this over ssh on each
        machine, which is what stops a worker with no fairchem from silently
        rejecting every structure it is given.
        """
        try:
            import fairchem.core  # noqa: F401
        except ImportError as e:
            raise ValueError(
                "energyBackend = uma, but fairchem-core is not importable here "
                f"({e}). Install it into the interpreter that runs CrYAL: "
                "`pip install fairchem-core`. The ORCA external-tools "
                "installation is a different thing and is not required — "
                "see INSTALL.md.") from None

        task = (getattr(cfg, "uma_task", "") or "").strip()
        tasks = _installed_tasks()
        if not task:
            raise ValueError(
                "umaTask is mandatory and has no default: each UMA head was "
                "trained on a different domain and has its own zero of energy, "
                f"so guessing one silently biases every number. Choose from: "
                f"{', '.join(tasks)}. For organic molecular crystals that is "
                "'omc'.")
        if task not in tasks:
            raise ValueError(
                f"umaTask '{task}' is not offered by the installed fairchem — "
                f"available: {', '.join(tasks)}")

        model = (getattr(cfg, "uma_model", "") or "").strip()
        models = _available_models()
        if models and model not in models:
            raise ValueError(
                f"umaModel '{model}' is not offered by the installed fairchem — "
                f"available: {', '.join(models)}")

        device = (getattr(cfg, "uma_device", "cuda") or "cuda").strip()
        if device not in ("cuda", "cpu"):
            raise ValueError(f"umaDevice must be 'cuda' or 'cpu', got '{device}'")

        if getattr(cfg, "ase_optimizer", "FIRE") not in cls._optimizers():
            raise ValueError(
                f"aseOptimizer '{cfg.ase_optimizer}' not recognised — "
                f"choose one of: {', '.join(cls._optimizers())}")

    @staticmethod
    def _optimizers():
        from .ase_calculator import _OPTIMIZERS
        return _OPTIMIZERS

    # -- description -------------------------------------------------------

    def describe(self) -> str:
        cell = "cell+positions" if self.cfg.ase_relax_cell else "positions only"
        return (f"{self.name} — {self.cfg.uma_model} task={self.cfg.uma_task} "
                f"on {self.cfg.uma_device}, {self.cfg.ase_optimizer} to "
                f"fmax={self.cfg.ase_fmax} eV/Å ({cell})")

    # -- calculator --------------------------------------------------------

    @property
    def calculator(self):
        """The shared UMA calculator, built on first access.

        Deliberately the one-call `from_model_checkpoint`: assembling the
        predict unit by hand invites the InferenceSettings trap, where a
        hand-built settings object leaves `external_graph_gen=None` and the
        first evaluation dies with 'No edges found in input system'.
        """
        if self._calc is None:
            from fairchem.core import FAIRChemCalculator
            cfg = self.cfg
            self._calc = FAIRChemCalculator.from_model_checkpoint(
                cfg.uma_model,
                task_name=cfg.uma_task,
                device=cfg.uma_device,
                inference_settings=cfg.uma_inference_settings)
            if self.logger:
                self.logger.info(
                    f"UMA ready: {cfg.uma_model} / {cfg.uma_task} on {cfg.uma_device}")
        return self._calc
