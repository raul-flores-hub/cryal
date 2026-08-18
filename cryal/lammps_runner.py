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
lammps_runner.py — compatibility shim.

The LAMMPS/MACE-OFF relaxation now lives in `cryal.backends.lammps_mace`, one
of several selectable energy backends (see `cryal/backends/base.py` for the
contract). This module keeps the original module-level API working, because it
is the interface documented in the released README and the one external scripts
were told to call:

    from cryal.lammps_runner import evaluate_structure, get_reference_bonds

evaluate_structure() below is exactly the LAMMPS backend, gates included, so
code written against v1.0.0 behaves as it did. New code should go through
`cryal.backends.get_backend(cfg)` instead, which honours `energyBackend`.
"""

from .backends.base import get_reference_bonds, check_molecular_integrity
from .backends.lammps_mace import (LammpsMaceBackend, _lammps_env,
                                   _write_lammps_data, _z_of_type)
from .config import Config


def evaluate_structure(atoms, cfg: Config, step_dir: str,
                       ref_bonds, mol_size: int,
                       logger=None):
    """
    Run a LAMMPS relaxation on a single structure.

    Kept for backwards compatibility; equivalent to

        LammpsMaceBackend(cfg).evaluate(atoms, step_dir, ref_bonds, mol_size)

    and unaffected by `energyBackend` — it is the LAMMPS path by definition.

    Returns (relaxed_atoms, energy_eV) or (None, None) on failure.
    """
    return LammpsMaceBackend(cfg, logger).evaluate(
        atoms, step_dir, ref_bonds, mol_size, logger=logger)


__all__ = ["evaluate_structure", "get_reference_bonds",
           "check_molecular_integrity", "LammpsMaceBackend",
           "_lammps_env", "_write_lammps_data", "_z_of_type"]
