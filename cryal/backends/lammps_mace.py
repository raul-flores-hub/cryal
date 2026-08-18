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
lammps_mace.py — relaxation by an external LAMMPS process.

This is the backend used for the run reported in the accompanying article:
LAMMPS driving MACE-OFF23 through the mliap-unified pair style, with the
MD-NPT + minimization protocol written in the user's LAMMPS input script.

Each relax() call:
  1. writes the structure as a LAMMPS data file (structure.data)
  2. copies the user's input script into the step directory
  3. runs LAMMPS there
  4. reads back the relaxed structure (minimized.data) and energy (energy.dat)

The pre-relaxation gate, the energy sanity bound and the molecular-integrity
check are not here: they belong to every backend and live in base.evaluate().
"""

import os
import sysconfig
import subprocess
import shutil
import warnings

from ase.io import read, write

from . import register
from .base import EnergyBackend

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_lammps_data(atoms, filepath: str, spec_order):
    """
    Write ASE Atoms to a LAMMPS data file (atomic style, triclinic cell).
    spec_order maps element symbols to LAMMPS atom types.
    """
    write(filepath, atoms, format='lammps-data', specorder=spec_order)


def _z_of_type(spec_order):
    """Map LAMMPS atom types to atomic numbers for ASE reader."""
    from ase.data import atomic_numbers
    return {i + 1: atomic_numbers.get(sym, 6)
            for i, sym in enumerate(spec_order)}


def _lammps_env():
    """
    Build the environment for the LAMMPS subprocess.

    The MACE-OFF mliap-unified pair style runs inside LAMMPS's embedded Python
    interpreter, which must `import lammps`. That module lives in the venv that
    runs CrYAL (mace-env), so we expose its site-packages via PYTHONPATH. This
    makes the run work whether or not the venv was activated in the shell.
    """
    env = os.environ.copy()
    purelib = sysconfig.get_paths().get("purelib")
    parts = [p for p in (purelib, env.get("PYTHONPATH")) if p]
    if parts:
        env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

@register
class LammpsMaceBackend(EnergyBackend):
    """LAMMPS as an external process, driven by the user's input script."""

    name = "lammps_mace"
    aliases = ("lammps",)

    #: seconds before a single structure is abandoned
    timeout = 1800

    @classmethod
    def validate_config(cls, cfg):
        if not os.path.exists(cfg.lammps_input):
            raise FileNotFoundError(f"lammpsInput not found: {cfg.lammps_input}")

    def describe(self) -> str:
        return f"{self.name} — {self.cfg.lammps_command} -in {self.cfg.lammps_input}"

    def relax(self, atoms, step_dir: str, logger=None):
        logger = logger or self.logger
        cfg = self.cfg

        struct_path = os.path.join(step_dir, "structure.data")
        minim_path  = os.path.join(step_dir, "minimized.data")
        energy_path = os.path.join(step_dir, "energy.dat")
        lammps_out  = os.path.join(step_dir, "lammps.output")

        # Write input structure
        try:
            _write_lammps_data(atoms, struct_path, cfg.spec_order)
        except Exception as e:
            if logger:
                logger.debug(f"  LAMMPS: failed to write structure.data — {e}")
            return None, None

        # Copy LAMMPS input to step directory
        lmp_input_local = os.path.join(step_dir, os.path.basename(cfg.lammps_input))
        shutil.copy(cfg.lammps_input, lmp_input_local)

        # Run LAMMPS
        cmd = f"{cfg.lammps_command} -in {os.path.basename(cfg.lammps_input)}"
        try:
            with open(lammps_out, 'w') as out:
                result = subprocess.run(
                    cmd, shell=True, cwd=step_dir,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    env=_lammps_env(),
                    timeout=self.timeout
                )
        except subprocess.TimeoutExpired:
            if logger:
                logger.debug(f"  LAMMPS: timeout in {step_dir}")
            return None, None
        except Exception as e:
            if logger:
                logger.debug(f"  LAMMPS: execution error — {e}")
            return None, None

        if result.returncode != 0:
            if logger:
                logger.debug(f"  LAMMPS: non-zero exit code ({result.returncode})")
            return None, None

        # Read results
        if not os.path.exists(minim_path) or not os.path.exists(energy_path):
            if logger:
                logger.debug(f"  LAMMPS: output files missing in {step_dir}")
            return None, None

        try:
            relaxed = read(minim_path, format='lammps-data',
                           style='atomic', Z_of_type=_z_of_type(cfg.spec_order))
            with open(energy_path) as f:
                energy = float(f.read().strip())
        except Exception as e:
            if logger:
                logger.debug(f"  LAMMPS: could not read output — {e}")
            return None, None

        return relaxed, energy
