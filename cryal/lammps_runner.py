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
lammps_runner.py — LAMMPS interface for single-structure relaxation.

Each call to evaluate_structure():
  1. Creates a temporary working directory under output_dir/steps/
  2. Writes the structure as a LAMMPS data file (structure.data)
  3. Runs LAMMPS with the user-supplied input script
  4. Reads the relaxed structure (minimized.data) and energy (energy.dat)
  5. Checks molecular integrity (optional): rejects structures where
     MACE-OFF23 formed new covalent bonds during relaxation
  6. Returns (relaxed_atoms, energy_eV) or (None, None) on failure

The molecular integrity check compares bond lengths in the relaxed structure
against reference bonds from molecule.xyz. A bond is considered broken if
its length exceeds integrity_max_bond_ratio × sum_of_covalent_radii.
"""

import os
import sys
import sysconfig
import subprocess
import shutil
import numpy as np
import warnings
from typing import Optional, Tuple

from ase.io import read, write
from ase.neighborlist import NeighborList, natural_cutoffs

from .utils import get_molecules_robust, compute_density, check_bond_integrity
from .config import Config

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Reference bond detection (from molecule.xyz)
# ---------------------------------------------------------------------------

def get_reference_bonds(mol_file: str, max_ratio: float = 2.0):
    """
    Detect all covalent bonds in the isolated molecule and compute their
    maximum allowed length (used for integrity checks after relaxation).

    Returns list of (i, j, max_allowed_distance).
    """
    mol = read(mol_file)
    cutoffs = natural_cutoffs(mol, mult=1.2)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=False)
    nl.update(mol)

    bonds = []
    for i in range(len(mol)):
        nbrs, offsets = nl.get_neighbors(i)
        for j, offset in zip(nbrs, offsets):
            if j > i:
                pos_j = mol.positions[j] + np.dot(offset, mol.cell.array)
                d = np.linalg.norm(pos_j - mol.positions[i])
                # max_allowed = max_ratio × (sum of covalent radii)
                max_d = (cutoffs[i] + cutoffs[j]) * max_ratio / 1.2
                bonds.append((i, j, max_d))
    return bonds


def check_molecular_integrity(atoms, mol_size: int, ref_bonds) -> bool:
    """
    Return True if all reference bonds are intact in the relaxed crystal.
    Uses MIC distances to handle atoms wrapped across periodic boundaries.
    """
    if not ref_bonds or mol_size is None:
        return True
    n_mol = len(atoms) // mol_size
    for m in range(n_mol):
        offset = m * mol_size
        for i, j, max_d in ref_bonds:
            d = atoms.get_distance(offset + i, offset + j, mic=True)
            if d > max_d:
                return False
    return True


# ---------------------------------------------------------------------------
# LAMMPS data file writer
# ---------------------------------------------------------------------------

def _write_lammps_data(atoms, filepath: str, spec_order):
    """
    Write ASE Atoms to a LAMMPS data file (atomic style, triclinic cell).
    spec_order maps element symbols to LAMMPS atom types.
    """
    write(filepath, atoms, format='lammps-data', specorder=spec_order)


# ---------------------------------------------------------------------------
# Subprocess environment
# ---------------------------------------------------------------------------

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
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_structure(atoms, cfg: Config, step_dir: str,
                       ref_bonds, mol_size: int,
                       logger=None):
    """
    Run LAMMPS relaxation on a single structure.

    Parameters
    ----------
    atoms     : ASE Atoms — structure to relax
    cfg       : Config — run parameters
    step_dir  : str — working directory for this evaluation
    ref_bonds : list — from get_reference_bonds()
    mol_size  : int — atoms per molecule
    logger    : logging.Logger

    Returns
    -------
    (relaxed_atoms, energy_eV) or (None, None) on failure.
    """
    os.makedirs(step_dir, exist_ok=True)

    # Reliable pre-LAMMPS gate. MACE-OFF produces ~1e6 eV/Å forces on
    # sub-Ångström atomic overlaps; the minimizer cannot escape them and the
    # NPT stage then blows up ("lost atoms" / "tilted box too far"). Catch them
    # here with a PBC-correct NeighborList check (unlike the MIC-based contact
    # checks) and skip the ~5 min LAMMPS run on a structure that cannot relax.
    if not check_bond_integrity(atoms):
        if logger:
            logger.debug("  Pre-LAMMPS: rejected — atomic overlap < 0.8 Å (would explode)")
        return None, None

    struct_path = os.path.join(step_dir, "structure.data")
    minim_path  = os.path.join(step_dir, "minimized.data")
    energy_path = os.path.join(step_dir, "energy.dat")
    lammps_out  = os.path.join(step_dir, "lammps.output")
    lammps_log  = os.path.join(step_dir, "log.lammps")

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
        result = subprocess.run(
            cmd, shell=True, cwd=step_dir,
            stdout=open(lammps_out, 'w'),
            stderr=subprocess.STDOUT,
            env=_lammps_env(),
            timeout=1800   # 30 min per structure
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

    # Sanity check on energy
    if energy > -100.0:
        if logger:
            logger.debug(f"  LAMMPS: unrealistic energy {energy:.2f} eV")
        return None, None

    # Molecular integrity check
    if cfg.check_molecular_integrity and ref_bonds:
        if not check_molecular_integrity(relaxed, mol_size, ref_bonds):
            if logger:
                logger.debug("  LAMMPS: molecular integrity failed (covalent bond broken)")
            return None, None

    return relaxed, energy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _z_of_type(spec_order):
    """Map LAMMPS atom types to atomic numbers for ASE reader."""
    from ase.data import atomic_numbers
    return {i + 1: atomic_numbers.get(sym, 6)
            for i, sym in enumerate(spec_order)}
