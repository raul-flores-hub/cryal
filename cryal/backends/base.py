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
base.py — the contract every energy backend honours.

A backend turns a candidate structure into a relaxed structure and a total
energy. The active-learning loop knows nothing else about it: the surrogate
consumes only cell parameters, density and energy, so any engine that can
relax a periodic molecular crystal can stand in here.

Subclasses implement one method, relax(). Everything that must happen around
it — and that we learned the hard way must happen — lives in evaluate(), which
is not meant to be overridden:

  1. the pre-relaxation integrity gate, which rejects structures carrying
     sub-Angstrom atomic overlaps before any expensive engine is started;
  2. a sanity bound on the returned energy;
  3. the molecular-integrity check, which rejects relaxations that formed or
     broke covalent bonds.

Step 1 is the one worth insisting on. Overlapping atoms make a machine-learned
potential produce forces of order 1e5 eV/A; the minimizer cannot escape them
and the run dies minutes later with an error that points at the relaxation
instead of at the geometry. Catching it here costs 5 s per bad structure
instead of ~5 min, and every backend inherits that for free.
"""

import os
from abc import ABC, abstractmethod

import numpy as np
from ase.io import read
from ase.neighborlist import NeighborList, natural_cutoffs

from ..utils import check_bond_integrity


# ---------------------------------------------------------------------------
# Molecular integrity (backend-independent)
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
# The backend interface
# ---------------------------------------------------------------------------

class EnergyBackend(ABC):
    """
    Base class for relaxation/energy engines.

    Attributes
    ----------
    name    : str — the value of `energyBackend` in INPUT.txt that selects it
    aliases : tuple — alternative spellings accepted for `name`
    """

    name: str = ""
    aliases: tuple = ()

    #: May two evaluate() calls run at the same time in this process?
    #:
    #: Only a backend that keeps no mutable state between calls can say yes.
    #: The LAMMPS backend can: every call is a fresh subprocess whose whole
    #: world is its step directory. An in-process calculator usually cannot,
    #: because it holds one model instance whose `atoms` and `results` two
    #: threads would overwrite for each other — and because a second thread
    #: would be queueing on the same GPU anyway. cryal.parallel refuses to
    #: open more than one local slot on a backend that leaves this False,
    #: rather than letting the corruption happen quietly.
    thread_safe: bool = False

    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.logger = logger

    # -- optional hooks ----------------------------------------------------

    @classmethod
    def validate_config(cls, cfg):
        """
        Raise if the configuration cannot drive this backend (missing input
        script, missing calculator, ...). Called at load_config() time so a
        misconfigured run dies in the first second instead of the first cycle.
        """
        return None

    @classmethod
    def job_files(cls, cfg) -> dict:
        """
        Files a machine other than this one needs before it can run relax().

        Maps a Config attribute holding a path to that path. cryal.parallel
        copies each file to the remote worker once per run and rewrites the
        attribute in the worker's copy of the configuration to point at it, so
        the remote engine reads the same input script as the local one instead
        of whatever happens to sit at that path over there.

        Returns {} for a backend that needs nothing shipped: the ASE backend
        names its calculator by dotted path, and model weights are the
        worker's own installation to provide.
        """
        return {}

    @classmethod
    def required_commands(cls, cfg) -> list:
        """
        External programs that must be on a worker's PATH for relax() to work.

        Checked once per machine before the run starts. A non-interactive ssh
        session does not read ~/.bashrc, so an engine that is there when you
        log in by hand can still be missing when cryal calls it — a warning at
        second one beats a hundred identical failures at hour three.
        """
        return []

    def describe(self) -> str:
        """One line for the run log and the configuration summary."""
        return self.name

    # -- the one method a backend must provide -----------------------------

    @abstractmethod
    def relax(self, atoms, step_dir: str, logger=None):
        """
        Relax one structure and return (relaxed_atoms, energy_eV), or
        (None, None) if the relaxation failed.

        `step_dir` exists already and belongs to this evaluation: any file the
        engine needs to write goes there. Failures are reported by returning
        (None, None) rather than by raising — a single bad candidate must not
        end the search.
        """
        raise NotImplementedError

    # -- the fixed part of every evaluation --------------------------------

    def evaluate(self, atoms, step_dir: str, ref_bonds, mol_size: int,
                 logger=None):
        """
        Full evaluation of one candidate: gate, relax, check.

        Returns (relaxed_atoms, energy_eV) or (None, None) on rejection or
        failure. This is the single call the active-learning loop makes.
        """
        logger = logger or self.logger
        os.makedirs(step_dir, exist_ok=True)

        # Reliable pre-relaxation gate. A PBC-correct NeighborList check
        # (unlike the MIC-based contact checks) on overlaps that no minimizer
        # can undo — see the module docstring for why this runs first.
        if not check_bond_integrity(atoms):
            if logger:
                logger.debug("  Pre-relaxation: rejected — atomic overlap "
                             "< 0.8 Å (would explode)")
            return None, None

        relaxed, energy = self.relax(atoms, step_dir, logger=logger)
        if relaxed is None or energy is None:
            return None, None

        # Sanity check on energy. The bound is a configuration knob because it
        # is only meaningful against a given potential's zero of energy.
        limit = getattr(self.cfg, "energy_sanity_max", None)
        if limit is not None and energy > limit:
            if logger:
                logger.debug(f"  {self.name}: unrealistic energy {energy:.2f} eV")
            return None, None

        # Molecular integrity: did the relaxation break the molecule?
        if getattr(self.cfg, "check_molecular_integrity", False) and ref_bonds:
            if not check_molecular_integrity(relaxed, mol_size, ref_bonds):
                if logger:
                    logger.debug(f"  {self.name}: molecular integrity failed "
                                 "(covalent bond broken)")
                return None, None

        return relaxed, energy
