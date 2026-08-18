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
utils.py — Common utilities shared across CrYAL modules.

Includes:
  - Logging setup
  - Molecule identification (connectivity + sequential fallback)
  - Intermolecular contact checks
  - Molecular unwrapping (MIC)
  - Density computation
  - Cell parameter extraction
"""

import logging
import numpy as np
from ase.neighborlist import NeighborList, natural_cutoffs


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(name: str, log_file: str, level: str = "INFO") -> logging.Logger:
    """Create a logger that writes to both console and file."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Molecule identification
# ---------------------------------------------------------------------------

def get_molecules_by_connectivity(atoms):
    """
    Identify molecules using covalent bond connectivity.
    Returns a list of numpy arrays of atom indices, one per molecule.
    """
    cutoffs = natural_cutoffs(atoms)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    visited = set()
    molecules = []
    for start in range(len(atoms)):
        if start in visited:
            continue
        stack, component = [start], []
        while stack:
            i = stack.pop()
            if i in visited:
                continue
            visited.add(i)
            component.append(i)
            nbrs, _ = nl.get_neighbors(i)
            stack.extend(j for j in nbrs if j not in visited)
        molecules.append(np.array(sorted(component), dtype=int))
    return molecules


def get_molecules_by_size(atoms, mol_size: int):
    """
    Sequential molecule assignment: atoms 0..mol_size-1 → mol 0, etc.
    Used as fallback when connectivity detection fails (e.g. unwrapped atoms
    spanning periodic boundaries cause incorrect bond distances).
    """
    n = len(atoms)
    if n % mol_size != 0:
        raise ValueError(f"Atom count {n} not divisible by mol_size {mol_size}")
    return [np.arange(i * mol_size, (i + 1) * mol_size, dtype=int)
            for i in range(n // mol_size)]


def get_molecules_robust(atoms, mol_size: int):
    """
    Try connectivity-based detection; fall back to sequential if it fails.
    Returns (molecules: list[np.ndarray], method: str).
    """
    mols = get_molecules_by_connectivity(atoms)
    n_exp = len(atoms) // mol_size
    if len(mols) == n_exp and all(len(m) == mol_size for m in mols):
        return mols, "connectivity"
    return get_molecules_by_size(atoms, mol_size), "sequential"


# ---------------------------------------------------------------------------
# Periodic boundary utilities
# ---------------------------------------------------------------------------

def unwrap_molecule(positions: np.ndarray, mol_indices: np.ndarray,
                    cell: np.ndarray) -> np.ndarray:
    """
    Unwrap a molecule so all atoms are contiguous in Cartesian space.
    Uses the minimum image convention relative to the first atom.
    Returns unwrapped positions (shape: [n_atoms_mol, 3]).
    """
    cell_T = cell.T
    cell_T_inv = np.linalg.inv(cell_T)
    ref = positions[mol_indices[0]].copy()
    unwrapped = np.empty((len(mol_indices), 3))
    unwrapped[0] = ref
    for k, idx in enumerate(mol_indices[1:], start=1):
        diff = positions[idx] - ref
        frac = cell_T_inv @ diff
        frac -= np.round(frac)
        unwrapped[k] = ref + cell_T @ frac
    return unwrapped


def mic_vector(pos_from: np.ndarray, pos_to: np.ndarray,
               cell_T: np.ndarray, cell_T_inv: np.ndarray) -> np.ndarray:
    """Minimum image convention vector from pos_from to pos_to."""
    diff = pos_to - pos_from
    frac = cell_T_inv @ diff
    frac -= np.round(frac)
    return cell_T @ frac


# ---------------------------------------------------------------------------
# Contact checks
# ---------------------------------------------------------------------------

def check_close_contacts(atoms, molecules, cfg) -> bool:
    """
    Return True if no intermolecular contacts violate minimum distances.
    Uses MIC distances (atoms.get_all_distances(mic=True)).
    """
    symbols  = np.array(atoms.get_chemical_symbols())
    atom_mol = np.empty(len(atoms), dtype=int)
    for mid, mol_idx in enumerate(molecules):
        atom_mol[mol_idx] = mid

    all_dist = atoms.get_all_distances(mic=True)
    np.fill_diagonal(all_dist, np.inf)

    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            if atom_mol[i] == atom_mol[j]:
                continue
            d = all_dist[i, j]
            si, sj = symbols[i], symbols[j]
            if si == "H" and sj == "H":
                if d < cfg.min_hh_distance:
                    return False
            elif si != "H" and sj != "H":
                if d < cfg.min_heavy_distance:
                    return False
            else:
                if d < cfg.min_hx_distance:
                    return False
    return True


def push_apart_overlaps(atoms, molecules, cfg, max_iter: int = 300,
                        verbose: bool = False, logger=None) -> tuple:
    """
    Iteratively push overlapping molecules apart.
    Each iteration displaces molecular COMs proportionally to the overlap magnitude.
    Returns (success: bool, n_initial_violations: int).
    """
    cell_T     = atoms.cell.array.T
    cell_T_inv = np.linalg.inv(cell_T)
    symbols    = np.array(atoms.get_chemical_symbols())

    atom_mol = np.zeros(len(atoms), dtype=int)
    for mid, mol_idx in enumerate(molecules):
        atom_mol[mol_idx] = mid

    thresh_hh    = cfg.min_hh_distance    + 0.5
    thresh_heavy = cfg.min_heavy_distance + 0.5
    thresh_hx    = cfg.min_hx_distance    + 0.5

    n_violations_0 = None

    for iteration in range(max_iter):
        all_dist = atoms.get_all_distances(mic=True)
        np.fill_diagonal(all_dist, np.inf)
        mol_disp   = [np.zeros(3) for _ in molecules]
        violations = 0

        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                if atom_mol[i] == atom_mol[j]:
                    continue
                d = all_dist[i, j]
                si, sj = symbols[i], symbols[j]
                thresh = (thresh_hh    if si == "H" and sj == "H" else
                          thresh_heavy if si != "H" and sj != "H" else
                          thresh_hx)
                if d < thresh:
                    violations += 1
                    v = mic_vector(atoms.positions[j], atoms.positions[i],
                                   cell_T, cell_T_inv)
                    v_len = np.linalg.norm(v)
                    if v_len < 1e-8:
                        v = np.random.randn(3); v_len = np.linalg.norm(v)
                    direction = v / v_len
                    magnitude = (thresh - d) * 0.5
                    mol_disp[atom_mol[i]] +=  direction * magnitude
                    mol_disp[atom_mol[j]] -= direction * magnitude

        if n_violations_0 is None:
            n_violations_0 = violations

        if violations == 0:
            return True, n_violations_0

        for mol_id, mol_indices in enumerate(molecules):
            d_vec = mol_disp[mol_id]
            if np.linalg.norm(d_vec) > 1e-8:
                _translate_molecule(atoms, mol_indices, d_vec)

    ok = check_close_contacts(atoms, molecules, cfg)
    return ok, n_violations_0


def _translate_molecule(atoms, mol_indices: np.ndarray, translation: np.ndarray):
    """Translate molecule and wrap its COM back into the unit cell."""
    atoms.positions[mol_indices] += translation
    com = atoms.positions[mol_indices].mean(axis=0)
    frac_com  = np.linalg.inv(atoms.cell.array.T) @ com
    wrap_shift = np.floor(frac_com)
    atoms.positions[mol_indices] -= atoms.cell.array.T @ wrap_shift


# ---------------------------------------------------------------------------
# Cell and bond checks
# ---------------------------------------------------------------------------

def check_cell_axes(atoms, min_length: float) -> bool:
    """Return True if all cell axes >= min_length Å."""
    return all(c >= min_length for c in atoms.cell.lengths())


def check_bond_integrity(atoms, min_bond: float = 0.8) -> bool:
    """
    Return True if all covalent bonds (detected via natural_cutoffs) have
    length >= min_bond Å. Catches periodic-image overlaps in tight cells.
    Uses the correct MIC distance from NeighborList offsets.
    """
    cutoffs = natural_cutoffs(atoms)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=False)
    nl.update(atoms)
    cell = atoms.cell.array
    for i in range(len(atoms)):
        nbrs, offsets = nl.get_neighbors(i)
        for j, offset in zip(nbrs, offsets):
            # No `j > i` filter here. With bothways=False ASE already reports
            # every pair exactly once, but not necessarily from the lower
            # index: a contact across the periodic boundary can be listed at
            # the higher index with a non-zero offset. Filtering on j > i
            # therefore skipped exactly the periodic overlaps this gate exists
            # to catch. It also drops the case j == i, an atom clashing with
            # its own image in an absurdly small cell.
            pos_j_image = atoms.positions[j] + np.dot(offset, cell)
            d = np.linalg.norm(pos_j_image - atoms.positions[i])
            if d < min_bond:
                return False
    return True


# ---------------------------------------------------------------------------
# Physical utilities
# ---------------------------------------------------------------------------

def compute_density(atoms, Z: int, mol_weight: float) -> float:
    """Compute crystal density in g/cm³."""
    NA = 6.02214076e23
    return (Z * mol_weight / NA) / (atoms.get_volume() * 1e-24)


def cell_params_dict(atoms) -> dict:
    """Return cell parameters as a dict with keys a,b,c,alpha,beta,gamma,volume."""
    cp = atoms.cell.cellpar()
    return {'a': cp[0], 'b': cp[1], 'c': cp[2],
            'alpha': cp[3], 'beta': cp[4], 'gamma': cp[5],
            'volume': atoms.get_volume()}


def inter_min_distance(atoms, molecules) -> float:
    """Return the minimum intermolecular distance (MIC)."""
    atom_mol = np.zeros(len(atoms), dtype=int)
    for mid, mol_idx in enumerate(molecules):
        atom_mol[mol_idx] = mid
    all_dist = atoms.get_all_distances(mic=True)
    np.fill_diagonal(all_dist, np.inf)
    min_d = np.inf
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            if atom_mol[i] != atom_mol[j] and all_dist[i, j] < min_d:
                min_d = all_dist[i, j]
    return min_d
