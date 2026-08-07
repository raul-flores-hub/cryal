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
chpi_optimizer.py — Optional geometric CH-pi contact optimizer.

Adjusts molecular positions to maximize H...aromatic centroid contacts
in the equilibrium range [d_min, d_max]. Activated by useChpiOptimizer = True
in INPUT.txt.

The optimizer is a greedy iterative procedure:
  1. Find all intermolecular CH-pi contacts within d_search.
  2. For contacts outside [d_min, d_max], translate the H-bearing molecule
     so that H moves to d_target from the aromatic centroid.
  3. Accept the move only if: no new close contacts are created AND
     the number of equilibrium contacts does not decrease.
  4. Repeat until convergence (max_no_improve consecutive iterations
     without an improvement in the contact count).

Requires RDKit for aromatic ring and C-H bond detection.
"""

import numpy as np
from typing import List, Tuple, Optional

try:
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False

from .utils import (get_molecules_robust, unwrap_molecule, mic_vector,
                    check_close_contacts, push_apart_overlaps,
                    inter_min_distance, compute_density, _translate_molecule)
from .config import Config


# ---------------------------------------------------------------------------
# Molecule analysis (RDKit)
# ---------------------------------------------------------------------------

def analyze_molecule(mol_file: str, logger=None):
    """
    Detect aromatic rings and C-H pairs in the molecule using RDKit.

    Returns:
      ring_info : list of {'atom_indices', 'centroid', 'normal'}
      ch_pairs  : list of (h_index, c_index)
    """
    if not _HAS_RDKIT:
        raise ImportError("RDKit is required for CH-pi optimization. "
                          "Install with: pip install rdkit")

    mol = Chem.MolFromXYZFile(mol_file)
    if mol is None:
        raise RuntimeError(f"RDKit could not read {mol_file}")
    rdDetermineBonds.DetermineBonds(mol, charge=0)

    conf = mol.GetConformer()
    positions = np.array([[conf.GetAtomPosition(i).x,
                           conf.GetAtomPosition(i).y,
                           conf.GetAtomPosition(i).z]
                          for i in range(mol.GetNumAtoms())])

    ring_info = []
    for ring in mol.GetRingInfo().AtomRings():
        if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        ring = list(ring)
        coords = positions[ring]
        centroid = coords.mean(axis=0)
        v1 = coords[1] - coords[0]
        v2 = coords[2] - coords[0]
        normal = np.cross(v1, v2)
        n_len = np.linalg.norm(normal)
        if n_len < 1e-8:
            continue
        ring_info.append({'atom_indices': ring,
                          'centroid': centroid,
                          'normal': normal / n_len})

    ch_pairs = []
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        s1 = mol.GetAtomWithIdx(a1).GetSymbol()
        s2 = mol.GetAtomWithIdx(a2).GetSymbol()
        if s1 == "H" and s2 == "C":
            ch_pairs.append((a1, a2))
        elif s1 == "C" and s2 == "H":
            ch_pairs.append((a2, a1))

    if logger:
        logger.debug(f"  Molecule: {len(ring_info)} aromatic rings, "
                     f"{len(ch_pairs)} C-H pairs")
    return ring_info, ch_pairs


# ---------------------------------------------------------------------------
# Crystal-level geometry
# ---------------------------------------------------------------------------

def _get_crystal_rings(atoms, molecules, mol_ring_info):
    """Compute aromatic ring centroids and normals for all molecules."""
    ring_data = []
    for mol_id, mol_indices in enumerate(molecules):
        unwrapped = unwrap_molecule(atoms.positions, mol_indices, atoms.cell.array)
        for ring_idx, ring in enumerate(mol_ring_info):
            ring_pos  = unwrapped[ring['atom_indices']]
            centroid  = ring_pos.mean(axis=0)
            v1 = ring_pos[1] - ring_pos[0]
            v2 = ring_pos[2] - ring_pos[0]
            normal = np.cross(v1, v2)
            n_len  = np.linalg.norm(normal)
            ring_data.append({'mol_id': mol_id, 'ring_idx': ring_idx,
                               'centroid': centroid,
                               'normal': normal / n_len if n_len > 1e-8
                                         else ring['normal'].copy()})
    return ring_data


def _get_crystal_ch(atoms, molecules, ch_pairs):
    """Compute C-H pair positions for all molecules."""
    ch_data = []
    for mol_id, mol_indices in enumerate(molecules):
        unwrapped = unwrap_molecule(atoms.positions, mol_indices, atoms.cell.array)
        for h_local, c_local in ch_pairs:
            ch_data.append({'mol_id': mol_id,
                             'h_pos': unwrapped[h_local].copy(),
                             'c_pos': unwrapped[c_local].copy()})
    return ch_data


def _find_contacts(atoms, molecules, ring_data, ch_data, cfg, d_search=None):
    """Find all intermolecular CH-pi contacts within d_search."""
    if d_search is None:
        d_search = cfg.d_search
    cell_T     = atoms.cell.array.T
    cell_T_inv = np.linalg.inv(cell_T)
    contacts   = []

    for ch in ch_data:
        mol_h = ch['mol_id']
        h_pos = ch['h_pos']
        v_ch  = h_pos - ch['c_pos']
        len_ch = np.linalg.norm(v_ch)
        if len_ch < 1e-8:
            continue
        v_ch_unit = v_ch / len_ch

        for ring in ring_data:
            if ring['mol_id'] == mol_h:
                continue
            v_hc  = mic_vector(h_pos, ring['centroid'], cell_T, cell_T_inv)
            dist  = np.linalg.norm(v_hc)
            if dist < 0.5 or dist > d_search:
                continue
            v_hc_unit = v_hc / dist
            angle = np.degrees(np.arccos(
                np.clip(np.dot(v_ch_unit, v_hc_unit), -1.0, 1.0)))
            if angle < cfg.angle_ch_centroid_min:
                continue
            ang_n = np.degrees(np.arccos(
                np.clip(abs(np.dot(v_hc_unit, ring['normal'])), 0.0, 1.0)))
            if ang_n > cfg.angle_from_normal_max:
                continue
            contacts.append({'h_mol': mol_h, 'ring_mol': ring['mol_id'],
                              'ring_idx': ring['ring_idx'],
                              'h_pos': h_pos.copy(), 'c_pos': ch['c_pos'].copy(),
                              'centroid': ring['centroid'].copy(),
                              'normal': ring['normal'].copy(),
                              'distance': dist, 'angle': angle, 'angle_normal': ang_n})
    contacts.sort(key=lambda x: x['distance'])
    return contacts


def _count_eq(contacts, cfg):
    return sum(1 for c in contacts if cfg.d_min <= c['distance'] <= cfg.d_max)


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

def optimize_ch_pi(atoms, mol_ring_info, ch_pairs, cfg: Config,
                   mol_size: Optional[int] = None,
                   verbose: bool = False, logger=None):
    """
    Run the CH-pi geometric optimizer on a crystal structure.

    Returns (optimized_atoms, n_eq_contacts, history_list).
    Returns (atoms, 0, []) if the structure cannot be optimized.
    """
    atoms = atoms.copy()

    # Identify molecules
    if mol_size is not None:
        molecules, method = get_molecules_robust(atoms, mol_size)
    else:
        from .utils import get_molecules_by_connectivity
        molecules = get_molecules_by_connectivity(atoms)
        method = "connectivity"

    if len(molecules) < 2:
        if verbose and logger:
            logger.debug("  CH-pi: only 1 molecule detected — invalid structure")
        return atoms, 0, []

    # Resolve any close contacts before optimizing
    if not check_close_contacts(atoms, molecules, cfg):
        ok, n_viol = push_apart_overlaps(atoms, molecules, cfg, verbose=verbose,
                                         logger=logger)
        if not ok:
            if verbose and logger:
                logger.debug(f"  CH-pi: unresolvable overlaps ({n_viol} violations)")
            return atoms, 0, []

    cell_T     = atoms.cell.array.T
    cell_T_inv = np.linalg.inv(cell_T)
    history        = []
    blocked_pairs  = set()
    best_n_eq      = 0
    no_improve_cnt = 0

    for iteration in range(cfg.max_chpi_iter):
        ring_data    = _get_crystal_rings(atoms, molecules, mol_ring_info)
        ch_data      = _get_crystal_ch(atoms, molecules, ch_pairs)
        all_contacts = _find_contacts(atoms, molecules, ring_data, ch_data, cfg)
        in_eq  = [c for c in all_contacts if cfg.d_min <= c['distance'] <= cfg.d_max]
        out_eq = [c for c in all_contacts
                  if c['distance'] < cfg.d_min or c['distance'] > cfg.d_max]
        n_eq   = len(in_eq)
        history.append(n_eq)

        if not out_eq:
            break
        if n_eq > best_n_eq:
            best_n_eq = n_eq; no_improve_cnt = 0
        else:
            no_improve_cnt += 1
            if no_improve_cnt >= cfg.max_no_improve:
                break

        out_eq.sort(key=lambda c: abs(c['distance'] - cfg.d_target))
        made_move = False

        for contact in out_eq:
            key = (contact['h_mol'], contact['ring_mol'], contact['ring_idx'])
            if key in blocked_pairs:
                continue
            dist = contact['distance']
            if abs(dist - cfg.d_target) < 0.05:
                continue
            v_hc   = mic_vector(contact['h_pos'], contact['centroid'],
                                 cell_T, cell_T_inv)
            v_unit = v_hc / dist
            delta  = dist - cfg.d_target
            MAX_T  = 1.5   # Å
            trans  = v_unit * (delta if abs(delta) <= MAX_T
                               else np.sign(delta) * MAX_T)

            atoms_trial = atoms.copy()
            _translate_molecule(atoms_trial, molecules[contact['h_mol']], trans)

            if not check_close_contacts(atoms_trial, molecules, cfg):
                blocked_pairs.add(key)
                continue

            rd_t = _get_crystal_rings(atoms_trial, molecules, mol_ring_info)
            cd_t = _get_crystal_ch(atoms_trial, molecules, ch_pairs)
            in_t = _find_contacts(atoms_trial, molecules, rd_t, cd_t, cfg,
                                   d_search=cfg.d_max + 0.2)
            if _count_eq(in_t, cfg) < n_eq:
                blocked_pairs.add(key)
                continue

            atoms = atoms_trial
            made_move = True
            break

        if not made_move:
            break

    # Final contact count
    rd_f = _get_crystal_rings(atoms, molecules, mol_ring_info)
    cd_f = _get_crystal_ch(atoms, molecules, ch_pairs)
    final_contacts = _find_contacts(atoms, molecules, rd_f, cd_f, cfg,
                                    d_search=cfg.d_max + 0.2)
    n_final = _count_eq(final_contacts, cfg)

    if verbose and logger:
        logger.debug(f"  CH-pi: {n_final} contacts in [{cfg.d_min},{cfg.d_max}] Å")

    return atoms, n_final, history
