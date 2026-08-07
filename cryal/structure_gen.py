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
structure_gen.py — Crystal structure generation using PyXtal with USPEX-like
volume targeting.

Key design (blind search):
  - Volume range estimated from molecular weight + physical density bounds.
    No experimental structure knowledge is used.
  - Beta angle for monoclinic SGs: uniform in [betaMin, betaMax] by default.
    The GP will discover which beta values are energetically favorable.
  - If PyXtal cannot place molecules at the target volume, the cell is expanded
    in small steps (VOLUME_EXPANSION_FACTORS) until placement succeeds.
    This mirrors USPEX's approach: start dense, expand minimally.

Biased generation (used in AL cycles > 0):
  - The GP suggests target regions as (beta, volume) pairs.
  - Structures are generated with beta and volume distributions centered on
    those suggestions, allowing the search to focus on promising regions.
"""

import os
import random
import warnings
import numpy as np
from typing import List, Optional, Tuple, Dict

from ase.io import read
from pyxtal import pyxtal
from pyxtal.lattice import Lattice

from .utils import (check_cell_axes, check_bond_integrity,
                    check_close_contacts, push_apart_overlaps,
                    get_molecules_robust, inter_min_distance,
                    compute_density, cell_params_dict, _translate_molecule)
from .config import Config

warnings.filterwarnings('ignore')

# Crystal system classification
_MONOCLINIC   = frozenset({4, 9, 14, 15})
_ORTHORHOMBIC = frozenset({19, 33, 61})
_TRICLINIC    = frozenset({2})


# ---------------------------------------------------------------------------
# Lattice generation
# ---------------------------------------------------------------------------

def random_lattice_for_sg(sg: int, volume: float, cfg: Config,
                           beta_override: Optional[float] = None
                           ) -> Tuple[float, float, float, float, float, float]:
    """
    Generate random cell parameters (a, b, c, alpha, beta, gamma) for a
    given space group and target volume.

    Axial ratios are sampled from a log-normal distribution (sigma=0.5),
    favoring moderately anisotropic cells while allowing a wide range.
    All axes are guaranteed >= cfg.min_cell_axis.

    Beta angle behavior:
      - If beta_override is provided: beta = beta_override
      - Otherwise: sample from the distribution specified in cfg
        (uniform or triangular, blind by default)
    """
    for _ in range(300):
        if sg in _MONOCLINIC:
            alpha = gamma = 90.0
            if beta_override is not None:
                beta = float(beta_override)
            elif cfg.beta_distribution == "triangular" and cfg.beta_mode is not None:
                beta = random.triangular(cfg.beta_min, cfg.beta_max, cfg.beta_mode)
            else:
                beta = random.uniform(cfg.beta_min, cfg.beta_max)
            sin_b = np.sin(np.radians(beta))
            r1 = np.exp(random.gauss(0.0, 0.5))
            r2 = np.exp(random.gauss(0.0, 0.5))
            a = (volume / (r1 * r2 * sin_b)) ** (1.0 / 3.0)
            b, c = a * r1, a * r2

        elif sg in _ORTHORHOMBIC:
            alpha = beta = gamma = 90.0
            r1 = np.exp(random.gauss(0.0, 0.5))
            r2 = np.exp(random.gauss(0.0, 0.5))
            a = (volume / (r1 * r2)) ** (1.0 / 3.0)
            b, c = a * r1, a * r2

        else:  # triclinic (SG=2)
            alpha = random.uniform(70.0, 110.0)
            beta  = random.uniform(70.0, 110.0)
            gamma = random.uniform(70.0, 110.0)
            ca = np.cos(np.radians(alpha))
            cb = np.cos(np.radians(beta))
            cg = np.cos(np.radians(gamma))
            det = 1 - ca**2 - cb**2 - cg**2 + 2*ca*cb*cg
            if det < 0.01:
                continue
            vol_factor = np.sqrt(det)
            r1 = np.exp(random.gauss(0.0, 0.4))
            r2 = np.exp(random.gauss(0.0, 0.4))
            a = (volume / (r1 * r2 * vol_factor)) ** (1.0 / 3.0)
            b, c = a * r1, a * r2

        if min(a, b, c) >= cfg.min_cell_axis:
            return a, b, c, alpha, beta, gamma

    # Fallback: cubic cell
    a = volume ** (1.0 / 3.0)
    return a, a, a, 90.0, 90.0, 90.0


def _try_pyxtal(mol_file: str, sg: int, Z: int, a, b, c, alpha, beta, gamma,
                expected: int) -> Optional:
    """
    Attempt to generate one PyXtal structure with the given lattice.
    Returns ASE Atoms or None on failure.
    """
    try:
        lat  = Lattice.from_para(a, b, c, alpha, beta, gamma)
        xtal = pyxtal(molecular=True)
        xtal.from_random(3, sg, [mol_file], [Z], lattice=lat, conventional=True)
        atoms = xtal.to_ase(resort=False)
        if len(atoms) != expected:
            return None
        return atoms
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_structures(mol_file: str, cfg: Config, n: int,
                        suggestions: Optional[List[Dict]] = None,
                        logger=None) -> List[Dict]:
    """
    Generate n crystal structures.

    In cycle 0 (suggestions=None):
      - SG sampled from cfg.space_groups / cfg.sg_weights (CSD statistics)
      - Volume sampled uniformly from [volume_min, volume_max]
      - Beta sampled according to cfg.beta_distribution (uniform by default)

    In cycles > 0 (suggestions provided):
      - Half the batch is biased toward GP suggestions (beta, volume targets)
      - Half is sampled randomly for exploration

    Returns list of dicts: {'atoms': ASE Atoms, 'space_group': int, 'source': str}
    """
    mol_abs     = os.path.abspath(mol_file)
    n_atoms_mol = len(read(mol_abs))
    expected    = n_atoms_mol * cfg.Z
    vol_min     = cfg.volume_min()
    vol_max     = cfg.volume_max()

    structures = []
    attempts   = 0
    errors: Dict[str, int] = {}

    # Split into biased and unbiased if suggestions are available
    n_biased   = (n // 2) if suggestions else 0
    n_unbiased = n - n_biased

    def _attempt(sg, volume, beta_override=None):
        """One generation attempt with USPEX-like expansion loop."""
        base_vol = volume
        for expansion in cfg.volume_expansion_factors:
            vol = base_vol * expansion
            a, b, c, alpha, beta, gamma = random_lattice_for_sg(
                sg, vol, cfg, beta_override=beta_override)
            atoms = _try_pyxtal(mol_abs, sg, cfg.Z, a, b, c, alpha, beta, gamma, expected)
            if atoms is None:
                continue
            if not check_cell_axes(atoms, cfg.min_cell_axis):
                errors['axis_short'] = errors.get('axis_short', 0) + 1
                continue
            if not check_bond_integrity(atoms):
                errors['bond_integrity'] = errors.get('bond_integrity', 0) + 1
                continue
            return atoms, sg, expansion
        return None, sg, None

    # --- Unbiased generation ---
    count_unbiased = 0
    while count_unbiased < n_unbiased and attempts < n * 60:
        attempts += 1
        sg = random.choices(cfg.space_groups, weights=cfg.sg_weights, k=1)[0]
        volume = random.uniform(vol_min, vol_max)
        atoms, sg, exp = _attempt(sg, volume)
        if atoms is None:
            errors['all_exp_failed'] = errors.get('all_exp_failed', 0) + 1
            continue
        structures.append({'atoms': atoms, 'space_group': sg,
                           'source': f'random_sg{sg}',
                           'expansion': exp})
        count_unbiased += 1

    # --- Biased generation (from GP suggestions) ---
    if suggestions:
        count_biased = 0
        for suggestion in (suggestions * (n_biased // len(suggestions) + 2)):
            if count_biased >= n_biased:
                break
            attempts += 1
            beta_t   = suggestion.get('beta', None)
            vol_t    = suggestion.get('volume', random.uniform(vol_min, vol_max))
            # Prefer monoclinic SG when suggestion has an informative beta
            if beta_t is not None and beta_t > 92.0:
                mono_sgs = [sg for sg in cfg.space_groups if sg in _MONOCLINIC]
                mono_w   = [cfg.sg_weights[i] for i, sg in enumerate(cfg.space_groups)
                            if sg in _MONOCLINIC]
                sg = random.choices(mono_sgs, weights=mono_w, k=1)[0] if mono_sgs else \
                     random.choices(cfg.space_groups, weights=cfg.sg_weights, k=1)[0]
                beta_override = random.triangular(
                    max(90.0, beta_t - 10), min(150.0, beta_t + 10), beta_t)
            else:
                sg = random.choices(cfg.space_groups, weights=cfg.sg_weights, k=1)[0]
                beta_override = None
            # Volume close to suggestion
            vol = random.uniform(vol_t * 0.9, vol_t * 1.1)
            vol = np.clip(vol, vol_min * 0.85, vol_max * 1.2)
            atoms, sg, exp = _attempt(sg, vol, beta_override)
            if atoms is None:
                continue
            structures.append({'atoms': atoms, 'space_group': sg,
                               'source': f'biased_sg{sg}_b{beta_t:.0f}' if beta_t else f'biased_sg{sg}',
                               'expansion': exp})
            count_biased += 1

    if logger:
        logger.info(f"  Generated {len(structures)}/{n} structures "
                    f"({attempts} attempts, {n_biased} biased)")
        if errors:
            top_errors = sorted(errors.items(), key=lambda x: -x[1])[:3]
            for k, v in top_errors:
                logger.debug(f"    [{v}x] {k}")
        betas = [s['atoms'].cell.cellpar()[4] for s in structures]
        rhos  = [compute_density(s['atoms'], cfg.Z,
                                 cfg.mol_weight) for s in structures]
        logger.info(f"  Volume: [{min(s['atoms'].get_volume() for s in structures):.0f}, "
                    f"{max(s['atoms'].get_volume() for s in structures):.0f}] Å³")
        logger.info(f"  Density: [{min(rhos):.3f}, {max(rhos):.3f}] g/cm³")
        logger.info(f"  Beta: [{min(betas):.1f}°, {max(betas):.1f}°]")

    return structures


# ---------------------------------------------------------------------------
# Seed loading and perturbation
# ---------------------------------------------------------------------------

def _rotation_matrix(axis: np.ndarray, theta_deg: float) -> np.ndarray:
    """Rodrigues rotation matrix."""
    theta = np.radians(theta_deg)
    axis  = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa,bb,cc,dd = a*a,b*b,c*c,d*d
    bc,ad,ac,ab,bd,cd = b*c,a*d,a*c,a*b,b*d,c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


def _perturb_one(seed_atoms, molecules, cfg) -> Tuple:
    """Apply random rotation + translation to a fraction of molecules."""
    atoms_p   = seed_atoms.copy()
    n_perturb = max(1, int(round(len(molecules) * cfg.frac_mols_perturb)))
    for mol_id in random.sample(range(len(molecules)), n_perturb):
        pos = atoms_p.positions[molecules[mol_id]]
        com = pos.mean(axis=0)
        axis = np.random.rand(3) - 0.5
        axis /= np.linalg.norm(axis)
        R = _rotation_matrix(axis, random.uniform(-cfg.max_rot_perturb, cfg.max_rot_perturb))
        atoms_p.positions[molecules[mol_id]] = (pos - com) @ R.T + com
        atoms_p.positions[molecules[mol_id]] += np.random.uniform(
            -cfg.max_trans_perturb, cfg.max_trans_perturb, 3)
    ok = check_close_contacts(atoms_p, molecules, cfg)
    return atoms_p, ok


def load_seeds(cfg: Config, mol_size: int, logger=None) -> List[Dict]:
    """
    Load CIF files from cfg.seeds_folder and generate perturbations.
    If the folder is missing or empty, return an empty list silently.
    """
    if not cfg.use_seeds_folder:
        return []

    folder = cfg.seeds_folder
    if not os.path.isdir(folder):
        if logger:
            logger.info(f"  Seeds folder '{folder}' not found — starting without seeds")
        return []

    import glob
    cif_files = sorted(glob.glob(os.path.join(folder, "*.cif")) +
                       glob.glob(os.path.join(folder, "*.POSCAR")) +
                       glob.glob(os.path.join(folder, "POSCAR*")))

    if not cif_files:
        if logger:
            logger.info(f"  Seeds folder '{folder}' is empty — starting without seeds")
        return []

    all_structs = []
    for cif in cif_files:
        try:
            seed_atoms = read(cif)
        except Exception as e:
            if logger:
                logger.warning(f"  Could not load seed {cif}: {e}")
            continue

        molecules, method = get_molecules_robust(seed_atoms, mol_size)
        rho  = compute_density(seed_atoms, cfg.Z, cfg.mol_weight)
        beta = seed_atoms.cell.cellpar()[4]
        if logger:
            logger.info(f"  Seed {os.path.basename(cif)}: "
                        f"rho={rho:.3f} g/cm³  beta={beta:.1f}°  [{method}]")

        # Add original seed
        all_structs.append({'atoms': seed_atoms, 'space_group': None,
                            'source': f'seed_{os.path.basename(cif)}'})

        # Generate perturbations
        generated = 0
        max_att   = cfg.num_perturbations_per_seed * 10
        for _ in range(max_att):
            if generated >= cfg.num_perturbations_per_seed:
                break
            atoms_p, ok = _perturb_one(seed_atoms, molecules, cfg)
            if ok:
                all_structs.append({'atoms': atoms_p, 'space_group': None,
                                    'source': f'perturb_{os.path.basename(cif)}'})
                generated += 1

        if logger:
            logger.info(f"    Perturbations: {generated}/{cfg.num_perturbations_per_seed}")

    return all_structs
