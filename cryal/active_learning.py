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
active_learning.py — Main active learning orchestrator.

Active Learning CSP Loop
========================
Cycle 0 (initialization):
  1. Generate initial_structures random crystal structures (blind, no prior knowledge)
  2. Load seeds from Seeds/ folder if available (optional)
  3. Apply CH-pi optimizer to each structure (optional)
  4. Evaluate all structures with LAMMPS/MACE-OFF23
  5. Save results to database

Cycle k (k = 1 ... num_cycles):
  1. Train GP surrogate on all evaluated structures
  2. Maximize Expected Improvement to identify promising regions
     (the GP autonomously discovers favorable beta/density combinations)
  3. Generate new structures: half biased toward EI suggestions,
     half random (to maintain exploration)
  4. Apply CH-pi optimizer (optional)
  5. Evaluate with LAMMPS
  6. Update database
  7. Report: best energy, GP feature importance, coverage

Convergence:
  Stop early if the best energy does not improve for max_no_improve_cycles
  consecutive cycles.

The pool mechanism copies the best pool_size structures from each cycle
into the Seeds/ working folder so they can bias the next cycle's generation.
"""

import os
import json
import glob
import time
import random
import shutil
import numpy as np
from typing import List, Dict, Optional

from ase.io import write

from .config import Config, print_config_summary
from .utils import (setup_logger, get_molecules_robust,
                    compute_density, cell_params_dict, check_bond_integrity)
from .structure_gen import generate_structures, load_seeds
from .backends import get_backend, get_reference_bonds
from .gp_model import GPSurrogate, maximize_ei, gp_report

try:
    from .chpi_optimizer import analyze_molecule, optimize_ch_pi
    _CHPI_AVAILABLE = True
except ImportError:
    _CHPI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

class CrystalDatabase:
    """
    In-memory + on-disk database of evaluated crystal structures.

    Each record:
      id, cycle, source, space_group,
      a, b, c, alpha, beta, gamma, volume, density,
      energy, n_chpi_contacts (optional),
      cif_path
    """

    def __init__(self, db_path: str, cif_dir: str):
        self.db_path = db_path
        self.cif_dir = cif_dir
        self.records: List[Dict] = []
        os.makedirs(cif_dir, exist_ok=True)
        if os.path.exists(db_path):
            self._load()

    def _load(self):
        with open(self.db_path) as f:
            self.records = json.load(f)

    def save(self):
        with open(self.db_path, 'w') as f:
            json.dump(self.records, f, indent=2)

    def add(self, record: Dict):
        self.records.append(record)

    def best_energy(self) -> Optional[float]:
        if not self.records:
            return None
        return min(r['energy'] for r in self.records)

    def best_record(self) -> Optional[Dict]:
        if not self.records:
            return None
        return min(self.records, key=lambda r: r['energy'])

    def top_n(self, n: int) -> List[Dict]:
        return sorted(self.records, key=lambda r: r['energy'])[:n]

    def __len__(self):
        return len(self.records)


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------

def _apply_chpi(struct_list, mol_ring_info, ch_pairs, mol_size, cfg, logger):
    """Apply CH-pi optimizer to a list of structures. Returns updated list."""
    if not cfg.use_chpi_optimizer or not _CHPI_AVAILABLE:
        return struct_list
    optimized = []
    n_reverted = 0
    for i, s in enumerate(struct_list):
        orig_atoms = s['atoms']
        try:
            opt_atoms, n_contacts, history = optimize_ch_pi(
                orig_atoms, mol_ring_info, ch_pairs, cfg,
                mol_size=mol_size, verbose=False, logger=logger)
            # The CH-pi optimizer's overlap guard / push_apart use MIC distances
            # (get_all_distances(mic=True)), which can miss periodic-image
            # overlaps in skewed triclinic cells. As a result it can return a
            # structure with sub-Ångström atomic clashes that make MACE-OFF
            # explode in LAMMPS. Never forward a structure that the optimizer
            # broke: if the input was clean and the result is not, keep the
            # input geometry. check_bond_integrity uses a PBC-correct
            # NeighborList and reliably catches these clashes.
            if check_bond_integrity(opt_atoms) or not check_bond_integrity(orig_atoms):
                use_atoms = opt_atoms
            else:
                use_atoms = orig_atoms
                n_contacts = 0
                n_reverted += 1
            s = dict(s)
            s['atoms']           = use_atoms
            s['n_chpi_contacts'] = n_contacts
        except Exception as e:
            logger.debug(f"  CH-pi error on structure {i+1}: {e}")
            s = dict(s)
            s['n_chpi_contacts'] = 0
        optimized.append(s)
    if n_reverted and logger:
        logger.info(f"  CH-pi: reverted {n_reverted}/{len(struct_list)} structures "
                    f"that developed atomic overlaps (kept pre-optimization geometry)")
    return optimized


def _evaluate_batch(struct_list, cfg, backend, cycle: int, batch_start: int,
                    ref_bonds, mol_size: int,
                    db: CrystalDatabase, logger) -> int:
    """
    Evaluate a batch of structures with the energy backend and add the results
    to the database. Returns the number of successful evaluations.
    """
    n_ok = 0
    for i, s in enumerate(struct_list):
        idx       = batch_start + i
        step_name = f"cycle{cycle:03d}_struct{idx:05d}"
        step_dir  = os.path.join(cfg.output_dir, "steps", step_name)

        logger.info(f"  Evaluating {step_name}  source={s.get('source','?')}  "
                    f"SG={s.get('space_group','?')}")

        relaxed, energy = backend.evaluate(
            s['atoms'], step_dir, ref_bonds, mol_size, logger=logger)

        if relaxed is None:
            logger.info(f"    → FAILED (relaxation or integrity)")
            continue

        cp   = cell_params_dict(relaxed)
        rho  = compute_density(relaxed, cfg.Z, cfg.mol_weight)
        rec_id = len(db) + 1

        # Save CIF
        cif_path = os.path.join(db.cif_dir, f"{step_name}.cif")
        relaxed.wrap()
        write(cif_path, relaxed)

        record = {
            'id':           rec_id,
            'cycle':        cycle,
            'source':       s.get('source', 'unknown'),
            'space_group':  s.get('space_group'),
            'n_chpi':       s.get('n_chpi_contacts', None),
            'energy':       round(energy, 6),
            'density':      round(rho, 4),
            'volume':       round(cp['volume'], 2),
            'a':            round(cp['a'], 4),
            'b':            round(cp['b'], 4),
            'c':            round(cp['c'], 4),
            'alpha':        round(cp['alpha'], 3),
            'beta':         round(cp['beta'], 3),
            'gamma':        round(cp['gamma'], 3),
            'cif_path':     cif_path,
        }
        db.add(record)
        db.save()
        n_ok += 1

        logger.info(f"    → E={energy:.4f} eV  rho={rho:.3f} g/cm³  "
                    f"beta={cp['beta']:.1f}°  V={cp['volume']:.0f} Å³")

    return n_ok


# ---------------------------------------------------------------------------
# Main active learning runner
# ---------------------------------------------------------------------------

def run(cfg: Config):
    """
    Execute the full active learning CSP loop.
    """
    # --- Setup ---
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "steps"), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "structures"), exist_ok=True)

    logger = setup_logger(
        "cryal",
        os.path.join(cfg.output_dir, "cryal.log"),
        cfg.log_level
    )
    print_config_summary(cfg, logger)

    # Initialize database (loads existing records if database.json is present)
    db = CrystalDatabase(
        db_path=os.path.join(cfg.output_dir, "database.json"),
        cif_dir=os.path.join(cfg.output_dir, "structures")
    )

    # --- Resume detection ---------------------------------------------------
    # When cfg.resume is set and an existing database is found, skip cycle 0
    # and every cycle already present on disk, then continue from the next one.
    # The interrupted (partial) cycle's evaluations stay in the database and
    # feed the GP; we simply move on to the following cycle (per design).
    resume = bool(cfg.resume) and len(db) > 0
    start_cycle = 1
    best_energy_prev = None
    no_improve_count = 0
    if cfg.resume and len(db) == 0:
        logger.warning(f"resume=True but no existing database in {db.db_path} — "
                       "starting a fresh run from cycle 0")
    if resume:
        last_cycle = max(r['cycle'] for r in db.records)
        start_cycle = last_cycle + 1
        best_energy_prev, no_improve_count = _reconstruct_resume_state(db, start_cycle)
        br = db.best_record()
        logger.info("=" * 60)
        logger.info("RESUME — continuing an existing run")
        logger.info("=" * 60)
        logger.info(f"Loaded {len(db)} evaluated structures from {db.db_path}")
        logger.info(f"Last cycle on disk = {last_cycle}  →  continuing at cycle {start_cycle}")
        logger.info(f"Best so far: E={br['energy']:.4f} eV  rho={br['density']:.3f}  "
                    f"beta={br['beta']:.1f}°  V={br['volume']:.0f} Å³  (cycle {br['cycle']})")
        logger.info(f"Reconstructed no_improve_count = {no_improve_count}/"
                    f"{cfg.max_no_improve_cycles}")
        if start_cycle > cfg.num_cycles:
            logger.info(f"All {cfg.num_cycles} cycles already complete — "
                        "writing final report and exiting.")

    # Seed RNGs. On resume, offset the seed by the start cycle so that a
    # restarted run explores fresh structures instead of replaying the exact
    # same random stream that seed alone would regenerate.
    seed = cfg.random_seed + (start_cycle * 7919 if resume else 0)
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Molecule analysis
    mol_size = len(__import__('ase').io.read(cfg.molecule_file))
    logger.info(f"Molecule: {mol_size} atoms/molecule  MW={cfg.mol_weight:.2f} g/mol")
    logger.info(f"Volume range: [{cfg.volume_min():.0f}, {cfg.volume_max():.0f}] Å³")

    # The one object that knows how a structure becomes an energy. Built
    # once: an in-process backend would otherwise reload its potential for
    # every candidate.
    backend = get_backend(cfg, logger)
    logger.info(f"Energy backend: {backend.describe()}")
    if cfg.energy_sanity_max is not None:
        # Worth stating: the bound is calibrated for MACE-OFF total energies,
        # and a different potential's zero of energy can make it reject
        # everything silently.
        logger.info(f"  energies above {cfg.energy_sanity_max:.1f} eV count "
                    "as a failed relaxation (energySanityMax)")

    ref_bonds = get_reference_bonds(cfg.molecule_file,
                                     cfg.integrity_max_bond_ratio) \
                if cfg.check_molecular_integrity else []
    if ref_bonds:
        logger.info(f"Integrity check: {len(ref_bonds)} reference bonds")

    # CH-pi setup
    mol_ring_info = ch_pairs = None
    if cfg.use_chpi_optimizer and _CHPI_AVAILABLE:
        mol_ring_info, ch_pairs = analyze_molecule(cfg.molecule_file, logger)
        logger.info(f"CH-pi optimizer: ON  "
                    f"({len(mol_ring_info)} rings, {len(ch_pairs)} C-H pairs)")
    elif cfg.use_chpi_optimizer:
        logger.warning("CH-pi optimizer requested but RDKit not available — disabled")

    gp = GPSurrogate(kernel_type=cfg.gp_kernel, n_restarts=cfg.gp_n_restarts)
    suggestions = None

    # ====================================================================
    # Cycle 0: Initialization  (skipped when resuming)
    # ====================================================================
    if not resume:
        logger.info("=" * 60)
        logger.info("CYCLE 0 — Initialization (blind random generation)")
        logger.info("=" * 60)
        t0 = time.time()

        # Generate random structures
        logger.info(f"Generating {cfg.initial_structures} random structures...")
        structs = generate_structures(cfg.molecule_file, cfg, cfg.initial_structures,
                                      suggestions=None, logger=logger)

        # Load seeds
        seeds = load_seeds(cfg, mol_size, logger)
        if seeds:
            logger.info(f"Loaded {len(seeds)} seed/perturbation structures")
            structs.extend(seeds)

        logger.info(f"Total structures to evaluate: {len(structs)}")

        # Optional CH-pi optimization
        structs = _apply_chpi(structs, mol_ring_info, ch_pairs, mol_size, cfg, logger)

        # Evaluate
        n_ok = _evaluate_batch(structs, cfg, backend, cycle=0, batch_start=0,
                               ref_bonds=ref_bonds, mol_size=mol_size,
                               db=db, logger=logger)

        logger.info(f"Cycle 0 complete: {n_ok}/{len(structs)} OK  "
                    f"[{time.time()-t0:.0f}s]")
        if db.best_energy() is not None:
            br = db.best_record()
            logger.info(f"Best so far: E={br['energy']:.4f} eV  "
                        f"rho={br['density']:.3f}  beta={br['beta']:.1f}°")

    # ====================================================================
    # Cycles start_cycle .. num_cycles  (start_cycle = 1 unless resuming)
    # ====================================================================
    for cycle in range(start_cycle, cfg.num_cycles + 1):
        logger.info("=" * 60)
        logger.info(f"CYCLE {cycle}/{cfg.num_cycles} — Active Learning")
        logger.info("=" * 60)
        t0 = time.time()

        # Early stopping
        best_now = db.best_energy()
        if best_energy_prev is not None and best_now is not None:
            if abs(best_now - best_energy_prev) < 1e-4:
                no_improve_count += 1
                logger.info(f"No improvement ({no_improve_count}/"
                             f"{cfg.max_no_improve_cycles})")
                if no_improve_count >= cfg.max_no_improve_cycles:
                    logger.info("Convergence criterion met — stopping early")
                    break
            else:
                no_improve_count = 0
        best_energy_prev = best_now

        # --- Train GP ---
        if not cfg.use_gp:
            logger.info("GP disabled (useGP = false) — candidates generated "
                        "without EI bias")
            suggestions = None
        elif len(db) >= 5:
            logger.info(f"Training GP on {len(db)} structures...")
            try:
                gp.fit(db.records)
                suggestions = maximize_ei(gp, db.records, cfg, rng)
                gp_report(gp, suggestions, db.best_energy(), logger)
                # Save GP suggestions for this cycle
                _save_suggestions(suggestions, cycle, cfg.output_dir)
            except Exception as e:
                logger.warning(f"GP training failed: {e} — using random generation")
                suggestions = None
        else:
            logger.info("Not enough data for GP — random generation")
            suggestions = None

        # --- Generate new structures ---
        logger.info(f"Generating {cfg.structures_per_cycle} structures "
                    f"({'half biased' if suggestions else 'random'})...")
        structs = generate_structures(cfg.molecule_file, cfg,
                                      cfg.structures_per_cycle,
                                      suggestions=suggestions, logger=logger)

        # Use top-N from database as additional seeds for this cycle
        pool_cifs = _export_pool(db, cfg, cycle)
        if pool_cifs:
            cfg_tmp = _config_with_seeds(cfg, pool_cifs)
            pool_structs = load_seeds(cfg_tmp, mol_size, logger)
            if pool_structs:
                logger.info(f"Pool seeds: {len(pool_structs)} perturbations")
                structs.extend(pool_structs[:cfg.structures_per_cycle // 4])

        # CH-pi
        structs = _apply_chpi(structs, mol_ring_info, ch_pairs, mol_size, cfg, logger)

        # Evaluate
        batch_start = sum(1 for r in db.records if r['cycle'] < cycle)
        n_ok = _evaluate_batch(structs, cfg, backend, cycle=cycle,
                               batch_start=batch_start,
                               ref_bonds=ref_bonds, mol_size=mol_size,
                               db=db, logger=logger)

        logger.info(f"Cycle {cycle} complete: {n_ok}/{len(structs)} OK  "
                    f"[{time.time()-t0:.0f}s]")

        if db.best_energy() is not None:
            br = db.best_record()
            logger.info(f"Best so far: E={br['energy']:.4f} eV  "
                        f"rho={br['density']:.3f}  beta={br['beta']:.1f}°  "
                        f"V={br['volume']:.0f} Å³  (cycle {br['cycle']})")

        # Save per-cycle report
        _save_cycle_report(db, cycle, cfg.output_dir)

    # ====================================================================
    # Final summary
    # ====================================================================
    logger.info("=" * 60)
    logger.info("ACTIVE LEARNING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total structures evaluated: {len(db)}")

    top5 = db.top_n(5)
    logger.info("Top 5 structures by energy:")
    logger.info(f"  {'#':>3} {'E(eV)':>14} {'rho':>8} {'beta':>8} "
                f"{'V':>8} {'cycle':>6} source")
    for rank, r in enumerate(top5, 1):
        logger.info(f"  {rank:>3} {r['energy']:>14.4f} {r['density']:>8.3f} "
                    f"{r['beta']:>8.1f} {r['volume']:>8.0f} "
                    f"{r['cycle']:>6}  {r['source']}")

    _save_final_report(db, gp, cfg)
    logger.info(f"\nResults in: {cfg.output_dir}/")
    logger.info(f"Database:   {cfg.output_dir}/database.json")
    logger.info(f"Best CIF:   {top5[0]['cif_path']}")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _reconstruct_resume_state(db: CrystalDatabase, start_cycle: int):
    """
    Rebuild the early-stopping state (best_energy_prev, no_improve_count) as if
    the main loop had run uninterrupted up to (but not including) start_cycle.

    Mirrors the loop exactly: at cycle c the loop compares
        best_now = best energy over records with cycle <= c-1
    against best_energy_prev (the previous cycle's best_now, i.e. cycle <= c-2),
    incrementing no_improve_count when they match within 1e-4 and resetting it
    otherwise. Replaying that here makes a resumed run converge identically to
    one that was never interrupted.
    """
    def cumbest(c):
        vals = [r['energy'] for r in db.records if r['cycle'] <= c]
        return min(vals) if vals else None

    no_improve = 0
    prev = None
    for c in range(1, start_cycle):
        best_now = cumbest(c - 1)
        if prev is not None and best_now is not None:
            if abs(best_now - prev) < 1e-4:
                no_improve += 1
            else:
                no_improve = 0
        prev = best_now
    return prev, no_improve


def _save_suggestions(suggestions, cycle, output_dir):
    path = os.path.join(output_dir, f"cycle{cycle:03d}_suggestions.json")
    with open(path, 'w') as f:
        json.dump(suggestions, f, indent=2)


def _export_pool(db: CrystalDatabase, cfg: Config, cycle: int) -> List[str]:
    """Export top pool_size CIF files from the database for use as seeds."""
    top = db.top_n(cfg.pool_size)
    pool_dir = os.path.join(cfg.output_dir, f"pool_cycle{cycle:03d}")
    os.makedirs(pool_dir, exist_ok=True)
    cif_paths = []
    for r in top:
        if os.path.exists(r['cif_path']):
            dst = os.path.join(pool_dir, os.path.basename(r['cif_path']))
            shutil.copy(r['cif_path'], dst)
            cif_paths.append(dst)
    return cif_paths


def _config_with_seeds(cfg: Config, seed_cifs: List[str]) -> Config:
    """Create a temporary config copy pointing to a specific set of seed files."""
    import copy, tempfile
    tmp_cfg = copy.deepcopy(cfg)
    # Write seed paths to a temporary folder by symlinking
    tmp_dir = tempfile.mkdtemp(prefix="cryal_pool_")
    for cif in seed_cifs:
        dst = os.path.join(tmp_dir, os.path.basename(cif))
        shutil.copy(cif, dst)
    tmp_cfg.seeds_folder              = tmp_dir
    tmp_cfg.use_seeds_folder          = True
    tmp_cfg.num_perturbations_per_seed = max(2, cfg.num_perturbations_per_seed // 3)
    return tmp_cfg


def _save_cycle_report(db: CrystalDatabase, cycle: int, output_dir: str):
    """Save a JSON summary of the current cycle's results."""
    report = {
        'cycle':           cycle,
        'total_evaluated': len(db),
        'best_energy':     db.best_energy(),
        'best_record':     db.best_record(),
        'density_stats':   _stats([r['density'] for r in db.records]),
        'beta_stats':      _stats([r['beta'] for r in db.records]),
        'energy_stats':    _stats([r['energy'] for r in db.records]),
    }
    path = os.path.join(output_dir, f"cycle{cycle:03d}_report.json")
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)


def _save_final_report(db: CrystalDatabase, gp: GPSurrogate, cfg: Config):
    """Save the final JSON report."""
    report = {
        'total_evaluated':  len(db),
        'num_cycles_run':   max(r['cycle'] for r in db.records) if db.records else 0,
        'best_energy':      db.best_energy(),
        'top_10':           db.top_n(10),
        'gp_feature_importance': gp.feature_importance() if gp.gp else None,
        'config': {
            'molecule':   cfg.molecule_file,
            'Z':          cfg.Z,
            'density_min': cfg.density_min,
            'density_max': cfg.density_max,
            'space_groups': cfg.space_groups,
            'sg_weights':   cfg.sg_weights,
            'beta_distribution': cfg.beta_distribution,
        }
    }
    path = os.path.join(cfg.output_dir, "final_report.json")
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)


def _stats(values):
    if not values:
        return {}
    arr = np.array(values)
    return {'min': float(arr.min()), 'max': float(arr.max()),
            'mean': float(arr.mean()), 'median': float(np.median(arr))}
