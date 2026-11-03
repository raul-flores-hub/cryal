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
search.py — the loop every search other than BACH runs in.

BACH keeps its own loop in active_learning.run, the one the article was
produced with. The alternatives (general.algorithm = "random", "ga",
"basin_hopping") share this one, and differ only in a Strategy: how the next
cycle's candidates are proposed, and what, if anything, happens once they
have been evaluated. Everything else is common, and the same as BACH's:

  - cycle 0 is BACH's cycle 0: initial_structures from the generator, plus
    the seeds of [seeds], evaluated the same way;
  - every candidate goes through the same CH–π step, backend, gate,
    integrity check and parallel pool, into the same database;
  - [active_learning] sets the budget (cycles, structures_per_cycle) and the
    stopping rule (max_no_improve_cycles), and the early-stopping state is
    rebuilt on resume exactly as BACH rebuilds it.

So a comparison between searches run from one cryal.toml with one key changed
compares only the way candidates are proposed.

Each cycle's log lists what every source of candidates contributed (see
source_family), and final_report.json keeps the totals.
"""

import json
import os
import random
import time
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np

from .active_learning import (CrystalDatabase, _apply_chpi, _evaluate_batch,
                              _reconstruct_resume_state, _save_cycle_report)
from .backends import get_backend, get_reference_bonds
from .config import GA_OPERATORS, Config, print_config_summary
from .parallel import build_pool
from .structure_gen import generate_structures, load_seeds
from .utils import setup_logger

try:
    from .chpi_optimizer import analyze_molecule
    _CHPI_AVAILABLE = True
except ImportError:
    _CHPI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Where a candidate came from
# ---------------------------------------------------------------------------

#: The families of source, in the order the reports list them.
FAMILIES = ("biased", "random", "perturb", "seed") + GA_OPERATORS + ("hop", "pso", "other")


def source_family(source: str) -> str:
    """
    Where a record came from, from its source: a GA operator, a basin-hopping
    move ('hop'), a particle-swarm step ('pso'), or for the generator's structures 'biased' (toward a GP
    suggestion), 'random', 'perturb' (a perturbed seed or, in BACH from cycle
    1 on, a perturbed incumbent of the pool) and 'seed'.
    """
    if source.startswith('ga_'):
        for op in GA_OPERATORS:
            if source.startswith(f'ga_{op}'):
                return op
    if source.startswith('hop_'):
        return 'hop'
    if source == 'pso':
        return 'pso'
    for family in ("biased", "random", "perturb", "seed"):
        if source.startswith(family):
            return family
    return 'other'


def source_stats(records: List[Dict], cycle: Optional[int] = None) -> Dict[str, Dict]:
    """Per source family: candidates kept and their best energy (one cycle or all)."""
    stats: Dict[str, Dict] = {}
    for r in records:
        if cycle is not None and r['cycle'] != cycle:
            continue
        s = stats.setdefault(source_family(r['source']), {'kept': 0, 'best_energy': None})
        s['kept'] += 1
        if s['best_energy'] is None or r['energy'] < s['best_energy']:
            s['best_energy'] = r['energy']
    return {f: stats[f] for f in FAMILIES if f in stats}


def report_cycle(records: List[Dict], cycle: int, best_before, logger):
    """What each source contributed this cycle, and which one found a new best."""
    for f, s in source_stats(records, cycle).items():
        logger.info(f"    {f:<12} kept {s['kept']:>3}   best E={s['best_energy']:.4f} eV")
    if not records:
        return
    br = min(records, key=lambda r: r['energy'])
    if br['cycle'] == cycle and (best_before is None or br['energy'] < best_before):
        parents = f" (parents {br['parents']})" if br.get('parents') else ""
        logger.info(f"  New best from {source_family(br['source'])}{parents}")


# ---------------------------------------------------------------------------
# What a search is
# ---------------------------------------------------------------------------

class Strategy:
    """
    How one search proposes candidates. Subclasses set `name` (the value of
    general.algorithm), `title` (for the log) and implement propose().

    The context passed to every method carries cfg, db, logger, rng (a numpy
    Generator seeded from general.random_seed), mol (cryal.genetic.Molecule)
    and mol_size.
    """

    name = ""
    title = ""
    unit = "CYCLE"          # what the log calls one round

    def start(self, ctx) -> None:
        """Once, after setup and before the first cycle this run evaluates."""

    def propose(self, ctx, cycle: int) -> List[Dict]:
        """The candidates of `cycle`, as dicts the evaluation pipeline takes."""
        raise NotImplementedError

    def after_cycle(self, ctx, cycle: int) -> None:
        """Once a cycle's candidates are in the database."""

    def settings(self, cfg: Config) -> Dict:
        """This search's own settings, for final_report.json."""
        return {}


def _final_report(db: CrystalDatabase, cfg: Config, strategy: Strategy):
    report = {
        'algorithm':       strategy.name,
        'total_evaluated': len(db),
        'num_cycles_run':  max(r['cycle'] for r in db.records) if db.records else 0,
        'best_energy':     db.best_energy(),
        'top_10':          db.top_n(10),
        'source_stats':    source_stats(db.records),
        'config': dict({
            'molecule':          cfg.molecule_file,
            'Z':                 cfg.Z,
            'density_min':       cfg.density_min,
            'density_max':       cfg.density_max,
            'space_groups':      cfg.space_groups,
            'sg_weights':        cfg.sg_weights,
            'structures_per_cycle': cfg.structures_per_cycle,
        }, **strategy.settings(cfg)),
    }
    with open(os.path.join(cfg.output_dir, "final_report.json"), 'w') as f:
        json.dump(report, f, indent=2)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def run_search(cfg: Config, strategy: Strategy):
    """Run one search. Mirrors active_learning.run in everything but proposing."""
    from .genetic import Molecule          # genetic imports this module

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "steps"), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "structures"), exist_ok=True)

    logger = setup_logger("cryal", os.path.join(cfg.output_dir, "cryal.log"),
                          cfg.log_level)
    print_config_summary(cfg, logger)

    db = CrystalDatabase(db_path=os.path.join(cfg.output_dir, "database.json"),
                         cif_dir=os.path.join(cfg.output_dir, "structures"))
    unit = strategy.unit.lower()

    # --- Resume: the same rule as BACH.
    resume = bool(cfg.resume) and len(db) > 0
    start_cycle = 1
    best_energy_prev = None
    no_improve_count = 0
    if cfg.resume and len(db) == 0:
        logger.warning(f"resume=True but no existing database in {db.db_path} — "
                       f"starting a fresh run from {unit} 0")
    if resume:
        last_cycle = max(r['cycle'] for r in db.records)
        start_cycle = last_cycle + 1
        best_energy_prev, no_improve_count = _reconstruct_resume_state(db, start_cycle)
        logger.info(f"RESUME — {len(db)} structures loaded; continuing at "
                    f"{unit} {start_cycle} (no_improve_count = {no_improve_count})")

    # Seeded exactly as active_learning.run is. That does not make cycle 0
    # the same structures as BACH's: PyXtal places molecules with a generator
    # of its own, seeded from the operating system (INSTALL.md, "Version drift").
    seed = cfg.random_seed + (start_cycle * 7919 if resume else 0)
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    mol = Molecule.from_file(cfg.molecule_file)
    logger.info(f"Molecule: {mol.size} atoms/molecule  MW={cfg.mol_weight:.2f} g/mol")
    logger.info(f"Volume range: [{cfg.volume_min():.0f}, {cfg.volume_max():.0f}] Å³")

    backend = get_backend(cfg, logger)
    logger.info(f"Energy backend: {backend.describe()}")
    ref_bonds = get_reference_bonds(cfg.molecule_file, cfg.integrity_max_bond_ratio) \
        if cfg.check_molecular_integrity else []
    pool = build_pool(cfg, backend, logger)
    ctx = SimpleNamespace(cfg=cfg, db=db, logger=logger, rng=rng, mol=mol,
                          mol_size=mol.size)

    try:
        mol_ring_info = ch_pairs = None
        if cfg.use_chpi_optimizer and _CHPI_AVAILABLE:
            mol_ring_info, ch_pairs = analyze_molecule(cfg.molecule_file, logger)
        elif cfg.use_chpi_optimizer:
            logger.warning("CH-pi optimizer requested but RDKit not available — disabled")

        def evaluate(structs, cycle):
            structs = _apply_chpi(structs, mol_ring_info, ch_pairs, mol.size, cfg, logger)
            batch_start = sum(1 for r in db.records if r['cycle'] < cycle)
            return _evaluate_batch(structs, cfg, backend, cycle=cycle,
                                   batch_start=batch_start, ref_bonds=ref_bonds,
                                   mol_size=mol.size, db=db, logger=logger, pool=pool)

        # ================================================================
        # Cycle 0: BACH's cycle 0
        # ================================================================
        if not resume:
            logger.info("=" * 60)
            logger.info(f"{strategy.unit} 0 — initial structures (blind random generation)")
            logger.info("=" * 60)
            t0 = time.time()
            structs = generate_structures(cfg.molecule_file, cfg, cfg.initial_structures,
                                          suggestions=None, logger=logger)
            seeds = load_seeds(cfg, mol.size, logger)
            if seeds:
                logger.info(f"Loaded {len(seeds)} seed/perturbation structures")
                structs.extend(seeds)
            n_ok = evaluate(structs, 0)
            logger.info(f"{strategy.unit.capitalize()} 0 complete: "
                        f"{n_ok}/{len(structs)} OK  [{time.time()-t0:.0f}s]")

        strategy.start(ctx)

        # ================================================================
        # Cycles start_cycle .. num_cycles
        # ================================================================
        for cycle in range(start_cycle, cfg.num_cycles + 1):
            logger.info("=" * 60)
            logger.info(f"{strategy.unit} {cycle}/{cfg.num_cycles} — {strategy.title}")
            logger.info("=" * 60)
            t0 = time.time()

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

            structs = strategy.propose(ctx, cycle)
            n_ok = evaluate(structs, cycle)
            logger.info(f"{strategy.unit.capitalize()} {cycle} complete: "
                        f"{n_ok}/{len(structs)} OK  [{time.time()-t0:.0f}s]")
            strategy.after_cycle(ctx, cycle)
            report_cycle(db.records, cycle, best_now, logger)
            if db.best_energy() is not None:
                br = db.best_record()
                logger.info(f"Best so far: E={br['energy']:.4f} eV  "
                            f"rho={br['density']:.3f}  V={br['volume']:.0f} Å³  "
                            f"({unit} {br['cycle']}, {source_family(br['source'])})")
            _save_cycle_report(db, cycle, cfg.output_dir)

        logger.info("=" * 60)
        logger.info(f"{strategy.title.upper()} COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total structures evaluated: {len(db)}")
        for f, s in source_stats(db.records).items():
            logger.info(f"  {f:<12} kept {s['kept']:>4}   best E={s['best_energy']:.4f} eV")
        _final_report(db, cfg, strategy)
        if db.records:
            logger.info(f"Best CIF:   {db.best_record()['cif_path']}")
    finally:
        if pool is not None:
            pool.close()


# ---------------------------------------------------------------------------
# Pure random search
# ---------------------------------------------------------------------------

class RandomSearch(Strategy):
    """
    Generate, relax, keep the best: every cycle is structures_per_cycle fresh
    structures from the generator, as in cycle 0, and nothing learnt from the
    ones before (AIRSS-like). Not the same as gp.enabled = false, which still
    adds BACH's perturbed incumbents of the pool to every cycle. This is the
    baseline any other search has to beat for its guidance to mean anything.
    """

    name = "random"
    title = "pure random search"

    def propose(self, ctx, cycle):
        return generate_structures(ctx.cfg.molecule_file, ctx.cfg,
                                   ctx.cfg.structures_per_cycle,
                                   suggestions=None, logger=ctx.logger)


def run_random(cfg: Config):
    run_search(cfg, RandomSearch())
