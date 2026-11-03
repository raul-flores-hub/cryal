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
basin_hopping.py — basin hopping over molecular crystals
(`general.algorithm = "basin_hopping"`).

Basin hopping (Wales and Doye, 1997) walks from one relaxed minimum to the
next: perturb the current structure, relax it, and move there by the
Metropolis rule at a fictitious temperature. It is the undirected reference
for BACH, which the README calls basin hopping with jumps directed by a
surrogate; this is the same walk with the jumps left to chance.

Walkers
=======
One chain evaluates one candidate at a time, and a cycle's candidates are
relaxed together — on several machines with [parallel]. So `walkers` chains
run side by side, started from the `walkers` lowest-energy distinct
structures of cycle 0, and each cycle every walker makes its share of the
structures_per_cycle trial moves from its current minimum. When the cycle is
evaluated, each walker looks at its best trial and moves there if

    dE = (E_trial - E_current) / Z  <=  0,   or   u < exp(-dE / kT),

with u uniform in [0, 1) and kT = `temperature` in eV per molecule
(0.01 eV ≈ 1 kJ/mol). Taking the best of several trials is the batched form
of the walk; with one trial per walker it is the textbook one.

Moves are the GA's mutations (cryal/genetic.py), from one parent: strain of
the cell at constant volume, rotation and translation of some molecules,
chosen at random by `moves` weight. Children go through the same repair —
push rigid molecules apart, stretch a cell a molecule meets itself in,
expand — before they are relaxed like any other candidate.

State
=====
All of it is in the database: a trial's record carries its `walker`, its
`parents` (the walker's minimum it was made from) and, once the cycle is
settled, `accepted`. A walker is at its last accepted record, or at its
starting structure. So resuming needs nothing else, and a cycle interrupted
after its trials were evaluated but before they were settled is settled
when the run resumes.
"""

from typing import Dict, List

import numpy as np

from .config import BH_MOVES, Config
from .genetic import (MAX_ATTEMPTS, distinct_lowest, duplicate_settings,
                      load_parents, realize, rotate, strain, translate)
from .search import Strategy, run_search
from .structure_gen import generate_structures


def accept(dE: float, kT: float, u: float) -> bool:
    """Metropolis: downhill always, uphill with probability exp(-dE / kT)."""
    if dE <= 0:
        return True
    if kT <= 0:
        return False
    return u < np.exp(-dE / kT)


def starting_points(records: List[Dict], cfg: Config, mol=None) -> List[Dict]:
    """The walkers' starting structures: cycle 0's lowest distinct ones."""
    return distinct_lowest([r for r in records if r['cycle'] == 0], cfg.bh_walkers,
                           cfg, mol)


def positions(records: List[Dict], cfg: Config, mol=None,
              before_cycle=None) -> Dict[int, Dict]:
    """
    Where each walker is: its last accepted record (of the cycles before
    `before_cycle`, if given), or its starting structure.
    """
    here = dict(enumerate(starting_points(records, cfg, mol)))
    for r in sorted(records, key=lambda r: r['id']):
        if r.get('accepted') and r.get('walker') in here:
            if before_cycle is None or r['cycle'] < before_cycle:
                here[r['walker']] = r
    return here


def shares(n: int, walkers: int, cycle: int) -> List[int]:
    """n trials over the walkers, as evenly as can be; the remainder rotates."""
    base, extra = divmod(n, walkers)
    return [base + ((w - cycle) % walkers < extra) for w in range(walkers)]


class BasinHopping(Strategy):

    name = "basin_hopping"
    title = "basin hopping"

    def start(self, ctx):
        # A cycle whose trials were evaluated but not settled: the run
        # stopped between the two. Settle it now, before anything moves on.
        pending = sorted({r['cycle'] for r in ctx.db.records
                          if 'walker' in r and 'accepted' not in r})
        for cycle in pending:
            ctx.logger.info(f"Settling cycle {cycle}, evaluated before the run stopped")
            self.settle(ctx, cycle)

    def propose(self, ctx, cycle):
        cfg, logger, rng = ctx.cfg, ctx.logger, ctx.rng
        n = cfg.structures_per_cycle
        here = positions(ctx.db.records, cfg, ctx.mol)
        genomes = load_parents(list(here.values()), ctx.mol, logger)
        walkers = [w for w in sorted(here) if here[w]['id'] in genomes]
        out: List[Dict] = []
        if walkers:
            names = [m for m in BH_MOVES if cfg.bh_moves[m] > 0]
            p = np.array([cfg.bh_moves[m] for m in names])
            p /= p.sum()
            energies = [here[w]['energy'] for w in walkers]
            logger.info(f"Walkers: {len(walkers)} at E in [{min(energies):.4f}, "
                        f"{max(energies):.4f}] eV")
            failed: Dict[str, int] = {}
            for w, k in zip(walkers, shares(n, len(walkers), cycle)):
                parent = here[w]
                made = 0
                for _ in range(k * MAX_ATTEMPTS):
                    if made >= k:
                        break
                    move = names[rng.choice(len(names), p=p)]
                    g = genomes[parent['id']]
                    if move == "strain":
                        child = strain(g, cfg.bh_strain_sigma, rng)
                    elif move == "rotation":
                        child = rotate(g, cfg.bh_fraction_mutated, cfg.bh_max_rotation, rng)
                    else:
                        child = translate(g, cfg.bh_fraction_mutated,
                                          cfg.bh_max_translation, rng)
                    atoms, why = realize(child, ctx.mol, cfg)
                    if atoms is None:
                        failed[why] = failed.get(why, 0) + 1
                        continue
                    out.append({'atoms': atoms, 'space_group': None,
                                'source': f'hop_{move}', 'parents': [parent['id']],
                                'walker': w})
                    made += 1
            if failed and len(out) < n:
                logger.warning(f"  basin hopping: {n - len(out)} trial(s) not made ("
                               + ", ".join(f"{r} x{c}" for r, c in failed.items()) + ")")
        if len(out) < n:
            # No walker could move (nothing readable to start from), or some
            # trials could not be made: the budget is spent on random
            # structures, which belong to no walker.
            out += generate_structures(cfg.molecule_file, cfg, n - len(out),
                                       suggestions=None, logger=logger)
        return out

    def after_cycle(self, ctx, cycle):
        self.settle(ctx, cycle)

    def settle(self, ctx, cycle):
        """Each walker of `cycle` moves to its best trial, or stays, by Metropolis."""
        cfg, db = ctx.cfg, ctx.db
        trials = [r for r in db.records if r['cycle'] == cycle and 'walker' in r]
        if not trials:
            return
        here = positions(db.records, cfg, ctx.mol, before_cycle=cycle)
        moved = 0
        walkers = sorted({r['walker'] for r in trials})
        for w in walkers:
            group = [r for r in trials if r['walker'] == w]
            best = min(group, key=lambda r: r['energy'])
            current = here.get(w)
            ok = True if current is None else accept(
                (best['energy'] - current['energy']) / cfg.Z, cfg.bh_temperature,
                ctx.rng.random())
            for r in group:
                r['accepted'] = bool(ok and r is best)
            moved += ok
        db.save()
        ctx.logger.info(f"  basin hopping: {moved}/{len(walkers)} walkers moved "
                        f"(kT = {cfg.bh_temperature:g} eV/molecule)")

    def settings(self, cfg):
        return {
            'walkers':          cfg.bh_walkers,
            'temperature':      cfg.bh_temperature,
            'moves':            cfg.bh_moves,
            'strain_sigma':     cfg.bh_strain_sigma,
            'max_rotation':     cfg.bh_max_rotation,
            'max_translation':  cfg.bh_max_translation,
            'fraction_mutated': cfg.bh_fraction_mutated,
            'duplicates':       duplicate_settings(cfg),
        }


def run(cfg: Config):
    """Execute basin hopping (cryal/search.py runs the loop)."""
    run_search(cfg, BasinHopping())
