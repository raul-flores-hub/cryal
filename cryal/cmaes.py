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
cmaes.py — the covariance matrix adaptation evolution strategy over molecular
crystals (`general.algorithm = "cmaes"`).

CMA-ES (Hansen and Ostermeier, 2001) samples each generation from a
multivariate normal, ranks the samples, moves the mean toward the best and
adapts the step size and the covariance to the directions that paid off: in
effect it learns, generation by generation, which combinations of variables
the landscape is sensitive to. The strategy below is the standard
(mu/mu_w, lambda)-CMA-ES of Hansen's tutorial (arXiv:1604.00772), its
default parameters derived from the dimension and lambda, with no tuning.

The search space
================
A crystal is the genome of cryal/genetic.py — a cell and Z rigid molecules —
written as a vector around a starting structure:

  - the cell, as the six entries of its lower-triangular matrix (a along x,
    b in the xy plane), with logarithms on the diagonal, so that every
    vector is a valid, right-handed cell: (ln a_x, ln b_y, ln c_z, b_x/a_x,
    c_x/a_x, c_y/b_y);
  - each molecular centre, in fractional coordinates;
  - each molecule's orientation, as a rotation vector applied to the shape
    it has in the starting structure (which also gives its conformation).

Each block is scaled so that one unit means a comparable change — 5 % of
strain (`strain_scale`), half an ångström of displacement (`shift_scale`),
ten degrees of turn (`turn_scale`) — and the strategy works in those units,
starting at the starting structure with step size `sigma`. With Z = 4 the
space has 30 dimensions.

A sample is decoded, repaired as the GA repairs its children (push rigid
molecules apart, stretch a cell a molecule meets itself in, expand), and
relaxed; its fitness is the relaxed energy, while the sample stays where it
was drawn — the Baldwinian form, the one that keeps CMA-ES's assumptions. A
sample that cannot be built is drawn again; one whose relaxation fails is
left out of that generation's ranking.

Generations, instances, restarts
================================
One cycle is one generation of lambda = structures_per_cycle samples. An
instance starts from cycle 0's lowest structure, and restarts — from the
lowest distinct structure not started from yet — when its step size has
collapsed (sigma times the largest axis of the distribution below
`restart_sigma`) or its best has not improved for `stall` generations.

The budget matters more here than for the other searches: CMA-ES learns
its covariance over generations, and the 10 cycles of the default budget
leave little room to learn it in 30 dimensions. More cycles of fewer
structures suit it better, at the same cost.

State
=====
Every sample's record carries `cma`: the instance it belongs to and the
sample itself, in the strategy's units. The strategy's state — mean, step
size, covariance, evolution paths, restarts — is rebuilt from the database
by replaying the updates generation by generation, exactly, so a resumed
run is in the state an uninterrupted one would have been.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from ase.io import read
from scipy.spatial.transform import Rotation

from .config import Config
from .genetic import (MAX_ATTEMPTS, Genome, distinct_lowest, duplicate_settings,
                      genome_from_atoms, realize, standardize)
from .search import Strategy, run_search
from .structure_gen import generate_structures


# ---------------------------------------------------------------------------
# The strategy, on vectors
# ---------------------------------------------------------------------------

class CMA:
    """(mu/mu_w, lambda)-CMA-ES, minimizing. Hansen's tutorial, section C."""

    def __init__(self, mean: np.ndarray, sigma: float, lam: int):
        n = len(mean)
        self.n, self.lam = n, lam
        self.mean = np.array(mean, dtype=float)
        self.sigma = float(sigma)
        self.C = np.eye(n)
        self.ps = np.zeros(n)
        self.pc = np.zeros(n)
        self.generation = 0
        self.chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self._weights(lam // 2)

    def _weights(self, mu: int):
        n = self.n
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        m = self.mueff
        self.cs = (m + 2) / (n + m + 5)
        self.ds = 1 + 2 * max(0.0, np.sqrt((m - 1) / (n + 1)) - 1) + self.cs
        self.cc = (4 + m / n) / (n + 4 + 2 * m / n)
        self.c1 = 2 / ((n + 1.3) ** 2 + m)
        self.cmu = min(1 - self.c1, 2 * (m - 2 + 1 / m) / ((n + 2) ** 2 + m))

    def axes(self) -> Tuple[np.ndarray, np.ndarray]:
        """(B, D): C = B diag(D^2) B^T."""
        C = 0.5 * (self.C + self.C.T)
        vals, B = np.linalg.eigh(C)
        return B, np.sqrt(np.maximum(vals, 1e-20))

    def ask(self, rng: np.random.Generator) -> np.ndarray:
        """One sample from N(mean, sigma^2 C)."""
        B, D = self.axes()
        return self.mean + self.sigma * (B @ (D * rng.standard_normal(self.n)))

    def tell(self, samples: List[np.ndarray], fitness: List[float]) -> None:
        """
        Update from this generation's evaluated samples. The best mu of them
        are recombined; when fewer than lambda // 2 came back, the weights
        are those of the number that did. With none, nothing changes but the
        generation count.
        """
        self.generation += 1
        if not samples:
            return
        mu_full = self.lam // 2
        mu = min(mu_full, len(samples))
        self._weights(mu)
        order = np.argsort(fitness)[:mu]
        Y = (np.array([samples[i] for i in order]) - self.mean) / self.sigma
        yw = self.w @ Y
        B, D = self.axes()
        C_inv_sqrt = B @ np.diag(1 / D) @ B.T
        self.mean = self.mean + self.sigma * yw
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) \
            * (C_inv_sqrt @ yw)
        norm_ps = np.linalg.norm(self.ps)
        hs = norm_ps / np.sqrt(1 - (1 - self.cs) ** (2 * self.generation)) \
            < (1.4 + 2 / (self.n + 1)) * self.chi_n
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        rank_mu = (self.w[:, None] * Y).T @ Y
        self.C = ((1 - self.c1 - self.cmu + (1 - hs) * self.c1 * self.cc * (2 - self.cc)) * self.C
                  + self.c1 * np.outer(self.pc, self.pc) + self.cmu * rank_mu)
        self.sigma *= np.exp((self.cs / self.ds) * (norm_ps / self.chi_n - 1))
        self._weights(mu_full)

    def spread(self) -> float:
        """sigma times the longest axis of the distribution."""
        return float(self.sigma * self.axes()[1].max())


# ---------------------------------------------------------------------------
# Crystals as vectors
# ---------------------------------------------------------------------------

def cell_to_u(L: np.ndarray) -> np.ndarray:
    """Lower-triangular cell (rows a, b, c) -> six unconstrained numbers."""
    ax, bx, by, cx, cy, cz = L[0, 0], L[1, 0], L[1, 1], L[2, 0], L[2, 1], L[2, 2]
    return np.array([np.log(ax), np.log(by), np.log(cz), bx / ax, cx / ax, cy / by])


def u_to_cell(u: np.ndarray) -> np.ndarray:
    ax, by, cz = np.exp(u[:3])
    return np.array([[ax, 0, 0], [u[3] * ax, by, 0], [u[4] * ax, u[5] * by, cz]])


class Encoding:
    """
    The vector a sample is, around one starting structure: x = 0 is the
    starting structure itself. See the module docstring for the blocks.
    """

    def __init__(self, start: Genome, cfg: Config):
        self.start = start
        self.Z = start.Z
        self.u0 = cell_to_u(start.cell)
        lengths = np.linalg.norm(start.cell, axis=1)
        # The shift scale is in Å; in fractional units it depends on the axis.
        self.scale = np.concatenate([
            np.full(6, cfg.cma_strain_scale),
            np.tile(cfg.cma_shift_scale / lengths, self.Z),
            np.full(3 * self.Z, np.radians(cfg.cma_turn_scale)),
        ])
        self.n = len(self.scale)

    def decode(self, x: np.ndarray) -> Genome:
        p = x * self.scale
        Z = self.Z
        cell = u_to_cell(self.u0 + p[:6])
        frac = (self.start.frac + p[6:6 + 3 * Z].reshape(Z, 3)) % 1.0
        turns = p[6 + 3 * Z:].reshape(Z, 3)
        shapes = np.array([self.start.shapes[m] @ Rotation.from_rotvec(turns[m]).as_matrix().T
                           for m in range(Z)])
        return Genome(cell, frac, shapes)


# ---------------------------------------------------------------------------
# Replaying the run
# ---------------------------------------------------------------------------

class Instance:
    """One CMA-ES run from one starting record, and what it has done."""

    def __init__(self, number: int, start: Dict, n: int, cfg: Config):
        self.number = number
        self.start = start
        self.cma = CMA(np.zeros(n), cfg.cma_sigma, cfg.structures_per_cycle)
        self.best = start['energy']
        self.stalled = 0


def replay(records: List[Dict], cfg: Config, mol, up_to: int) -> Tuple[Optional["Instance"], List[int]]:
    """
    The instance that proposes generation `up_to`, in the state it reaches
    after every earlier generation, and the ids of every record started from.
    Deterministic from the database: each step is the one the run took.
    """
    starts: List[int] = []
    inst = _restart(records, cfg, mol, 1, starts, 0)
    if inst is None:
        return None, starts
    for cycle in range(1, up_to):
        mine = [r for r in records if r['cycle'] == cycle and r.get('cma', {}).get('instance')
                == inst.number]
        inst.cma.tell([np.array(r['cma']['x']) for r in mine], [r['energy'] for r in mine])
        best = min([r['energy'] for r in mine], default=np.inf)
        if best < inst.best - 1e-6:
            inst.best, inst.stalled = best, 0
        else:
            inst.stalled += 1
        if inst.cma.spread() < cfg.cma_restart_sigma or inst.stalled >= cfg.cma_stall:
            nxt = _restart(records, cfg, mol, cycle + 1, starts, inst.number)
            if nxt is not None:
                inst = nxt
    return inst, starts


def _restart(records, cfg, mol, cycle, starts, last_number) -> Optional[Instance]:
    """A new instance from the lowest distinct record before `cycle` not started from."""
    earlier = [r for r in records if r['cycle'] < cycle]
    for r in distinct_lowest(earlier, len(earlier), cfg, mol):
        if r['id'] not in starts:
            starts.append(r['id'])
            n = 6 + 6 * cfg.Z
            return Instance(last_number + 1, r, n, cfg)
    return None


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------

class CMAES(Strategy):

    name = "cmaes"
    title = "CMA-ES"
    unit = "GENERATION"

    def start(self, ctx):
        self.mol = ctx.mol

    def propose(self, ctx, cycle):
        cfg, logger, rng = ctx.cfg, ctx.logger, ctx.rng
        n_want = cfg.structures_per_cycle
        inst, _ = replay(ctx.db.records, cfg, ctx.mol, cycle)
        out: List[Dict] = []
        start = self.genome(inst.start) if inst is not None else None
        if start is not None:
            enc = Encoding(start, cfg)
            logger.info(f"CMA-ES instance {inst.number} from record {inst.start['id']}: "
                        f"generation {inst.cma.generation + 1}, sigma {inst.cma.sigma:.3f}, "
                        f"spread {inst.cma.spread():.3f}, best E={inst.best:.4f} eV")
            failed: Dict[str, int] = {}
            for _ in range(n_want * MAX_ATTEMPTS):
                if len(out) >= n_want:
                    break
                x = inst.cma.ask(rng)
                try:
                    atoms, why = realize(enc.decode(x), ctx.mol, cfg)
                except Exception as exc:
                    atoms, why = None, type(exc).__name__
                if atoms is None:
                    failed[why] = failed.get(why, 0) + 1
                    continue
                out.append({'atoms': atoms, 'space_group': None, 'source': 'cmaes',
                            'parents': [inst.start['id']],
                            'cma': {'instance': inst.number, 'x': [float(v) for v in x]}})
            if failed:
                logger.info("  CMA-ES: resampled " + ", ".join(
                    f"{k} x{c}" for k, c in failed.items()))
        if len(out) < n_want:
            out += generate_structures(cfg.molecule_file, cfg, n_want - len(out),
                                       suggestions=None, logger=logger)
        return out

    def genome(self, record):
        try:
            g = genome_from_atoms(read(record['cif_path']), self.mol)
        except Exception:
            return None
        return None if g is None else standardize(g)

    def settings(self, cfg):
        return {
            'lambda':         cfg.structures_per_cycle,
            'sigma':          cfg.cma_sigma,
            'strain_scale':   cfg.cma_strain_scale,
            'shift_scale':    cfg.cma_shift_scale,
            'turn_scale':     cfg.cma_turn_scale,
            'restart_sigma':  cfg.cma_restart_sigma,
            'stall':          cfg.cma_stall,
            'duplicates':     duplicate_settings(cfg),
        }


def run(cfg: Config):
    """Execute CMA-ES (cryal/search.py runs the loop)."""
    run_search(cfg, CMAES())
