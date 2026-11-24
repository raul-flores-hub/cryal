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


"""--resume completes the cycle a run was interrupted in.

A cycle's candidates are written to cycle_in_progress.json before any is
evaluated (CyclePlan), and each outcome as it arrives. A resumed run
evaluates the candidates without one and closes the cycle; finished cycles
are listed in cycles.json. A run interrupted here is simulated by a backend
that raises KeyboardInterrupt on its n-th relaxation.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from cryal import active_learning as AL
from cryal import basin_hopping as BH
from cryal import search as S
from cryal.active_learning import (LEDGER_FILE, PLAN_FILE, CrystalDatabase,
                                   CyclePlan, _evaluate_batch)
from cryal.backends import get_backend
from cryal.structure_gen import generate_structures
from tests.test_genetic import BENZENE, run_config
from tests.test_search import reset_logger


class Interrupting:
    """A backend that stops the run on its n-th evaluation, before relaxing it."""

    def __init__(self, inner, stop_at):
        self.inner, self.stop_at, self.calls = inner, stop_at, 0

    def evaluate(self, *args, **kwargs):
        self.calls += 1
        if self.calls == self.stop_at:
            raise KeyboardInterrupt
        return self.inner.evaluate(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def interrupted(run, cfg, stop_at):
    """Run until the backend's stop_at-th evaluation, as a power cut would."""
    def backend(c, logger):
        return Interrupting(get_backend(c, logger), stop_at)
    with mock.patch.object(AL, "get_backend", backend), \
            mock.patch.object(S, "get_backend", backend):
        try:
            run(cfg)
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("the run was not interrupted")
    reset_logger()


class ResumeCase(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cryal_resume_")
        self.out = os.path.join(self.tmp, "run")
        reset_logger()

    def tearDown(self):
        reset_logger()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def records(self):
        with open(os.path.join(self.out, "database.json")) as f:
            return json.load(f)

    def plan(self):
        with open(os.path.join(self.out, PLAN_FILE)) as f:
            return json.load(f)

    def ledger(self):
        with open(os.path.join(self.out, LEDGER_FILE)) as f:
            return json.load(f)

    def resume(self, run, **kw):
        run(run_config(self.out, log_level="WARNING", resume=True, **kw))
        reset_logger()

    def check_completed(self, cycle, plan_before):
        """The cycle holds what it planned, each candidate once, nothing else."""
        self.assertFalse(os.path.exists(os.path.join(self.out, PLAN_FILE)))
        recs = self.records()
        steps = [os.path.basename(r['cif_path']) for r in recs]
        self.assertEqual(len(steps), len(set(steps)))           # none twice
        done = {e['cycle']: e for e in self.ledger()}
        self.assertEqual(done[cycle]['planned'], len(plan_before['candidates']))
        self.assertEqual(done[cycle]['kept'], sum(r['cycle'] == cycle for r in recs))
        sources = sorted(r['source'] for r in recs if r['cycle'] == cycle)
        planned = sorted(c['source'] for c in plan_before['candidates'])
        self.assertTrue(set(sources) <= set(planned))
        return recs


class TestSerial(ResumeCase):

    def test_random_search_completes_the_interrupted_cycle(self):
        # Cycle 0 has 6 candidates; stop on the 4th of cycle 1.
        interrupted(S.run_random, run_config(self.out, log_level="WARNING"), 6 + 4)
        plan = self.plan()
        self.assertEqual(plan['cycle'], 1)
        self.assertEqual(sum(o is not None for o in plan['outcomes']), 3)
        before = self.records()
        self.assertEqual([e['cycle'] for e in self.ledger()], [0])

        self.resume(S.run_random)
        recs = self.check_completed(1, plan)
        self.assertEqual(recs[:len(before)], before)            # nothing rewritten
        self.assertEqual([e['cycle'] for e in self.ledger()], [0, 1, 2])
        self.assertEqual(sum(r['cycle'] == 1 for r in recs), 6)

    def test_an_interrupted_cycle_0_is_completed(self):
        interrupted(S.run_random, run_config(self.out, log_level="WARNING"), 3)
        plan = self.plan()
        self.assertEqual(plan['cycle'], 0)
        self.resume(S.run_random)
        self.check_completed(0, plan)
        self.assertEqual([e['cycle'] for e in self.ledger()], [0, 1, 2])

    def test_bach_completes_the_interrupted_cycle(self):
        interrupted(AL.run, run_config(self.out, log_level="WARNING"), 6 + 2)
        plan = self.plan()
        self.assertEqual(plan['cycle'], 1)
        n = len(plan['candidates'])
        self.resume(AL.run)
        recs = self.check_completed(1, plan)
        self.assertEqual(sum(r['cycle'] == 1 for r in recs), n)
        self.assertTrue(os.path.exists(os.path.join(self.out, "cycle001_report.json")))
        self.assertEqual([e['cycle'] for e in self.ledger()], [0, 1, 2])

    def test_basin_hopping_settles_the_completed_cycle_once(self):
        cfg = run_config(self.out, log_level="WARNING", bh_walkers=3)
        interrupted(BH.run, cfg, 6 + 4)
        plan = self.plan()
        # The trials evaluated before the stop are not settled at start().
        self.resume(BH.run, bh_walkers=3)
        recs = self.check_completed(1, plan)
        trials = [r for r in recs if 'walker' in r]
        self.assertTrue(trials and all('accepted' in r for r in trials))
        for cycle in (1, 2):
            moved = [r for r in trials if r['cycle'] == cycle and r['accepted']]
            self.assertLessEqual(len({r['walker'] for r in moved}), 3)
            self.assertEqual(len(moved), len({r['walker'] for r in moved}))

    def test_a_run_from_before_the_ledger_keeps_the_old_rule(self):
        S.run_random(run_config(self.out, log_level="WARNING", num_cycles=1))
        reset_logger()
        os.remove(os.path.join(self.out, LEDGER_FILE))
        with self.assertLogs("cryal", level="WARNING") as logs:
            S.run_random(run_config(self.out, log_level="WARNING", resume=True))
        reset_logger()
        self.assertTrue(any(LEDGER_FILE in m for m in logs.output))
        self.assertEqual({r['cycle'] for r in self.records()}, {0, 1, 2})

    def test_a_fresh_run_discards_a_stale_plan(self):
        interrupted(S.run_random, run_config(self.out, log_level="WARNING"), 3)
        # Cycle 0 never finished, so there is no ledger yet.
        shutil.rmtree(os.path.join(self.out, "structures"))
        os.remove(os.path.join(self.out, "database.json"))
        S.run_random(run_config(self.out, log_level="WARNING"))
        reset_logger()
        self.assertFalse(os.path.exists(os.path.join(self.out, PLAN_FILE)))
        self.assertEqual([e['cycle'] for e in self.ledger()], [0, 1, 2])


class FakePool:
    """Relaxes with a backend in this thread; records which tasks it got."""

    n_slots = 2

    def __init__(self, backend):
        self.backend, self.seen = backend, []

    def describe(self):
        return "fake"

    def evaluate_all(self, tasks, ref_bonds, mol_size, on_result=None):
        out = []
        for index, step_dir, atoms in tasks:
            self.seen.append(index)
            relaxed, energy = self.backend.evaluate(atoms, step_dir, ref_bonds, mol_size)
            on_result(index, relaxed, energy, "")
            out.append((index, relaxed, energy))
        return out


class TestParallelBatch(ResumeCase):

    def test_only_unevaluated_candidates_reach_the_pool(self):
        import logging
        cfg = run_config(self.out, log_level="WARNING")
        os.makedirs(os.path.join(self.out, "steps"), exist_ok=True)
        logger = logging.getLogger("cryal.test")
        backend = get_backend(cfg, logger)
        db = CrystalDatabase(os.path.join(self.out, "database.json"),
                             os.path.join(self.out, "structures"))
        structs = generate_structures(cfg.molecule_file, cfg, 4, suggestions=None,
                                      logger=logger)
        plan = CyclePlan.create(self.out, 0, 0, structs)
        # Two relaxed before the stop, never recorded: a parallel batch
        # records only at its end.
        for i in (0, 2):
            relaxed, energy = backend.evaluate(structs[i]['atoms'],
                                               os.path.join(self.out, "steps", f"s{i}"),
                                               [], 12)
            plan.note(i, relaxed, energy)
        resumed = CyclePlan.load(self.out)
        pool = FakePool(backend)
        n_ok = _evaluate_batch(None, cfg, backend, 0, 0, [], 12, db, logger,
                               pool=pool, plan=resumed)
        self.assertEqual(pool.seen, [1, 3])
        self.assertEqual(n_ok, len(db))
        # Recorded in candidate order, as an uninterrupted batch would be.
        steps = [os.path.basename(r['cif_path']) for r in db.records]
        self.assertEqual(steps, sorted(steps))


if __name__ == "__main__":
    unittest.main()
