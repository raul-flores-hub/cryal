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

"""What a cycle writes down about the candidates it did not keep.

A rejected candidate leaves no record but a line in the log -- there is no row
in the database to go back to -- so that line is the only thing standing
between "this search is going badly" and "this search is misconfigured".

The case these tests exist for is `aseMaxSteps`. A relaxation that runs out of
steps is discarded, correctly: it was not evaluated at a minimum. But the
budget does not carry between engines -- from the same geometries MACE
converged in about 25 steps where UMA needed 39 to 582 -- so a value brought
over from another engine throws away structures that a slightly larger number
would have kept, and every one of them looks like an ordinary loss.
"""

import logging
import os
import shutil
import tempfile
import unittest

import numpy as np
from ase import Atoms

from cryal.active_learning import CrystalDatabase, _evaluate_batch
from cryal.backends import get_backend
from cryal.config import Config


def candidate(shift: float = 0.0) -> dict:
    """Two C2 'molecules', clear of the pre-relaxation gate.

    `shift` moves one of them: the stub calculator answers from its cache when
    handed a structure it has already seen, so each candidate has to differ.
    """
    atoms = Atoms("C4",
                  positions=[[2.0, 2.0, 2.0], [3.5, 2.0, 2.0],
                             [2.0, 8.0, 2.0], [3.5, 8.0, 2.0]],
                  cell=np.eye(3) * 15.0, pbc=True)
    atoms.positions[2][1] += shift
    return {"atoms": atoms, "source": "test", "space_group": 2}


class TestCycleReport(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cryal_cycle_")
        self.cfg = Config(energy_backend="ase",
                          ase_calculator="tests.stub_calculator.StubCalculator",
                          ase_calculator_kwargs={"energy": -1000.0},
                          ase_optimizer="FIRE", ase_max_steps=8,
                          check_molecular_integrity=False,
                          output_dir=self.tmp, Z=2, mol_weight=24.0)
        self.db = CrystalDatabase(os.path.join(self.tmp, "database.json"),
                                  os.path.join(self.tmp, "structures"))
        self.backend = get_backend(self.cfg)
        self.logger = logging.getLogger("cryal.test.cycle")
        self.logger.propagate = False

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _cycle(self, cycle: int, n: int, first_shift: float = 0.0):
        structs = [candidate(first_shift + i * 0.1) for i in range(n)]
        with self.assertLogs(self.logger, level=logging.DEBUG) as caught:
            kept = _evaluate_batch(structs, self.cfg, self.backend, cycle, 0,
                                   [], 4, self.db, self.logger)
        return kept, caught.records

    def _at_least(self, records, level):
        return [r for r in records if r.levelno >= level]

    def test_a_kept_candidate_is_recorded_and_nothing_is_warned_about(self):
        kept, records = self._cycle(0, 3)
        self.assertEqual(kept, 3)
        self.assertEqual(len(self.db), 3)
        self.assertEqual(self._at_least(records, logging.WARNING), [])

    def test_a_discarded_candidate_says_which_setting_discarded_it(self):
        self.backend.calculator.constant_force = 1.0    # nothing can converge
        kept, records = self._cycle(1, 2, first_shift=1.0)
        self.assertEqual(kept, 0)
        lines = [r.getMessage() for r in self._at_least(records, logging.INFO)]
        self.assertTrue(any("FAILED" in ln and "aseMaxSteps" in ln
                            for ln in lines), lines)

    def test_the_cycle_warns_once_however_many_structures_it_lost(self):
        # Once, not per structure: the actionable fact is the pattern, and a
        # warning repeated forty times is one nobody finishes reading.
        self.backend.calculator.constant_force = 1.0
        _, records = self._cycle(1, 4, first_shift=1.0)
        warnings = [r.getMessage() for r in self._at_least(records,
                                                           logging.WARNING)]
        self.assertEqual(len(warnings), 1, warnings)
        self.assertIn("4 of 4", warnings[0])
        self.assertIn("aseMaxSteps", warnings[0])

    def test_each_cycle_reports_its_own_losses(self):
        # The tally is read at the end of a cycle and starts again: a cycle
        # that lost nothing must not inherit the last one's warning.
        self.backend.calculator.constant_force = 1.0
        self._cycle(1, 2, first_shift=1.0)
        self.backend.calculator.constant_force = 0.0
        _, records = self._cycle(2, 2, first_shift=3.0)
        self.assertEqual(self._at_least(records, logging.WARNING), [])
