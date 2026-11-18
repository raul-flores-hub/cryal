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


"""The nvalchemi backend: batched evaluation through a worker process.

Most of these run CrYAL's side against tests/fake_nvalchemi_worker.py, which
speaks the same task.json / result.json as the real worker and needs no
nvalchemi. The last class runs the real worker, and is skipped unless
CRYAL_NVALCHEMI_PYTHON (the interpreter of an environment with
nvalchemi-toolkit) and CRYAL_NVALCHEMI_MODEL (a MACE .model file) are set.
"""

import json
import logging
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from ase.io import read

from cryal import active_learning as AL
from cryal import search as S
from cryal.active_learning import LEDGER_FILE, PLAN_FILE
from cryal.backends import backend_class, get_backend
from cryal.backends import nvalchemi as NVA
from cryal.config import load_config
from cryal.structure_gen import generate_structures
from tests.test_genetic import BENZENE, run_config
from tests.test_search import reset_logger

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = os.path.join(HERE, "fake_nvalchemi_worker.py")


def nva_config(out, model, **kw):
    """run_config with the nvalchemi engine and the fake worker's interpreter."""
    return run_config(out, energy_backend="nvalchemi", nva_python=sys.executable,
                      nva_model=model, nva_batch_size=4, log_level="WARNING", **kw)


class NvaCase(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cryal_nva_")
        self.out = os.path.join(self.tmp, "run")
        self.model = os.path.join(self.tmp, "fake.model")
        open(self.model, "w").close()
        self.calls = os.path.join(self.tmp, "calls.txt")
        self.env = mock.patch.dict(os.environ, {"FAKE_NVA_LOG": self.calls})
        self.env.start()
        self.worker = mock.patch.object(NVA, "WORKER", FAKE)
        self.worker.start()
        reset_logger()

    def tearDown(self):
        self.worker.stop()
        self.env.stop()
        reset_logger()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def batches(self):
        if not os.path.exists(self.calls):
            return []
        return [line.strip().split(",") for line in open(self.calls)]

    def records(self):
        return json.load(open(os.path.join(self.out, "database.json")))


class TestConfig(NvaCase):

    def toml(self, body):
        path = os.path.join(self.tmp, "cryal.toml")
        with open(path, "w") as f:
            f.write(f'[general]\nmolecule = "{BENZENE}"\n[backend]\nengine = "nvalchemi"\n'
                    f'[backend.nvalchemi]\npython = "{sys.executable}"\n'
                    f'model = "{self.model}"\n' + body)
        return path

    def test_defaults_are_the_validated_protocol(self):
        cfg = load_config(self.toml(""))
        self.assertEqual(cfg.nva_stages, [
            {"temperature": 20.0, "pressure": 3000.0, "steps": 6000},
            {"temperature": 10.0, "pressure": 1.0, "steps": 6000}])
        self.assertEqual((cfg.nva_dtype, cfg.nva_batch_size, cfg.nva_fmax,
                          cfg.nva_max_steps, cfg.nva_pressure_coupling),
                         ("float32", 32, 1e-3, 10000, "triclinic"))

    def test_stages_are_read_and_checked(self):
        cfg = load_config(self.toml(
            "stages = [{ temperature = 300, pressure = 1, steps = 100 }]\n"))
        self.assertEqual(cfg.nva_stages, [{"temperature": 300.0, "pressure": 1.0,
                                           "steps": 100}])
        self.assertEqual(load_config(self.toml("stages = []\n")).nva_stages, [])
        for bad in ('stages = [{ temperature = 300, pressure = 1 }]',
                    'stages = [{ temperature = 300, pressure = 1, steps = 1, dt = 2 }]',
                    'stages = [{ temperature = -5, pressure = 1, steps = 1 }]',
                    'stages = [{ temperature = 5, pressure = 1, steps = 1.5 }]'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                load_config(self.toml(bad + "\n"))

    def test_python_and_model_are_required(self):
        path = os.path.join(self.tmp, "cryal.toml")
        with open(path, "w") as f:
            f.write(f'[general]\nmolecule = "{BENZENE}"\n[backend]\nengine = "nvalchemi"\n')
        with self.assertRaises(ValueError):
            load_config(path)
        path = self.toml("")
        with open(path) as f:
            text = f.read().replace(self.model, self.model + ".missing")
        with open(path, "w") as f:
            f.write(text)
        with self.assertRaises(FileNotFoundError):
            load_config(path)

    def test_parallel_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            load_config(self.toml("[parallel]\nenabled = true\nlocal_slots = 2\n"))
        self.assertIn("batches", str(ctx.exception))

    def test_unknown_key_is_named(self):
        with self.assertRaises(ValueError):
            load_config(self.toml("batchsize = 8\n"))


class TestEvaluateBatch(NvaCase):

    def setUp(self):
        super().setUp()
        self.cfg = nva_config(self.out, self.model)
        self.logger = logging.getLogger("cryal.test")
        self.backend = get_backend(self.cfg, self.logger)
        self.structs = generate_structures(BENZENE, self.cfg, 3, suggestions=None,
                                           logger=self.logger)
        self.dirs = [os.path.join(self.tmp, "steps", f"s{i}") for i in range(3)]

    def test_a_batch_comes_back_in_order(self):
        res = self.backend.evaluate_batch([s["atoms"] for s in self.structs], self.dirs,
                                          [], 12)
        self.assertEqual(len(res), 3)
        for (relaxed, energy, reason), s in zip(res, self.structs):
            self.assertIsNone(reason)
            np.testing.assert_allclose(relaxed.positions, s["atoms"].positions, atol=1e-9)
            self.assertAlmostEqual(energy, -1000.0 + 1e-3 * s["atoms"].get_volume(), 6)
        self.assertEqual(self.batches(), [["s0", "s1", "s2"]])
        task = json.load(open(os.path.join(self.tmp, "steps", "batch_s0", "task.json")))
        self.assertEqual(task["stages"], self.cfg.nva_stages)
        self.assertEqual(task["model"], os.path.abspath(self.model))

    def test_a_structure_with_overlaps_never_reaches_the_worker(self):
        bad = self.structs[1]["atoms"].copy()
        bad.positions[1] = bad.positions[0] + 0.1
        res = self.backend.evaluate_batch([self.structs[0]["atoms"], bad,
                                           self.structs[2]["atoms"]], self.dirs, [], 12)
        self.assertIn("overlap", res[1][2])
        self.assertEqual(self.batches(), [["s0", "s2"]])

    def test_an_exploded_structure_fails_alone(self):
        with mock.patch.dict(os.environ, {"FAKE_NVA_FAIL": "s1"}):
            res = self.backend.evaluate_batch([s["atoms"] for s in self.structs],
                                              self.dirs, [], 12)
        self.assertIsNone(res[0][2])
        self.assertIn("exploded", res[1][2])
        self.assertIsNone(res[2][2])

    def test_a_worker_that_dies_fails_the_batch_with_its_reason(self):
        with mock.patch.dict(os.environ, {"FAKE_NVA_EXIT": "3"}):
            res = self.backend.evaluate_batch([s["atoms"] for s in self.structs],
                                              self.dirs, [], 12)
        self.assertTrue(all(r[0] is None and "exited 3" in r[2] for r in res))

    def test_a_batch_past_its_timeout_fails(self):
        self.cfg.nva_timeout = 1
        with mock.patch.dict(os.environ, {"FAKE_NVA_SLEEP": "5"}):
            res = self.backend.evaluate_batch([s["atoms"] for s in self.structs],
                                              self.dirs, [], 12)
        self.assertTrue(all("timed out" in r[2] for r in res))

    def test_relative_step_directories(self):
        # output_dir is usually relative; the worker runs inside the batch's
        # directory, so the paths it is given must not be.
        cwd = os.getcwd()
        os.chdir(self.tmp)
        try:
            res = self.backend.evaluate_batch([s["atoms"] for s in self.structs],
                                              [os.path.join("rel", f"s{i}") for i in range(3)],
                                              [], 12)
        finally:
            os.chdir(cwd)
        self.assertTrue(all(r[2] is None for r in res), res)

    def test_the_worker_gets_expandable_segments(self):
        # Without them PyTorch's fragmented cache starves nvalchemi's Warp
        # kernels of GPU memory mid-stage, and the whole batch fails.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            self.assertEqual(NVA.worker_env()["PYTORCH_CUDA_ALLOC_CONF"],
                             "expandable_segments:True")
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
            self.assertEqual(NVA.worker_env()["PYTORCH_CUDA_ALLOC_CONF"],
                             "max_split_size_mb:128")

    def test_relax_of_one_structure_is_a_batch_of_one(self):
        relaxed, energy = self.backend.relax(self.structs[0]["atoms"], self.dirs[0])
        self.assertIsNotNone(relaxed)
        self.assertEqual(self.batches(), [["s0"]])


class TestRuns(NvaCase):

    def test_a_bach_run_relaxes_each_cycle_in_batches(self):
        # 6 initial structures in batches of 4: 4 + 2; cycles of 6 the same way.
        AL.run(nva_config(self.out, self.model, num_cycles=1))
        sizes = [len(b) for b in self.batches()]
        self.assertEqual(sizes[:2], [4, 2])
        recs = self.records()
        self.assertEqual(len(recs), sum(sizes))
        steps = [os.path.basename(r["cif_path"]) for r in recs]
        self.assertEqual(steps, sorted(steps))                 # candidate order

    def test_an_interrupted_cycle_resumes_after_its_last_finished_batch(self):
        # Stop inside the second batch of cycle 0: the first batch's outcomes
        # are in the plan, and only the rest is sent again.
        cfg = nva_config(self.out, self.model, num_cycles=0)
        real = NVA.NvalchemiBackend.relax_batch
        calls = {"n": 0}

        def stop_on_second(self_, *a, **k):
            calls["n"] += 1
            if calls["n"] == 2:
                raise KeyboardInterrupt
            return real(self_, *a, **k)
        with mock.patch.object(NVA.NvalchemiBackend, "relax_batch", stop_on_second):
            with self.assertRaises(KeyboardInterrupt):
                S.run_random(cfg)
        reset_logger()
        plan = json.load(open(os.path.join(self.out, PLAN_FILE)))
        self.assertEqual(sum(o is not None for o in plan["outcomes"]), 4)
        S.run_random(nva_config(self.out, self.model, num_cycles=0, resume=True))
        reset_logger()
        self.assertEqual([len(b) for b in self.batches()], [4, 2])
        self.assertEqual(len(self.records()), 6)
        self.assertEqual([e["cycle"] for e in json.load(
            open(os.path.join(self.out, LEDGER_FILE)))], [0])


@unittest.skipUnless(os.environ.get("CRYAL_NVALCHEMI_PYTHON")
                     and os.environ.get("CRYAL_NVALCHEMI_MODEL"),
                     "set CRYAL_NVALCHEMI_PYTHON and CRYAL_NVALCHEMI_MODEL to run the "
                     "real nvalchemi worker")
class TestRealWorker(unittest.TestCase):
    """The real worker on two benzene crystals, with a short protocol."""

    def test_two_crystals(self):
        self.two_crystals("float32", True)

    def test_two_crystals_in_float64(self):
        # float64 without cuEquivariance: every float field of the batch must
        # follow, or Warp's float64 kernels refuse the float32 masses.
        self.two_crystals("float64", False)

    def two_crystals(self, dtype, cueq):
        tmp = tempfile.mkdtemp(prefix="cryal_nva_real_")
        try:
            cfg = run_config(tmp, energy_backend="nvalchemi", nva_dtype=dtype, nva_cueq=cueq,
                             nva_python=os.environ["CRYAL_NVALCHEMI_PYTHON"],
                             nva_model=os.environ["CRYAL_NVALCHEMI_MODEL"],
                             nva_stages=[{"temperature": 20.0, "pressure": 1.0, "steps": 20}],
                             nva_max_steps=50, nva_fmax=0.05)
            logger = logging.getLogger("cryal.test")
            backend = get_backend(cfg, logger)
            structs = generate_structures(BENZENE, cfg, 2, suggestions=None, logger=logger)
            dirs = [os.path.join(tmp, "steps", f"s{i}") for i in range(2)]
            res = backend.relax_batch([s["atoms"] for s in structs], dirs, logger)
            log = open(os.path.join(tmp, "steps", "batch_s0", "worker.log")).read()
            # Room for 0.4 atoms/Å³ within MACE-OFF23's 4.5 Å cutoff: nvalchemi's
            # own 80 overflowed for a cell compressed at 3000 bar.
            self.assertIn("neighbour matrix: at least 160 neighbours per atom", log)
            for relaxed, energy, reason in res:
                self.assertIsNone(reason, log)
                self.assertTrue(np.isfinite(energy))
                self.assertEqual(len(relaxed), len(structs[0]["atoms"]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
