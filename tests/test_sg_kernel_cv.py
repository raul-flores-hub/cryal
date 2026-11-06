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

"""tools/sg_kernel_cv.py: the protocol is what the docstring says it is.

The data rules (eligibility, lineages, units), the splits (identical for
every variant), every model builds and scores, and the decision rule picks
the simplest variant that is not significantly worse than the best.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import sg_kernel_cv as cv   # noqa: E402


def _records(n, groups, rng, cycles=4, offset=0.0):
    out = []
    for i in range(n):
        cell = rng.uniform(5, 15, 3)
        g = groups[i % len(groups)]
        out.append({"cycle": i % cycles, "space_group": g,
                    "energy": float(-100 - cell[0] + 0.5 * (g == 14) + offset + rng.normal(0, 0.05)),
                    "a": cell[0], "b": cell[1], "c": cell[2], "alpha": 90.0,
                    "beta": float(rng.uniform(90, 120)), "gamma": 90.0,
                    "density": float(rng.uniform(1.0, 1.4))})
    return out


class TestData(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        rng = np.random.default_rng(0)
        a = _records(150, [14, 2, 4], rng)
        b = a + _records(80, [14, 2, 4], rng, offset=0.1)      # continues a
        c = _records(150, [14], rng)                            # one group only
        d = _records(150, [14, 61], rng) + [dict(a[0], space_group=None)]
        runs = {"a": (a, "terminada", "X"), "b": (b, "terminada", "X"),
                "c": (c, "terminada", "Y"), "d": (d, "terminada", None),
                "e": (a, "en curso", "Z")}
        manifest = {"runs": []}
        for name, (recs, status, mol) in runs.items():
            os.makedirs(os.path.join(self.dir, name))
            with open(os.path.join(self.dir, name, "database.json"), "w") as f:
                json.dump(recs, f)
            manifest["runs"].append({"run": name, "status": status, "molecule": mol,
                                     "formula": "C6H6", "potential": "P"})
        with open(os.path.join(self.dir, "MANIFEST.json"), "w") as f:
            json.dump(manifest, f)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_eligibility_lineage_and_units(self):
        runs = cv.load_runs(self.dir, log=lambda *a: None)
        # a is continued by b (kept: larger); c has one group; e is running
        self.assertEqual(sorted(runs), ["b", "d"])
        self.assertEqual(runs["b"]["unit"], "X | P")
        self.assertEqual(runs["d"]["unit"], "C6H6 | P")         # no name: formula
        self.assertTrue(all(r["space_group"] is not None for r in runs["d"]["records"]))

    def test_folds_are_fixed_and_capped(self):
        recs = _records(900, [14, 2, 4], np.random.default_rng(1), cycles=3)
        f1, f2 = cv.folds("r", recs), cv.folds("r", recs)
        self.assertEqual([(s, f) for s, f, _, _ in f1], [(s, f) for s, f, _, _ in f2])
        for (s, f, tr, te), (_, _, tr2, te2) in zip(f1, f2):
            np.testing.assert_array_equal(tr, tr2)
            self.assertLessEqual(len(tr), cv.MAX_TRAIN)
            self.assertFalse(set(tr) & set(te))
        cycles = [f for s, f, _, _ in f1 if s == "cycle"]
        self.assertEqual(cycles, ["1", "2"])                     # cycle 0 has no past
        self.assertEqual(sorted(f for s, f, _, _ in f1 if s == "logo"), ["14", "2", "4"])


class TestModels(unittest.TestCase):

    def test_every_variant_fits_and_scores(self):
        rng = np.random.default_rng(2)
        recs = _records(70, [14, 2, 4, 15, 61], rng)
        X = np.array([[r[k] for k in cv.CELL] + [r["space_group"]] for r in recs])
        y = np.array([r["energy"] for r in recs])
        train, test = np.arange(55), np.arange(55, 70)
        for v in [cv.REFERENCE, "prod-X-mle", "prod-LF-map", "sum-LFPC-mle", "sum-LFC-map"]:
            with self.subTest(variant=v):
                row = cv.fit_and_score(("r", "u", "kfold", "0", v, X, y, train, test, 0))
                self.assertEqual(row["error"], "")
                self.assertTrue(np.isfinite(row["nlpd"]))
                if v != cv.REFERENCE:
                    shares = json.loads(row["shares"])
                    self.assertEqual(len(shares), 2 if v.startswith("sum") else 1)

    def test_map_prior_touches_only_space_group_weights(self):
        # With a flat likelihood the prior alone acts: it pulls the log of
        # every space-group weight to 0 and leaves the rest where they start.
        k = cv.build_kernel("prod-LFPC-map")
        w, i = [], 0                        # positions in theta, not in the list
        for h in (h for h in k.hyperparameters if not h.fixed):
            if h.name.rsplit("__", 1)[-1].startswith("w_"):
                w.extend(range(i, i + h.n_elements))
            i += h.n_elements
        self.assertEqual(len(w), 5)
        start = np.full(len(k.theta), 2.0)
        x, _ = cv.map_optimizer(k)(lambda t, eval_gradient=True: (0.0, np.zeros_like(t)),
                                   start, k.bounds)
        others = [i for i in range(len(start)) if i not in w]
        np.testing.assert_allclose(x[w], 0.0, atol=1e-4)
        np.testing.assert_allclose(x[others], 2.0)


class TestDecision(unittest.TestCase):

    def _table(self, worse_by):
        rng = np.random.default_rng(3)
        table = {}
        for u in range(20):
            base = rng.normal(0, 0.01)
            table[f"u{u}"] = {v: base + worse_by.get(v, 0.05) + rng.normal(0, 0.001)
                              for v in cv.VARIANTS + [cv.REFERENCE]}
        return table

    def test_simplest_not_worse_wins(self):
        # a simpler variant ties the complex best (differences symmetric about
        # zero); everything else is worse by far more than the noise
        table = self._table({"sum-LFPC-map": 0.0})
        for i, u in enumerate(sorted(table)):
            table[u]["prod-LF-mle"] = table[u]["sum-LFPC-map"] + (0.001 if i % 2 else -0.001)
        res = cv.decide(table)
        self.assertIn(res["best"], ("sum-LFPC-map", "prod-LF-mle"))
        self.assertEqual(res["chosen"], "prod-LF-mle")

    def test_complexity_is_kept_when_the_data_ask_for_it(self):
        table = self._table({"sum-LFPC-map": 0.0})
        self.assertEqual(cv.decide(table)["chosen"], "sum-LFPC-map")


if __name__ == "__main__":
    unittest.main()
