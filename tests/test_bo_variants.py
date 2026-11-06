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

"""Variants of BACH's acquisition (gp.acquisition, gp.batch).

The first thing guarded is that they are variants: with the keys left out,
maximize_ei must be the function the article ran, suggestion for suggestion
and draw for draw. PUBLISHED below is that function's selection, frozen as it
was before the variants existed; the test fails if the default path ever
drifts from it.
"""

import glob
import json
import logging
import os
import shutil
import tempfile
import unittest
import warnings

import numpy as np

from cryal import gp_model as GP
from cryal.config import Config, load_config, print_config_summary
from tests.test_genetic import BENZENE, run_config

warnings.filterwarnings("ignore")


def PUBLISHED(gp, database_records, cfg, rng):
    """maximize_ei as of the article (v1.0.0 and up to this change), verbatim."""
    from scipy.stats import norm
    best_y = min(r['energy'] for r in database_records)
    NA = 6.02214076e23
    n = cfg.num_ei_candidates
    lo_ax, hi_ax = cfg.min_cell_axis, 35.0
    C = np.column_stack([
        rng.uniform(lo_ax, hi_ax, n), rng.uniform(lo_ax, hi_ax, n),
        rng.uniform(lo_ax, hi_ax, n), rng.uniform(70.0, 110.0, n),
        rng.uniform(cfg.beta_min, cfg.beta_max, n), rng.uniform(70.0, 110.0, n),
        rng.uniform(cfg.density_min, cfg.density_max * 1.1, n),
    ])
    if gp.use_anisotropy:
        C = np.column_stack([C, C[:, :3].max(axis=1) / C[:, :3].min(axis=1)])
    V_from_density = (cfg.Z * cfg.mol_weight / NA) / (C[:, 6] * 1e-24)
    ratio = V_from_density / (C[:, 0] * C[:, 1] * C[:, 2])
    C = C[(ratio > 0.4) & (ratio < 2.5)]
    if len(C) == 0:
        return []
    mu, std = gp.predict(C)
    improvement = best_y - mu - cfg.ei_xi
    Zs = improvement / (std + 1e-9)
    ei = improvement * norm.cdf(Zs) + std * norm.pdf(Zs)
    ei[std < 1e-10] = 0.0
    X_scaled = gp.scaler_X.transform(C)
    out, used = [], []
    for i in np.argsort(-ei):
        if len(out) >= cfg.num_ei_suggestions:
            break
        if any(np.linalg.norm(X_scaled[i] - u) < 0.5 for u in used):
            continue
        used.append(X_scaled[i])
        V_i = (cfg.Z * cfg.mol_weight / NA) / (C[i, 6] * 1e-24)
        s = {'a': float(C[i, 0]), 'b': float(C[i, 1]), 'c': float(C[i, 2]),
             'alpha': float(C[i, 3]), 'beta': float(C[i, 4]), 'gamma': float(C[i, 5]),
             'density': float(C[i, 6]), 'volume': float(V_i), 'E_pred': float(mu[i]),
             'E_std': float(std[i]), 'EI': float(ei[i])}
        if gp.use_anisotropy:
            s['anisotropy'] = float(C[i, 7])
        out.append(s)
    return out


def toy_records(n=60, seed=3):
    """A landscape with a minimum near 1.25 g/cm3 and beta 105 degrees."""
    rng = np.random.default_rng(seed)
    recs = []
    for _ in range(n):
        a, b, c = rng.uniform(5, 12, 3)
        beta, rho = rng.uniform(90, 130), rng.uniform(0.8, 1.6)
        recs.append(dict(a=a, b=b, c=c, alpha=90.0, beta=beta, gamma=90.0, density=rho,
                         energy=-100 + 30 * (rho - 1.25) ** 2 + 0.02 * abs(beta - 105)
                         + rng.normal(0, 0.05)))
    return recs


RECS = toy_records()


def fitted(anisotropy=False):
    gp = GP.GPSurrogate(n_restarts=1, use_anisotropy=anisotropy)
    gp.fit(RECS)
    return gp


GP0 = fitted()


def cfg(**kw):
    c = Config(Z=4, mol_weight=78.11, min_cell_axis=5.0, num_ei_candidates=20000)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def scaled(gp, suggestions):
    names = ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'density']
    return gp.scaler_X.transform(np.array([[s[k] for k in names] for s in suggestions]))


class TestPublishedPath(unittest.TestCase):

    def test_the_default_is_the_published_function(self):
        for anisotropy in (False, True):
            gp = GP0 if not anisotropy else fitted(True)
            r1, r2 = np.random.default_rng(11), np.random.default_rng(11)
            self.assertEqual(GP.maximize_ei(gp, RECS, cfg(), r1),
                             PUBLISHED(gp, RECS, cfg(), r2))
            self.assertEqual(r1.random(), r2.random())        # same draws consumed

    def test_the_summary_says_nothing_new_by_default(self):
        lines = []

        class L:
            def info(self, s):
                lines.append(s)
        print_config_summary(cfg(), L())
        self.assertFalse(any("Acquisition" in l for l in lines))
        lines.clear()
        print_config_summary(cfg(gp_acquisition="ucb"), L())
        self.assertTrue(any("UCB" in l for l in lines))


class TestVariants(unittest.TestCase):

    def run_variant(self, seed=5, **kw):
        return GP.maximize_ei(GP0, RECS, cfg(**kw), np.random.default_rng(seed))

    def test_ucb_kappa_moves_from_exploitation_to_exploration(self):
        greedy = self.run_variant(gp_acquisition="ucb", ucb_kappa=0.0)
        bold = self.run_variant(gp_acquisition="ucb", ucb_kappa=50.0)
        self.assertEqual(greedy[0]['acquisition'], "ucb")
        self.assertNotIn('EI', greedy[0])
        # kappa = 0 ranks by predicted energy alone; a large kappa by spread.
        self.assertEqual(greedy[0]['score'], -greedy[0]['E_pred'])
        self.assertLess(np.mean([s['E_pred'] for s in greedy]),
                        np.mean([s['E_pred'] for s in bold]))
        self.assertGreater(np.mean([s['E_std'] for s in bold]),
                           np.mean([s['E_std'] for s in greedy]))

    def test_believer_batch_spreads_out(self):
        top = self.run_variant()
        believed = self.run_variant(gp_batch="believer", gp_pool=500)
        self.assertEqual(len(believed), cfg().num_ei_suggestions)
        # The first pick is a best candidate either way (far from the data,
        # many tie at the prior: the same score, not always the same point).
        self.assertAlmostEqual(believed[0]['EI'], top[0]['EI'], places=9)

        def spacing(s):
            X = scaled(GP0, s)
            D = np.linalg.norm(X[:, None] - X[None], axis=2)
            np.fill_diagonal(D, np.inf)
            return D.min()
        self.assertGreater(spacing(believed), 0.0)
        self.assertEqual(len({(s['a'], s['b']) for s in believed}), len(believed))
        ucb = self.run_variant(gp_acquisition="ucb", gp_batch="believer", gp_pool=500)
        self.assertEqual(len(ucb), cfg().num_ei_suggestions)

    def test_thompson(self):
        a = self.run_variant(gp_acquisition="thompson", gp_pool=500)
        b = self.run_variant(gp_acquisition="thompson", gp_pool=500)
        self.assertEqual(a, b)                                # reproducible from the rng
        self.assertGreater(len(a), 0)
        self.assertLessEqual(len(a), cfg().num_ei_suggestions)
        self.assertEqual(len({(s['a'], s['b']) for s in a}), len(a))
        self.assertTrue(all(s['acquisition'] == "thompson" for s in a))
        for s in a:                                           # inside the candidate box
            self.assertTrue(5.0 <= min(s['a'], s['b'], s['c']) and
                            max(s['a'], s['b'], s['c']) <= 35.0)

    def test_report_names_the_acquisition(self):
        text = GP.gp_report(GP0, self.run_variant(gp_acquisition="ucb"), -100.0)
        self.assertIn("ucb suggestions", text)
        self.assertIn("EI suggestions", GP.gp_report(GP0, self.run_variant(), -100.0))


class TestConfig(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cryal_bo_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def load(self, body):
        path = os.path.join(self.tmp, "c.toml")
        with open(path, "w") as f:
            f.write(f'[general]\nmolecule = "{BENZENE}"\n[backend]\nengine = "ase"\n'
                    '[backend.ase]\ncalculator = "tests.stub_calculator.StubCalculator"\n'
                    + body)
        return load_config(path)

    def test_keys(self):
        c = self.load("")
        self.assertEqual((c.gp_acquisition, c.gp_batch), ("ei", "diverse"))
        c = self.load('[gp]\nacquisition = "ucb"\nucb_kappa = 1.5\nbatch = "believer"\n'
                      'pool = 300\n')
        self.assertEqual((c.gp_acquisition, c.ucb_kappa, c.gp_batch, c.gp_pool),
                         ("ucb", 1.5, "believer", 300))
        for body, says in [('[gp]\nacquisition = "pi"\n', "acquisition"),
                           ('[gp]\nacquisition = "thompson"\nbatch = "believer"\n',
                            "Thompson")]:
            with self.subTest(body=body):
                with self.assertRaises(ValueError) as ctx:
                    self.load(body)
                self.assertIn(says, str(ctx.exception))


class TestBachRun(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cryal_bo_run_")

    def tearDown(self):
        lg = logging.getLogger("cryal")
        for h in list(lg.handlers):
            lg.removeHandler(h)
            h.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_bach_runs_with_each_variant(self):
        from cryal.active_learning import run
        for acq, batch in [("thompson", "diverse"), ("ucb", "believer")]:
            with self.subTest(acquisition=acq, batch=batch):
                lg = logging.getLogger("cryal")
                for h in list(lg.handlers):
                    lg.removeHandler(h)
                    h.close()
                out = os.path.join(self.tmp, f"{acq}_{batch}")
                run(run_config(out, log_level="WARNING", num_cycles=1, gp_n_restarts=0,
                               num_ei_candidates=5000, gp_pool=300,
                               gp_acquisition=acq, gp_batch=batch))
                files = glob.glob(os.path.join(out, "cycle001_suggestions.json"))
                self.assertEqual(len(files), 1)
                with open(files[0]) as f:
                    sugg = json.load(f)
                self.assertTrue(sugg)
                self.assertTrue(all(s['acquisition'] == acq for s in sugg))


if __name__ == "__main__":
    unittest.main()
