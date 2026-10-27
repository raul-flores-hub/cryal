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

"""The hierarchical space-group kernel.

What it must guarantee, whatever the optimiser does with its weights: a
valid covariance (unit diagonal, positive semi-definite, correlations
ordered by what two groups share), a gradient the optimiser can trust, and
the reason it exists -- a group with no data is predicted from the groups
that share its lattice before the ones that share nothing.
"""

import unittest

import numpy as np
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from cryal.sg_kernel import (OnColumns, SpaceGroupKernel, _CENTERING,
                             bravais_lattice, crystal_family)


def _column(groups):
    return np.asarray(groups, dtype=float).reshape(-1, 1)


class TestClassification(unittest.TestCase):

    def test_centering_table_matches_pyxtal(self):
        from pyxtal.symmetry import Group
        for sg in range(1, 231):
            with self.subTest(sg=sg):
                self.assertEqual(_CENTERING[sg - 1], Group(sg).symbol[0])

    def test_examples(self):
        expected = {1: "aP", 14: "mP", 15: "mS", 19: "oP", 38: "oS", 64: "oS",
                    70: "oF", 74: "oI", 88: "tI", 146: "hR", 167: "hR",
                    143: "hP", 194: "hP", 205: "cP", 225: "cF", 230: "cI"}
        for sg, lattice in expected.items():
            with self.subTest(sg=sg):
                self.assertEqual(bravais_lattice(sg), lattice)

    def test_fourteen_lattices_nested_in_six_families(self):
        lattices = {bravais_lattice(sg) for sg in range(1, 231)}
        self.assertEqual(len(lattices), 14)
        for sg in range(1, 231):
            self.assertEqual(bravais_lattice(sg)[0], crystal_family(sg))

    def test_invalid_numbers_are_refused(self):
        for sg in (0, 231, -1, 14.0):
            with self.subTest(sg=sg):
                with self.assertRaises(ValueError):
                    bravais_lattice(sg)


class TestCovariance(unittest.TestCase):

    def test_each_level_gets_its_correlation(self):
        k = SpaceGroupKernel(w_lattice=0.5, w_family=2.0, w_any=0.25)
        rho = k.correlations()
        K = k(_column([14, 14, 4, 15, 61, 2]))
        self.assertEqual(K[0, 1], 1.0)                              # same group
        self.assertAlmostEqual(K[0, 2], rho["same lattice"])        # 14, 4: mP
        self.assertAlmostEqual(K[0, 3], rho["same family"])         # 14, 15: m
        self.assertAlmostEqual(K[0, 4], rho["nothing shared"])      # 14, 61
        self.assertAlmostEqual(K[4, 5], rho["nothing shared"])      # 61, 2
        np.testing.assert_array_equal(np.diag(K), 1.0)
        np.testing.assert_array_equal(k.diag(_column([14, 61])), 1.0)

    def test_variance_shares(self):
        k = SpaceGroupKernel(w_lattice=1.0, w_family=2.0, w_any=4.0)
        self.assertEqual(k.variance_shares(), {"group": 0.125, "lattice": 0.125,
                                               "family": 0.25, "all groups": 0.5})

    def test_correlations_are_ordered_for_any_weights(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            w = np.exp(rng.uniform(np.log(1e-5), np.log(1e5), 3))
            rho = SpaceGroupKernel(w_lattice=w[0], w_family=w[1], w_any=w[2]).correlations()
            self.assertTrue(1.0 >= rho["same lattice"] >= rho["same family"]
                            >= rho["nothing shared"] >= 0.0)

    def test_positive_semidefinite_for_any_weights_and_groups(self):
        rng = np.random.default_rng(1)
        for _ in range(100):
            w = np.exp(rng.uniform(np.log(1e-5), np.log(1e5), 3))
            groups = rng.integers(1, 231, size=60)
            K = SpaceGroupKernel(w_lattice=w[0], w_family=w[1], w_any=w[2])(_column(groups))
            self.assertGreaterEqual(np.linalg.eigvalsh(K).min(), -1e-10)

    def test_cross_covariance_matches_the_full_matrix(self):
        k = SpaceGroupKernel(w_lattice=0.3, w_family=0.7, w_any=0.1)
        a, b = [14, 2, 61], [4, 14, 167, 19]
        full = k(_column(a + b))
        np.testing.assert_allclose(k(_column(a), _column(b)), full[:3, 3:])

    def test_reads_only_its_own_column(self):
        X = np.array([[9.1, 14.0, 1.2], [3.3, 4.0, 0.9]])
        k = SpaceGroupKernel(column=1, w_lattice=0.5)
        np.testing.assert_allclose(k(X), SpaceGroupKernel(w_lattice=0.5)(_column([14, 4])))

    def test_a_scaled_or_non_group_column_is_refused(self):
        k = SpaceGroupKernel()
        for bad in ([14, 0.37], [14, 0], [231, 14]):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "space-group numbers"):
                    k(_column(bad))


class TestGradient(unittest.TestCase):

    def _numeric(self, kernel, X, eps=1e-6):
        theta = kernel.theta.copy()
        grads = []
        for i in range(len(theta)):
            up, down = theta.copy(), theta.copy()
            up[i] += eps
            down[i] -= eps
            kernel.theta = up
            K_up = kernel(X)
            kernel.theta = down
            K_down = kernel(X)
            grads.append((K_up - K_down) / (2 * eps))
        kernel.theta = theta
        return np.stack(grads, axis=2)

    def test_gradient_matches_finite_differences(self):
        X = _column([14, 4, 15, 61, 19, 2, 14, 167])
        k = SpaceGroupKernel(w_lattice=0.4, w_family=1.7, w_any=0.05)
        K, grad = k(X, eval_gradient=True)
        np.testing.assert_allclose(K, k(X))
        np.testing.assert_allclose(grad, self._numeric(k, X), atol=1e-8)

    def test_fixed_weights_drop_out_of_the_gradient(self):
        X = _column([14, 4, 61])
        k = SpaceGroupKernel(w_family_bounds="fixed")
        self.assertEqual(len(k.theta), 2)
        _, grad = k(X, eval_gradient=True)
        self.assertEqual(grad.shape, (3, 3, 2))
        np.testing.assert_allclose(grad, self._numeric(k, X), atol=1e-8)

    def test_gradient_of_the_full_product_kernel(self):
        rng = np.random.default_rng(2)
        X = np.column_stack([rng.uniform(5, 20, 8), rng.uniform(80, 120, 8),
                             [14, 4, 15, 61, 19, 2, 14, 167]])
        k = (ConstantKernel(2.0) * OnColumns(Matern(length_scale=[3.0, 10.0], nu=2.5), [0, 1])
             * SpaceGroupKernel(column=2, w_lattice=0.4, w_family=1.7, w_any=0.05))
        _, grad = k(X, eval_gradient=True)
        np.testing.assert_allclose(grad, self._numeric(k, X), atol=1e-6)


class TestOnColumns(unittest.TestCase):

    def test_sees_only_its_columns(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(6, 4))
        base = Matern(length_scale=[1.0, 2.0], nu=2.5)
        np.testing.assert_allclose(OnColumns(base, [0, 2])(X), base(X[:, [0, 2]]))
        np.testing.assert_allclose(OnColumns(base, [0, 2]).diag(X), np.ones(6))

    def test_hyperparameters_are_the_base_kernel_s(self):
        k = OnColumns(Matern(length_scale=[1.0, 2.0]), [0, 1])
        self.assertEqual([h.name for h in k.hyperparameters], ["kernel__length_scale"])
        np.testing.assert_allclose(k.theta, np.log([1.0, 2.0]))
        k.theta = np.log([3.0, 4.0])
        np.testing.assert_allclose(k.kernel.length_scale, [3.0, 4.0])

    def test_clone_round_trips(self):
        k = (ConstantKernel(1.0) * OnColumns(Matern(length_scale=[1.0, 2.0]), [0, 1])
             * SpaceGroupKernel(column=2, w_any=0.1))
        c = clone(k)
        self.assertEqual(c, k)
        self.assertIsNot(c.k2, k.k2)


class TestLearning(unittest.TestCase):
    """The point of the hierarchy: a group never seen borrows from its kin.

    A level can only be learnt where at least two groups with data share it:
    with one group per lattice, an offset of that lattice and an offset of
    that group give the data the same likelihood, and nothing can tell them
    apart. Both cases below are built with that in mind.
    """

    def _data(self, groups, offset, level, n_per_group, rng):
        X, y = [], []
        for sg in groups:
            cell = rng.uniform(0.0, 1.0, (n_per_group, 2))
            X.append(np.column_stack([cell, np.full(n_per_group, sg)]))
            y.append(np.sin(3 * cell[:, 0]) + cell[:, 1] + offset[level(sg)])
        return np.vstack(X), np.concatenate(y)

    def _fit(self, X, y):
        kernel = (ConstantKernel(1.0) * OnColumns(Matern(length_scale=[1.0, 1.0], nu=2.5), [0, 1])
                  * SpaceGroupKernel(column=2) + WhiteKernel(1e-3))
        return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                        n_restarts_optimizer=3, random_state=0).fit(X, y)

    def test_an_unseen_group_is_predicted_from_its_own_family(self):
        # Energy offsets by crystal family; P2_1 (4) has no training data.
        rng = np.random.default_rng(4)
        offset = {"m": 0.0, "o": 3.0, "a": -2.0}
        X, y = self._data([14, 15, 61, 19, 2], offset, crystal_family, 12, rng)
        gp = self._fit(X, y)

        probe = np.array([[0.5, 0.5, 4], [0.5, 0.5, 61]])
        truth = np.sin(1.5) + 0.5
        pred = gp.predict(probe)
        self.assertLess(abs(pred[0] - (truth + offset["m"])), 0.5)
        self.assertLess(abs(pred[1] - (truth + offset["o"])), 0.5)

        # The offsets are by family and nothing else, and the fit says so:
        # a family-level share well above the lattice-level one.
        shares = gp.kernel_.k1.k2.variance_shares()
        self.assertGreater(shares["family"], 10 * shares["lattice"])
        self.assertAlmostEqual(sum(shares.values()), 1.0)

    def test_an_unseen_group_is_predicted_from_its_own_lattice(self):
        # mS sits 2 above mP, both monoclinic; two groups with data in each
        # lattice (mP: 14, 7; mS: 15, 5), and none in P2_1 (4, mP) or Cc (9, mS).
        rng = np.random.default_rng(0)
        offset = {"mP": 0.0, "mS": 2.0, "oP": 0.0, "aP": 0.0}
        X, y = self._data([14, 7, 15, 5, 61, 19, 2], offset, bravais_lattice, 12, rng)
        gp = self._fit(X, y)

        truth = np.sin(1.5) + 0.5
        pred = gp.predict(np.array([[0.5, 0.5, 4], [0.5, 0.5, 9]]))
        self.assertLess(abs(pred[0] - (truth + offset["mP"])), 0.3)
        self.assertLess(abs(pred[1] - (truth + offset["mS"])), 0.3)

        shares = gp.kernel_.k1.k2.variance_shares()
        self.assertGreater(shares["lattice"], 3 * shares["group"])
        self.assertGreater(shares["lattice"], 100 * shares["family"])


if __name__ == "__main__":
    unittest.main()
