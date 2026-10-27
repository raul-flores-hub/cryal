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
sg_kernel.py — a hierarchical categorical kernel over space groups.

The space group a structure was generated in is a category, not a number:
14 and 15 are not "close" and 2 and 61 are not "far". This kernel gives the
GP a correlation between two groups that depends on what they share:

    k(s, s') = 1          same space group
             = rho_lat    same Bravais lattice          (mP, mS, oP, ... cF)
             = rho_fam    same crystal family           (m, o, t, h, c, a)
             = rho_any    nothing in common

with 1 >= rho_lat >= rho_fam >= rho_any >= 0. The levels nest: each of the
14 Bravais lattices lies in one of the 6 crystal families (the hexagonal
family holding both hP and hR), which is what keeps the ordering, and
positive semi-definiteness, true by construction rather than by constraint.
A crystal system level is deliberately absent: hP spans the trigonal and
hexagonal systems, so lattice and system do not nest.

Why hierarchical. A run may draw from every group valid for its Z, thirty to
a hundred of them, with a handful of structures each and none in some. With
one correlation for every pair of distinct groups, a group without data is
as unknown as any other, its uncertainty is large, and Expected Improvement
spends the budget visiting each one. Here P2_1 (4) learns from P2_1/c (14),
which shares its lattice, more than from Pbca (61), which shares nothing.

Parametrisation. Three positive weights, one per level below "same group":

    k(s, s') = (d_grp + w_lattice d_lat + w_family d_fam + w_any) / S,
    S        = 1 + w_lattice + w_family + w_any

where d_* is 1 when s and s' share that level. Each d_* is a block
indicator matrix, so k is a positively weighted sum of PSD matrices, and
k(s, s) = 1 for every group: the amplitude belongs to the continuous part
of the product (ConstantKernel), not to this one. The weight of "same
group" is fixed at 1; letting it float too would only rescale S, a flat
direction the optimiser would wander along. The correlations follow:

    rho_any = w_any / S
    rho_fam = (w_family + w_any) / S
    rho_lat = (w_lattice + w_family + w_any) / S

Reading a fitted kernel. In a product with ConstantKernel(A), the covariance
splits into independent parts, one per level: A/S of it is particular to
each group, A w_lattice/S is shared by a lattice, A w_family/S by a family
and A w_any/S by every group. variance_shares() reports those fractions,
and they, not the correlations, say where the energy differences live. A
correlation near 1 at every level is not a failed fit: it means a large
component common to all groups (the energy's dependence on the cell, say)
dominates, and the shares then show which level the remaining differences
belong to. A level is only learnt where the data hold at least two groups
that share it: with a single group of some lattice, an offset of that
lattice and one of that group fit the data equally well, and the share
between those two levels is the prior's call, not the data's.

Use. The kernel reads the space-group number (1-230) from one column of X,
unscaled, and ignores the others; OnColumns restricts a continuous kernel to
the cell columns, so the two multiply cleanly:

    k = ConstantKernel() * OnColumns(Matern(...), cell_columns) \\
        * SpaceGroupKernel(column=sg_column) + WhiteKernel()
"""

import numpy as np
from sklearn.gaussian_process.kernels import Hyperparameter, Kernel


# ---------------------------------------------------------------------------
# Crystallographic classification of the 230 groups
# ---------------------------------------------------------------------------

# First letter of the Hermann-Mauguin symbol of each group 1-230, in its
# standard setting (from PyXtal; tests/test_sg_kernel.py checks it against
# PyXtal). A, B and C are one Bravais type, side-centred (S): the A groups
# 38-41 are base-centred orthorhombic in another setting, not a new lattice.
_CENTERING = (
    "PPPPCPPCCPPCPPCPPPPCCFIIPPPPPPPPPPCCCAAAAFFIIIPPPPPPPPPPPPPPPPCCCCCCFFIIII"
    "PPPPIIPIPPPPIIPPPPPPPPIIPPPPPPPPIIIIPPPPPPPPIIIIPPPPPPPPPPPPPPPPIIIIPPPRPR"
    "PPPPPPRPPPPRRPPPPRRPPPPPPPPPPPPPPPPPPPPPPPPPPPPFIPIPPFFIPIPPFFIPPIPFIPFIPP"
    "PPFFFFII"
)
assert len(_CENTERING) == 230

# Crystal families by space-group number. Trigonal and hexagonal are one
# family; the rhombohedral (R) groups are its second lattice.
_FAMILIES = (("a", 1, 2), ("m", 3, 15), ("o", 16, 74), ("t", 75, 142),
             ("h", 143, 194), ("c", 195, 230))


def crystal_family(sg: int) -> str:
    """Crystal family of a space group: a, m, o, t, h or c."""
    _check_group(sg)
    for name, lo, hi in _FAMILIES:
        if lo <= sg <= hi:
            return name
    raise AssertionError(sg)   # unreachable: the ranges cover 1-230


def bravais_lattice(sg: int) -> str:
    """Bravais lattice of a space group, e.g. 'mP', 'oS', 'hR', 'cF'."""
    family = crystal_family(sg)          # validates sg
    letter = _CENTERING[sg - 1]
    return family + ("S" if letter in "ABC" else letter)


def _check_group(sg) -> None:
    if not (isinstance(sg, (int, np.integer)) and 1 <= sg <= 230):
        raise ValueError(f"{sg!r} is not a space group: valid numbers are 1-230")


# Integer codes per group, index 0 unused, so a column of group numbers can
# be turned into lattice and family codes with one fancy-indexing step.
_FAMILY_CODE  = np.zeros(231, dtype=int)
_LATTICE_CODE = np.zeros(231, dtype=int)
_lattices = {}
for _sg in range(1, 231):
    _FAMILY_CODE[_sg] = "amothc".index(crystal_family(_sg))
    _LATTICE_CODE[_sg] = _lattices.setdefault(bravais_lattice(_sg), len(_lattices))
assert len(_lattices) == 14
del _sg, _lattices


def _groups(X, column: int) -> np.ndarray:
    """The space-group column of X as integers, refusing anything else."""
    values = np.asarray(X, dtype=float)[:, column]
    groups = np.rint(values).astype(int)
    bad = (values != groups) | (groups < 1) | (groups > 230)
    if bad.any():
        raise ValueError(
            f"column {column} must hold space-group numbers 1-230; got "
            f"{values[bad][:5].tolist()} (is the column being scaled?)")
    return groups


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------

class SpaceGroupKernel(Kernel):
    """
    Hierarchical correlation between space groups; see the module docstring.

    Parameters
    ----------
    column : int
        Column of X that holds the space-group number. Other columns are
        ignored.
    w_lattice, w_family, w_any : float > 0
        Weights of the shared-lattice, shared-family and baseline levels,
        relative to the fixed weight 1 of "same group".
    *_bounds : pair of floats, or "fixed"
        Bounds on each weight for the hyperparameter optimiser.
    """

    def __init__(self, column=0, w_lattice=1.0, w_family=1.0, w_any=1.0,
                 w_lattice_bounds=(1e-5, 1e5), w_family_bounds=(1e-5, 1e5),
                 w_any_bounds=(1e-5, 1e5)):
        self.column = column
        self.w_lattice = w_lattice
        self.w_family = w_family
        self.w_any = w_any
        self.w_lattice_bounds = w_lattice_bounds
        self.w_family_bounds = w_family_bounds
        self.w_any_bounds = w_any_bounds

    @property
    def hyperparameter_w_lattice(self):
        return Hyperparameter("w_lattice", "numeric", self.w_lattice_bounds)

    @property
    def hyperparameter_w_family(self):
        return Hyperparameter("w_family", "numeric", self.w_family_bounds)

    @property
    def hyperparameter_w_any(self):
        return Hyperparameter("w_any", "numeric", self.w_any_bounds)

    def correlations(self) -> dict:
        """The correlation between two groups at each level of sharing."""
        S = 1.0 + self.w_lattice + self.w_family + self.w_any
        return {"same group": 1.0,
                "same lattice": (self.w_lattice + self.w_family + self.w_any) / S,
                "same family": (self.w_family + self.w_any) / S,
                "nothing shared": self.w_any / S}

    def variance_shares(self) -> dict:
        """
        Fraction of the variance that belongs to each level: particular to a
        group, shared within a lattice, within a family, or by all groups.
        """
        S = 1.0 + self.w_lattice + self.w_family + self.w_any
        return {"group": 1.0 / S, "lattice": self.w_lattice / S,
                "family": self.w_family / S, "all groups": self.w_any / S}

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return k(X, Y) and, optionally, its gradient in log-hyperparameters."""
        g1 = _groups(X, self.column)
        if Y is None:
            g2 = g1
        elif eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")
        else:
            g2 = _groups(Y, self.column)

        same = {
            "group":   (g1[:, None] == g2[None, :]).astype(float),
            "lattice": (_LATTICE_CODE[g1][:, None] == _LATTICE_CODE[g2][None, :]).astype(float),
            "family":  (_FAMILY_CODE[g1][:, None] == _FAMILY_CODE[g2][None, :]).astype(float),
            "any":     np.ones((len(g1), len(g2))),
        }
        weights = {"lattice": self.w_lattice, "family": self.w_family, "any": self.w_any}
        S = 1.0 + sum(weights.values())
        K = (same["group"] + sum(w * same[level] for level, w in weights.items())) / S

        if not eval_gradient:
            return K
        # d K / d log w = w (d_level - K) / S, for each weight left free, in
        # the order of self.hyperparameters (alphabetical), which is the
        # order of theta
        grads = [weights[h.name[2:]] * (same[h.name[2:]] - K) / S
                 for h in self.hyperparameters if not h.fixed]
        if not grads:
            return K, np.empty((len(g1), len(g1), 0))
        return K, np.stack(grads, axis=2)

    def diag(self, X):
        _groups(X, self.column)
        return np.ones(len(X))

    def is_stationary(self):
        return False

    @property
    def requires_vector_input(self):
        return True

    def __repr__(self):
        rho = self.correlations()
        return (f"SpaceGroupKernel(column={self.column}, "
                f"rho_lattice={rho['same lattice']:.3g}, "
                f"rho_family={rho['same family']:.3g}, "
                f"rho_any={rho['nothing shared']:.3g})")


class OnColumns(Kernel):
    """
    A kernel evaluated on some columns of X only.

    scikit-learn kernels see every column they are given; this one hands its
    base kernel the listed columns, so a Matérn over the cell parameters
    does not also measure distances along the space-group column.
    Hyperparameters are the base kernel's, named kernel__<name>.
    """

    def __init__(self, kernel, columns):
        self.kernel = kernel
        self.columns = columns

    def _take(self, X):
        return None if X is None else np.asarray(X)[:, list(self.columns)]

    def get_params(self, deep=True):
        params = dict(kernel=self.kernel, columns=self.columns)
        if deep:
            params.update(("kernel__" + k, v)
                          for k, v in self.kernel.get_params().items())
        return params

    @property
    def hyperparameters(self):
        return [Hyperparameter("kernel__" + h.name, h.value_type, h.bounds,
                               h.n_elements)
                for h in self.kernel.hyperparameters]

    @property
    def theta(self):
        return self.kernel.theta

    @theta.setter
    def theta(self, theta):
        self.kernel.theta = theta

    @property
    def bounds(self):
        return self.kernel.bounds

    def __call__(self, X, Y=None, eval_gradient=False):
        return self.kernel(self._take(X), self._take(Y), eval_gradient=eval_gradient)

    def diag(self, X):
        return self.kernel.diag(self._take(X))

    def is_stationary(self):
        return self.kernel.is_stationary()

    @property
    def requires_vector_input(self):
        return True

    def __eq__(self, b):
        return (type(self) is type(b) and self.kernel == b.kernel
                and list(self.columns) == list(b.columns))

    def __repr__(self):
        return f"OnColumns({self.kernel}, columns={list(self.columns)})"
