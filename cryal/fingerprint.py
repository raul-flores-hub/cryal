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
fingerprint.py — telling two crystal structures apart by their packing.

Two relaxed structures with the same energy per molecule and the same
density are usually, but not always, the same minimum: a search that finds
one deep minimum twenty times must count it once, and two polymorphs that
happen to agree in both must count twice. The fingerprint here decides.

It is the one of Oganov and Valle (J. Chem. Phys. 130, 104504, 2009): for
every pair of elements A, B, the radial distribution of B atoms around A
atoms out to `rmax`, normalized so that an ideal gas gives zero,

    F_AB(R) = sum_{i in A, j in B} delta_sigma(R - R_ij)
              / (4 pi R^2 (N_A N_B / V) Delta)  -  1,

binned every Delta and smeared by a Gaussian of width sigma, and two
structures are compared by the cosine distance

    D = (1 - sum_AB w_AB F1_AB . F2_AB
             / sqrt(sum_AB w_AB F1_AB . F1_AB  *  sum_AB w_AB F2_AB . F2_AB)) / 2,

w_AB = N_A N_B / sum N_A N_B: 0 for identical structures, 1 at most. It is
built from distances alone, so it does not change with the cell's setting,
its origin, the order of the atoms or the orientation of the crystal.

One departure, for molecular crystals: only distances between atoms of
different molecules count (and an atom's distances to its own molecule's
periodic images). Distances within a molecule are nearly the same in every
polymorph of it and would fill the short-range part of F with the molecule
rather than the packing, which is what tells polymorphs apart. Atoms are
taken as laid out molecule by molecule, as every structure in a CrYAL
database is (see genetic.py), and each molecule is first made whole along
its bonds: a pair within one molecule is told from a pair with its periodic
image by geometry, and a cell whose atoms were wrapped one by one splits
molecules across its faces.
"""

from typing import Dict, Tuple

import numpy as np

#: Defaults: Oganov and Valle's range, finer bins for molecular contacts.
RMAX, DELTA, SIGMA = 10.0, 0.05, 0.05


def fingerprint(atoms, mol, rmax: float = RMAX, delta: float = DELTA,
                sigma: float = SIGMA) -> Dict[Tuple[str, str], np.ndarray]:
    """
    {(A, B): F_AB} for every pair of elements A <= B, intermolecular
    distances only, plus the weights under the key 'w'. `mol` is the run's
    genetic.Molecule, whose bonds make each molecule whole.
    """
    from .genetic import _whole, image_pairs    # genetic imports nothing from here

    symbols = np.array(atoms.get_chemical_symbols())
    n = len(symbols)
    if n % mol.size:
        raise ValueError(f"{n} atoms are not whole molecules of {mol.size}")
    molid = np.arange(n) // mol.size
    cell = atoms.cell.array
    pos = np.vstack([_whole(atoms.positions[m:m + mol.size], cell, mol)
                     for m in range(0, n, mol.size)])
    i, j, d, _, S = image_pairs(pos, cell, rmax)
    inter = (molid[i] != molid[j]) | np.any(S != 0, axis=1)
    i, j, d = i[inter], j[inter], d[inter]

    volume = atoms.get_volume()
    edges = np.arange(0.0, rmax + delta, delta)
    centres = 0.5 * (edges[1:] + edges[:-1])
    # Gaussian smearing as a convolution over the bins.
    half = int(np.ceil(4 * sigma / delta))
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) * delta / sigma) ** 2)
    kernel /= kernel.sum()

    elements = sorted(set(symbols))
    counts = {e: int(np.sum(symbols == e)) for e in elements}
    out, weights = {}, {}
    for ai, A in enumerate(elements):
        for B in elements[ai:]:
            sel = (symbols[i] == A) & (symbols[j] == B)
            hist, _ = np.histogram(d[sel], bins=edges)
            hist = np.convolve(hist.astype(float), kernel, mode="same")
            norm = 4 * np.pi * centres ** 2 * (counts[A] * counts[B] / volume) * delta
            out[(A, B)] = hist / norm - 1.0
            weights[(A, B)] = counts[A] * counts[B]
    total = sum(weights.values())
    out['w'] = {k: w / total for k, w in weights.items()}
    return out


def distance(f1, f2) -> float:
    """Oganov–Valle cosine distance between two fingerprints, in [0, 1]."""
    w = f1['w']
    keys = [k for k in w if k in f2]
    dot = sum(w[k] * float(f1[k] @ f2[k]) for k in keys)
    n1 = sum(w[k] * float(f1[k] @ f1[k]) for k in keys)
    n2 = sum(w[k] * float(f2[k] @ f2[k]) for k in keys)
    if n1 <= 0 or n2 <= 0:
        return 0.0 if n1 == n2 else 1.0
    return float(np.clip(0.5 * (1.0 - dot / np.sqrt(n1 * n2)), 0.0, 1.0))
