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

"""A trivial ASE calculator, used to exercise the `ase` backend in tests.

It is deliberately not physics: the point is to drive the backend's plumbing
(dotted-path import, keyword arguments, optimizer selection, convergence
handling, step-directory output) without pulling in a machine-learned
potential, a GPU, or any dependency the test suite does not already have.

`force` = 0 makes the optimizer converge on its first evaluation; a non-zero
`force` never converges, which is how the non-convergence path is tested.
"""

import numpy as np
from ase.calculators.calculator import Calculator, all_changes


class StubCalculator(Calculator):
    """Constant energy, uniform forces, zero stress."""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, energy=-1000.0, force=0.0, **kwargs):
        super().__init__(**kwargs)
        self.constant_energy = float(energy)
        self.constant_force = float(force)

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        n = len(self.atoms)
        forces = np.zeros((n, 3))
        forces[:, 0] = self.constant_force
        self.results = {
            "energy": self.constant_energy,
            "free_energy": self.constant_energy,
            "forces": forces,
            "stress": np.zeros(6),
        }


def make_stub(**kwargs):
    """Factory form of the same calculator (backends accept either)."""
    return StubCalculator(**kwargs)


class VolumeStub(Calculator):
    """Energy that falls with density, zero forces and stress.

    The optimizer converges at once, so a relaxation returns the structure it
    was given, but structures now differ in energy: denser is lower. That is
    enough of a landscape for a search to have something to find, which a
    constant energy is not.

    With `v0` (Å³) the energy is lowest at that volume instead, rising
    quadratically on both sides. Without it, a search drives its population
    to the densest, thinnest cells the generator can make, which no real
    potential rewards and where hardly any child fits.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, energy=-1000.0, per_A3=0.01, v0=None, **kwargs):
        super().__init__(**kwargs)
        self.e0 = float(energy)
        self.per_A3 = float(per_A3)
        self.v0 = None if v0 is None else float(v0)

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        v = self.atoms.get_volume()
        if self.v0 is None:
            e = self.e0 + self.per_A3 * v
        else:
            e = self.e0 + self.per_A3 * (v - self.v0) ** 2 / self.v0
        self.results = {
            "energy": e,
            "free_energy": e,
            "forces": np.zeros((len(self.atoms), 3)),
            "stress": np.zeros(6),
        }
