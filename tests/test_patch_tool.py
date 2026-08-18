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

"""The ML-IAP model patcher.

The buffer-detection logic is pure and is tested without torch. The tests that
actually register buffers need torch and skip when it is absent, so the suite
still runs in an environment that only has the CrYAL dependencies.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "tools"))

from patch_mliap_model import BUFFERS, apply_patch, default_output, missing_buffers  # noqa: E402

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


class Bare:
    """Stand-in for a model exported before the buffers existed."""


class Complete:
    total_charge = 0.0
    total_spin = 1.0


class TestDetection(unittest.TestCase):

    def test_reports_both_buffers_missing(self):
        self.assertEqual(missing_buffers(Bare()), list(BUFFERS))

    def test_reports_nothing_missing_on_a_current_model(self):
        self.assertEqual(missing_buffers(Complete()), [])

    def test_reports_only_what_is_absent(self):
        half = Bare()
        half.total_charge = 0.0
        self.assertEqual(missing_buffers(half), ["total_spin"])


class TestOutputNaming(unittest.TestCase):

    def test_appends_v2_before_the_extension(self):
        self.assertEqual(default_output("MACE-OFF23_small.model-mliap_lammps.pt"),
                         "MACE-OFF23_small.model-mliap_lammps_v2.pt")

    def test_keeps_the_directory(self):
        self.assertEqual(default_output("/models/m.pt"), "/models/m_v2.pt")


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestPatching(unittest.TestCase):

    def _module(self):
        return torch.nn.Module()

    def test_adds_both_buffers_with_the_neutral_defaults(self):
        m = self._module()
        added = apply_patch(m, torch=torch)
        self.assertEqual(added, list(BUFFERS))
        self.assertAlmostEqual(float(m.total_charge), 0.0)
        self.assertAlmostEqual(float(m.total_spin), 1.0)

    def test_honours_explicit_charge_and_spin(self):
        m = self._module()
        apply_patch(m, charge=-1.0, spin=2.0, torch=torch)
        self.assertAlmostEqual(float(m.total_charge), -1.0)
        self.assertAlmostEqual(float(m.total_spin), 2.0)

    def test_is_idempotent(self):
        m = self._module()
        apply_patch(m, torch=torch)
        self.assertEqual(apply_patch(m, torch=torch), [])

    def test_buffers_are_registered_not_plain_attributes(self):
        # register_buffer is what makes them move with .to(device) and appear
        # in the state dict; a plain attribute would not survive the transfer.
        m = self._module()
        apply_patch(m, torch=torch)
        self.assertEqual(set(dict(m.named_buffers())), set(BUFFERS))


if __name__ == "__main__":
    unittest.main()
