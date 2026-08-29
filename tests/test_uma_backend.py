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

"""UMA as a named backend, and the one thing it must not let you do silently.

UMA is an ASE calculator, so the `ase` backend could in principle drive it.
It gets a name of its own because of `umaTask`: each head was trained on a
different domain and each has its own zero of energy -- for one periodic
C50Au4 cell, `omat` gives -466.55 eV and `oc20` -447.06 eV. A run that mixes
two heads produces numbers that are not comparable, and no inspection of the
database would reveal it. So the task is mandatory, it is checked against the
installed fairchem, and the check runs on every remote worker too.

fairchem is not a dependency of CrYAL, so everything that needs it skips when
it is absent.
"""

import unittest

from cryal.backends import available_backends, backend_class
from cryal.backends.ase_calculator import AseCalculatorBackend
from cryal.backends.uma import UmaBackend, _installed_tasks
from cryal.config import Config

try:
    import fairchem.core  # noqa: F401
    HAVE_FAIRCHEM = True
except ImportError:
    HAVE_FAIRCHEM = False


class TestRegistration(unittest.TestCase):

    def test_selectable_by_name(self):
        self.assertIn("uma", available_backends())
        self.assertIs(backend_class("uma"), UmaBackend)
        self.assertIs(backend_class("fairchem"), UmaBackend)

    def test_reuses_the_ase_relaxation(self):
        # The point is the calculator and its task, not a second copy of the
        # relaxation loop.
        self.assertTrue(issubclass(UmaBackend, AseCalculatorBackend))
        self.assertIs(UmaBackend.relax, AseCalculatorBackend.relax)

    def test_not_safe_to_run_twice_in_one_process(self):
        # One shared predict unit, same as the ase backend. cryal.parallel
        # clamps local slots to 1 on this; remote workers are unaffected.
        self.assertFalse(UmaBackend.thread_safe)

    def test_needs_nothing_shipped_to_a_worker(self):
        # The weights are the worker's own installation to provide; there is
        # no input script to copy.
        cfg = Config(energy_backend="uma", uma_task="omc")
        self.assertEqual(UmaBackend.job_files(cfg), {})
        self.assertEqual(UmaBackend.required_commands(cfg), [])

    def test_the_default_backend_is_still_lammps(self):
        self.assertEqual(Config().energy_backend, "lammps_mace")


class TestConfig(unittest.TestCase):

    def test_task_has_no_default(self):
        # Deliberate: a default would pick a domain for the user in silence.
        self.assertEqual(Config().uma_task, "")

    def test_keys_are_parsed(self):
        import os, tempfile
        from cryal.config import load_config
        text = ("% GENERAL\nmoleculeFile = examples/benzene.xyz\nZ = 4\n"
                "energyBackend = uma\numaModel = uma-s-1p1\numaTask = omc\n"
                "umaDevice = cpu\numaInferenceSettings = default\n")
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(text); path = f.name
        try:
            if not HAVE_FAIRCHEM:
                self.skipTest("fairchem-core not installed")
            cfg = load_config(path)
            self.assertEqual(cfg.uma_model, "uma-s-1p1")
            self.assertEqual(cfg.uma_task, "omc")
            self.assertEqual(cfg.uma_device, "cpu")
        finally:
            os.unlink(path)


@unittest.skipUnless(HAVE_FAIRCHEM, "fairchem-core not installed")
class TestValidation(unittest.TestCase):
    """Every one of these fails at load time, and at preflight on each worker."""

    def _check(self, **kw):
        UmaBackend.validate_config(Config(energy_backend="uma", **kw))

    def test_a_workable_configuration_passes(self):
        self._check(uma_task="omc")

    def test_a_missing_task_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            self._check(uma_task="")
        self.assertIn("mandatory", str(cm.exception))
        # The message must name the choices, or the user cannot act on it.
        self.assertIn("omc", str(cm.exception))

    def test_an_unknown_task_names_the_alternatives(self):
        with self.assertRaises(ValueError) as cm:
            self._check(uma_task="molecular_crystals")
        self.assertIn("molecular_crystals", str(cm.exception))
        self.assertIn("omc", str(cm.exception))

    def test_an_unknown_model_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            self._check(uma_task="omc", uma_model="uma-xl-9")
        self.assertIn("uma-xl-9", str(cm.exception))

    def test_a_bad_device_is_refused(self):
        with self.assertRaises(ValueError):
            self._check(uma_task="omc", uma_device="tpu")

    def test_a_bad_optimizer_is_refused(self):
        with self.assertRaises(ValueError):
            self._check(uma_task="omc", ase_optimizer="gradient_descent")

    def test_the_task_list_comes_from_the_installation(self):
        # Hardcoding it would reject a valid head on a newer fairchem: 2.13.0
        # and 2.21.0 do not offer the same set.
        tasks = _installed_tasks()
        self.assertIn("omc", tasks)
        self.assertIn("omat", tasks)


class TestMissingFairchem(unittest.TestCase):

    @unittest.skipIf(HAVE_FAIRCHEM, "fairchem-core is installed here")
    def test_the_error_says_how_to_fix_it(self):
        with self.assertRaises(ValueError) as cm:
            UmaBackend.validate_config(Config(energy_backend="uma", uma_task="omc"))
        msg = str(cm.exception)
        self.assertIn("pip install fairchem-core", msg)
        # And that the ORCA external-tools install is a different thing.
        self.assertIn("ORCA", msg)


if __name__ == "__main__":
    unittest.main()
