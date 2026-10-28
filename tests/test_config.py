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


"""cryal.toml loading, the conversion of old inputs, and the gp.enabled
ablation switch.

The backward-compatibility tests here are the ones that matter most: the run
reported in the accompanying article was produced by an input in the old
format (% SECTION headers), from before the useGP key existed. CrYAL no
longer runs from that format, so `cryal convert` must turn it into a
cryal.toml that means the same run -- surrogate on included -- or the
published result would no longer be reproducible from its own archive.
"""

import os
import tempfile
import unittest

from cryal.config import Config, load_config
from cryal.legacy import convert, parse_input

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_input(body: str, suffix: str = "_cryal.toml") -> str:
    """Write body to a temporary file and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as f:
        f.write(body)
    return path


MINIMAL = """
[general]
molecule = "examples/benzene.xyz"
Z = 4
"""


class _TempFiles(unittest.TestCase):

    def setUp(self):
        self._paths = []

    def tearDown(self):
        for p in self._paths:
            os.unlink(p)

    def _write(self, body, suffix="_cryal.toml"):
        p = write_input(body, suffix)
        self._paths.append(p)
        return p

    def _load(self, body):
        return load_config(self._write(body))


class TestLoader(_TempFiles):

    def test_values_reach_the_config(self):
        cfg = self._load(MINIMAL + "\n[active_learning]\ncycles = 7  # seven\n")
        self.assertEqual((cfg.Z, cfg.num_cycles), (4, 7))

    def test_missing_keys_keep_their_defaults(self):
        self.assertEqual(self._load(MINIMAL).pool_size, Config().pool_size)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_config("/nonexistent/cryal.toml")

    def test_space_group_table_keeps_file_order(self):
        cfg = self._load(MINIMAL + "\n[space_groups]\nweights = { 61 = 2, 14 = 36.5 }\n")
        self.assertEqual((cfg.space_groups, cfg.sg_weights), ([61, 14], [2.0, 36.5]))

    def test_ranges_fill_both_ends(self):
        cfg = self._load(MINIMAL + "\n[volume]\ndensity = [0.9, 1.4]\n"
                         "\n[monoclinic_beta]\nrange = [95, 120]\n")
        self.assertEqual((cfg.density_min, cfg.density_max), (0.9, 1.4))
        self.assertEqual((cfg.beta_min, cfg.beta_max), (95.0, 120.0))

    def test_volume_without_bounds_is_estimated(self):
        cfg = self._load(MINIMAL)
        self.assertIsNone(cfg.target_volume_min)
        self.assertGreater(cfg.volume_min(), 0)
        self.assertGreater(cfg.volume_max(), cfg.volume_min())


class TestLoaderRefusesWhatItCannotMean(_TempFiles):
    """Every one of these used to be read as a default, without a word."""

    def _refused(self, body, *fragments):
        with self.assertRaises(ValueError) as ctx:
            self._load(MINIMAL + body)
        for fragment in fragments:
            self.assertIn(fragment, str(ctx.exception))

    def test_a_misspelt_key_is_named_with_the_right_one(self):
        self._refused("\n[active_learning]\ncycle = 20\n",
                      "active_learning.cycle", "did you mean 'cycles'")

    def test_a_key_in_the_wrong_table_is_sent_home(self):
        self._refused("\n[gp]\ncycles = 20\n", "gp.cycles", "belongs in [active_learning]")

    def test_a_misspelt_table_is_named(self):
        self._refused("\n[genral]\nZ = 2\n", "genral", "did you mean 'general'")

    def test_a_key_outside_any_table(self):
        with self.assertRaises(ValueError) as ctx:
            load_config(self._write('Z = 4\n'))
        self.assertIn("belongs in [general]", str(ctx.exception))

    def test_a_string_where_a_boolean_goes(self):
        # the old format read `useGP = ture` as false
        self._refused('\n[gp]\nenabled = "ture"\n', "gp.enabled", "true or false")

    def test_a_float_where_an_integer_goes(self):
        self._refused("\n[active_learning]\ncycles = 7.5\n", "active_learning.cycles")

    def test_an_unknown_choice(self):
        # the old format ran any kernel name but 'rbf' as Matérn
        self._refused('\n[gp]\nkernel = "rbff"\n', "gp.kernel", "'matern25' or 'rbf'")

    def test_a_space_group_out_of_range(self):
        self._refused("\n[space_groups]\nweights = { 14 = 1, 231 = 1 }\n",
                      "'231' is not a space group")

    def test_a_range_upside_down(self):
        self._refused("\n[volume]\ndensity = [1.6, 0.8]\n", "volume.density")

    def test_a_syntax_error_names_the_file(self):
        with self.assertRaises(ValueError) as ctx:
            load_config(self._write("[general\n"))
        self.assertIn("_cryal.toml", str(ctx.exception))


class TestUseGP(_TempFiles):
    """The ablation switch for the Bayesian guidance."""

    def test_default_is_on(self):
        self.assertTrue(Config().use_gp)

    def test_input_without_the_key_keeps_the_surrogate_on(self):
        self.assertTrue(self._load(MINIMAL).use_gp)

    def test_false_switches_the_surrogate_off(self):
        self.assertFalse(self._load(MINIMAL + "\n[gp]\nenabled = false\n").use_gp)


OLD = """
% GENERAL
moleculeFile = examples/benzene.xyz
Z = 4
"""


class TestOldInputs(_TempFiles):
    """The format before cryal.toml: refused by the loader, converted by
    `cryal convert` into the run it described."""

    def _converted(self, body):
        return load_config(self._write(convert(self._write(OLD + body, "_old.in"))))

    def test_the_loader_points_at_the_converter(self):
        with self.assertRaises(ValueError) as ctx:
            load_config(self._write(OLD, "_old.in"))
        self.assertIn("cryal convert", str(ctx.exception))

    def test_an_old_input_without_usegp_keeps_the_surrogate_on(self):
        # This is the backward-compatibility contract: every input written
        # before the flag existed -- including the one archived with v1.0.0
        # -- must still mean "surrogate on".
        self.assertTrue(self._converted("").use_gp)

    def test_old_boolean_spellings_mean_what_they_meant(self):
        for value, meant in (("true", True), ("True", True), ("1", True),
                             ("yes", True), ("false", False), ("no", False),
                             ("ture", False)):
            with self.subTest(value=value):
                self.assertIs(self._converted(f"useGP = {value}\n").use_gp, meant)

    def test_old_values_are_typed_as_the_old_loader_typed_them(self):
        cfg = self._converted(
            "numCycles = 7  # seven cycles\n"
            "spaceGroups = 14 2 64\nsgWeights = 3 2 1\n"
            "targetVolumeMin = 400\ntargetVolumeMax = auto\ndensityMin = 0.9\n"
            "energySanityMax = none\ngpKernel = Matern\n")
        self.assertEqual(cfg.num_cycles, 7)
        self.assertEqual((cfg.space_groups, cfg.sg_weights), ([14, 2, 64], [3.0, 2.0, 1.0]))
        self.assertEqual((cfg.target_volume_min, cfg.target_volume_max), (400.0, None))
        self.assertEqual((cfg.density_min, cfg.density_max), (0.9, Config().density_max))
        self.assertIsNone(cfg.energy_sanity_max)
        self.assertEqual(cfg.gp_kernel, "matern25")   # anything but rbf was Matérn

    def test_keys_the_old_loader_ignored_are_listed_not_dropped(self):
        text = convert(self._write(OLD + "someKeyNobodyKnows = 3\n", "_old.in"))
        self.assertIn("somekeynobodyknows = 3", text)

    def test_old_parser_basics(self):
        raw = parse_input(self._write("\n# c\n% GENERAL\nMoLeCuLeFiLe = foo.xyz  # x\n",
                                      "_old.in"))
        self.assertEqual(raw, {"moleculefile": "foo.xyz"})


class TestShippedInput(unittest.TestCase):

    def test_the_shipped_cryal_toml_loads(self):
        cwd = os.getcwd()
        os.chdir(ROOT)
        try:
            cfg = load_config("cryal.toml")
        finally:
            os.chdir(cwd)
        self.assertEqual(cfg.molecule_file, "examples/benzene.xyz")


class TestPublishedInputs(unittest.TestCase):
    """Guard the two runs reported in the article, when they are on disk.

    These live outside the repository, so the tests skip when the paths are
    absent -- they are a local safety net, not part of the portable suite.
    They are in the old format, so they go through the converter.
    """

    RUNS = {
        "run2 (BACH, article)":
            "/home/ibeth/RAUL/uspexrunz/run2-USPEX-lammpsinput/INPUT.txt",
        "run3 (no-GP ablation)":
            "/home/ibeth/RAUL/uspexrunz/run3-ablation-noGP/INPUT.txt",
    }

    def _use_gp(self, path):
        toml_path = write_input(convert(path))
        cwd = os.getcwd()
        os.chdir(os.path.dirname(path))
        try:
            return load_config(toml_path).use_gp
        finally:
            os.chdir(cwd)
            os.unlink(toml_path)

    def test_run2_has_the_surrogate_on(self):
        path = self.RUNS["run2 (BACH, article)"]
        if not os.path.exists(path):
            self.skipTest("run2 input not available on this machine")
        self.assertTrue(self._use_gp(path))

    def test_ablation_has_the_surrogate_off(self):
        path = self.RUNS["run3 (no-GP ablation)"]
        if not os.path.exists(path):
            self.skipTest("ablation input not available on this machine")
        self.assertFalse(self._use_gp(path))


if __name__ == "__main__":
    unittest.main()
