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

"""INPUT.txt parsing and the useGP ablation switch.

The backward-compatibility test here is the one that matters most: the
configuration archived with v1.0.0 predates the useGP key, and the run
reported in the accompanying article was produced by an input that does not
contain it. If that input ever stopped meaning "surrogate on", the published
result would no longer be reproducible from its own archive.
"""

import os
import tempfile
import unittest

from cryal.config import Config, load_config, parse_input


def write_input(body: str) -> str:
    """Write body to a temporary INPUT.txt and return its path."""
    fd, path = tempfile.mkstemp(suffix="_INPUT.txt")
    with os.fdopen(fd, "w") as f:
        f.write(body)
    return path


MINIMAL = """
% GENERAL
moleculeFile = examples/benzene.xyz
Z = 4
"""


class TestParser(unittest.TestCase):

    def setUp(self):
        self._paths = []

    def tearDown(self):
        for p in self._paths:
            os.unlink(p)

    def _write(self, body):
        p = write_input(body)
        self._paths.append(p)
        return p

    def test_section_headers_are_dropped(self):
        raw = parse_input(self._write(MINIMAL))
        self.assertNotIn("% general", raw)
        self.assertEqual(raw["z"], "4")

    def test_keys_are_case_insensitive(self):
        raw = parse_input(self._write("% GENERAL\nMoLeCuLeFiLe = foo.xyz\n"))
        self.assertEqual(raw["moleculefile"], "foo.xyz")

    def test_inline_comments_are_stripped(self):
        cfg = load_config(self._write(MINIMAL + "numCycles = 7  # seven cycles\n"))
        self.assertEqual(cfg.num_cycles, 7)

    def test_blank_and_comment_lines_are_ignored(self):
        raw = parse_input(self._write("\n\n# a comment\n% GENERAL\n\nZ = 2\n"))
        self.assertEqual(len(raw), 1)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            parse_input("/nonexistent/INPUT.txt")

    def test_unknown_key_does_not_break_the_load(self):
        cfg = load_config(self._write(MINIMAL + "someKeyNobodyKnows = 3\n"))
        self.assertEqual(cfg.Z, 4)


class TestUseGP(unittest.TestCase):
    """The ablation switch for the Bayesian guidance."""

    def setUp(self):
        self._paths = []

    def tearDown(self):
        for p in self._paths:
            os.unlink(p)

    def _load(self, body):
        p = write_input(body)
        self._paths.append(p)
        return load_config(p)

    def test_default_is_on(self):
        self.assertTrue(Config().use_gp)

    def test_input_without_the_key_keeps_the_surrogate_on(self):
        # This is the backward-compatibility contract: every input written
        # before the flag existed -- including the one archived with v1.0.0 --
        # must still mean "surrogate on".
        self.assertTrue(self._load(MINIMAL).use_gp)

    def test_false_switches_the_surrogate_off(self):
        self.assertFalse(self._load(MINIMAL + "useGP = false\n").use_gp)

    def test_truthy_spellings(self):
        for value in ("true", "True", "1", "yes"):
            with self.subTest(value=value):
                self.assertTrue(self._load(MINIMAL + f"useGP = {value}\n").use_gp)

    def test_falsy_spellings(self):
        for value in ("false", "False", "0", "no"):
            with self.subTest(value=value):
                self.assertFalse(self._load(MINIMAL + f"useGP = {value}\n").use_gp)


class TestPublishedInputs(unittest.TestCase):
    """Guard the two runs reported in the article, when they are on disk.

    These live outside the repository, so the tests skip when the paths are
    absent -- they are a local safety net, not part of the portable suite.
    """

    RUNS = {
        "run2 (BACH, article)":
            "/home/ibeth/RAUL/uspexrunz/run2-USPEX-lammpsinput/INPUT.txt",
        "run3 (no-GP ablation)":
            "/home/ibeth/RAUL/uspexrunz/run3-ablation-noGP/INPUT.txt",
    }

    def test_run2_has_the_surrogate_on(self):
        path = self.RUNS["run2 (BACH, article)"]
        if not os.path.exists(path):
            self.skipTest("run2 input not available on this machine")
        self.assertTrue(load_config(path).use_gp)

    def test_ablation_has_the_surrogate_off(self):
        path = self.RUNS["run3 (no-GP ablation)"]
        if not os.path.exists(path):
            self.skipTest("ablation input not available on this machine")
        self.assertFalse(load_config(path).use_gp)


if __name__ == "__main__":
    unittest.main()
