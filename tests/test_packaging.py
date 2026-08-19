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

"""What `pip install` will produce, checked without building anything.

Packaging metadata rots quietly: a new subpackage that nobody listed is simply
missing from the wheel, and the failure surfaces on someone else's machine as
an ImportError. These tests read pyproject.toml and compare it against what is
actually on disk, so that adding a module without packaging it fails here.

They also pin two facts that are easy to break by accident: the console scripts
must point at functions that exist, and CITATION.cff must keep describing the
*archived* release (1.0.0, the DOI the article cites) rather than drifting with
the development version.
"""

import importlib
import os
import re
import tomllib
import unittest

import cryal


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_pyproject():
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        return tomllib.load(f)


def packages_on_disk():
    """Every importable subpackage under cryal/, as dotted names."""
    found = set()
    for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, "cryal")):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        if "__init__.py" in filenames:
            rel = os.path.relpath(dirpath, ROOT)
            found.add(rel.replace(os.sep, "."))
    return found


class TestProjectMetadata(unittest.TestCase):

    def setUp(self):
        self.pyproject = load_pyproject()
        self.project = self.pyproject["project"]

    def test_the_version_comes_from_the_package(self):
        attr = self.pyproject["tool"]["setuptools"]["dynamic"]["version"]["attr"]
        self.assertEqual(attr, "cryal.__version__")
        self.assertRegex(cryal.__version__, r"^\d+\.\d+\.\d+")

    def test_the_development_version_is_ahead_of_the_archived_release(self):
        self.assertNotEqual(cryal.__version__, "1.0.0")

    def test_citation_still_describes_the_archived_release(self):
        # The article cites version 1.0.0 and its Zenodo DOI. This file must
        # keep pointing there until a new release is actually archived.
        with open(os.path.join(ROOT, "CITATION.cff")) as f:
            citation = f.read()
        self.assertIn("version: 1.0.0", citation)
        self.assertIn("10.5281/zenodo.21896733", citation)

    def test_declared_files_exist(self):
        paths = [self.project["readme"]] + list(self.project["license-files"])
        for path in paths:
            with self.subTest(path=path):
                self.assertTrue(os.path.exists(os.path.join(ROOT, path)))

    def test_python_requirement_is_declared(self):
        self.assertRegex(self.project["requires-python"], r">=3\.\d+")


class TestPackagesAreComplete(unittest.TestCase):
    """A subpackage that nobody listed is missing from the wheel."""

    def setUp(self):
        self.declared = set(load_pyproject()["tool"]["setuptools"]["packages"])

    def test_every_subpackage_on_disk_is_declared(self):
        self.assertEqual(packages_on_disk() - self.declared, set())

    def test_nothing_declared_is_missing_from_disk(self):
        self.assertEqual(self.declared - packages_on_disk(), set())

    def test_tests_and_tools_are_not_shipped_as_top_level_packages(self):
        for name in ("tests", "tools"):
            self.assertNotIn(name, self.declared)


class TestConsoleScripts(unittest.TestCase):

    def setUp(self):
        self.scripts = load_pyproject()["project"]["scripts"]

    def test_the_entry_points_resolve_to_callables(self):
        for name, target in self.scripts.items():
            with self.subTest(script=name):
                module_name, _, func_name = target.partition(":")
                module = importlib.import_module(module_name)
                self.assertTrue(callable(getattr(module, func_name)))

    def test_the_run_command_is_provided(self):
        self.assertEqual(self.scripts["cryal"], "cryal.cli:main")

    def test_the_checkout_script_and_the_command_share_one_implementation(self):
        # run_cryal.py must stay a wrapper: two copies of the argument
        # handling would drift, and the article documents the script.
        with open(os.path.join(ROOT, "run_cryal.py")) as f:
            source = f.read()
        self.assertIn("from cryal.cli import main", source)

    def test_the_patcher_wrapper_still_works_from_a_checkout(self):
        with open(os.path.join(ROOT, "tools", "patch_mliap_model.py")) as f:
            source = f.read()
        self.assertIn("from cryal.tools.patch_mliap_model import main", source)


class TestDependencies(unittest.TestCase):
    """pyproject declares bounds; requirements.txt pins. Same five names."""

    @staticmethod
    def _name(spec):
        return re.split(r"[<>=!~\[ ]", spec.strip(), maxsplit=1)[0].lower()

    def setUp(self):
        self.declared = {self._name(d)
                         for d in load_pyproject()["project"]["dependencies"]}
        with open(os.path.join(ROOT, "requirements.txt")) as f:
            self.pinned = {self._name(line) for line in f
                           if line.strip() and not line.startswith("#")}

    def test_the_two_lists_name_the_same_packages(self):
        self.assertEqual(self.declared, self.pinned)

    def test_the_declared_dependencies_are_lower_bounds(self):
        for spec in load_pyproject()["project"]["dependencies"]:
            with self.subTest(spec=spec):
                self.assertIn(">=", spec)

    def test_torch_is_not_a_dependency(self):
        # Pulling torch into this environment is what broke the exported MACE
        # model once already; it belongs to the LAMMPS side, not here.
        for name in ("torch", "mace-torch", "pymatgen", "rdkit"):
            self.assertNotIn(name, self.declared)


class TestSourceDistribution(unittest.TestCase):

    def test_manifest_lists_files_that_exist(self):
        with open(os.path.join(ROOT, "MANIFEST.in")) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2 and parts[0] == "include":
                    with self.subTest(path=parts[1]):
                        self.assertTrue(
                            os.path.exists(os.path.join(ROOT, parts[1])))


if __name__ == "__main__":
    unittest.main()
