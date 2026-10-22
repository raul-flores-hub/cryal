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

"""The run log must say each thing once.

A duplicated log is not merely untidy: these logs are how a long search is
audited afterwards, and a reader who cannot tell one evaluation from two
printings of the same evaluation cannot count anything in them.
"""

import logging
import os
import shutil
import tempfile
import unittest

from cryal.utils import setup_logger


class TestLoggerIsolation(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cryal_log_")
        self.log_file = os.path.join(self.tmp, "run.log")
        self.name = "cryal.test.isolation"
        logging.getLogger(self.name).handlers.clear()

    def tearDown(self):
        logging.getLogger(self.name).handlers.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_records_do_not_reach_the_root_logger(self):
        # fairchem calls logging.basicConfig() while building the UMA
        # calculator, so by the first local relaxation the root logger has a
        # handler whether CrYAL asked for one or not. Propagating there prints
        # every line a second time as `INFO:cryal:...`.
        root = logging.getLogger()
        seen = []

        class Capture(logging.Handler):
            def emit(self, record):
                seen.append(record.getMessage())

        handler = Capture()
        root.addHandler(handler)
        try:
            logger = setup_logger(self.name, self.log_file)
            logger.info("one evaluation")
        finally:
            root.removeHandler(handler)

        self.assertEqual(seen, [])
        self.assertFalse(logger.propagate)

    def test_the_line_is_still_written_once(self):
        # Not propagating must not mean not logging.
        logger = setup_logger(self.name, self.log_file)
        logger.info("one evaluation")
        for h in logger.handlers:
            h.flush()
        with open(self.log_file) as f:
            body = f.read()
        self.assertEqual(body.count("one evaluation"), 1)


if __name__ == "__main__":
    unittest.main()
