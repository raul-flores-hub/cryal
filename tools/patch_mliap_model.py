#!/usr/bin/env python3
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
tools/patch_mliap_model.py — wrapper kept for the documented path.

The patcher moved into the package (`cryal/tools/patch_mliap_model.py`) so an
installed CrYAL carries it as well, where it is available as the command
`cryal-patch-mliap-model`. INSTALL.md documents this path, and running

    python tools/patch_mliap_model.py <model.pt>

from a checkout must keep working.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from cryal.tools.patch_mliap_model import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
