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
run_cryal.py — CrYAL entry point, for running from a checkout.

Usage:
    python run_cryal.py              # reads INPUT.txt in current directory
    python run_cryal.py my_INPUT.txt # reads a custom input file
    python run_cryal.py --resume     # continue an interrupted run (e.g. after
                                     # a power outage) from its output_dir

This is the invocation the article and the README document, and it works from
a plain checkout with no installation. The command line itself lives in
`cryal/cli.py`, so that `pip install cryal` also provides a `cryal` command;
both run exactly the same code.

See INPUT.txt for all configurable parameters.
"""

import sys

from cryal.cli import main

if __name__ == "__main__":
    sys.exit(main())
