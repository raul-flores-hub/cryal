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
CrYAL — Crystal Structure Prediction via Active Learning
=========================================================
Blind crystal structure prediction combining:
  - PyXtal random structure generation with USPEX-like volume targeting
  - LAMMPS / MACE-OFF23 energy evaluation
  - Gaussian Process surrogate model with Expected Improvement
  - Active learning loop that discovers favorable regions autonomously

Usage:
    python run_cryal.py [INPUT.txt]
"""

__version__ = "1.0.0"
__author__  = "CrYAL contributors"
