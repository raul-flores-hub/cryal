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

"""Maintenance utilities shipped with CrYAL.

They live inside the package so that an installed copy has them too: an
installation has no `tools/` directory, and the model patcher is exactly the
kind of thing one needs on a machine where CrYAL was installed rather than
cloned.
"""
