# ===----------------------------------------------------------------------=== #
# Nabla 2025
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
# ===----------------------------------------------------------------------=== #

"""
Provides information about MAX Engine, such as the version.
"""
from ._engine_impl import _EngineImpl, _get_engine_path


fn get_version() raises -> String:
    """Returns the current MAX Engine version.

    Returns:
        Version as string.
    """
    var version = _EngineImpl(_get_engine_path()).get_version()
    return version
