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

from memory import ArcPointer
from utils import Variant

from nabla.api.array import Array
from nabla.core.device_array import DeviceArray, zeros_like
from nabla.engine.utils import (
    TrafoMeta,
    std_basis,
    get_full_trace_recursively_jvp,
    Callable,
    callable,
)
from nabla.api.utils import ExecutionContext


fn set_execution_context_recursively(
    mut callable: ArcPointer[Callable],
    execution_context: ExecutionContext,
) raises -> None:
    if callable[].execution_context:
        return

    callable[].execution_context = execution_context
    for child in callable[].trafos:
        set_execution_context_recursively(child[], execution_context)
