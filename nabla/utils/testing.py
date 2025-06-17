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

"""Testing utilities for Nabla arrays."""

from typing import TYPE_CHECKING, Union

import numpy as np

# Import Array directly for runtime isinstance checks
from ..core.array import Array

if TYPE_CHECKING:
    pass  # Array already imported above


def allclose(
    a: Union[Array, np.ndarray, float, int],
    b: Union[Array, np.ndarray, float, int],
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    """
    Returns True if two arrays are element-wise equal within a tolerance.

    This function automatically converts Nabla Arrays to numpy arrays using
    .to_numpy() before comparison, providing a convenient way to compare
    Nabla arrays with each other or with numpy arrays/scalars.

    Args:
        a: Input array or scalar
        b: Input array or scalar
        rtol: Relative tolerance parameter
        atol: Absolute tolerance parameter
        equal_nan: Whether to compare NaN's as equal

    Returns:
        bool: True if the arrays are equal within the given tolerance

    Examples:
        >>> import nabla as nb
        >>> a = nb.array([1.0, 2.0, 3.0])
        >>> b = nb.array([1.0, 2.0, 3.000001])
        >>> nb.allclose(a, b)
        True
        >>> nb.allclose(a, np.array([1.0, 2.0, 3.0]))
        True
    """
    # Convert Nabla Arrays to numpy arrays
    if isinstance(a, Array):
        a = a.to_numpy()
    if isinstance(b, Array):
        b = b.to_numpy()

    # Use numpy's allclose for the actual comparison
    return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
