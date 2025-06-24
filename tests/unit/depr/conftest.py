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

"""Pytest configuration and fixtures for unit tests."""

import numpy as np
import pytest

# Import from test_utils but avoid circular imports
DTYPES_TO_TEST = [np.float32, np.float64]

# Tolerance configurations
RTOL_F32_VAL, ATOL_F32_VAL = 1e-5, 1e-6
RTOL_F64_VAL, ATOL_F64_VAL = 1e-7, 1e-8
RTOL_F32_GRAD, ATOL_F32_GRAD = 1e-4, 1e-5
RTOL_F64_GRAD, ATOL_F64_GRAD = 1e-6, 1e-7


def get_tolerances(dtype, is_gradient=False):
    """Get appropriate tolerances for the given dtype."""
    is_complex = np.issubdtype(dtype, np.complexfloating)
    is_float = np.issubdtype(dtype, np.floating)

    if is_complex or is_float:
        if np.dtype(dtype).itemsize >= 8:  # float64 or complex128
            return (
                (RTOL_F64_GRAD, ATOL_F64_GRAD)
                if is_gradient
                else (RTOL_F64_VAL, ATOL_F64_VAL)
            )
        else:  # float32 or complex64
            return (
                (RTOL_F32_GRAD, ATOL_F32_GRAD)
                if is_gradient
                else (RTOL_F32_VAL, ATOL_F32_VAL)
            )
    return (0, 0)  # For exact comparison (e.g., int, bool)


# --- Pytest Fixtures ---
@pytest.fixture(params=DTYPES_TO_TEST, ids=[d.__name__ for d in DTYPES_TO_TEST])
def dtype(request):
    """Parametrized fixture for data types."""
    return request.param


@pytest.fixture
def tolerances(dtype):
    """Fixture providing tolerances for the given dtype."""
    return {
        "value": get_tolerances(dtype, is_gradient=False),
        "gradient": get_tolerances(dtype, is_gradient=True),
    }
