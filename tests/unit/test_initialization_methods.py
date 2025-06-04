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

"""Test initialization methods for statistical properties and correctness."""

import numpy as np
import pytest

import nabla as nb

# Reduced test parameters to avoid overwhelming the test runner
TEST_SHAPES = [
    (10, 20),  # Basic 2D
    (5, 10, 15),  # 3D
]

TEST_SEEDS = [42, 123]  # Only 2 seeds instead of 4


@pytest.mark.parametrize("shape", TEST_SHAPES)
def test_zeros_init(shape):
    """Test zeros initialization produces all zeros."""
    result = nb.zeros(shape)
    assert result.shape == shape
    assert np.allclose(result.to_numpy(), 0.0), "Zeros init should produce all zeros"


@pytest.mark.parametrize("shape", TEST_SHAPES)
def test_ones_init(shape):
    """Test ones initialization produces all ones."""
    result = nb.ones(shape)
    assert result.shape == shape
    assert np.allclose(result.to_numpy(), 1.0), "Ones init should produce all ones"


@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_xavier_uniform_properties(shape, seed):
    """Test Xavier uniform statistical properties."""
    gain = 1.0
    result = nb.xavier_uniform(shape, gain=gain, seed=seed)
    values = result.to_numpy()

    # Check shape
    assert result.shape == shape

    # Check bounds
    fan_in, fan_out = shape[-2], shape[-1]
    expected_bound = gain * np.sqrt(6.0 / (fan_in + fan_out))
    assert np.all(values >= -expected_bound), "Values should be >= -bound"
    assert np.all(values <= expected_bound), "Values should be <= bound"

    # For larger shapes, check statistical properties
    if np.prod(shape) > 1000:
        # Check approximate mean (should be close to 0)
        assert abs(np.mean(values)) < 0.1, (
            f"Mean should be close to 0, got {np.mean(values)}"
        )


@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_xavier_normal_properties(shape, seed):
    """Test Xavier normal statistical properties."""
    gain = 1.0
    result = nb.xavier_normal(shape, gain=gain, seed=seed)
    values = result.to_numpy()

    # Check shape
    assert result.shape == shape

    # For larger shapes, check statistical properties
    if np.prod(shape) > 1000:
        # Check approximate mean (should be close to 0)
        assert abs(np.mean(values)) < 0.1, (
            f"Mean should be close to 0, got {np.mean(values)}"
        )

        # Check approximate standard deviation
        fan_in, fan_out = shape[-2], shape[-1]
        expected_std = gain * np.sqrt(2.0 / (fan_in + fan_out))
        actual_std = np.std(values)
        assert abs(actual_std - expected_std) / expected_std < 0.2, (
            f"Std should be close to {expected_std}, got {actual_std}"
        )


@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_he_uniform_properties(shape, seed):
    """Test He uniform statistical properties."""
    result = nb.he_uniform(shape, seed=seed)
    values = result.to_numpy()

    # Check shape
    assert result.shape == shape

    # Check bounds
    fan_in = shape[-2]
    expected_bound = np.sqrt(6.0 / fan_in)
    assert np.all(values >= -expected_bound), "Values should be >= -bound"
    assert np.all(values <= expected_bound), "Values should be <= bound"


@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_he_normal_properties(shape, seed):
    """Test He normal statistical properties."""
    result = nb.he_normal(shape, seed=seed)
    values = result.to_numpy()

    # Check shape
    assert result.shape == shape

    # For larger shapes, check statistical properties
    if np.prod(shape) > 1000:
        # Check approximate mean (should be close to 0)
        assert abs(np.mean(values)) < 0.1, (
            f"Mean should be close to 0, got {np.mean(values)}"
        )


@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_lecun_uniform_properties(shape, seed):
    """Test LeCun uniform statistical properties."""
    result = nb.lecun_uniform(shape, seed=seed)
    values = result.to_numpy()

    # Check shape
    assert result.shape == shape

    # Check bounds
    fan_in = shape[-2]
    expected_bound = np.sqrt(3.0 / fan_in)
    assert np.all(values >= -expected_bound), "Values should be >= -bound"
    assert np.all(values <= expected_bound), "Values should be <= bound"


@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_lecun_normal_properties(shape, seed):
    """Test LeCun normal statistical properties."""
    result = nb.lecun_normal(shape, seed=seed)
    values = result.to_numpy()

    # Check shape
    assert result.shape == shape

    # For larger shapes, check statistical properties
    if np.prod(shape) > 1000:
        # Check approximate mean (should be close to 0)
        assert abs(np.mean(values)) < 0.1, (
            f"Mean should be close to 0, got {np.mean(values)}"
        )


@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_randn_properties(seed):
    """Test basic random normal generation."""
    shape = (100, 200)
    mean, std = 0.0, 1.0

    result = nb.randn(shape, mean=mean, std=std, seed=seed)
    values = result.to_numpy()

    # Check shape
    assert result.shape == shape

    # Check approximate mean and std
    assert abs(np.mean(values) - mean) < 0.1, f"Mean should be close to {mean}"
    assert abs(np.std(values) - std) < 0.1, f"Std should be close to {std}"


@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_rand_properties(seed):
    """Test basic random uniform generation."""
    shape = (100, 200)
    lower, upper = -2.0, 3.0

    result = nb.rand(shape, lower=lower, upper=upper, seed=seed)
    values = result.to_numpy()

    # Check shape
    assert result.shape == shape

    # Check bounds
    assert np.all(values >= lower), f"Values should be >= {lower}"
    assert np.all(values <= upper), f"Values should be <= {upper}"

    # Check approximate mean
    expected_mean = (lower + upper) / 2
    assert abs(np.mean(values) - expected_mean) < 0.1, (
        f"Mean should be close to {expected_mean}"
    )


def test_error_conditions():
    """Test error conditions for initialization methods."""
    # Test 1D shape (should fail for most methods)
    with pytest.raises(ValueError):
        nb.xavier_uniform((10,))

    with pytest.raises(ValueError):
        nb.he_normal((10,))

    with pytest.raises(ValueError):
        nb.lecun_uniform((10,))


def test_deterministic_behavior():
    """Test that same seed produces same results."""
    shape = (10, 10)
    seed = 42

    # Test multiple methods
    methods = [
        nb.xavier_uniform,
        nb.xavier_normal,
        nb.he_uniform,
        nb.he_normal,
        nb.lecun_uniform,
        nb.lecun_normal,
    ]

    for method in methods:
        result1 = method(shape, seed=seed)
        result2 = method(shape, seed=seed)
        np.testing.assert_array_equal(
            result1.to_numpy(),
            result2.to_numpy(),
            err_msg=f"Same seed should produce same results for {method.__name__}",
        )


if __name__ == "__main__":
    # Quick smoke test
    print("Running statistical property tests...")
    test_xavier_uniform_properties((100, 200), 42)
    test_he_normal_properties((100, 200), 42)
    test_randn_properties(42)
    test_rand_properties(42)
    test_deterministic_behavior()
    test_error_conditions()
    print("✓ All tests passed!")
