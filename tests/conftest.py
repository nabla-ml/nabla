# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Shared fixtures and utilities for comprehensive ops tests.

Test Hierarchy:
1. test_physical_ops.py - Physical ops (foundation, must pass first)
2. test_logical_ops.py - Logical ops without batch_dims
3. test_vmap_ops.py - vmap transforms (uses vmap directly, not manual batch_dims)
4. test_sharding_ops.py - Sharding operations and vmap+sharding combo
"""


import pytest
import numpy as np
from typing import Callable

import nabla
from nabla import Tensor, DeviceMesh, P
from nabla.sharding.spec import DimSpec


# =============================================================================
# Array Creation Helpers
# =============================================================================

def make_array(*shape: int, seed: int = 42, dtype=np.float32) -> np.ndarray:
    """Create a deterministic random array."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(dtype)


def make_positive_array(*shape: int, seed: int = 42, dtype=np.float32) -> np.ndarray:
    """Create a deterministic positive random array (for ops needing positive inputs)."""
    rng = np.random.default_rng(seed)
    return np.abs(rng.standard_normal(shape)).astype(dtype) + 0.1


def tensor_from_numpy(arr: np.ndarray) -> Tensor:
    """Create a nabla Tensor from numpy array."""
    return Tensor.from_dlpack(arr)


def to_numpy(t: Tensor) -> np.ndarray:
    """Extract numpy array from Tensor.
    
    Uses Tensor.to_numpy() which handles:
    - Realization of lazy tensors
    - Gathering of sharded tensors
    - Proper device transfer to CPU
    """
    return t.to_numpy()


# =============================================================================
# Assertion Helpers
# =============================================================================

def assert_allclose(result: Tensor, expected: np.ndarray, rtol: float = 1e-5, atol: float = 1e-6):
    """Assert tensor values match expected numpy array."""
    actual = to_numpy(result)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def assert_shape(result: Tensor, expected_shape: tuple):
    """Assert tensor.shape matches expected (logical shape)."""
    actual = tuple(int(d) for d in result.shape)
    assert actual == expected_shape, f"Shape mismatch: got {actual}, expected {expected_shape}"


def assert_physical_shape(result: Tensor, expected_shape: tuple):
    """Assert tensor's physical shape (global_shape) matches expected."""
    actual = result.global_shape or result.local_shape
    actual = tuple(int(d) for d in actual)
    assert actual == expected_shape, f"Physical shape mismatch: got {actual}, expected {expected_shape}"


def assert_dtype(result: Tensor, expected_dtype):
    """Assert tensor dtype matches expected."""
    assert result.dtype == expected_dtype, f"Dtype mismatch: got {result.dtype}, expected {expected_dtype}"


def assert_batch_dims(result: Tensor, expected: int):
    """Assert tensor batch_dims matches expected."""
    actual = result._impl.batch_dims
    assert actual == expected, f"batch_dims mismatch: got {actual}, expected {expected}"


def assert_is_sharded(result: Tensor, expected: bool = True):
    """Assert tensor is/isn't sharded."""
    actual = result._impl.is_sharded
    assert actual == expected, f"is_sharded mismatch: got {actual}, expected {expected}"


# =============================================================================
# Mesh Fixtures
# =============================================================================

# --- 1D Meshes ---

@pytest.fixture
def mesh_1d():
    """1D mesh with 4 devices named 'dp'."""
    return DeviceMesh("mesh_1d", (4,), ("dp",))


@pytest.fixture
def mesh_1d_2():
    """1D mesh with 2 devices named 'dp'."""
    return DeviceMesh("mesh_1d_2", (2,), ("dp",))


@pytest.fixture
def mesh_1d_8():
    """1D mesh with 8 devices named 'dp'."""
    return DeviceMesh("mesh_1d_8", (8,), ("dp",))


# --- 2D Meshes ---

@pytest.fixture
def mesh_2d():
    """2D mesh with shape (2, 2) named ('dp', 'tp')."""
    return DeviceMesh("mesh_2d", (2, 2), ("dp", "tp"))


@pytest.fixture
def mesh_2d_2x4():
    """2D mesh with shape (2, 4) named ('dp', 'tp')."""
    return DeviceMesh("mesh_2d_2x4", (2, 4), ("dp", "tp"))


@pytest.fixture
def mesh_2d_4x2():
    """2D mesh with shape (4, 2) named ('dp', 'tp')."""
    return DeviceMesh("mesh_2d_4x2", (4, 2), ("dp", "tp"))


@pytest.fixture
def mesh_2d_1x4():
    """2D mesh with shape (1, 4) - degenerate 2D."""
    return DeviceMesh("mesh_2d_1x4", (1, 4), ("dp", "tp"))


# --- Asymmetric 2D Meshes (critical for proper testing) ---

@pytest.fixture
def mesh_2x4():
    """Asymmetric 2D mesh with shape (2, 4)."""
    return DeviceMesh("mesh_2x4", (2, 4), ("dp", "tp"))


@pytest.fixture
def mesh_4x2():
    """Asymmetric 2D mesh with shape (4, 2)."""
    return DeviceMesh("mesh_4x2", (4, 2), ("dp", "tp"))


@pytest.fixture
def mesh_3x2():
    """Asymmetric 2D mesh with shape (3, 2) - non-power-of-2."""
    return DeviceMesh("mesh_3x2", (3, 2), ("dp", "tp"))


# --- 3D Meshes ---

@pytest.fixture
def mesh_3d():
    """3D mesh with shape (2, 2, 2) named ('dp', 'tp', 'pp')."""
    return DeviceMesh("mesh_3d", (2, 2, 2), ("dp", "tp", "pp"))


@pytest.fixture
def mesh_3d_2x2x4():
    """3D mesh with shape (2, 2, 4) named ('dp', 'tp', 'pp')."""
    return DeviceMesh("mesh_3d_2x2x4", (2, 2, 4), ("dp", "tp", "pp"))


@pytest.fixture 
def mesh_3d_4x2x2():
    """3D mesh with shape (4, 2, 2) named ('dp', 'tp', 'pp')."""
    return DeviceMesh("mesh_3d_4x2x2", (4, 2, 2), ("dp", "tp", "pp"))


@pytest.fixture
def mesh_3d_2x4x2():
    """3D mesh with shape (2, 4, 2) named ('dp', 'tp', 'pp')."""
    return DeviceMesh("mesh_3d_2x4x2", (2, 4, 2), ("dp", "tp", "pp"))


@pytest.fixture
def mesh_3d_2x2x4():
    """3D mesh with shape (2, 2, 4) named ('dp', 'tp', 'pp')."""
    return DeviceMesh("mesh_3d_2x2x4", (2, 2, 4), ("dp", "tp", "pp"))


# =============================================================================
# Sharding Helpers
# =============================================================================

def shard_on_axis(tensor: Tensor, mesh: DeviceMesh, axis: int, mesh_axis: int = 0) -> Tensor:
    """Shard tensor on a specific axis using specified mesh dimension.
    
    Args:
        tensor: The tensor to shard
        mesh: The device mesh
        axis: The tensor axis to shard
        mesh_axis: Which mesh axis to use for sharding (default 0 = first axis name)
    """
    rank = len(tensor.shape)
    # Default to open specs so they can accept further sharding (e.g. broadcasting)
    specs = [DimSpec([], is_open=True) for _ in range(rank)]
    # The sharded axis is fixed (closed) unless explicitly opened (sharding constraint)
    specs[axis] = DimSpec([mesh.axis_names[mesh_axis]], is_open=False)
    return tensor.shard(mesh, specs)


def shard_on_axes(tensor: Tensor, mesh: DeviceMesh, axis_mapping: dict[int, int]) -> Tensor:
    """Shard tensor on multiple axes with specific mesh axis mapping.
    
    Args:
        tensor: The tensor to shard
        mesh: The device mesh
        axis_mapping: Dict mapping tensor_axis -> mesh_axis
                     e.g., {0: 0, 1: 1} shards tensor axis 0 on mesh axis 0,
                           tensor axis 1 on mesh axis 1
    """
    rank = len(tensor.shape)
    # Default to open specs
    specs = [DimSpec([], is_open=True) for _ in range(rank)]
    for tensor_axis, mesh_axis in axis_mapping.items():
        specs[tensor_axis] = DimSpec([mesh.axis_names[mesh_axis]], is_open=False)
    return tensor.shard(mesh, specs)


def replicated(tensor: Tensor, mesh: DeviceMesh) -> Tensor:
    """Create a fully replicated sharded tensor (open to sharding propagation)."""
    rank = len(tensor.shape)
    # Replicated starts with no axes but should be Open to allow it to match sharded inputs
    specs = [DimSpec([], is_open=True) for _ in range(rank)]
    return tensor.shard(mesh, specs)
