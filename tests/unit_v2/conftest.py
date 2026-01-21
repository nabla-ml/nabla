# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import pytest
from nabla import DeviceMesh

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
