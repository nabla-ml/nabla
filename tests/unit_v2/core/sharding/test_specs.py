# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import pytest
import numpy as np
from nabla.core.sharding.spec import (
    DeviceMesh, 
    DimSpec, 
    ShardingSpec, 
    parse_sub_axis,
    validate_sub_axes_non_overlapping,
    check_sub_axes_maximality
)

class TestDeviceMesh:
    def test_mesh_1d(self):
        mesh = DeviceMesh("mesh_1d", (4,), ("d",))
        assert len(mesh.devices) == 4
        assert mesh.get_axis_size("d") == 4
        assert mesh.get_coordinate(device_id=2, axis_name="d") == 2
        
    def test_mesh_2d(self):
        mesh = DeviceMesh("mesh_2d", (2, 4), ("x", "y"))
        assert len(mesh.devices) == 8
        assert mesh.get_coordinate(5, "x") == 1
        assert mesh.get_coordinate(5, "y") == 1
        
    def test_sub_axis_logic(self):
        mesh = DeviceMesh("mesh_sub", (8,), ("data",))
        assert mesh.get_axis_size("data:(2)4") == 4
        assert mesh.get_coordinate(7, "data:(2)4") == 3
        assert mesh.get_coordinate(2, "data:(2)4") == 2

class TestDimSpec:
    def test_repr_and_parsing(self):
        d = DimSpec(["x", "y"])
        assert str(d) == "{'x', 'y'}"
        d_open = DimSpec(["z"], is_open=True)
        assert str(d_open) == "{'z', ?}"
        d_prio = DimSpec(["x"], priority=1)
        assert str(d_prio) == "{'x'}p1"

    def test_validation(self):
        with pytest.raises(ValueError, match="cannot have non-zero priority"):
            DimSpec([], priority=1)

class TestShardingSpec:
    def test_basic_spec(self):
        mesh = DeviceMesh("m", (4,), ("x",))
        spec = ShardingSpec(mesh, [DimSpec(["x"]), DimSpec([])])
        assert str(spec) == "sharding<@m, [{'x'}, {}]>"
        
    def test_overlap_validation(self):
        mesh = DeviceMesh("m", (4,), ("x",))
        with pytest.raises(ValueError, match="used multiple times"):
            ShardingSpec(mesh, [DimSpec(["x"]), DimSpec(["x"])])
        with pytest.raises(ValueError, match="explicitly replicated"):
            ShardingSpec(mesh, [DimSpec(["x"])], replicated_axes={"x"})

    def test_sub_axis_overlap(self):
        mesh = DeviceMesh("m", (4,), ("x",))
        ShardingSpec(mesh, [DimSpec(["x:(1)2"]), DimSpec(["x:(2)2"])])
        with pytest.raises(ValueError, match="Sub-axes overlap"):
            ShardingSpec(mesh, [DimSpec(["x:(1)4"]), DimSpec(["x:(2)2"])])
