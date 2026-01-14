import itertools
import math
from typing import Dict, List, Tuple, Optional, Union

# --- --- --- --- --- ---
#      TYPE ALIASES
# --- --- --- --- --- ---
# Type Aliases for clarity, establishing a domain-specific vocabulary.
# Using Tuple enforces the immutability of shape and coordinate structures.
DeviceCoord = Tuple[int, ...]
TensorShape = Tuple[int, ...]
MeshShape = Tuple[int, ...]
ShardingSpec = Tuple[Optional[Union[int, Tuple[int, ...]]], ...]
PlacementSlices = Tuple[slice, ...]
Placements = Dict[DeviceCoord, PlacementSlices]


# --- --- --- --- --- ---
#   VALIDATION LOGIC
# --- --- --- --- --- ---

def _validate_inputs(
    tensor_shape: TensorShape, mesh_shape: MeshShape, sharding_spec: ShardingSpec
) -> None:
    """
    Performs robust checks on the sharding configuration.

    Raises:
        ValueError: If any part of the configuration is invalid.
    """
    if len(tensor_shape) != len(sharding_spec):
        raise ValueError("tensor_shape and sharding_spec must have the same length.")

    # A set to track mesh axes that have already been used for sharding.
    used_mesh_axes: set[int] = set()
    num_mesh_dims: int = len(mesh_shape)

    # Iterate over each dimension's sharding specification.
    for spec in sharding_spec:
        if spec is None:  # This dimension is replicated, no validation needed.
            continue

        # Coerce spec to a tuple for consistent processing.
        mesh_axes_for_dim: Tuple[int, ...] = (spec,) if isinstance(spec, int) else spec

        for axis in mesh_axes_for_dim:
            # 1. Check if the specified mesh axis is valid.
            if not (0 <= axis < num_mesh_dims):
                raise ValueError(
                    f"Invalid mesh axis '{axis}' in sharding_spec. "
                    f"It is out of bounds for mesh_shape with {num_mesh_dims} dimensions."
                )
            # 2. Check if the mesh axis has already been used by another tensor dimension.
            if axis in used_mesh_axes:
                raise ValueError(
                    f"Mesh axis '{axis}' is used to shard multiple tensor dimensions. "
                    "Each mesh axis can only be used once."
                )
            used_mesh_axes.add(axis)


# --- --- --- --- --- ---
#  CORE SHARDING LOGIC
# --- --- --- --- --- ---

def get_device_placements(
    tensor_shape: TensorShape, mesh_shape: MeshShape, sharding_spec: ShardingSpec
) -> Placements:
    """
    Calculates the tensor slice for each device in a mesh using a concise, functional style.

    This is the main entry point. It delegates the complex work to helper functions.
    """
    # Perform all input validation upfront.
    _validate_inputs(tensor_shape, mesh_shape, sharding_spec)

    # Precisely annotating the iterator type for maximum clarity.
    all_devices: itertools.product[DeviceCoord] = itertools.product(
        *(range(s) for s in mesh_shape)
    )

    # A declarative core: "For each device, calculate its tuple of slices."
    return {
        device: _get_slices_for_device(device, tensor_shape, mesh_shape, sharding_spec)
        for device in all_devices
    }


def _get_slices_for_device(
    device_coord: DeviceCoord,
    tensor_shape: TensorShape,
    mesh_shape: MeshShape,
    sharding_spec: ShardingSpec,
) -> PlacementSlices:
    """Orchestrates the slice calculation for all dimensions of a tensor for a single device."""

    # A tuple comprehension that reads like a sentence:
    # "For each dimension, get a full slice if its spec is None, otherwise calculate the partitioned slice."
    return tuple(
        slice(None)
        if spec is None
        else _get_partitioned_slice(dim_size, spec, device_coord, mesh_shape)
        for dim_size, spec in zip(tensor_shape, sharding_spec)
    )


def _get_partitioned_slice(
    dim_size: int,
    spec: Union[int, Tuple[int, ...]],
    device_coord: DeviceCoord,
    mesh_shape: MeshShape,
) -> slice:
    """
    Calculates the slice for a SINGLE tensor dimension that is being partitioned.
    This version supports uneven sharding.
    """
    # The type of spec is narrowed from the calling function.
    mesh_axes: Tuple[int, ...] = (spec,) if isinstance(spec, int) else spec
    num_shards: int = math.prod(mesh_shape[axis] for axis in mesh_axes)

    # Calculate base size and the remainder for uneven distribution.
    base_shard_size: int = dim_size // num_shards
    remainder: int = dim_size % num_shards

    # Calculate the device's 1D index on the flattened sub-grid defined by the mesh axes.
    linear_index: int = 0
    stride: int = 1
    for axis in reversed(mesh_axes):
        linear_index += device_coord[axis] * stride
        stride *= mesh_shape[axis]

    # Distribute the remainder elements to the first few devices.
    shard_size: int = base_shard_size + (1 if linear_index < remainder else 0)
    start: int = (linear_index * base_shard_size) + min(linear_index, remainder)

    return slice(start, start + shard_size)


def pretty_print_placements(placements: Placements, title: str = "") -> None:
    """Helper function to format the output for readability."""
    if title:
        print(title)

    # Annotating local variables enhances readability and helps static analysis.
    sorted_placements: List[Tuple[DeviceCoord, PlacementSlices]] = sorted(placements.items())
    for device, slices in sorted_placements:
        slice_str: str = ", ".join(f"{s.start or 0}:{s.stop or 'end'}" for s in slices)
        # Pad device string for alignment
        device_str: str = str(device).ljust(12)
        print(f"Device {device_str}: tensor[{slice_str}]")


# --- --- --- --- --- ---
#      MAIN EXAMPLES & TESTS
# --- --- --- --- --- ---
if __name__ == "__main__":

    # --- Test Suite 1: 2D Tensor on a 1D Mesh (Even Sharding) ---
    print("=" * 60)
    print("### Test Suite 1: 2D Tensor on a 1D Mesh ###")
    print("=" * 60)
    TENSOR_SHAPE_1D: TensorShape = (4096, 1024)
    MESH_SHAPE_1D: MeshShape = (4,)

    specs_1d: Dict[str, ShardingSpec] = {
        "Data Parallelism (shard batch)": (0, None),
        "Tensor Parallelism (shard features)": (None, 0),
    }

    for desc, spec in specs_1d.items():
        title = f"\n--- {desc} ---\nConfig: tensor={TENSOR_SHAPE_1D}, mesh={MESH_SHAPE_1D}, spec={spec}"
        placements = get_device_placements(TENSOR_SHAPE_1D, MESH_SHAPE_1D, spec)
        pretty_print_placements(placements, title)

    # --- Test Suite 2: 3D Tensor on a 2D Mesh (Even Sharding) ---
    print("\n" + "=" * 60)
    print("### Test Suite 2: 3D Tensor on an Asymmetric 2D Mesh ###")
    print("=" * 60)
    TENSOR_SHAPE_2D: TensorShape = (1024, 2048, 512)
    MESH_SHAPE_2D: MeshShape = (4, 2)

    specs_2d: Dict[str, ShardingSpec] = {
        "Shard dim 0 on axis 0, dim 1 on axis 1": (0, 1, None),
        "Shard dim 0 over all 8 devices": ((0, 1), None, None),
    }

    for desc, spec in specs_2d.items():
        title = f"\n--- {desc} ---\nConfig: tensor={TENSOR_SHAPE_2D}, mesh={MESH_SHAPE_2D}, spec={spec}"
        placements = get_device_placements(TENSOR_SHAPE_2D, MESH_SHAPE_2D, spec)
        pretty_print_placements(placements, title)

    # --- Test Suite 3: 4D Tensor on a 3D Mesh (Even Sharding) ---
    print("\n" + "=" * 60)
    print("### Test Suite 3: 4D Tensor on a 3D Mesh ###")
    print("=" * 60)
    TENSOR_SHAPE_3D: TensorShape = (128, 64, 32, 16)
    MESH_SHAPE_3D: MeshShape = (2, 2, 2)

    specs_3d: Dict[str, ShardingSpec] = {
        "3D Parallelism (shard dims 0,1,2 on axes 0,1,2)": (0, 1, 2, None),
        "Shard dim 1 over all 8 devices (row-major)": (None, (0, 1, 2), None, None),
    }

    for desc, spec in specs_3d.items():
        title = f"\n--- {desc} ---\nConfig: tensor={TENSOR_SHAPE_3D}, mesh={MESH_SHAPE_3D}, spec={spec}"
        placements = get_device_placements(TENSOR_SHAPE_3D, MESH_SHAPE_3D, spec)
        pretty_print_placements(placements, title)

    # --- Test Suite 4: Uneven Sharding Scenarios ---
    print("\n" + "=" * 60)
    print("### Test Suite 4: Uneven Sharding Scenarios ###")
    print("=" * 60)

    # Scenario A: Simple case - 100 is not divisible by 8
    # Expected: 100 = 8 * 12 + 4. First 4 devices get 13 items, the rest get 12.
    TENSOR_A: TensorShape = (100, 512)
    MESH_A: MeshShape = (8,)
    SPEC_A: ShardingSpec = (0, None)
    title_a = f"\n--- Simple Uneven ---\nConfig: tensor={TENSOR_A}, mesh={MESH_A}, spec={SPEC_A}"
    placements_a = get_device_placements(TENSOR_A, MESH_A, SPEC_A)
    pretty_print_placements(placements_a, title_a)

    # Scenario B: Complex case - shard dim of size 17 over a 2x3=6 device mesh
    # Expected: 17 = 6 * 2 + 5. First 5 devices get 3 items, the last one gets 2.
    TENSOR_B: TensorShape = (17, 1024)
    MESH_B: MeshShape = (2, 3)
    SPEC_B: ShardingSpec = ((0, 1), None)
    title_b = f"\n--- Complex Uneven ---\nConfig: tensor={TENSOR_B}, mesh={MESH_B}, spec={SPEC_B}"
    placements_b = get_device_placements(TENSOR_B, MESH_B, SPEC_B)
    pretty_print_placements(placements_b, title_b)

    # --- Test Suite 5: Invalid Configurations (Demonstrates new validation) ---
    print("\n" + "=" * 60)
    print("### Test Suite 5: Invalid Configurations ###")
    print("=" * 60)

    try:
        # Test 1: Using a mesh axis that is out of bounds (e.g., axis 2 in a 2D mesh)
        TENSOR_INVALID_1: TensorShape = (1024, 2048)
        MESH_INVALID_1: MeshShape = (4, 2)
        SPEC_INVALID_1: ShardingSpec = (2, None) # Invalid: '2' is not a valid axis
        print("\n--- Testing out-of-bounds mesh axis ---")
        get_device_placements(TENSOR_INVALID_1, MESH_INVALID_1, SPEC_INVALID_1)
    except ValueError as e:
        print(f"Successfully caught expected error: {e}")

    try:
        # Test 2: Using the same mesh axis (0) for two different tensor dimensions
        TENSOR_INVALID_2: TensorShape = (1024, 2048)
        MESH_INVALID_2: MeshShape = (4, 2)
        SPEC_INVALID_2: ShardingSpec = (0, 0) # Invalid: mesh axis 0 used twice
        print("\n--- Testing reused mesh axis ---")
        get_device_placements(TENSOR_INVALID_2, MESH_INVALID_2, SPEC_INVALID_2)
    except ValueError as e:
        print(f"Successfully caught expected error: {e}")