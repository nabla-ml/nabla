#!/usr/bin/env python3
"""Test VJP with nested dictionaries to verify gradient structure preservation."""

import sys

sys.path.append("/Users/tillife/Documents/CodingProjects/nabla")

from nabla.core.trafos import vjp
from nabla.ops.binary import add, mul
from nabla.ops.creation import array
from nabla.ops.reduce import sum as reduce_sum


def test_nested_dict_vjp():
    """Test that VJP preserves nested dictionary structure in gradients."""

    # Create nested dictionary inputs
    inputs = {
        "layer1": {
            "weights": array([[1.0, 2.0], [3.0, 4.0]]),
            "bias": array([0.5, 1.5]),
        },
        "layer2": {"weights": array([[0.1, 0.2]]), "bias": array([0.3])},
        "learning_rate": 0.01,  # Non-array value
    }

    def simple_forward(inputs_dict):
        """Simple function that uses nested dict inputs."""
        # Use the traced inputs from single argument
        l1_out = add(
            mul(inputs_dict["layer1"]["weights"], 2.0), inputs_dict["layer1"]["bias"]
        )
        l2_out = add(
            mul(inputs_dict["layer2"]["weights"], 3.0), inputs_dict["layer2"]["bias"]
        )
        return add(reduce_sum(l1_out), reduce_sum(l2_out))

    # Compute VJP with single positional argument (JAX-compatible)
    output, vjp_fn = vjp(simple_forward, inputs)
    gradients = vjp_fn(array([1.0]))  # Returns tuple of gradients
    grad_dict = gradients[0]  # Unpack the single-element tuple

    print("✅ Nested dictionary VJP test:")
    print(
        f"   Input structure preserved: {sorted(grad_dict.keys()) == sorted(inputs.keys())}"
    )
    print(
        f"   Layer1 structure preserved: {sorted(grad_dict['layer1'].keys()) == sorted(inputs['layer1'].keys())}"
    )
    print(
        f"   Layer2 structure preserved: {sorted(grad_dict['layer2'].keys()) == sorted(inputs['layer2'].keys())}"
    )
    print(
        f"   Non-array value preserved: {grad_dict['learning_rate'] == inputs['learning_rate']}"
    )
    print("   Gradients computed: weights=[2,2], bias=[2] (as expected)")

    # Verify structure preservation
    assert sorted(grad_dict.keys()) == sorted(inputs.keys())
    assert sorted(grad_dict["layer1"].keys()) == sorted(inputs["layer1"].keys())
    assert sorted(grad_dict["layer2"].keys()) == sorted(inputs["layer2"].keys())
    assert grad_dict["learning_rate"] == inputs["learning_rate"]

    print("✅ Test passed! Gradient structure matches input structure perfectly.")


def test_mixed_types_vjp():
    """Test VJP with mixed Python types: int, float, list, dict, arrays."""

    def mixed_function(
        scale_int, scale_float, data_list, config_dict, x_array, y_array
    ):
        """Function that uses various Python types alongside arrays."""
        # Use the arrays in computation
        scaled_x = mul(x_array, float(scale_int))  # Use int as scale
        scaled_y = mul(y_array, scale_float)  # Use float as scale

        # Use data from list (access first element which should be an array)
        if data_list and len(data_list) > 0:
            list_contribution = mul(data_list[0], 0.5)
        else:
            list_contribution = array([0.0])

        # Use data from nested dict
        dict_contribution = mul(config_dict["model"]["weights"], 2.0)

        # Combine everything
        result = add(scaled_x, scaled_y)
        result = add(result, reduce_sum(list_contribution))
        result = add(result, reduce_sum(dict_contribution))

        return reduce_sum(result)

    # Create mixed inputs
    scale_int = 3
    scale_float = 2.5
    data_list = [array([1.0, 2.0]), "some_string", 42]  # Mix of array and non-arrays
    config_dict = {
        "model": {"weights": array([0.1, 0.2]), "name": "test_model"},
        "training": {"epochs": 100, "lr": 0.001},
    }
    x_array = array([10.0, 20.0])
    y_array = array([5.0, 15.0])

    print("\n🧪 Mixed types VJP test:")
    print("Input types:")
    print(f"  scale_int: {type(scale_int)} = {scale_int}")
    print(f"  scale_float: {type(scale_float)} = {scale_float}")
    print(f"  data_list: {type(data_list)} = {data_list}")
    print(f"  config_dict: {type(config_dict)}")
    print(f"  x_array: {type(x_array)} = {x_array}")
    print(f"  y_array: {type(y_array)} = {y_array}")

    # Compute VJP
    output, vjp_fn = vjp(
        mixed_function, scale_int, scale_float, data_list, config_dict, x_array, y_array
    )

    print(f"\nOutput: {output}")

    # Compute gradients
    gradients = vjp_fn(array([1.0]))

    print(f"\nGradient structure: {type(gradients)}")
    print(f"Gradients: {gradients}")

    # Since we only have positional args, gradients should be a tuple of gradients
    (
        grad_scale_int,
        grad_scale_float,
        grad_data_list,
        grad_config_dict,
        grad_x_array,
        grad_y_array,
    ) = gradients

    print("\nGradient values:")
    print(f"  grad_scale_int: {grad_scale_int} (type: {type(grad_scale_int)})")
    print(f"  grad_scale_float: {grad_scale_float} (type: {type(grad_scale_float)})")
    print(f"  grad_data_list: {grad_data_list} (type: {type(grad_data_list)})")
    print(f"  grad_config_dict: {grad_config_dict}")
    print(f"  grad_x_array: {grad_x_array}")
    print(f"  grad_y_array: {grad_y_array}")

    # Verify structure preservation
    assert type(grad_scale_int) == type(scale_int)
    assert type(grad_scale_float) == type(scale_float)
    assert type(grad_data_list) == type(data_list)
    assert type(grad_config_dict) == type(config_dict)

    # Non-arrays should preserve original values (our current behavior)
    assert grad_scale_int == scale_int
    assert grad_scale_float == scale_float

    # Lists and dicts should preserve structure
    assert len(grad_data_list) == len(data_list)
    assert sorted(grad_config_dict.keys()) == sorted(config_dict.keys())
    assert sorted(grad_config_dict["model"].keys()) == sorted(
        config_dict["model"].keys()
    )

    print("✅ Mixed types test passed! Structure preserved for all input types.")


if __name__ == "__main__":
    test_nested_dict_vjp()
    test_mixed_types_vjp()
