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

"""Linear layer implementations."""

import nabla as nb


def linear_forward(
    x: nb.Array, weight: nb.Array, bias: nb.Array | None = None
) -> nb.Array:
    """Forward pass through a linear layer.

    Computes: output = x @ weight + bias

    Args:
        x: Input tensor of shape (batch_size, in_features)
        weight: Weight tensor of shape (in_features, out_features)
        bias: Optional bias tensor of shape (1, out_features) or (out_features,)

    Returns:
        Output tensor of shape (batch_size, out_features)
    """
    output = nb.matmul(x, weight)
    if bias is not None:
        output = output + bias
    return output


def mlp_forward(x: nb.Array, params: list[nb.Array]) -> nb.Array:
    """MLP forward pass through all layers.

    This is the original MLP forward function from mlp_train_jit.py.
    Applies ReLU activation to all layers except the last.

    Args:
        x: Input tensor of shape (batch_size, input_dim)
        params: List of parameters [W1, b1, W2, b2, ..., Wn, bn]

    Returns:
        Output tensor of shape (batch_size, output_dim)
    """
    output = x
    for i in range(0, len(params) - 1, 2):
        w, b = params[i], params[i + 1]
        output = nb.matmul(output, w) + b
        # Apply ReLU to all layers except the last
        if i < len(params) - 2:
            output = nb.relu(output)
    return output


def mlp_forward_with_activations(
    x: nb.Array,
    params: list[nb.Array],
    activation: str = "relu",
    final_activation: str | None = None,
) -> nb.Array:
    """MLP forward pass with configurable activations.

    Args:
        x: Input tensor of shape (batch_size, input_dim)
        params: List of parameters [W1, b1, W2, b2, ..., Wn, bn]
        activation: Activation function for hidden layers ("relu", "tanh", "sigmoid")
        final_activation: Optional activation for final layer

    Returns:
        Output tensor of shape (batch_size, output_dim)
    """
    output = x

    # Activation function mapping
    from .activations import gelu, leaky_relu, relu, sigmoid, silu, tanh

    activation_fns = {
        "relu": relu,
        "leaky_relu": leaky_relu,
        "sigmoid": sigmoid,
        "tanh": tanh,
        "gelu": gelu,
        "silu": silu,
        "swish": silu,  # alias for silu
    }

    if activation not in activation_fns:
        raise ValueError(
            f"Unsupported activation: {activation}. Supported: {list(activation_fns.keys())}"
        )

    act_fn = activation_fns[activation]

    for i in range(0, len(params) - 1, 2):
        w, b = params[i], params[i + 1]
        output = nb.matmul(output, w) + b

        # Apply activation to hidden layers
        if i < len(params) - 2:
            output = act_fn(output)
        elif final_activation is not None:
            # Apply final activation if specified
            final_act_fn = activation_fns.get(final_activation)
            if final_act_fn is not None:
                output = final_act_fn(output)

    return output
