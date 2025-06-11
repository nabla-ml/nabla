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
"""Neural Network module for Nabla."""

# Import submodules for easy access
from . import losses
from . import optim
from . import init
from . import layers
from . import architectures
from . import utils

# Import commonly used functions for convenience
from .losses import (
    mean_squared_error, mean_absolute_error, huber_loss,
    cross_entropy_loss, sparse_cross_entropy_loss, 
    binary_cross_entropy_loss, softmax_cross_entropy_loss
)
from .optim import (
    adamw_step, init_adamw_state, 
    adam_step, init_adam_state,
    sgd_step, init_sgd_state,
    learning_rate_schedule, exponential_decay_schedule,
    cosine_annealing_schedule, warmup_cosine_schedule
)
from .init import initialize_mlp_params, he_normal, xavier_normal, lecun_normal
from .layers import (
    mlp_forward, linear_forward, mlp_forward_with_activations,
    relu, leaky_relu, sigmoid, tanh, gelu, silu, swish, softmax, log_softmax
)
from .architectures import create_mlp_config, MLPBuilder
from .utils import (
    value_and_grad, create_sin_dataset, create_dataset,
    accuracy, precision, recall, f1_score,
    mean_squared_error_metric, mean_absolute_error_metric, 
    r_squared, pearson_correlation,
    dropout, l1_regularization, l2_regularization, 
    elastic_net_regularization, gradient_clipping
)

__all__ = [
    # Submodules
    "losses",
    "optim", 
    "init",
    "layers",
    "architectures",
    "utils",
    
    # Commonly used functions
    # Loss functions
    "mean_squared_error",
    "mean_absolute_error",
    "huber_loss",
    "cross_entropy_loss", 
    "sparse_cross_entropy_loss",
    "binary_cross_entropy_loss",
    "softmax_cross_entropy_loss",
    
    # Optimizers
    "adamw_step",
    "init_adamw_state",
    "adam_step", 
    "init_adam_state",
    "sgd_step",
    "init_sgd_state",
    "learning_rate_schedule",
    "exponential_decay_schedule",
    "cosine_annealing_schedule",
    "warmup_cosine_schedule",
    
    # Initialization
    "initialize_mlp_params",
    "he_normal",
    "xavier_normal",
    "lecun_normal",
    
    # Layers and activations
    "mlp_forward",
    "linear_forward",
    "mlp_forward_with_activations",
    "relu",
    "leaky_relu",
    "sigmoid",
    "tanh",
    "gelu", 
    "silu",
    "swish",
    "softmax",
    "log_softmax",
    
    # Architectures
    "create_mlp_config",
    "MLPBuilder",
    
    # Utilities
    "value_and_grad",
    "create_sin_dataset",
    "create_dataset",
    "accuracy",
    "precision",
    "recall", 
    "f1_score",
    "mean_squared_error_metric",
    "mean_absolute_error_metric",
    "r_squared",
    "pearson_correlation",
    "dropout",
    "l1_regularization",
    "l2_regularization",
    "elastic_net_regularization",
    "gradient_clipping",
]
