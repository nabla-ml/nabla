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

"""Base operation utilities and registration functions."""

from __future__ import annotations

# Global execution mode flag, TODO: remove global flag and apply model compiling more elegantly
EAGERMODE: bool = False


# def _validate_binary_args(args: list[Array], op_name: str) -> None:
#     """Validate arguments for binary operations."""
#     if len(args) != 2:
#         raise ValueError(f"{op_name} operation requires 2 arguments, got {len(args)}")
#     if not all(isinstance(arg, Array) for arg in args):
#         raise TypeError(
#             f"All arguments must be instances of Array, got {[type(arg) for arg in args]}"
#         )
#     if args[0].dtype != args[1].dtype:
#         raise ValueError(
#             f"Dtypes {args[0].dtype} and {args[1].dtype} are not compatible for {op_name}."
#         )
#     if args[0].device != args[1].device:
#         raise ValueError(
#             f"Devices {args[0].device} and {args[1].device} are not compatible for {op_name}."
#         )


# def _validate_unary_arg(arg: Array, op_name: str) -> None:
#     """Validate argument for unary operations."""
#     if not isinstance(arg, Array):
#         raise TypeError(f"Argument must be an instance of Array, got {type(arg)}")


# def _validate_callables(maxpr, eagerxpr, vjp_rule, jvp_rule) -> None:
#     """Validate that all operation functions are callable."""
#     if not callable(maxpr):
#         raise TypeError(f"maxpr must be callable, got {type(maxpr)}")
#     if not callable(eagerxpr):
#         raise TypeError(f"eagerxpr must be callable, got {type(eagerxpr)}")
#     if not callable(vjp_rule):
#         raise TypeError(f"vjp_rule must be callable, got {type(vjp_rule)}")
#     if not callable(jvp_rule):
#         raise TypeError(f"jvp_rule must be callable, got {type(jvp_rule)}")


# def register_binary_op(
#     args: list[Array],
#     op_name: str,
#     maxpr: MaxprCallable,
#     eagerxpr: Callable[[list[Array], Array], None],
#     vjp_rule: VJPRule,
#     jvp_rule: JVPRule,
# ) -> Array:
#     """Register a binary operation with validation and broadcasting."""
#     _validate_binary_args(args, op_name)
#     _validate_callables(maxpr, eagerxpr, vjp_rule, jvp_rule)

#     target_shape = get_broadcasted_shape(args[0].shape, args[1].shape)
#     arg0_broadcasted = broadcast_to(args[0], target_shape)
#     arg1_broadcasted = broadcast_to(args[1], target_shape)

#     target_batch_dims = get_batch(args[0].batch_dims, args[1].batch_dims)
#     arg0_broadcasted = broadcast_batch_dims(arg0_broadcasted, target_batch_dims)
#     arg1_broadcasted = broadcast_batch_dims(arg1_broadcasted, target_batch_dims)

#     res = Array(
#         shape=target_shape,
#         dtype=args[0].dtype,
#         device=args[0].device,
#         materialize=False,
#         name=op_name,
#     )
#     res.set_maxpr(maxpr)
#     res.add_argument(arg0_broadcasted)
#     res.add_argument(arg1_broadcasted)
#     res.vjp_rule = vjp_rule
#     res.jvp_rule = jvp_rule

#     if EAGERMODE:
#         eagerxpr([arg0_broadcasted, arg1_broadcasted], res)

#     return res


# def register_unary_op(
#     arg: Array,
#     op_name: str,
#     maxpr: MaxprCallable,
#     eagerxpr: Callable[[list[Array], Array], None],
#     vjp_rule: VJPRule,
#     jvp_rule: JVPRule,
#     output_shape_fn: Callable[[tuple], tuple] | None = None,
#     output_dtype: DType | None = None,
# ) -> Array:
#     """Register a unary operation with validation."""
#     _validate_unary_arg(arg, op_name)
#     _validate_callables(maxpr, eagerxpr, vjp_rule, jvp_rule)

#     final_shape = (
#         output_shape_fn(arg.shape) if output_shape_fn is not None else arg.shape
#     )
#     final_dtype = output_dtype if output_dtype is not None else arg.dtype

#     res = Array(
#         shape=final_shape,
#         dtype=final_dtype,
#         device=arg.device,
#         materialize=False,
#         name=op_name,
#     )
#     res.set_maxpr(maxpr)
#     res.add_argument(arg)
#     res.vjp_rule = vjp_rule
#     res.jvp_rule = jvp_rule

#     if EAGERMODE:
#         eagerxpr([arg], res)

#     return res
