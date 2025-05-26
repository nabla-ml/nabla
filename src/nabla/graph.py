from __future__ import annotations
import numpy as np
import time
from pathlib import Path
from typing import (
    List,
    Final,
    ClassVar,
    Union,
    Tuple,
    Type,
    Set,
    Callable,
    Optional,
    Sequence,
    Dict,
    Protocol,
    TypedDict,
    cast,
    Any,
)
from collections import deque

from max.engine import InferenceSession, Model
from max.driver import Tensor, CPU, Accelerator, Device
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops, Value

# Define type aliases for better readability
MaxprCallable = Callable[[List[Value]], Value]
VJPRule = Callable[[List[Value], List[Value]], Value]
JVPRule = Callable[[List[Value], List[Value]], Value]
Shape = Tuple[int, ...]

# Execution mode flag
EAGERMODE: bool = True

# Global model cache with proper typing
global_execution_context: Dict[int, Model] = {}


def get_broadcasted_shape(
    shape1: Shape,
    shape2: Shape,
    ignore_axes: List[int] = [],
    replace_ignored_dims: List[int] = [],
) -> Shape:
    if len(replace_ignored_dims) != len(ignore_axes):
        raise ValueError(
            "replace_ignored_dims must have the same length as ignore_axes"
        )

    s1_len = len(shape1)
    s2_len = len(shape2)
    max_rank = max(s1_len, s2_len)

    # Initialize result shape. We'll fill it. Using 1s is a common default for broadcasting.
    res_shape_list = [1] * max_rank

    # Normalize ignore_axes to positive indices and store their replacement values.
    # These normalized indices refer to positions in the `max_rank` shape.
    normalized_ignored_map = {}  # Stores {normalized_idx: replacement_dim}

    for i in range(len(ignore_axes)):
        axis_spec = ignore_axes[i]
        replacement_dim = replace_ignored_dims[i]

        # Validate and normalize the axis_spec relative to max_rank
        if not (-max_rank <= axis_spec < max_rank):
            raise ValueError(
                f"ignore_axis {axis_spec} is out of bounds for max_rank {max_rank}"
            )

        normalized_idx = axis_spec if axis_spec >= 0 else max_rank + axis_spec

        # If multiple ignore_axes entries map to the same normalized_idx (e.g. 0 and -max_rank),
        # the last one in the list will win. This is typical Python dict behavior.
        normalized_ignored_map[normalized_idx] = replacement_dim
        res_shape_list[normalized_idx] = replacement_dim

    # Pad original shapes with leading 1s to align them to max_rank for broadcasting logic
    padded_shape1_list = [1] * (max_rank - s1_len) + list(shape1)
    padded_shape2_list = [1] * (max_rank - s2_len) + list(shape2)

    # Perform broadcasting for non-ignored axes
    # Iterate from the leftmost dimension of the padded shapes
    for i in range(max_rank):
        if i in normalized_ignored_map:
            # This dimension's value in res_shape_list is already set by replace_ignored_dims
            continue

        d1 = padded_shape1_list[i]
        d2 = padded_shape2_list[i]

        if d1 == d2:
            res_shape_list[i] = d1
        elif d1 == 1:
            res_shape_list[i] = d2
        elif d2 == 1:
            res_shape_list[i] = d1
        else:
            # Dimensions are different and neither is 1, broadcasting error.
            raise ValueError(
                f"Shapes {shape1} and {shape2} cannot be broadcast at dimension index {i} "
                f"(0-indexed from left of max_rank {max_rank} shape). "
                f"Padded values at this index are {d1} (from shape1) and {d2} (from shape2)."
            )

    return tuple(res_shape_list)


class Array:
    name: str
    impl: Optional[Tensor]
    args: List[Array]
    visited: bool
    shape: Shape
    dtype: DType
    device: Device
    tensor_value: Optional[Value]
    maxpr: Optional[MaxprCallable]
    vjp_rule: Optional[VJPRule]
    jvp_rule: Optional[JVPRule]
    batch_dim_ctr: int
    op_params: Optional[Dict[str, Any]]
    _numpy_cache: Optional[np.ndarray]

    def __init__(
        self,
        shape: Shape,
        dtype: DType = DType.float32,
        device: Device = CPU(),
        materialize: bool = False,
        name: str = "",
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.name = name
        self.args = []
        self.visited = False
        self.tensor_value = None
        self.maxpr = None
        self.vjp_rule = None
        self.jvp_rule = None
        self.batch_dim_ctr = 0
        self.op_params = None
        self._numpy_cache = None

        if materialize:
            self.impl = Tensor(dtype, shape, device=device)
        else:
            self.impl = None

    @classmethod
    def from_impl(cls, impl: Tensor, name: str = "") -> Array:
        if not isinstance(impl, Tensor):
            raise TypeError(f"Data must be a MAX Tensor, got {type(impl)}")
        if not impl.shape:
            raise ValueError("Cannot create Array from empty shape Tensor")

        instance = cls(
            shape=impl.shape, dtype=impl.dtype, device=impl.device, materialize=True
        )
        instance.impl = impl if impl else None
        instance.name = name
        return instance

    def copy_from(self, other: Array) -> None:
        if self.shape != other.shape or self.dtype != other.dtype:
            raise ValueError("Shape or dtype mismatch for copy")

        self.impl = other.impl.copy()

    def add_argument(self, arg_node: Array) -> None:
        if not isinstance(arg_node, Array):
            raise TypeError(
                f"Argument must be an instance of Array, got {type(arg_node)}"
            )
        self.args.append(arg_node)

    def realize(self) -> None:
        realize_([self,])
        if self.impl is None:
            raise ValueError("Data is None after realization")

    def get_numpy(self) -> np.ndarray:
        if self._numpy_cache is None:
            if self.impl is None:
                raise ValueError("Cannot get NumPy array from None impl")
            self._numpy_cache = self.impl.to_numpy()
        return self._numpy_cache

    def get_arguments(self) -> List[Array]:
        return list(self.args)

    def set_maxpr(self, fn: MaxprCallable) -> None:
        self.maxpr = fn

    def __repr__(self) -> str:
        self.realize()
        return self.impl.to(CPU()).to_numpy().__str__()

    def to(self, device: Device) -> Array:
        if self.impl is None:
            realize_(self)
        new_impl = self.impl.to(device)
        return Array.from_impl(new_impl, name=self.name)


def arange(shape: Shape, dtype: DType, device: Device = CPU()) -> Array:
    return Array.from_impl(
        Tensor.from_numpy(
            np.arange(np.prod(shape), dtype=DType.to_numpy(dtype)).reshape(shape)
        )
    ).to(device)


class Add:
    @staticmethod
    def maxpr(args: List[Value], output: Array) -> None:
        if len(args) != 2:
            raise ValueError(f"Add operation requires 2 arguments, got {len(args)}")
        output.tensor_value = ops.add(args[0], args[1])

    @staticmethod
    def eagerxpr(args: List[Array], output: Array) -> None:
        if len(args) != 2:
            raise ValueError(f"Add operation requires 2 arguments, got {len(args)}")
        np_result = np.add(args[0].get_numpy(), args[1].get_numpy())
        output.impl = Tensor.from_numpy(np_result)
    
    @staticmethod
    def vjp_rule(primals: List[Array], cotangent: Array, output: Array) -> List[Array]:
        if len(primals) != 2:
            raise ValueError(f"Add VJP rule requires 2 primals, got {len(primals)}")
        return [cotangent, cotangent]
    
    @staticmethod
    def jvp_rule(primals: List[Array], tangents: List[Array], output: Array) -> Array:
        if len(primals) != 2 or len(tangents) != 2:
            raise ValueError(f"Add JVP rule requires 2 primals and 2 tangents, got {len(primals)} and {len(tangents)}")
        return add(tangents[0], tangents[1])


def add(arg0: Array, arg1: Array) -> Array:
    if arg0.dtype != arg1.dtype:
        raise ValueError(
            f"Dtypes {arg0.dtype} and {arg1.dtype} are not compatible for multiplication."
        )
    
    res_shape = get_broadcasted_shape(arg0.shape, arg1.shape)
    res = Array(shape=res_shape, dtype=arg0.dtype, materialize=False, name="add")
    res.set_maxpr(Add.maxpr)
    res.add_argument(arg0)
    res.add_argument(arg1)
    res.vjp_rule = Add.vjp_rule
    res.jvp_rule = Add.jvp_rule

    if EAGERMODE:
        Add.eagerxpr([arg0, arg1], res)

    return res


class Mul:
    @staticmethod
    def maxpr(args: List[Value], output: Array) -> None:
        """MAX graph implementation of multiplication."""
        if len(args) != 2:
            raise ValueError(f"Mul operation requires 2 arguments, got {len(args)}")
        output.tensor_value = ops.mul(args[0], args[1])

    @staticmethod
    def eagerxpr(args: List[Array], output: Array) -> None:
        if len(args) != 2:
            raise ValueError(f"Mul operation requires 2 arguments, got {len(args)}")
        np_result = np.multiply(args[0].get_numpy(), args[1].get_numpy())
        output.impl = Tensor.from_numpy(np_result)
    
    @staticmethod
    def vjp_rule(primals: List[Array], cotangent: Array, output: Array) -> List[Array]:
        if len(primals) != 2:
            raise ValueError(f"Mul VJP rule requires 2 primals, got {len(primals)}")
        return [mul(cotangent, primals[1]), mul(cotangent, primals[0])]
    
    @staticmethod
    def jvp_rule(primals: List[Array], tangents: List[Array], output: Array) -> Array:
        if len(primals) != 2 or len(tangents) != 2:
            raise ValueError(f"Mul JVP rule requires 2 primals and 2 tangents, got {len(primals)} and {len(tangents)}")
        return ops.add(
            mul(primals[0], tangents[1]),
            mul(primals[1], tangents[0])
        )


def mul(arg0: Array, arg1: Array) -> Array:
    if arg0.dtype != arg1.dtype:
        raise ValueError(
            f"Dtypes {arg0.dtype} and {arg1.dtype} are not compatible for multiplication."
        )
    
    res_shape = get_broadcasted_shape(arg0.shape, arg1.shape)
    res = Array(shape=res_shape, dtype=arg0.dtype, materialize=False, name="mul")
    res.set_maxpr(Mul.maxpr)
    res.add_argument(arg0)
    res.add_argument(arg1)
    res.vjp_rule = Mul.vjp_rule
    res.jvp_rule = Mul.jvp_rule

    if EAGERMODE:
        Mul.eagerxpr([arg0, arg1], res)

    return res

class Transpose:
    @staticmethod
    def maxpr(args: List[Value], output: Array) -> None:
        if len(args) != 1:
            raise ValueError(f"Transpose operation requires 1 argument, got {len(args)}")
        output.tensor_value = ops.transpose(args[0])

    @staticmethod
    def eagerxpr(args: List[Array], output: Array) -> None:
        if len(args) != 1:
            raise ValueError(f"Transpose operation requires 1 argument, got {len(args)}")
        np_result = np.transpose(args[0].get_numpy())
        output.impl = Tensor.from_numpy(np_result)
    
    @staticmethod
    def vjp_rule(primals: List[Array], cotangent: Array, output: Array) -> List[Array]:
        if len(primals) != 1:
            raise ValueError(f"Transpose VJP rule requires 1 primal, got {len(primals)}")
        # The cotangent is the gradient of the output with respect to the input
        return [transpose(cotangent)]
    
    @staticmethod
    def jvp_rule(primals: List[Array], tangents: List[Array], output: Array) -> Array:
        if len(primals) != 1 or len(tangents) != 1:
            raise ValueError(f"Transpose JVP rule requires 1 primal and 1 tangent, got {len(primals)} and {len(tangents)}")
        # The JVP is simply the transpose of the tangent
        return transpose(tangents[0])
    
def transpose(arg: Array) -> Array: 
    res = Array(shape=arg.shape[::-1], dtype=arg.dtype, materialize=False, name="transpose")
    res.set_maxpr(Transpose.maxpr)
    res.add_argument(arg)
    res.vjp_rule = Transpose.vjp_rule
    res.jvp_rule = Transpose.jvp_rule

    if EAGERMODE:
        Transpose.eagerxpr([arg], res)

    return res

class MatMul:
    @staticmethod
    def maxpr(args: List[Value], output: Array) -> None:
        if len(args) != 2:
            raise ValueError(f"MatMul operation requires 2 arguments, got {len(args)}")

        x_val, y_val = args[0], args[1]
        x_shape_orig, y_shape_orig = x_val.shape, y_val.shape

        # K-dimension check (already in calling function, but good for safety)
        if x_shape_orig[-1] != y_shape_orig[-2]:
            raise ValueError(
                f"Shapes {x_shape_orig} and {y_shape_orig} are not compatible for matrix multiplication "
                f"(K-dimension mismatch: {x_shape_orig[-1]} vs {y_shape_orig[-2]})"
            )

        # 1. Determine the final N-D output shape.
        # This shape defines the target for broadcasting and the final result.
        # (M, K) @ (K, N) -> (M, N)
        # (B, M, K) @ (K, N) -> (B, M, N)
        # (B1, M, K) @ (B2, K, N) -> (B_broadcasted, M, N) if B1,B2 broadcast.
        output_shape_tuple = get_broadcasted_shape(
            x_shape_orig,
            y_shape_orig,
            ignore_axes=[-2, -1],
            replace_ignored_dims=[x_shape_orig[-2], y_shape_orig[-1]],
        )

        # Extract relevant dimensions
        m_dim = output_shape_tuple[-2]
        n_dim = output_shape_tuple[-1]
        k_dim = x_shape_orig[-1]  # K from input x

        output_batch_shape = output_shape_tuple[:-2]

        # 2. Broadcast inputs to align with the output_batch_shape.
        # x_val needs to become shape: output_batch_shape + (m_dim, k_dim)
        # y_val needs to become shape: output_batch_shape + (k_dim, n_dim)

        x_target_broadcast_shape = output_batch_shape + (m_dim, k_dim)
        if x_val.shape != x_target_broadcast_shape:
            x_val_b = ops.broadcast_to(x_val, x_target_broadcast_shape)
        else:
            x_val_b = x_val

        y_target_broadcast_shape = output_batch_shape + (k_dim, n_dim)
        if y_val.shape != y_target_broadcast_shape:
            y_val_b = ops.broadcast_to(y_val, y_target_broadcast_shape)
        else:
            y_val_b = y_val

        # At this point:
        # x_val_b.shape is output_batch_shape + (m_dim, k_dim)
        # y_val_b.shape is output_batch_shape + (k_dim, n_dim)
        # The leading output_batch_shape dimensions are identical for x_val_b and y_val_b.

        # 3. Reshape x_val_b and y_val_b to 4D for ops.matmul: (B_eff1, B_eff2, M, K)
        num_batch_dims = len(output_batch_shape)

        # Determine target shapes for ops.matmul input
        shape_for_x_matmul: Shape
        shape_for_y_matmul: Shape

        if num_batch_dims == 0:  # e.g., (M,K) input -> target (1,1,M,K) for matmul
            shape_for_x_matmul = (1, 1, m_dim, k_dim)
            shape_for_y_matmul = (1, 1, k_dim, n_dim)
        elif num_batch_dims == 1:  # e.g., (B0,M,K) -> target (B0,1,M,K)
            b0 = int(output_batch_shape[0])
            shape_for_x_matmul = (b0, 1, m_dim, k_dim)
            shape_for_y_matmul = (b0, 1, k_dim, n_dim)
        elif num_batch_dims == 2:  # e.g., (B0,B1,M,K) -> target (B0,B1,M,K) (no change)
            # x_val_b and y_val_b are already in the desired 4D format
            shape_for_x_matmul = x_val_b.shape
            shape_for_y_matmul = y_val_b.shape
        else:  # num_batch_dims > 2, e.g. (B0,B1,B2,M,K)
            # Flatten to (prod(B0..Bn-1), Bn, M,K)
            b_eff_1 = int(np.prod(output_batch_shape[:-1]))
            b_eff_2 = int(output_batch_shape[-1])
            shape_for_x_matmul = (b_eff_1, b_eff_2, m_dim, k_dim)
            shape_for_y_matmul = (b_eff_1, b_eff_2, k_dim, n_dim)

        # Perform the reshape if needed
        if x_val_b.shape == shape_for_x_matmul:
            x_for_matmul = x_val_b
        else:
            x_for_matmul = ops.reshape(x_val_b, shape_for_x_matmul)

        if y_val_b.shape == shape_for_y_matmul:
            y_for_matmul = y_val_b
        else:
            y_for_matmul = ops.reshape(y_val_b, shape_for_y_matmul)

        # 4. Perform the 4D matrix multiplication.
        # ops.matmul input shapes: (B_eff1, B_eff2, M, K) and (B_eff1, B_eff2, K, N)
        # ops.matmul output shape: (B_eff1, B_eff2, M, N)
        matmul_res_4d = ops.matmul(x_for_matmul, y_for_matmul)

        # 5. Reshape the 4D result back to the true N-D output_shape_tuple.
        # The shape of matmul_res_4d is (B_eff1_actual, B_eff2_actual, M, N).
        # We need to reshape it to output_shape_tuple = (BroadcastedBatchDims..., M, N)
        if matmul_res_4d.shape != output_shape_tuple:
            final_res = ops.reshape(matmul_res_4d, output_shape_tuple)
        else:
            # This condition implies output_shape_tuple was already 4D and matched
            # the (B_eff1, B_eff2, M, N) form, e.g., num_batch_dims == 2.
            final_res = matmul_res_4d

        output.tensor_value = final_res

    @staticmethod
    def eagerxpr(args: List[Array], output: Array) -> None:
        if len(args) != 2:
            raise ValueError(f"MatMul operation requires 2 arguments, got {len(args)}")

        arg0_numpy = args[0].get_numpy()
        arg1_numpy = args[1].get_numpy()

        if arg0_numpy.shape[-1] != arg1_numpy.shape[-2]:
            raise ValueError(
                f"Eager MatMul: Shapes {args[0].shape} and {args[1].shape} are not compatible for matrix multiplication."
            )

        np_result = np.matmul(arg0_numpy, arg1_numpy)
        output.impl = Tensor.from_numpy(np_result)
    
    @staticmethod
    def vjp_rule(primals: List[Array], cotangent: Array, output: Array) -> List[Array]:
        if len(primals) != 2:
            raise ValueError(f"MatMul VJP rule requires 2 primals, got {len(primals)}")
        
        x, y = primals
        return [matmul(cotangent, transpose(y)), matmul(transpose(x), cotangent)]
    
    @staticmethod
    def jvp_rule(primals: List[Array], tangents: List[Array], output: Array) -> Array:
        if len(primals) != 2 or len(tangents) != 2:
            raise ValueError(f"MatMul JVP rule requires 2 primals and 2 tangents, got {len(primals)} and {len(tangents)}")
        
        x, y = primals
        tx, ty = tangents
        return add(matmul(x, ty), matmul(tx, y))
      


def matmul(arg0: Array, arg1: Array) -> Array:
    if arg0.shape[-1] != arg1.shape[-2]:
        raise ValueError(
            f"Shapes {arg0.shape} and {arg1.shape} are not compatible for matrix multiplication."
        )

    res = Array(
        shape=get_broadcasted_shape(
            arg0.shape,
            arg1.shape,
            ignore_axes=[-2, -1],
            replace_ignored_dims=[arg0.shape[-2], arg1.shape[-1]],
        ),
        dtype=arg0.dtype,
        materialize=False,
        name="matmul",
    )
    res.set_maxpr(MatMul.maxpr)
    res.add_argument(arg0)
    res.add_argument(arg1)
    res.vjp_rule = MatMul.vjp_rule
    res.jvp_rule = MatMul.jvp_rule

    if EAGERMODE:
        MatMul.eagerxpr([arg0, arg1], res)

    return res


class Negate:
    @staticmethod
    def maxpr(args: List[Value], output: Array) -> None:
        if len(args) != 1:
            raise ValueError(f"Negate operation requires 1 argument, got {len(args)}")
        output.tensor_value = ops.negative(args[0])

    @staticmethod
    def eagerxpr(args: List[Array], output: Array) -> None:
        if len(args) != 1:
            raise ValueError(f"Negate operation requires 1 argument, got {len(args)}")
        np_result = np.negative(args[0].get_numpy())
        output.impl = Tensor.from_numpy(np_result)
    
    @staticmethod
    def vjp_rule(primals: List[Array], cotangent: Array, output: Array) -> List[Array]:
        if len(primals) != 1:
            raise ValueError(f"Negate VJP rule requires 1 primal, got {len(primals)}")
        return [negate(cotangent)]
    
    @staticmethod
    def jvp_rule(primals: List[Array], tangents: List[Array], output: Array) -> Array:
        if len(primals) != 1 or len(tangents) != 1:
            raise ValueError(f"Negate JVP rule requires 1 primal and 1 tangent, got {len(primals)} and {len(tangents)}")
        return negate(tangents[0])

    
def negate(arg: Array) -> Array:
    res = Array(shape=arg.shape, dtype=arg.dtype, materialize=False, name="negate")
    res.set_maxpr(Negate.maxpr)
    res.add_argument(arg)
    res.vjp_rule = Negate.vjp_rule
    res.jvp_rule = Negate.jvp_rule

    if EAGERMODE:
        Negate.eagerxpr([arg], res)

    return res


class Cos:
    @staticmethod
    def maxpr(args: List[Value], output: Array) -> None:
        if len(args) != 1:
            raise ValueError(f"Cos operation requires 1 argument, got {len(args)}")
        output.tensor_value = ops.cos(args[0])

    @staticmethod
    def eagerxpr(args: List[Array], output: Array) -> None:
        if len(args) != 1:
            raise ValueError(f"Cos operation requires 1 argument, got {len(args)}")
        np_result = np.cos(args[0].get_numpy())
        output.impl = Tensor.from_numpy(np_result)
    
    @staticmethod
    def vjp_rule(primals: List[Array], cotangent: Array, output: Array) -> List[Array]:
        if len(primals) != 1:
            raise ValueError(f"Cos VJP rule requires 1 primal, got {len(primals)}")
        return [negate(mul(cotangent, sin(primals[0])))]
    
    @staticmethod
    def jvp_rule(primals: List[Array], tangents: List[Array], output: Array) -> Array:
        if len(primals) != 1 or len(tangents) != 1:
            raise ValueError(f"Cos JVP rule requires 1 primal and 1 tangent, got {len(primals)} and {len(tangents)}")
        return negate(mul(tangents[0], sin(primals[0])))
    
def cos(arg: Array) -> Array:
    res = Array(shape=arg.shape, dtype=arg.dtype, materialize=False, name="cos")
    res.set_maxpr(Cos.maxpr)
    res.add_argument(arg)
    res.vjp_rule = Cos.vjp_rule
    res.jvp_rule = Cos.jvp_rule

    if EAGERMODE:
        Cos.eagerxpr([arg], res)

    return res

class Sin:
    @staticmethod
    def maxpr(args: List[Value], output: Array) -> None:
        if len(args) != 1:
            raise ValueError(f"Sin operation requires 1 argument, got {len(args)}")
        output.tensor_value = ops.sin(args[0])

    @staticmethod
    def eagerxpr(args: List[Array], output: Array) -> None:
        if len(args) != 1:
            raise ValueError(f"Sin operation requires 1 argument, got {len(args)}")
        np_result = np.sin(args[0].get_numpy())
        output.impl = Tensor.from_numpy(np_result)
    
    @staticmethod
    def vjp_rule(primals: List[Array], cotangent: Array, output: Array) -> List[Array]:
        if len(primals) != 1:
            raise ValueError(f"Sin VJP rule requires 1 primal, got {len(primals)}")
        return [mul(cotangent, cos(primals[0]))]
    
    @staticmethod
    def jvp_rule(primals: List[Array], tangents: List[Array], output: Array) -> Array:
        if len(primals) != 1 or len(tangents) != 1:
            raise ValueError(f"Sin JVP rule requires 1 primal and 1 tangent, got {len(primals)} and {len(tangents)}")
        return mul(tangents[0], cos(primals[0]))
    
def sin(arg: Array) -> Array:
    res = Array(shape=arg.shape, dtype=arg.dtype, materialize=False, name="sin")
    res.set_maxpr(Sin.maxpr)
    res.add_argument(arg)
    res.vjp_rule = Sin.vjp_rule
    res.jvp_rule = Sin.jvp_rule

    if EAGERMODE:
        Sin.eagerxpr([arg], res)

    return res

class AddOneCustom:
    @staticmethod
    def maxpr(args: List[Value], output: Array) -> None:
        output.tensor_value = ops.custom(
            name="add_one_custom",
            values=[args[0]],
            out_types=[TensorType(dtype=args[0].dtype, shape=args[0].tensor.shape, device=args[0].device)],
        )[0].tensor
    
    @staticmethod
    def eagerxpr(args: List[Array], output: Array) -> None:
        if len(args) != 1:
            raise ValueError(f"AddOneCustom operation requires 1 argument, got {len(args)}")
        np_result = args[0].get_numpy() + 1
        output.impl = Tensor.from_numpy(np_result)
    
    @staticmethod
    def vjp_rule(primals: List[Array], cotangent: Array, output: Array) -> List[Array]:
        raise NotImplementedError(
            "AddOneCustom does not support VJP rule. It is a custom operation."
        )
    
    @staticmethod
    def jvp_rule(primals: List[Array], tangents: List[Array], output: Array) -> Array:
        raise NotImplementedError(
            "AddOneCustom does not support JVP rule. It is a custom operation."
        )
    
def add_one_custom(arg: Array) -> Array:
    res = Array(shape=arg.shape, dtype=arg.dtype, materialize=False, name="add_one_custom")
    res.set_maxpr(AddOneCustom.maxpr)
    res.add_argument(arg)
    res.vjp_rule = AddOneCustom.vjp_rule
    res.jvp_rule = AddOneCustom.jvp_rule

    if EAGERMODE:
        AddOneCustom.eagerxpr([arg], res)

    return res

class IncrBatchDimCtr:
    @staticmethod
    def maxpr(args: List[Value], output: Array) -> None:
        output.tensor_value = args[0].tensor_value
    
    @staticmethod
    def eagerxpr(args: List[Array], output: Array) -> None:
        output.impl = args[0].impl
    
    @staticmethod
    def vjp_rule(primals: List[Array], cotangent: Array, output: Array) -> List[Array]:
        return incr_batch_dim_ctr(cotangent)
    
    @staticmethod
    def jvp_rule(primals: List[Array], tangents: List[Array], output: Array) -> Array:
        return decr_batch_dim_ctr(tangents[0])
    
def incr_batch_dim_ctr(arg: Array) -> Array:
    if EAGERMODE:
        IncrBatchDimCtr.eagerxpr([arg])
    else:
        res = Array(shape=arg.shape, dtype=arg.dtype, materialize=False, name="incr_batch_dim_ctr")
        res.set_maxpr(IncrBatchDimCtr.maxpr)

    res.add_argument(arg)
    res.vjp_rule = IncrBatchDimCtr.vjp_rule
    res.jvp_rule = IncrBatchDimCtr.jvp_rule
    res.batch_dim_ctr += 1
    return res

class DecrBatchDimCtr:
    @staticmethod
    def maxpr(args: List[Value], output: Array) -> None:
        output.tensor_value = args[0].tensor_value
    
    @staticmethod
    def eagerxpr(args: List[Array], output: Array) -> None:
        output.impl = args[0].impl 
    
    @staticmethod
    def vjp_rule(primals: List[Array], cotangent: Array, output: Array) -> List[Array]:
        return decr_batch_dim_ctr(cotangent)
    
    @staticmethod
    def jvp_rule(primals: List[Array], tangents: List[Array], output: Array) -> Array:
        return incr_batch_dim_ctr(tangents[0])
    
def decr_batch_dim_ctr(arg: Array) -> Array:
    if arg.batch_dim_ctr <= 0:
        raise ValueError("Cannot decrement batch_dim_ctr below 0")
    res = Array(shape=arg.shape, dtype=arg.dtype, materialize=False, name="decr_batch_dim_ctr")
    res.set_maxpr(DecrBatchDimCtr.maxpr)
    res.add_argument(arg)
    res.vjp_rule = DecrBatchDimCtr.vjp_rule
    res.jvp_rule = DecrBatchDimCtr.jvp_rule
    res.batch_dim_ctr = max(0, arg.batch_dim_ctr - 1)
    
    if EAGERMODE:
        DecrBatchDimCtr.eagerxpr([arg], res)

    return res

class Squeeze:
    # helper methods
    @staticmethod
    def get_axis(output: Array) -> int:
        op_params = output.op_params or {}
        if "axis" not in op_params:
            raise ValueError("Squeeze operation requires 'axis' parameter in op_params")
        axis = op_params["axis"]
        if not isinstance(axis, int):
            raise ValueError(f"Squeeze 'axis' must be an int, got {type(axis)}")
        return axis
    
    @staticmethod
    def get_shape(input_shape: Tuple[int, ...], axis: int) -> Tuple[int, ...]:
        if axis < 0:
            axis += len(input_shape)
        if axis < 0 or axis >= len(input_shape):
            raise ValueError(f"Axis {axis} is out of bounds for shape {input_shape}")
        return tuple(dim for i, dim in enumerate(input_shape) if i != axis or dim != 1)
    
    @staticmethod
    def maxpr(args: List[Value], output: Array) -> None:
        if len(args) != 1:
            raise ValueError(f"Squeeze operation requires 1 argument, got {len(args)}")
        output.tensor_value = ops.squeeze(args[0], axis=Squeeze.get_axis(output))
    
    @staticmethod
    def eagerxpr(args: List[Array], output: Array) -> None:
        if len(args) != 1:
            raise ValueError(f"Squeeze operation requires 1 argument, got {len(args)}")
        np_result = np.squeeze(args[0].get_numpy(), axis=Squeeze.get_axis(output))
        output.impl = Tensor.from_numpy(np_result)
    
    @staticmethod
    def vjp_rule(primals: List[Array], cotangent: Array, output: Array) -> List[Array]:
        if len(primals) != 1:
            raise ValueError(f"Squeeze VJP rule requires 1 primal, got {len(primals)}")
        return [unsqueeze(cotangent, axis=Squeeze.get_axis(output))]
    
    @staticmethod
    def jvp_rule(primals: List[Array], tangents: List[Array], output: Array) -> Array:
        if len(primals) != 1 or len(tangents) != 1:
            raise ValueError(f"Squeeze JVP rule requires 1 primal and 1 tangent, got {len(primals)} and {len(tangents)}")
        return squeeze(tangents[0], axis=Squeeze.get_axis(output))
    
def squeeze(arg: Array, axis: int) -> Array:
    res = Array(shape=Squeeze.get_shape(arg.shape, axis), dtype=arg.dtype, materialize=False, name=f"squeeze_{axis}")
    res.set_maxpr(Squeeze.maxpr)
    res.op_params = {"axis": axis}
    res.add_argument(arg)
    res.vjp_rule = Squeeze.vjp_rule
    res.jvp_rule = Squeeze.jvp_rule

    if EAGERMODE:
        Squeeze.eagerxpr([arg], res)

    return res

class Unsqueeze:
    # helper methods
    @staticmethod
    def get_axis(output: Array) -> int:
        op_params = output.op_params or {}
        if "axis" not in op_params:
            raise ValueError("Unsqueeze operation requires 'axis' parameter in op_params")
        axis = op_params["axis"]
        if not isinstance(axis, int):
            raise ValueError(f"Unsqueeze 'axis' must be an int, got {type(axis)}")
        return axis
    
    @staticmethod
    def get_shape(input_shape: Tuple[int, ...], axis: int) -> Tuple[int, ...]:
        if axis < 0:
            axis += len(input_shape) + 1
        return input_shape[:axis] + (1,) + input_shape[axis:]
    
    @staticmethod
    def maxpr(args: List[Value], output: Array) -> None:
        if len(args) != 1:
            raise ValueError(f"Unsqueeze operation requires 1 argument, got {len(args)}")
        output.tensor_value = ops.unsqueeze(args[0], axis=Unsqueeze.get_axis(output))
    
    @staticmethod
    def eagerxpr(args: List[Array], output: Array) -> None:
        if len(args) != 1:
            raise ValueError(f"Unsqueeze operation requires 1 argument, got {len(args)}")
        np_result = np.expand_dims(args[0].get_numpy(), axis=Unsqueeze.get_axis(output))
        output.impl = Tensor.from_numpy(np_result)
    
    @staticmethod
    def vjp_rule(primals: List[Array], cotangent: Array, output: Array) -> List[Array]:
        if len(primals) != 1:
            raise ValueError(f"Unsqueeze VJP rule requires 1 primal, got {len(primals)}")
        return [squeeze(cotangent, axis=Unsqueeze.get_axis(output))]
    
    @staticmethod
    def jvp_rule(primals: List[Array], tangents: List[Array], output: Array) -> Array:
        if len(primals) != 1 or len(tangents) != 1:
            raise ValueError(f"Unsqueeze JVP rule requires 1 primal and 1 tangent, got {len(primals)} and {len(tangents)}")
        return unsqueeze(tangents[0], axis=Unsqueeze.get_axis(output))
    
def unsqueeze(arg: Array, axis: int) -> Array:
    res = Array(shape=Unsqueeze.get_shape(arg.shape, axis), dtype=arg.dtype, materialize=False, name=f"unsqueeze_{axis}")
    res.set_maxpr(Unsqueeze.maxpr)
    res.op_params = {"axis": axis}
    res.add_argument(arg)
    res.vjp_rule = Unsqueeze.vjp_rule
    res.jvp_rule = Unsqueeze.jvp_rule

    if EAGERMODE:
        print("Unsqueeze eagerxpr called")
        Unsqueeze.eagerxpr([arg], res)

    return res

class BroadcastTo:
    # helper methods
    @staticmethod
    def get_target_shape(output: Array) -> Tuple[int, ...]:
        op_params = output.op_params or {}
        if "target_shape" not in op_params:
            raise ValueError("BroadcastTo operation requires 'shape' parameter in op_params")
        target_shape = op_params["target_shape"]
        if not isinstance(target_shape, tuple):
            raise ValueError(f"BroadcastTo 'shape' must be a tuple, got {type(target_shape)}")
        return target_shape
    
    @staticmethod
    def get_broadcasted_axes(input_shape: Tuple[int, ...], target_shape: Tuple[int, ...]) -> List[int]:
        if len(input_shape) > len(target_shape):
            raise ValueError(f"Input shape {input_shape} cannot be broadcast to target shape {target_shape}")
        
        broadcasted_axes = []
        for i in range(len(target_shape)):
            if i < len(input_shape):
                if input_shape[i] != target_shape[i] and input_shape[i] != 1:
                    raise ValueError(f"Cannot broadcast {input_shape} to {target_shape}")
            else:
                broadcasted_axes.append(i)
        
        return broadcasted_axes
    
    @staticmethod
    def maxpr(args: List[Value], output: Array) -> None:
        if len(args) != 2:
            raise ValueError(f"BroadcastTo operation requires 2 arguments, got {len(args)}")
        output.tensor_value = ops.broadcast_to(args[0], BroadcastTo.get_target_shape(output))
    
    @staticmethod
    def eagerxpr(args: List[Array], output: Array) -> None:
        if len(args) != 2:
            raise ValueError(f"BroadcastTo operation requires 2 arguments, got {len(args)}")
        target_shape = BroadcastTo.get_target_shape(output)
        np_result = np.broadcast_to(args[0].get_numpy(), target_shape)
        output.impl = Tensor.from_numpy(np_result)
    
    @staticmethod
    def vjp_rule(primals: List[Array], cotangent: Array, output: Array) -> List[Array]:
        # lets assume that sum operation is already impleted 
        if len(primals) != 2:
            raise ValueError(f"BroadcastTo VJP rule requires 2 primals, got {len(primals)}")
        target_shape = BroadcastTo.get_target_shape(output)
        brodcasted_axes = BroadcastTo.get_broadcasted_axes(primals[0].shape, target_shape)
        if not brodcasted_axes:
            raise ValueError(f"No broadcasted axes found for {primals[0].shape} to {target_shape}")
        # We need to sum over the broadcasted axes
        return [sum(cotangent, axes=brodcasted_axes)]
    
    @staticmethod
    def jvp_rule(primals: List[Array], tangents: List[Array], output: Array) -> Array:
        if len(primals) != 2 or len(tangents) != 2:
            raise ValueError(f"BroadcastTo JVP rule requires 2 primals and 2 tangents, got {len(primals)} and {len(tangents)}")
        target_shape = BroadcastTo.get_target_shape(output)
        return broadcast_to(tangents[0], target_shape)
    
def broadcast_to(arg: Array, target_shape: Tuple[int, ...]) -> Array:
    res = Array(shape=target_shape, dtype=arg.dtype, materialize=False, name=f"broadcast_to_{target_shape}")
    res.set_maxpr(BroadcastTo.maxpr)
    res.op_params = {"target_shape": target_shape}
    res.add_argument(arg)
    res.vjp_rule = BroadcastTo.vjp_rule
    res.jvp_rule = BroadcastTo.jvp_rule

    if EAGERMODE:
        BroadcastTo.eagerxpr([arg], res)
    return res

class Reshape:
    # helper methods 
    def get_target_shape(output: Array) -> Tuple[int, ...]:
        op_params = output.op_params or {}
        if "target_shape" not in op_params:
            raise ValueError("Reshape operation requires 'target_shape' parameter in op_params")
        target_shape = op_params["target_shape"]
        if not isinstance(target_shape, tuple):
            raise ValueError(f"Reshape 'target_shape' must be a tuple, got {type(target_shape)}")
        return target_shape
    
    def get_arg_shape(output: Array) -> Tuple[int, ...]:
        op_params = output.op_params or {}
        if "arg_shape" not in op_params:
            raise ValueError("Reshape operation requires 'arg_shape' parameter in op_params")
        arg_shape = op_params["arg_shape"]
        if not isinstance(arg_shape, tuple):
            raise ValueError(f"Reshape 'arg_shape' must be a tuple, got {type(arg_shape)}")
        return arg_shape
    
    @staticmethod
    def maxpr(args: List[Value], output: Array) -> None:
        if len(args) != 1:
            raise ValueError(f"Reshape operation requires 1 argument, got {len(args)}")
        output.tensor_value = ops.reshape(args[0], Reshape.get_target_shape(output))

    @staticmethod
    def eagerxpr(args: List[Array], output: Array) -> None:
        if len(args) != 1:
            raise ValueError(f"Reshape operation requires 1 argument, got {len(args)}")
        target_shape = Reshape.get_target_shape(output)
        np_result = np.reshape(args[0].get_numpy(), target_shape)
        output.impl = Tensor.from_numpy(np_result)

    @staticmethod
    def vjp_rule(primals: List[Array], cotangent: Array, output: Array) -> List[Array]:
        if len(primals) != 1:
            raise ValueError(f"Reshape VJP rule requires 1 primal, got {len(primals)}")
        arg_shape = Reshape.get_arg_shape(output)
        return [reshape(cotangent, arg_shape)]
    
    @staticmethod
    def jvp_rule(primals: List[Array], tangents: List[Array], output: Array) -> Array:
        if len(primals) != 1 or len(tangents) != 1:
            raise ValueError(f"Reshape JVP rule requires 1 primal and 1 tangent, got {len(primals)} and {len(tangents)}")
        target_shape = Reshape.get_target_shape(output)
        return reshape(tangents[0], target_shape)
    
def reshape(arg: Array, shape: Tuple[int, ...]) -> Array:
    if not isinstance(shape, tuple):
        raise ValueError(f"Reshape 'target_shape' must be a tuple, got {type(target_shape)}")
    
    res = Array(shape=shape, dtype=arg.dtype, materialize=False, name=f"reshape_{shape}")
    res.set_maxpr(Reshape.maxpr)
    res.op_params = {"target_shape": shape, "arg_shape": arg.shape}
    res.add_argument(arg)
    res.vjp_rule = Reshape.vjp_rule
    res.jvp_rule = Reshape.jvp_rule

    if EAGERMODE:
        Reshape.eagerxpr([arg], res)

    return res



class Sum:
    # helper methods
    @staticmethod
    def get_shape(arg_shape: Tuple[int, ...], axes: Union[int, List[int], None], keep_dim: bool = False) -> Tuple[int, ...]:
        if axes is None:
            return arg_shape
        if isinstance(axes, int):
            axes = [axes]
        if not isinstance(axes, list):
            raise ValueError(f"Sum 'axes' must be an int or a list of ints, got {type(axes)}")
        target_shape = []
        for i, dim in enumerate(arg_shape):

            if i in axes:
                if keep_dim:
                    target_shape.append(1)
            else:
                target_shape.append(dim)
        return tuple(target_shape)
                
    
    @staticmethod
    def maxpr(args: List[Value], output: Array) -> None:
        if len(args) != 1:
            raise ValueError(f"Sum operation requires 1 argument, got {len(args)}")
        axes = output.op_params.get("axes", None)
        if isinstance(axes, int):
            axes = [axes]
        axes = sorted(axes) if axes is not None else None
        output_symbol = args[0]
        for i in range(len(axes)):
            output_symbol = ops.sum(output_symbol, axis=axes[i] - i)
        output.tensor_value = output_symbol


    @staticmethod
    def eagerxpr(args: List[Array], output: Array) -> None:
        if len(args) != 1:
            raise ValueError(f"Sum operation requires 1 argument, got {len(args)}")
        axes = output.op_params.get("axes", None)
        keep_dims = output.op_params.get("keep_dims", False)
        np_result = np.sum(args[0].get_numpy(), axis=axes, keepdims=keep_dims)
        output.impl = Tensor.from_numpy(np_result)

    @staticmethod
    def vjp_rule(primals: List[Array], cotangent: Array, output: Array) -> List[Array]:
        if len(primals) != 1:
            raise ValueError(f"Sum VJP rule requires 1 primal, got {len(primals)}")
        op_params = output.op_params or {}
        # retreive "arg_shape" from params and brodcast cotangent to this shape 
        arg_shape = op_params.get("arg_shape", primals[0].shape)
        if not isinstance(arg_shape, tuple):
            raise ValueError(f"Sum 'arg_shape' must be a tuple, got {type(arg_shape)}")
        
        # We need to broadcast the cotangent to the arg_shape and then sum over the axes
        return [broadcast_to(cotangent, arg_shape)]
        
    @staticmethod
    def jvp_rule(primals: List[Array], tangents: List[Array], output: Array) -> Array:
        if len(primals) != 1 or len(tangents) != 1:
            raise ValueError(f"Sum JVP rule requires 1 primal and 1 tangent, got {len(primals)} and {len(tangents)}")
        axes = output.op_params.get("axes", None)
        return sum(tangents[0], axes=axes, keep_dims=output.op_params.get("keep_dims", False))
    
def sum(arg: Array, axes: Union[int, List[int], None] = None, keep_dims: bool = False) -> Array:
    res = Array(
        shape=Sum.get_shape(arg.shape, axes),
        dtype=arg.dtype,
        materialize=False,
        name=f"sum_{axes}_{keep_dims}",
    )
    res.set_maxpr(Sum.maxpr)
    res.op_params = {"axes": axes, "arg_shape": arg.shape, "keep_dims": keep_dims}
    res.add_argument(arg)
    res.vjp_rule = Sum.vjp_rule
    res.jvp_rule = Sum.jvp_rule

    if EAGERMODE:
        Sum.eagerxpr([arg], res)

    return res


def compute_node_hash(node: Array) -> int:
    components = [
        str(node.shape),
        str(node.dtype),
        node.name or "unnamed",
    ]
    node_str = "-".join(components)
    return hash(node_str)


def get_trace(nodes: Sequence[Array]) -> Tuple[List[Array], List[Array], int]:
    trace: List[Array] = []
    inputs: List[Array] = []
    visited: Set[Array] = set()

    # Iterative DFS using a stack
    for start_node in nodes:
        if start_node in visited:
            continue

        stack: List[Array] = [start_node]
        while stack:
            node = stack[-1]  # Peek at the top node

            if node in visited:
                stack.pop()  # Already processed this node
                continue

            # If this is a leaf node (has impl)
            if node.impl is not None:
                inputs.append(node)
                trace.append(node)
                visited.add(node)
                stack.pop()
                continue

            # Check if all children have been visited
            all_children_visited = True
            for arg in node.args:
                if arg not in visited:
                    all_children_visited = False
                    stack.append(arg)  # Add unvisited child to stack

            # If all children have been visited, we can process this node
            if all_children_visited:
                visited.add(node)
                trace.append(node)
                stack.pop()

    # Compute the key from the trace
    key: int = 0
    for node in trace:
        node_hash = compute_node_hash(node)
        key = key ^ (node_hash + 0x9E3779B9 + (key << 6) + (key >> 2))

    key = key % 1000000000

    return inputs, trace, key


def realize_(outputs: List[Array]) -> None:
    
    output_list = []

    # Check if there are outputs which need to be realized
    nothing_to_compute = True
    for output in outputs:
        if not isinstance(output, Array):
            raise TypeError("Outputs must be an instance of Array")
        if output.impl is None:
            output_list.append(output)
            nothing_to_compute = False

    if nothing_to_compute:
        return

    # Retrieve the trace and inputs which are the last realized values (i.e. leaves)
    inputs, trace, key = get_trace(output_list)

    if key in global_execution_context:
        model = global_execution_context[key]
    else:
        # Build input types for the Graph
        input_types = []
        devices = []
        for input_node in inputs:
            input_types.append(
                TensorType(
                    dtype=input_node.dtype,
                    shape=input_node.shape,
                    device=DeviceRef.from_device(input_node.device),
                )
            )
            if input_node.device not in devices:
                devices.append(input_node.device)

        # Define the MAX graph
        try:
            custom_op_package_path = Path(__file__).parent / "mojo_kernels"
            with Graph("max_graph", input_types=input_types, custom_extensions=[custom_op_package_path]) as graph:
                input_symbols = graph.inputs
                for i in range(len(input_symbols)):
                    inputs[i].tensor_value = input_symbols[i]

                for node in trace:
                    if node.tensor_value is not None:
                        continue

                    arg_symbols = []
                    for arg in node.get_arguments():
                        if arg.tensor_value is None:
                            raise ValueError(f"Error retrieving symbol for {arg.name}")
                        arg_symbols.append(arg.tensor_value)

                    if node.maxpr is None:
                        raise ValueError(f"Node {node.name} has no maxpr function")
                    node.maxpr(args=arg_symbols, output=node) # set tensor value for each node inplace wiht the respective maxpr rule

                output_symbols = []
                for output in output_list:
                    if output.tensor_value is None:
                        raise ValueError(f"Output {output.name} has no tensor value")
                    output_symbols.append(output.tensor_value)

                graph.output(*output_symbols)

            # Set up the MAX model and cache it
            session = InferenceSession(devices=devices)
            model = session.load(graph)
            global_execution_context[key] = model

        except Exception as e:
            raise ValueError(f"Failed to build computation graph: {e}")

    # Prepare input tensors
    tensor_inputs = []
    for input_node in inputs:
        if input_node.impl is None:
            raise ValueError(f"Input {input_node.name} has no impl")
        tensor_inputs.append(input_node.impl)

    # Execute the model and update outputs
    try:
        model_outputs = model.execute(*tensor_inputs)
        for i, output in enumerate(output_list):
            output.impl = model_outputs[i]
            output._numpy_cache = None  # Invalidate cache
    except Exception as e:
        raise ValueError(f"Error executing computation: {e}")