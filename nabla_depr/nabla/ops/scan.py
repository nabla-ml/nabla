
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..core.tensor import Tensor
from .operation import Operation
from max.dtype import DType
from max.graph import TensorValue
from max.dtype import DType

class TupleGetItemOp(Operation):
    def __init__(self, index: int, output_shape: tuple, output_dtype: DType, output_batch_dims: tuple = ()):
        super().__init__(f"getitem_{index}")
        self.index = index
        self._output_shape = output_shape
        self._output_dtype = output_dtype
        self._output_batch_dims = output_batch_dims

    def compute_output_shape(self, *input_shapes):
        return self._output_shape

    def forward(self, input_tensor: Tensor) -> Tensor:
        res = Tensor(
            shape=self._output_shape,
            dtype=self._output_dtype,
            device=input_tensor.logical_device,
            materialize=False,
            name=self.name,
            batch_dims=self._output_batch_dims,
        )
        res.set_maxpr(self.maxpr)
        res.add_arguments(input_tensor)
        res.creator_op = self
        
        if not res.stage_realization:
             self.eagerxpr([input_tensor], res)
             
        return res

    def maxpr(self, args: list[TensorValue], output: Tensor) -> None:
        # args[0] is the list/tuple from scan
        # We assume it supports indexing
        output.tensor_value = args[0][self.index]

    def eagerxpr(self, args: list[Tensor], output: Tensor) -> None:
        # args[0] is the container tensor.
        # Its impl should be the list of numpy arrays.
        # We access _impl directly because to_numpy() enforces array conversion and fails for lists.
        container = args[0]
        container_impl = container._impl
        
        if isinstance(container_impl, (list, tuple)):
            output.impl_(container_impl[self.index])
        else:
             # Fallback or error
             raise TypeError(f"Expected list/tuple in container impl, got {type(container_impl)}")

    def vjp_rule(self, primals, cotangent, output):
        # TODO: Implement VJP for TupleGetItemOp if needed.
        # For now, we assume ScanOp handles its own gradients or we don't differentiate through this yet.
        raise NotImplementedError("VJP for TupleGetItemOp not implemented yet")

    def jvp_rule(self, primals, tangents, output):
        raise NotImplementedError("JVP for TupleGetItemOp not implemented yet")


class ScanOp(Operation):
    def __init__(
        self,
        f: Callable,
        length: int,
        reverse: bool,
        unroll: int,
        init_structure: Any,
        xs_structure: Any,
        carry_structure: Any,
        y_structure: Any,
        flat_y_shapes: list[tuple],
        flat_y_dtypes: list[DType],
        num_init: int,
    ):
        super().__init__("scan")
        self.f = f
        self.length = length
        self.reverse = reverse
        self.unroll = unroll
        self.init_structure = init_structure
        self.xs_structure = xs_structure
        self.carry_structure = carry_structure
        self.y_structure = y_structure
        self.flat_y_shapes = flat_y_shapes
        self.flat_y_dtypes = flat_y_dtypes
        self.num_init = num_init

    def compute_output_shape(self, *input_shapes: tuple) -> tuple:
        # Not used for multi-output ScanOp in the same way
        return ()

    def forward(self, *args: Tensor) -> Tensor:
        # Split args into init and xs
        num_init = self.num_init
        
        flat_init = args[:num_init]
        flat_xs = args[num_init:]
        
        # Determine output shapes (for metadata, though container doesn't strictly need them)
        # But we need to compute batch dims for the container
        from ..utils.shape_utils import get_broadcasted_shape
        
        output_batch_dims = ()
        for arg in args:
            output_batch_dims = get_broadcasted_shape(output_batch_dims, arg.batch_dims)
        
        device = args[0].logical_device

        # Create the Container Tensor (scan_result)
        # This tensor will hold the list of results from maxpr.
        scan_result = Tensor(
            shape=(), 
            dtype=DType.float32, # Dummy
            device=device,
            materialize=False,
            name=f"{self.name}_result_container",
            batch_dims=output_batch_dims
        )
        scan_result.creator_op = self
        scan_result.add_arguments(*args)
        scan_result.set_maxpr(self.maxpr)

        # If eager execution is needed (not tracing)
        is_traced = any(arg.traced for arg in args)
        
        if not is_traced:
             self.eagerxpr(list(args), scan_result)
             
        return scan_result

    def eagerxpr(self, args: list[Tensor], output: Tensor) -> None:
        from ..transforms.utils import tree_unflatten, tree_flatten, tree_map
        from .indexing import gather
        from ..ops.creation import tensor
        from ..ops.view import stack
        
        # Reconstruct inputs
        num_init = self.num_init
        flat_init = args[:num_init]
        flat_xs = args[num_init:]
        
        init = tree_unflatten(self.init_structure, flat_init)
        xs = tree_unflatten(self.xs_structure, flat_xs)
        
        # Prepare eager loop
        carry = init
        ys_accum = []
        
        # Iterate
        loop_range = range(self.length)
        if self.reverse:
            loop_range = reversed(loop_range)
            
        for i in loop_range:
            # Slice xs at i
            idx = tensor(i, dtype=DType.int64, device=args[0].logical_device)
            
            def _slice(x):
                return gather(x, idx, axis=0)
            
            x_i = tree_map(_slice, xs)
            
            # Run f
            carry, y_i = self.f(carry, x_i)
            
            ys_accum.append(y_i)
            
        if self.reverse:
            ys_accum.reverse()
            
        # Stack ys
        flat_ys_accum = [tree_flatten(y)[0] for y in ys_accum]
        
        if not flat_ys_accum:
            stacked_leaves = []
        else:
            num_leaves = len(flat_ys_accum[0])
            transposed_leaves = [[flat_ys_accum[t][i] for t in range(self.length)] for i in range(num_leaves)]
            stacked_leaves = [stack(l, axis=0) for l in transposed_leaves]
        
        # Assign to outputs
        flat_final_carry, _ = tree_flatten(carry)
        
        # Assign to output container
        flat_final_carry, _ = tree_flatten(carry)
        
        # The result structure should match what maxpr returns: [index, *carry, *ys, *xs]
        # But wait, maxpr returns that because of while_loop.
        # Eager execution doesn't use while_loop in the same way.
        # However, TupleGetItemOp expects the same structure if we want consistency.
        # Or we can make TupleGetItemOp smart?
        # No, consistency is better.
        # So we should return a list: [final_index, *final_carry, *stacked_ys, *xs]
        
        # Final index
        final_index = self.length if not self.reverse else -1 # Approximate
        
        # We need to wrap these in something that to_numpy() can return?
        # Tensor.impl_ takes a numpy array or a list?
        # If we pass a list to impl_, it might try to convert it to a numpy array.
        # We want it to STORE the list.
        # Tensor implementation usually stores a numpy array or a device buffer.
        # Storing a list might break things if Tensor expects an array.
        # But `scan_result` is a dummy tensor.
        # We can hack it: `output.impl = result_list` (bypassing `impl_` checks if any).
        # Or `output.impl_` handles it?
        # Let's check Tensor.impl_.
        # Assuming we can store arbitrary python object if we are careful.
        
        # Construct the result list
        # Note: maxpr returns [index, *carry, *ys, *xs]
        # We should match this.
        # xs are passed through.
        
        # Convert xs to numpy for consistency? Or keep as tensors?
        # maxpr returns TensorValues.
        # eagerxpr should return numpy arrays (or whatever the impl is).
        
        xs_impls = [x.to_numpy() for x in flat_xs]
        carry_impls = [c.to_numpy() for c in flat_final_carry]
        ys_impls = [y.to_numpy() for y in stacked_leaves] # stacked_leaves are already tensors? No, they are stacked arrays?
        # In eagerxpr above: stacked_leaves = [stack(l).to_numpy() ...]
        # Wait, `stack` returns Tensor.
        # `stacked_leaves` in original code was: `[stack(l, axis=0) for l in transposed_leaves]`
        # So they are Tensors.
        ys_impls = [y.to_numpy() for y in stacked_leaves]
        
        result_list = [final_index] + carry_impls + ys_impls + xs_impls
        
        # Store in output
        # We use a private attribute or force impl?
        # output._impl = result_list
        # But `to_numpy()` calls `self.impl`.
        # If `impl` is a list, `to_numpy()` returns a list.
        output._impl = result_list

    def maxpr(self, args: list[TensorValue], output: Tensor) -> None:
        from max.graph import ops, TensorValue, DeviceRef
        from ..ops.creation import zeros, tensor
        from ..core.tensor import Tensor
        from ..transforms.utils import tree_flatten, tree_unflatten, tree_map
        
        # 1. Setup Inputs
        num_init = self.num_init
        flat_init_vals = args[:num_init]
        flat_xs_vals = args[num_init:]
        
        device = output.logical_device
        # Determine batch_dims from output (as a proxy for input batch_dims)
        batch_dims = output.batch_dims
        n_batch = len(batch_dims)
        
        # Helper to wrap value in Tensor
        def wrap(val: TensorValue, shape, dtype, b_dims=()) -> Tensor:
            t = Tensor(shape, dtype, device, materialize=False, batch_dims=b_dims)
            t.tensor_value = val
            t.traced = True 
            t.stage_realization = True # Prevent eager execution in body_fn
            return t

        # Reconstruct xs Tensors (captured by body_fn)
        flat_xs_vals_wrapped = []
        for val in flat_xs_vals:
             full_shape = tuple(int(d) for d in val.type.shape)
             if len(full_shape) >= n_batch:
                 shape = full_shape[n_batch:]
             else:
                 shape = full_shape
             
             t = wrap(val, shape, val.type.dtype, b_dims=batch_dims)
             flat_xs_vals_wrapped.append(t)

        # 2. Initialize Accumulators (ys)
        flat_ys_accum = []
        for shape, dtype in zip(self.flat_y_shapes, self.flat_y_dtypes):
            full_shape = (self.length, *shape)
            zero_scalar = ops.constant(0, dtype=dtype, device=DeviceRef.from_device(output.logical_device))
            target_shape = batch_dims + full_shape
            y_acc_val = ops.broadcast_to(zero_scalar, target_shape)
            flat_ys_accum.append(y_acc_val)
            
        # 3. Loop State
        start_idx = 0 if not self.reverse else self.length - 1
        index_val = ops.constant(start_idx, dtype=DType.int64, device=DeviceRef.from_device(device))
        
        # Broadcast index to include batch dimensions (if any)
        if n_batch > 0:
            index_val = ops.broadcast_to(index_val, batch_dims)
        
        # Broadcast init values to include batch dimensions
        # In vmap(scan), init might be created fresh (no batch dims) while xs is batched
        # We need all values in while_loop to have consistent batch dimensions
        flat_init_vals_broadcasted = []
        for i, val in enumerate(flat_init_vals):
            current_shape = tuple(int(d) for d in val.type.shape)
            # Target shape is batch_dims + current_shape
            target_shape = batch_dims + current_shape
            if current_shape != target_shape:
                # Need to broadcast
                val_broadcasted = ops.broadcast_to(val, target_shape)
                flat_init_vals_broadcasted.append(val_broadcasted)
            else:
                flat_init_vals_broadcasted.append(val)
        
        init_values = [index_val] + flat_init_vals_broadcasted + flat_ys_accum + flat_xs_vals
        
        # 4. Predicate Function
        def cond_fn(idx_val, *rest):
            # If idx_val is batched, extract first element (all batch elements have same index)
            idx_shape = idx_val.type.shape
            if len(idx_shape) > 0:
                # Batched index - take first element to get scalar
                idx_scalar = ops.gather(idx_val, ops.constant(0, dtype=DType.int64, device=DeviceRef.from_device(output.logical_device)), axis=0)
            else:
                idx_scalar = idx_val
            
            if not self.reverse:
                pred = idx_scalar < self.length
            else:
                pred = idx_scalar >= 0
            return pred

        # 5. Body Function
        def body_fn(idx_val, *state_vals):
            # Unpack state
            num_carry = num_init
            num_ys = len(flat_ys_accum)
            num_xs = len(flat_xs_vals)
            
            current_flat_carry_vals = state_vals[:num_carry]
            current_flat_ys_accum_vals = state_vals[num_carry : num_carry + num_ys]
            current_flat_xs_vals = state_vals[num_carry + num_ys : num_carry + num_ys + num_xs]
            
            # Wrap inputs for f
            flat_carry_tensors = []
            for val in current_flat_carry_vals:
                full_shape = tuple(int(d) for d in val.type.shape)
                if len(full_shape) >= n_batch:
                    shape = full_shape[n_batch:]
                else:
                    shape = full_shape
                t = wrap(val, shape, val.type.dtype, b_dims=batch_dims)
                flat_carry_tensors.append(t)
            
            carry = tree_unflatten(self.carry_structure, flat_carry_tensors)
            
            # Slice xs
            flat_x_slice_tensors = []
            
            # Extract scalar index for gathering (all batch elements use same index)
            idx_shape = idx_val.type.shape
            if len(idx_shape) > 0:
                # Batched index - extract first element to get scalar
                idx_scalar = ops.gather(idx_val, ops.constant(0, dtype=DType.int64, device=DeviceRef.from_device(output.logical_device)), axis=0)
            else:
                idx_scalar = idx_val
            
            for x_val in current_flat_xs_vals:
                x_slice_val = ops.gather(x_val, idx_scalar, axis=n_batch)
                
                full_shape = tuple(int(d) for d in x_slice_val.type.shape)
                if len(full_shape) >= n_batch:
                    shape = full_shape[n_batch:]
                else:
                    shape = full_shape
                t = wrap(x_slice_val, shape, x_slice_val.type.dtype, b_dims=batch_dims)
                flat_x_slice_tensors.append(t)
                
            x_slice = tree_unflatten(self.xs_structure, flat_x_slice_tensors)
            
            # Execute f (builds Nabla graph)
            new_carry, new_y = self.f(carry, x_slice)
            
            # Flatten outputs
            flat_new_carry, _ = tree_flatten(new_carry)
            flat_new_y, _ = tree_flatten(new_y)
            
            # Lowering: Recursively ensure all nodes have tensor_value
            def lower_node(node: Tensor):
                if node.tensor_value is not None:
                    return node.tensor_value
                
                # Handle constants (nodes with impl but no maxpr/tensor_value)
                if node.impl is not None:
                    # Create constant op
                    # _impl can be a MAXTensor or numpy array
                    if hasattr(node._impl, 'to_numpy'):
                        np_data = node._impl.to_numpy()
                    else:
                        np_data = node._impl
                    
                    # WORKAROUND: ops.constant has a bug where it interprets 
                    # np.array(1, dtype=float32) as shape [1] instead of []
                    # So for scalars, we convert to Python scalar using .item()
                    import numpy as np
                    if isinstance(np_data, np.ndarray) and np_data.shape == ():
                        np_data = np_data.item()
                    
                    node.tensor_value = ops.constant(
                        np_data, 
                        dtype=node.dtype, 
                        device=DeviceRef.from_device(node.logical_device)
                    )
                    return node.tensor_value

                # Handle operations
                if node.maxpr is None:
                    # This should not happen if it's not a constant and has no maxpr
                    raise ValueError(f"Node {node.name} ({type(node)}) has no maxpr and no impl!")
                
                # Recursively lower arguments
                arg_values = []
                for arg in node.args:
                    arg_values.append(lower_node(arg))
                
                # Execute maxpr
                node.maxpr(arg_values, node)
                
                if node.tensor_value is None:
                     raise ValueError(f"Node {node.name} maxpr did not set tensor_value")
                     
                return node.tensor_value

            # Lower all outputs
            flat_new_carry_vals = [lower_node(t) for t in flat_new_carry]
            flat_new_y_vals = [lower_node(t) for t in flat_new_y]
            
            # Update ys_accum
            new_ys_accum_vals = []
            for i, (acc_val, y_val) in enumerate(zip(current_flat_ys_accum_vals, flat_new_y_vals)):
                if acc_val.dtype != y_val.dtype:
                    y_val = ops.cast(y_val, acc_val.dtype)
                    
                # Use scatter_nd for robust update
                # indices needs to be (1,) for 1D scatter
                indices_nd = ops.unsqueeze(idx_val, 0) # (1,)
                indices_nd = ops.unsqueeze(indices_nd, 0) # (1, 1) - batch dim? No, wait.
                
                # scatter_nd(tensor, updates, indices)
                # tensor: acc_val (batch..., length, ...)
                # updates: y_val (batch..., ...) - needs to be expanded to match indices shape?
                # indices: (1, 1) -> points to [idx_val] in dim 0 (logical) which is dim n_batch (physical)
                
                # Actually, let's look at ScatterOp.maxpr again.
                # It uses ops.scatter for simple axis scatter if possible.
                # But here we are scattering a single slice into a larger tensor.
                # acc_val shape: (batch..., length, y_shape...)
                # y_val shape: (batch..., y_shape...)
                # idx_val: scalar
                
                # We want to update acc_val[:, ..., idx_val, ...] = y_val
                # axis is n_batch.
                
                # If ops.scatter failed, maybe we need to unsqueeze y_val to match rank?
                # The error was: 'rmo.mo.scatter' op operand #1 must be unranked or non-scalar tensor, but got '!mo.tensor<[], si32>'
                # Operand #1 is updates. y_val is scalar in the cumsum test case (int32).
                # So we need to unsqueeze y_val to make it a 1D tensor (rank 1) for scatter?
                # Or maybe ops.scatter requires updates to have same rank as input?
                
                # Let's try unsqueezing y_val to have a dimension of size 1 at the scatter axis.
                y_val_expanded = ops.unsqueeze(y_val, n_batch)
                
                # And we might need to unsqueeze idx_val too?
                # ops.scatter(input, updates, indices, axis)
                # indices should be 1D tensor of indices?
                idx_val_expanded = ops.unsqueeze(idx_val, 0)
                
                new_acc = ops.scatter(acc_val, y_val_expanded, idx_val_expanded, axis=n_batch)
                new_ys_accum_vals.append(new_acc)
            
            # Update index
            step = 1 if not self.reverse else -1
            new_idx_val = idx_val + step
            
            # Return updated state
            return [new_idx_val] + flat_new_carry_vals + new_ys_accum_vals + list(current_flat_xs_vals)

        # 6. Execute Loop
        results = ops.while_loop(init_values, cond_fn, body_fn)
        
        # Set output value
        output.tensor_value = results
        self._maxpr_done = True

    def vjp_rule(self, primals, cotangent, output):
        # TODO
        pass

    def jvp_rule(self, primals, tangents, output):
        # TODO
        pass


    @classmethod
    def run(
        cls,
        f: Callable,
        init: Any,
        xs: Any = None,
        length: int | None = None,
        reverse: bool = False,
        unroll: int = 1,
    ) -> tuple[Any, Any]:
        """
        Executes the scan operation.

        This method handles:
        1. Flattening pytree inputs (init, xs).
        2. Inferring output shapes by running `f` on the first slice of inputs.
        3. Creating the ScanOp instance.
        4. Executing the operation (eagerly or building a graph).
        5. Unflattening the results back to pytrees.
        """
        from ..transforms.utils import tree_flatten, tree_unflatten, tree_map
        from .indexing import gather
        from ..ops.creation import tensor

        # 1. Handle optional xs and determine length
        if xs is None:
            if length is None:
                raise ValueError("length must be specified if xs is None")
            xs = [None] * length
        
        # Flatten inputs to work with Tensors directly
        flat_init, init_structure = tree_flatten(init)
        flat_xs, xs_structure = tree_flatten(xs)

        if length is None:
            if not flat_xs:
                 raise ValueError("length must be specified if xs is empty")
            length = flat_xs[0].shape[0]

        # Validate that all xs have the correct leading dimension
        for x in flat_xs:
            if x.shape[0] != length:
                 raise ValueError(f"Leading axis size of xs must be equal to length {length}, got {x.shape[0]}")

        # 2. Infer output structure and shapes
        # We run the function `f` once on the first slice of `xs` to see what it returns.
        # This allows us to determine the structure (pytree) and shapes of the carry and outputs.
        # This is a standard technique in frameworks like JAX to avoid requiring the user to 
        # explicitly specify output signatures.
        def _get_first_slice(x):
            # Slice the first element (index 0) from the leading axis
            idx = tensor(0, dtype=DType.int64, device=x.logical_device)
            return gather(x, idx, axis=0)

        first_xs_slice = tree_map(_get_first_slice, xs)
        
        # Execute f with the first slice
        # TODO: In a compiled setting, this should ideally be done abstractly (without real data)
        # to avoid eager execution overhead, but for now we run it.
        # This "tracing" step is crucial for:
        # - Validating that f works with the provided init/xs structure.
        # - capturing the output structure (e.g. if f returns a dict or tuple).
        carry_out_example, y_out_example = f(init, first_xs_slice)
        
        flat_carry_out_example, carry_structure = tree_flatten(carry_out_example)
        flat_y_out_example, y_structure = tree_flatten(y_out_example)
        
        # Validate that the carry structure hasn't changed.
        # The carry must maintain the same structure (and ideally shape/dtype) throughout the scan.
        if len(flat_carry_out_example) != len(flat_init):
             raise ValueError("Carry output structure must match init structure")

        # 3. Create the ScanOp
        # We pass all the structural info needed to reconstruct outputs later.
        # The ScanOp itself will handle the actual execution (looping).
        op = cls(
            f=f,
            length=length,
            reverse=reverse,
            unroll=unroll,
            init_structure=init_structure,
            xs_structure=xs_structure,
            carry_structure=carry_structure,
            y_structure=y_structure,
            flat_y_shapes=[y.shape for y in flat_y_out_example],
            flat_y_dtypes=[y.dtype for y in flat_y_out_example],
            num_init=len(flat_init)
        )
        
        # 4. Execute the operation
        # 4. Execute the operation
        # forward() returns a single container Tensor
        scan_result = op.forward(*flat_init, *flat_xs)
        
        # 5. Unpack results using TupleGetItemOp
        # We need to construct the expected outputs.
        # The result list structure is: [index, *carry, *ys, *xs]
        # We want *carry (indices 1 to 1+num_carry) and *ys (indices 1+num_carry to 1+num_carry+num_ys)
        
        num_carry = len(flat_init)
        num_ys = len(flat_y_out_example)
        
        # Determine shapes/dtypes for outputs
        carry_shapes = [x.shape for x in flat_init]
        carry_dtypes = [x.dtype for x in flat_init]
        
        y_shapes = [(length, *y.shape) for y in flat_y_out_example]
        y_dtypes = [y.dtype for y in flat_y_out_example]
        
        output_shapes = carry_shapes + y_shapes
        output_dtypes = carry_dtypes + y_dtypes
        
        # Batch dims are broadcasted from inputs
        from ..utils.shape_utils import get_broadcasted_shape
        output_batch_dims = ()
        for arg in flat_init + flat_xs:
            output_batch_dims = get_broadcasted_shape(output_batch_dims, arg.batch_dims)
            
        outputs = []
        for i, (shape, dtype) in enumerate(zip(output_shapes, output_dtypes)):
            # Index in the container: 1 (skip index) + i
            idx = 1 + i
            getter = TupleGetItemOp(idx, shape, dtype, output_batch_dims)
            out = getter.forward(scan_result)
            outputs.append(out)
            
        # 6. Unflatten results
        flat_final_carry = outputs[:num_carry]
        flat_final_ys = outputs[num_carry:]
        
        final_carry = tree_unflatten(carry_structure, flat_final_carry)
        final_ys = tree_unflatten(y_structure, flat_final_ys)
        
        return final_carry, final_ys


def scan(
    f: Callable,
    init: Any,
    xs: Any = None,
    length: int | None = None,
    reverse: bool = False,
    unroll: int = 1,
) -> tuple[Any, Any]:
    """
    Scan a function over leading array axes while carrying along state.

    Args:
        f: A Python function to be scanned. It should accept `(carry, x)` and return `(new_carry, y)`.
        init: The initial carry value (can be a pytree).
        xs: The inputs to scan over (can be a pytree). The leading axis of leaves is scanned.
        length: The length of the scan. Required if `xs` is None.
        reverse: If True, scan in reverse order (not yet implemented).
        unroll: Loop unrolling factor (not yet implemented).

    Returns:
        A tuple `(final_carry, stacked_ys)` where `stacked_ys` matches the structure of `y` returned by `f`,
        but with an extra leading dimension of size `length`.
    """
    return ScanOp.run(f, init, xs, length, reverse, unroll)
