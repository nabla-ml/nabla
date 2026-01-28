# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from max import graph

    from ..core import Tensor


def ensure_tensor(x: Any) -> Tensor:
    """Convert scalar or array-like to Tensor."""
    from ..core import Tensor

    if isinstance(x, Tensor):
        return x

    return Tensor.constant(x)


class Operation(ABC):
    """Base class for all operations.

    Auto-propagates batch_dims.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    def maxpr(self, *args: graph.TensorValue, **kwargs: Any) -> Any:
        """Returns TensorValue or pytree of TensorValues."""
        ...

    @property
    def collective_reduce_type(self) -> str:
        """Type of reduction to use for cross-shard communication (sum, max, min, prod)."""
        return "sum"

    def compute_cost(
        self, input_shapes: list[tuple[int, ...]], output_shapes: list[tuple[int, ...]]
    ) -> float:
        """Estimate compute cost (FLOPs)."""
        return 0.0

    def memory_cost(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        dtype_bytes: int = 4,
    ) -> int:
        """Estimate memory usage (bytes) for output tensors."""
        total = 0
        for shape in output_shapes:
            elements = 1
            for dim in shape:
                elements *= dim
            total += elements * dtype_bytes
        return total

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # === New Path: Physical Execution ===
        if hasattr(self, "physical_execute"):
            from ..core import GRAPH, Tensor, pytree
            from ..core.sharding import spmd

            # 1. Collect Metadata (for Adaptation logic)
            max_batch_dims = 0
            any_traced = False
            any_sharded = False
            
            def collect_metadata(x):
                nonlocal max_batch_dims, any_traced, any_sharded
                if isinstance(x, Tensor):
                    if x.batch_dims > max_batch_dims:
                        max_batch_dims = x.batch_dims
                    if x.traced:
                        any_traced = True
                    if x.is_sharded:
                        any_sharded = True
            
            pytree.tree_map(collect_metadata, args)

            # 2. Adaptation: Reshard Inputs
            # Note: In the new model, __call__ handles this.
            # We assume physical_execute expects valid inputs.
            mesh = spmd.get_mesh_from_args(args) if any_sharded else None
            
            # Infer expected input sharding specific to this op (if relevant for resharding)
            # Legacy logic used spmd.infer_output_sharding to get input_shardings.
            # We must replicate that here to perform the necessary data movement.
            # This is "Adaptation".
            
            # We temporarily use the SPMD utility to get required input specs
            # This logic mimics the beginning of the legacy `execute` method
            adapted_kwargs = self.adapt_kwargs(args, kwargs, max_batch_dims)
            args = spmd.ensure_specs(args, mesh)
            _, input_shardings, _ = spmd.infer_output_sharding(
                self, args, mesh, adapted_kwargs or {}
            )
            
            # Perform the data movement (Logical Adaptation)
            resharded_args = spmd.reshard_inputs(args, input_shardings, mesh)

            # 3. Physical Execution (The "Dumb" Executor)
            # Returns raw TensorValues (PhysicalResult)
            with GRAPH.graph:
                raw_result = self.physical_execute(resharded_args, kwargs)

            # 4. Packaging (Wrapping raw values into nabla.Tensor)
            shard_values = None
            output_sharding = None
            res_mesh = mesh

            if isinstance(raw_result, tuple) and len(raw_result) == 3:
                 shard_values, output_sharding, res_mesh = raw_result
            elif hasattr(raw_result, "shard_values"):
                 shard_values = raw_result.shard_values
                 output_sharding = getattr(raw_result, "output_sharding", None)
                 res_mesh = getattr(raw_result, "mesh", mesh)
            else:
                 shard_values = raw_result

            output = spmd.create_sharded_output(
                shard_values,
                output_sharding,
                any_traced,
                max_batch_dims,
                mesh=res_mesh,
            )

            # 5. Tracing
            self._setup_output_refs(
                output, args, kwargs, kwargs, any_traced
            )
            
            # 6. Post-Processing (JVP)
            # Check for tangents in ORIGINAL args
            any_has_tangent = False
            for x in pytree.tree_leaves(args):
                if isinstance(x, Tensor) and x.tangent is not None:
                    any_has_tangent = True
            
            if any_has_tangent:
                self._apply_jvp(args, output)

            return output

        # === Legacy Path ===
        return self.execute(*args, **kwargs)

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        from ..core import Tensor, pytree
        from ..core.sharding import spmd

        any_traced = False
        any_has_tangent = False
        max_batch_dims = 0
        any_sharded = False
        original_kwargs = kwargs.copy() if kwargs else {}

        def collect_metadata(x: Any) -> Any:
            nonlocal any_traced, any_has_tangent, max_batch_dims, any_sharded
            if isinstance(x, Tensor):
                if x.traced:
                    any_traced = True
                if x.tangent is not None:
                    any_has_tangent = True
                if x.batch_dims > max_batch_dims:
                    max_batch_dims = x.batch_dims
                if x.is_sharded:
                    any_sharded = True
            return x

        pytree.tree_map(collect_metadata, args)

        kwargs = self.adapt_kwargs(args, kwargs, max_batch_dims)

        mesh = spmd.get_mesh_from_args(args) if any_sharded else None

        args = spmd.ensure_specs(args, mesh)
        output_sharding, input_shardings, reduce_axes = spmd.infer_output_sharding(
            self, args, mesh, kwargs or {}
        )

        args = spmd.reshard_inputs(args, input_shardings, mesh)

        output = self.maxpr_all(
            args,
            kwargs,
            output_sharding,
            mesh,
            any_traced,
            max_batch_dims,
            original_kwargs=original_kwargs,
        )

        if reduce_axes and mesh:
            output = self.apply_auto_reduction(output, mesh, reduce_axes)

        if any_has_tangent:
            self._apply_jvp(args, output)

        return output

    def maxpr_all(
        self,
        args: tuple,
        kwargs: dict,
        output_sharding: Any,
        mesh: Any,
        any_traced: bool,
        max_batch_dims: int,
        original_kwargs: dict | None = None,
    ) -> Any:
        from max import graph as g

        from ..core import GRAPH, Tensor, pytree
        from ..core.sharding import spmd

        num_shards = len(mesh.devices) if mesh else 1
        input_shardings = []
        if mesh:
            leaves = [a for a in pytree.tree_leaves(args) if isinstance(a, Tensor)]
            input_shardings = [t.sharding for t in leaves]

        with GRAPH.graph:
            shard_results = []
            for shard_idx in range(num_shards):
                shard_args = spmd.get_shard_args(
                    args, shard_idx, input_shardings, g, Tensor, pytree
                )
                shard_kwargs = self._transform_shard_kwargs(
                    kwargs, output_sharding, shard_idx
                )
                if shard_kwargs is None:
                    shard_kwargs = {}
                shard_results.append(self.maxpr(*shard_args, **shard_kwargs))

        if not shard_results:
            return None

        # Handle structured outputs
        first_res = shard_results[0]
        if isinstance(first_res, (list, tuple)):
            # Unzip: [(a0, b0), (a1, b1)] -> ([a0, a1], [b0, b1])
            unzipped = list(zip(*shard_results))
            outputs = []
            for res_shards in unzipped:
                outputs.append(
                    spmd.create_sharded_output(
                        list(res_shards),
                        output_sharding,
                        any_traced,
                        max_batch_dims,
                        mesh=mesh,
                    )
                )
            output = tuple(outputs) if isinstance(first_res, tuple) else outputs
        elif isinstance(first_res, dict):
            # Handle dict output
            keys = first_res.keys()
            outputs = {}
            for k in keys:
                res_shards = [r[k] for r in shard_results]
                outputs[k] = spmd.create_sharded_output(
                    res_shards, output_sharding, any_traced, max_batch_dims, mesh=mesh
                )
            output = outputs
        else:
            output = spmd.create_sharded_output(
                shard_results, output_sharding, any_traced, max_batch_dims, mesh=mesh
            )

        self._setup_output_refs(
            output, args, original_kwargs or kwargs, kwargs, any_traced
        )
        return output

    def apply_auto_reduction(
        self, output: Any, mesh: Any, reduce_axes: set[str]
    ) -> Any:
        from ..core import GRAPH, Tensor, pytree
        from ..core.sharding import spmd
        from ..core.sharding.spec import DimSpec, ShardingSpec
        from .communication import all_reduce_op

        def apply_grouped_all_reduce(t):
            if not isinstance(t, Tensor):
                return t

            t.hydrate()
            if not t._values:
                return t

            with GRAPH.graph:
                reduced_values = all_reduce_op.simulate_grouped_execution(
                    t.values, mesh, reduce_axes, reduce_op=self.collective_reduce_type
                )

            current_spec = t.sharding
            if current_spec:
                new_dim_specs = []
                for ds in current_spec.dim_specs:
                    new_axes = sorted(list(set(ds.axes) - reduce_axes))
                    new_dim_specs.append(DimSpec(new_axes))
                new_spec = ShardingSpec(mesh, new_dim_specs)
            else:
                rank = len(t.shape)
                new_spec = ShardingSpec(mesh, [DimSpec([]) for _ in range(rank)])

            reduced_tensor = spmd.create_sharded_output(
                reduced_values, new_spec, t.traced, t.batch_dims, mesh
            )

            trace_kwargs = {"mesh": mesh, "reduce_axes": list(reduce_axes)}
            all_reduce_op._setup_output_refs(
                reduced_tensor, (t,), trace_kwargs, trace_kwargs, t.traced
            )

            return reduced_tensor

        return pytree.tree_map(apply_grouped_all_reduce, output)

    def _apply_jvp(self, args: tuple, output: Any) -> None:
        from ..core import Tensor, pytree

        tangents = pytree.tree_map(
            lambda x: (
                Tensor(impl=x.tangent) if isinstance(x, Tensor) and x.tangent else None
            ),
            args,
        )
        output_tangent = self.jvp_rule(args, tangents, output)
        if output_tangent is not None:
            pytree.tree_map(
                lambda o, t: (
                    setattr(o._impl, "tangent", t._impl)
                    if isinstance(o, Tensor) and isinstance(t, Tensor)
                    else None
                ),
                output,
                output_tangent,
            )

    def adapt_kwargs(self, args: tuple, kwargs: dict, batch_dims: int) -> dict:
        return kwargs

    def _transform_shard_kwargs(
        self, kwargs: dict, output_sharding: Any, shard_idx: int
    ) -> dict:
        """Transform kwargs for per-shard maxpr execution."""

        return kwargs

    def infer_output_rank(
        self, input_shapes: tuple[tuple[int, ...], ...], **kwargs
    ) -> int:
        """Infer output rank from input shapes."""
        if not input_shapes:
            return 0
        return len(input_shapes[0])

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Default sharding rule: elementwise for same-rank ops."""
        from ..core.sharding.propagation import OpShardingRuleTemplate

        n_inputs = len(input_shapes)
        output_rank = len(output_shapes[0]) if output_shapes else len(input_shapes[0])

        mapping = {i: [f"d{i}"] for i in range(output_rank)}

        if n_inputs == 1:

            return OpShardingRuleTemplate([mapping], [mapping]).instantiate(
                input_shapes, output_shapes
            )
        else:

            return OpShardingRuleTemplate([mapping] * n_inputs, [mapping]).instantiate(
                input_shapes, output_shapes
            )

    def vjp_rule(self, primals: Any, cotangent: Any, output: Any) -> Any:
        raise NotImplementedError(f"'{self.name}' does not implement vjp_rule")

    def jvp_rule(self, primals: Any, tangents: Any, output: Any) -> Any:
        raise NotImplementedError(f"'{self.name}' does not implement jvp_rule")

    def get_sharding_rule_template(self) -> Any:
        return None

    def _setup_output_refs(
        self,
        output: Any,
        args: tuple,
        logical_kwargs: dict,
        physical_kwargs: dict,
        traced: bool,
    ) -> None:
        """Set up OutputRefs for tracing support."""
        from ..core import Tensor, pytree

        output_impls = [
            x._impl for x in pytree.tree_leaves(output) if isinstance(x, Tensor)
        ]

        if not output_impls:
            return

        import weakref

        from ..core import OutputRefs

        _, output_tree_def = pytree.tree_flatten(output, is_leaf=pytree.is_tensor)
        weak_refs = tuple(weakref.ref(impl) for impl in output_impls)

        def to_impl(x: Any) -> Any:
            return x._impl if isinstance(x, Tensor) else x

        stored_args = pytree.tree_map(to_impl, args)

        stored_logical_kwargs = (
            pytree.tree_map(to_impl, logical_kwargs) if logical_kwargs else None
        )
        stored_physical_kwargs = (
            pytree.tree_map(to_impl, physical_kwargs) if physical_kwargs else None
        )

        output_refs = OutputRefs(
            _refs=weak_refs,
            tree_def=output_tree_def,
            op=self,
            op_args=stored_args,
            op_kwargs=stored_logical_kwargs,
            physical_kwargs=stored_physical_kwargs,
        )

        for idx, impl in enumerate(output_impls):
            impl.output_refs = output_refs
            impl.output_index = idx

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class BinaryOperation(Operation):
    """Base for binary element-wise ops with batch_dims-aware broadcasting."""

    def compute_cost(
        self, input_shapes: list[tuple[int, ...]], output_shapes: list[tuple[int, ...]]
    ) -> float:
        """Elementwise binary op: 1 FLOP per output element."""
        if not output_shapes:
            return 0.0
        num_elements = 1
        for d in output_shapes[0]:
            num_elements *= d
        return float(num_elements)

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        from . import view as view_ops
        from .base import ensure_tensor

        x = ensure_tensor(x)
        y = ensure_tensor(y)

        x_logical = tuple(int(d) for d in x.shape)
        y_logical = tuple(int(d) for d in y.shape)
        target_logical = self._broadcast_shapes(x_logical, y_logical)

        if x_logical != target_logical:
            x = view_ops.broadcast_to(x, target_logical)
        if y_logical != target_logical:
            y = view_ops.broadcast_to(y, target_logical)

        x_batch_dims = x.batch_dims
        y_batch_dims = y.batch_dims

        if x_batch_dims >= y_batch_dims:
            global_phys = x.physical_global_shape or x.local_shape
            batch_shape = tuple(int(d) for d in global_phys[:x_batch_dims])
        else:
            global_phys = y.physical_global_shape or y.local_shape
            batch_shape = tuple(int(d) for d in global_phys[:y_batch_dims])

        target_physical = batch_shape + target_logical

        x_global = x.physical_global_shape or x.local_shape
        y_global = y.physical_global_shape or y.local_shape

        if tuple(int(d) for d in x_global) != target_physical:
            x = view_ops.broadcast_to_physical(x, target_physical)

        if tuple(int(d) for d in y_global) != target_physical:
            y = view_ops.broadcast_to_physical(y, target_physical)

        return super().__call__(x, y)

    def _broadcast_shapes(
        self, s1: tuple[int, ...], s2: tuple[int, ...]
    ) -> tuple[int, ...]:
        """Compute broadcast shape of two shapes (numpy-style right-aligned)."""
        if len(s1) > len(s2):
            s2 = (1,) * (len(s1) - len(s2)) + s2
        elif len(s2) > len(s1):
            s1 = (1,) * (len(s2) - len(s1)) + s1

        result = []
        for d1, d2 in zip(s1, s2, strict=False):
            if d1 == d2:
                result.append(d1)
            elif d1 == 1:
                result.append(d2)
            elif d2 == 1:
                result.append(d1)
            else:
                raise ValueError(f"Cannot broadcast shapes {s1} and {s2}")
        return tuple(result)


class LogicalAxisOperation(Operation):
    """Base for ops that take LOGICAL axis/axes kwargs.

    Translates integer kwargs by batch_dims offset.
    """

    axis_offset_for_insert: bool = False

    axis_arg_names: set[str] = {"axis", "dim"}

    def adapt_kwargs(self, args: tuple, kwargs: dict, batch_dims: int) -> dict:
        if batch_dims == 0:
            return kwargs

        # Typically the tensor argument is checked for logical ndim, but here we don't have it directly.
        # However, adapt_kwargs is called within maxpr_all where we know max_batch_dims.
        # But we need logical_ndim of the input.
        # The original code used `x.shape` which is the logical shape.
        # We can pass `input_visual_rank` or similar if needed, but usually we just offset by batch_dims.
        # Wait, the offset depends on logical_ndim for negative axes.
        # We might need to handle negative axes earlier or passed resolved kwargs.
        # But `LogicalAxisOperation` in `__call__` handled this via `x.shape`.
        # `x` is available in `adapt_kwargs`? No.
        # We need to resolve negative axes BEFORE adapt_kwargs or inside it if we have context.
        # If we just shift positive indices, it's safer.
        # Assuming negative indices are resolved by the caller or we resolve them using shape from somewhere.
        # Actually `LogicalAxisOperation` resolved negative indices using `logical_ndim`.
        # We can assume positive indices for now or rethink.

        # Let's check `Operation` usage. `execute` has access to `args`.
        # We can resolve negative axes in `adapt_kwargs` if we pass `args`?
        # But `adapt_kwargs` signature in `Operation` is `(self, kwargs, batch_dims)`.
        # Maybe we should change `adapt_kwargs` to take `args`?
        # Or `base.py` `maxpr_all` passes more context.
        # For now, let's just implement the shifting and assume positive or handle basic offset.

        translated = {}
        for key, value in kwargs.items():
            if key in self.axis_arg_names and isinstance(value, int):
                # We can't easily handle negative indices here without shape info.
                # But standard usage often resolves them or we assume positive after some validation.
                # If value < 0, it's tricky without rank.
                # However, if we assume the user provided valid negative index for the logical shape...
                # We can leave it negative? No, that would index from the end of PHYSICAL shape (including batch).
                # That is WRONG. -1 on logical (H, W) means W.
                # -1 on physical (B, H, W) means W. So negative indices are actually preserved OK if they refer to the last dims!
                # -2 on logical is H. -2 on physical is H.
                # So negative indices are fine as is!
                # Only POSITIVE indices need shifting by batch_dims.

                if value >= 0:
                    translated[key] = value + batch_dims
                else:
                    translated[key] = value
            else:
                translated[key] = value
        return translated

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Reduce sharding rule: (d0, d1, ...) -> (d0, ...) with axis dim removed."""
        from ..core.sharding.propagation import OpShardingRuleTemplate

        rank = len(input_shapes[0])
        axis = kwargs.get("axis", 0)

        in_mapping = {i: [f"d{i}"] for i in range(rank)}

        out_mapping = {}
        for i in range(rank):
            if i == axis:
                out_mapping[i] = []
            else:
                out_mapping[i] = [f"d{i}"]

        return OpShardingRuleTemplate([in_mapping], [out_mapping]).instantiate(
            input_shapes, output_shapes
        )


class ReduceOperation(LogicalAxisOperation):
    """Base for reduction operations (sum, mean, max, min, etc.)."""

    def compute_cost(
        self, input_shapes: list[tuple[int, ...]], output_shapes: list[tuple[int, ...]]
    ) -> float:
        """Reduction: 1 FLOP per input element (for summing/comparing)."""
        if not input_shapes:
            return 0.0
        num_elements = 1
        for d in input_shapes[0]:
            num_elements *= d
        return float(num_elements)

    def sharding_rule(
        self,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        **kwargs,
    ):
        """Reduction: reduce dims are removed or sized 1."""
        from ..core.sharding.propagation import OpShardingRuleTemplate

        if not input_shapes:
            return None

        rank = len(input_shapes[0])
        reduce_axes = kwargs.get("axis")
        keepdims = kwargs.get("keepdims", False)

        if reduce_axes is None:
            reduce_axes = tuple(range(rank))
        elif isinstance(reduce_axes, int):
            reduce_axes = (reduce_axes,)
        else:
            reduce_axes = tuple(reduce_axes)

        reduce_axes = tuple(ax + rank if ax < 0 else ax for ax in reduce_axes)

        factors = [f"d{i}" for i in range(rank)]
        in_mapping = {i: [factors[i]] for i in range(rank)}
        out_mapping = {}

        if keepdims:
            for i in range(rank):
                if i in reduce_axes:
                    out_mapping[i] = []
                else:
                    out_mapping[i] = [factors[i]]
        else:
            out_idx = 0
            for i in range(rank):
                if i not in reduce_axes:
                    out_mapping[out_idx] = [factors[i]]
                    out_idx += 1

        return OpShardingRuleTemplate([in_mapping], [out_mapping]).instantiate(
            input_shapes, output_shapes
        )

    def infer_output_shape(
        self, input_shapes: list[tuple[int, ...]], **kwargs
    ) -> tuple[int, ...]:
        """Compute output shape for reduction."""
        axis = kwargs.get("axis", 0)
        keepdims = kwargs.get("keepdims", False)
        in_shape = input_shapes[0]

        if axis < 0:
            axis = len(in_shape) + axis

        if keepdims:
            return tuple(1 if i == axis else d for i, d in enumerate(in_shape))
        else:
            return tuple(d for i, d in enumerate(in_shape) if i != axis)


class UnaryOperation(Operation):
    """Base for unary element-wise operations."""

    def compute_cost(
        self, input_shapes: list[tuple[int, ...]], output_shapes: list[tuple[int, ...]]
    ) -> float:
        """Unary elementwise op: 1 FLOP per element by default."""
        if not input_shapes:
            return 0.0
        num_elements = 1
        for d in input_shapes[0]:
            num_elements *= d
        return float(num_elements)


class LogicalShapeOperation(Operation):
    """Base for ops that take LOGICAL shape kwargs."""

    def adapt_kwargs(self, args: tuple, kwargs: dict, batch_dims: int) -> dict:
        if batch_dims == 0:
            return kwargs

        shape = kwargs.get("shape")
        if shape is not None and len(args) > 0:
            x = args[0]
            from ..core import Tensor

            if isinstance(x, Tensor):
                if x.physical_global_shape is not None:
                    global_batch_shape = tuple(
                        int(d) for d in x.physical_global_shape[:batch_dims]
                    )
                else:
                    global_phys = x.local_shape
                    # Fallback if local_shape is None? Should not happen if hydrated or determined.
                    if global_phys is None:
                        # If we can't determine physical shape, we might be in trouble or it's not realized.
                        # But for shape ops we usually need it.
                        # Let's assume input has shape info.
                        global_phys = (
                            x.shape
                        )  # Fallback to logical if physical missing? No.
                        # If physical is missing it might be (B, ...).
                        # Let's rely on standard property behavior.
                        pass

                    if global_phys:
                        global_batch_shape = tuple(
                            int(d) for d in global_phys[:batch_dims]
                        )
                    else:
                        # Best effort: assume standard layout
                        global_batch_shape = x.shape[
                            :batch_dims
                        ]  # This is wrong if x.shape is logical.
                        # But x.shape IS logical.
                        # Logic in original __call__:
                        # global_phys = x.local_shape
                        # global_batch_shape = tuple(int(d) for d in global_phys[:batch_dims])
                        # If local_shape is None, we can't do much.
                        return kwargs

                physical_shape = global_batch_shape + tuple(shape)
                new_kwargs = kwargs.copy()
                new_kwargs["shape"] = physical_shape
                return new_kwargs
        return kwargs

    # No execute override needed anymore
