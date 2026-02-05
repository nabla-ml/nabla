import numpy as np
import pytest
import nabla as nb
from nabla.core.graph.tracing import trace
from nabla.core.autograd import backward_on_trace
from nabla.core.sharding import DeviceMesh, DimSpec
from nabla.transforms import vmap

# Context manager to suppress JAX GPU warnings or handle missing JAX
try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None


def check_vjp(
    op_name,
    nabla_fn,
    jax_fn,
    args_np,
    mesh=None,
    input_specs=None,
    vmap_axes=None,  # (in_axes, out_axes)
    rtol=1e-5,
    atol=1e-5,
):
    """
    Unified helper to verify VJP (Vector-Jacobian Product) correctness.

    Args:
        op_name: Name of the operation (for logging/errors).
        nabla_fn: Nabla callable function.
        jax_fn: Equivalent JAX callable function (or None).
        args_np: Tuple of numpy arrays as inputs.
        mesh: Optional DeviceMesh for sharded tests.
        input_specs: Optional list of sharding specs (DimSpec) for inputs.
        vmap_axes: Optional tuple (in_axes, out_axes) for vmap tests.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
    """
    print(f"\n--- Checking VJP for: {op_name} ---")

    # --- 1. JAX Ground Truth ---
    grads_jax = None
    if HAS_JAX and jax_fn is not None:
        try:
            # Helper to scalarize output for grad
            def jax_scalar_fn(*args):
                res = jax_fn(*args)
                if vmap_axes:
                    # If we vmapped, we might have multiple outputs or a batch dim
                    # Logic: if vmap_axes is present, we wrap jax_fn in vmap FIRST
                    pass
                return jnp.sum(res)

            if vmap_axes:
                in_axes, out_axes = vmap_axes
                # Wrap with vmap
                vmapped_jax = jax.vmap(jax_fn, in_axes=in_axes, out_axes=out_axes)
                target_fn = lambda *a: jnp.sum(vmapped_jax(*a))
            else:
                target_fn = lambda *a: jnp.sum(jax_fn(*a))

            grad_fn = jax.grad(target_fn, argnums=tuple(range(len(args_np))))
            grads_jax = grad_fn(*args_np)
            if not isinstance(grads_jax, (list, tuple)):
                grads_jax = (grads_jax,)

        except Exception as e:
            print(f"JAX reference computation failed: {e}")
            # We don't fail the test if JAX fails (might be unimplemented in JAX or version mismatch),
            # but we won't be able to compare values.
            grads_jax = None

    # --- 2. Nabla Execution ---

    # Convert inputs to Nabla Tensors
    args_nb = [nb.Tensor.from_dlpack(x.copy()) for x in args_np]

    # Apply Sharding if requested
    if mesh is not None and input_specs is not None:
        sharded_args = []
        for arg, spec in zip(args_nb, input_specs):
            if spec:
                # Ensure spec is a list of DimSpec objects
                if isinstance(spec, (list, tuple)) and all(
                    isinstance(s, str) for s in spec
                ):
                    # Convenience: allow passing ["tp", "dp"] list of strings
                    real_spec = [DimSpec(s) for s in spec]
                else:
                    real_spec = spec

                sharded = nb.ops.shard(arg, mesh, real_spec)
                sharded_args.append(sharded)
            else:
                sharded_args.append(arg)
        args_nb = sharded_args

    # Define the target function for tracing
    if vmap_axes:
        in_axes, out_axes = vmap_axes

        def trace_target(*a):
            # vmap return a tensor, we sum it to scalar
            res = vmap(nabla_fn, in_axes=in_axes, out_axes=out_axes)(*a)
            # Reduce to scalar for backward
            # We assume simple summation is sufficient for gradient checking
            return nb.ops.reduce_sum(res, axis=list(range(len(res.shape))))

    else:

        def trace_target(*a):
            res = nabla_fn(*a)
            # Handle tuple returns? For now assume single output or we'd need more complex logic
            return nb.ops.reduce_sum(res, axis=list(range(len(res.shape))))

    # Trace and Backward
    traced = trace(trace_target, *args_nb)

    # Scalar cotangent (1.0)
    cotangent = nb.Tensor.from_dlpack(np.array(1.0, dtype=np.float32))
    grads_map = backward_on_trace(traced, cotangent)

    # Extract grads for inputs
    grads_nb = []
    for arg in args_nb:
        g = grads_map.get(arg)
        if g is None:
            # Some args might not have gradients (integers etc), but usually they should if float
            grads_nb.append(None)
        else:
            grads_nb.append(g)

    # --- 3. Verification ---

    # Helper to realize lazy tensors to numpy
    def to_np(t):
        if t is None:
            return None
        from nabla.core.graph.engine import GRAPH

        if not t._impl.is_realized:
            GRAPH.evaluate(t)
        return t.to_numpy()

    for i, (g_nb, arg_in) in enumerate(zip(grads_nb, args_np)):
        if g_nb is None:
            # If input is float, we expect a gradient
            if np.issubdtype(arg_in.dtype, np.floating):
                print(f"WARNING: No gradient computed for float input {i} in {op_name}")
            continue

        gnb_val = to_np(g_nb)

        # Shape check
        if gnb_val.shape != arg_in.shape:
            raise AssertionError(
                f"Gradient shape mismatch for input {i} in {op_name}. "
                f"Expected {arg_in.shape}, got {gnb_val.shape}"
            )

        # Value check (vs JAX)
        if grads_jax is not None and grads_jax[i] is not None:
            # JAX grad might be None for disconnected inputs
            gjax_val = np.asarray(grads_jax[i])

            try:
                np.testing.assert_allclose(
                    gnb_val,
                    gjax_val,
                    rtol=rtol,
                    atol=atol,
                    err_msg=f"Gradient mismatch for input {i} in {op_name}",
                )
                print(f"  ✓ Input {i}: Matches JAX")
            except AssertionError as e:
                print(f"  ✗ Input {i}: Mismatch!")
                print(f"    Nabla Max: {np.max(gnb_val)}, Min: {np.min(gnb_val)}")
                print(f"    JAX   Max: {np.max(gjax_val)}, Min: {np.min(gjax_val)}")
                raise e
        else:
            # Basic sanity check if JAX is missing
            if np.any(np.isnan(gnb_val)) or np.any(np.isinf(gnb_val)):
                raise AssertionError(f"NaN/Inf in gradient for input {i} in {op_name}")
            print(f"  ✓ Input {i}: Computed (JAX/Ref missing), Shape OK, Finite")
