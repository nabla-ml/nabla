# Failing Unit Tests

| Test File | Failure Mode | Root Cause (Estimated) |
|-----------|--------------|------------------------|
| `tests/unit/test_control_flow.py` | `IndexError: tuple index out of range` in `TestCondOp.test_cond_basic_true` | `max.driver.Buffer.from_dlpack` fails on scalar boolean arrays (`()` shape) during `nb.constant(True, dtype=nb.DType.bool)`. |
| `tests/unit/test_hessian_physical_ops.py` | `ValueError: Failed to run the MOToMGP pass manager: error: input shape not broadcastable to result shape` | Hessian calculation with `broadcast_to_physical` using `fwd_fwd` mode fails MAX compilation due to shape mismatch in generated MLIR. |
