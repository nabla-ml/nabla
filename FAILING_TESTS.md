# Failing Unit Tests

| Test File | Failure Mode | Root Cause (Estimated) |
|-----------|--------------|------------------------|
| `tests/unit/test_control_flow.py` | `IndexError: tuple index out of range` in `TestCondOp.test_cond_basic_true` | `max.driver.Buffer.from_dlpack` fails on scalar boolean arrays (`()` shape) during `nb.constant(True, dtype=nb.DType.bool)`. |
