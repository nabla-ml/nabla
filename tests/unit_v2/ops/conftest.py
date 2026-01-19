# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import pytest
import numpy as np
import nabla
from nabla.core import Tensor

@pytest.fixture
def op_verifier():
    class OpVerifier:
        def verify(self, op_fn, numpy_fn, input_shapes, input_data_fn=np.random.standard_normal, test_batch_dims=True, **kwargs):
            inputs_np = [input_data_fn(shape).astype(np.float32) for shape in input_shapes]
            inputs_nabla = [Tensor.from_dlpack(x) for x in inputs_np]
            out_nabla = op_fn(*inputs_nabla, **kwargs)
            out_np = numpy_fn(*inputs_np, **kwargs)
            
            op_name = getattr(op_fn, 'name', getattr(op_fn, '__name__', str(op_fn)))
            np.testing.assert_allclose(out_nabla.to_numpy(), out_np, rtol=1e-5, atol=1e-5, err_msg=f"Logical mismatch for {op_name}")
            
            if test_batch_dims:
                vmapped_op = nabla.vmap(lambda *args: op_fn(*args, **kwargs))
                inputs_np_batched = [np.stack([x, x + 0.1]) for x in inputs_np]
                inputs_batched_nabla = [Tensor.from_dlpack(x) for x in inputs_np_batched]
                out_batched_nabla = vmapped_op(*inputs_batched_nabla)
                out_batched_np = np.stack([
                    numpy_fn(inputs_np_batched[0][0], *[inp[0] for inp in inputs_np_batched[1:]], **kwargs),
                    numpy_fn(inputs_np_batched[0][1], *[inp[1] for inp in inputs_np_batched[1:]], **kwargs)
                ])
                np.testing.assert_allclose(out_batched_nabla.to_numpy(), out_batched_np, rtol=1e-5, atol=1e-5, err_msg=f"Batch (vmap) mismatch for {op_name}")

            try:
                inputs_clean = [Tensor.from_dlpack(input_data_fn(shape).astype(np.float32)) for shape in input_shapes]
                res_lazy = op_fn(*inputs_clean, **kwargs)
                out_tensors = res_lazy if isinstance(res_lazy, (tuple, list)) else [res_lazy]
                primary_out = out_tensors[0]
                if hasattr(primary_out, '_impl') and getattr(primary_out._impl, 'op', None) is not None:
                    op_instance = primary_out._impl.op
                    in_shapes = [tuple(int(d) for d in t.shape) for t in inputs_clean]
                    out_shapes = [tuple(int(d) for d in t.shape) for t in out_tensors]
                    rule = op_instance.sharding_rule(in_shapes, out_shapes, **kwargs)
                    assert rule is not None
                    if hasattr(rule, 'input_mappings'):
                        assert len(rule.input_mappings) == len(in_shapes)
                        assert len(rule.output_mappings) == len(out_shapes)
            except Exception as e:
                raise RuntimeError(f"Sharding rule verification failed for {op_name}: {e}") from e

    return OpVerifier()
