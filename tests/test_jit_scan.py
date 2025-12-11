
import unittest
import numpy as np
import nabla as nb
from nabla.ops.scan import scan
from nabla import jit

class TestJitScan(unittest.TestCase):
    def test_jit_scan_cumsum(self):
        # Simple cumsum: f(c, x) -> (c+x, c+x)
        def f(c, x):
            out = c + x
            return out, out
            
        @jit
        def run_scan(init, xs):
            return scan(f, init, xs)
            
        init = nb.tensor(0, dtype=nb.DType.int32)
        xs = nb.tensor([1, 2, 3, 4], dtype=nb.DType.int32)
        
        final_carry, ys = run_scan(init, xs)
        
        expected_ys = np.array([1, 3, 6, 10], dtype=np.int32)
        expected_carry = 10
        
        np.testing.assert_array_equal(ys.to_numpy(), expected_ys)
        np.testing.assert_array_equal(final_carry.to_numpy(), expected_carry)

    def test_jit_scan_pytree(self):
        # Test with pytree carry and inputs
        def f(carry, x):
            s, c = carry
            val = x['val']
            new_s = s + val
            new_c = c + 1
            return (new_s, new_c), new_s
            
        @jit
        def run_scan(init, xs):
            return scan(f, init, xs)
            
        init = (nb.tensor(0), nb.tensor(0))
        xs = {'val': nb.tensor([1, 2, 3])}
        
        final_carry, ys = run_scan(init, xs)
        
        expected_ys = np.array([1, 3, 6])
        expected_carry_s = 6
        expected_carry_c = 3
        
        np.testing.assert_array_equal(ys.to_numpy(), expected_ys)
        np.testing.assert_array_equal(final_carry[0].to_numpy(), expected_carry_s)
        np.testing.assert_array_equal(final_carry[1].to_numpy(), expected_carry_c)

    def test_jit_scan_reverse(self):
        # Test reverse scan
        def f(c, x):
            return c + x, c + x
            
        @jit
        def run_scan(init, xs):
            return scan(f, init, xs, reverse=True)
            
        init = nb.tensor(0, dtype=nb.DType.int32)
        xs = nb.tensor([1, 2, 3, 4], dtype=nb.DType.int32)
        
        # Reverse scan: processes 4, 3, 2, 1
        # c=0, x=4 -> c=4, y=4
        # c=4, x=3 -> c=7, y=7
        # c=7, x=2 -> c=9, y=9
        # c=9, x=1 -> c=10, y=10
        # Outputs reversed: [10, 9, 7, 4]
        
        final_carry, ys = run_scan(init, xs)
        
        expected_ys = np.array([10, 9, 7, 4], dtype=np.int32)
        expected_carry = 10
        
        np.testing.assert_array_equal(ys.to_numpy(), expected_ys)
        np.testing.assert_array_equal(final_carry.to_numpy(), expected_carry)

    def test_jit_vmap_scan(self):
        # jit(vmap(scan))
        # This tests if scan can be vmapped over and then compiled
        
        def f(c, x):
            return c + x, c + x
            
        def run_scan(x_seq):
            init = nb.tensor(0.0)
            return scan(f, init, x_seq)[1] # return ys
            
        @jit
        def run_vmap_scan(xs):
            return nb.vmap(run_scan)(xs)
            
        # xs: (batch, time)
        batch_size = 3
        time_steps = 4
        xs = nb.ones((batch_size, time_steps))
        
        ys = run_vmap_scan(xs)
        
        # Expected: (batch, time)
        # Each batch is cumsum of ones -> [1, 2, 3, 4]
        expected_ys = np.tile(np.array([1, 2, 3, 4], dtype=np.float32), (batch_size, 1))
        
        np.testing.assert_allclose(ys.to_numpy(), expected_ys)

if __name__ == '__main__':
    unittest.main()
