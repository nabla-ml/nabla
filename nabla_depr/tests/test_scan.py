
import unittest
import numpy as np
import nabla as nb
from nabla.ops.scan import scan

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

class TestScan(unittest.TestCase):
    def test_scan_cumsum(self):
        # Simple cumsum: f(c, x) -> (c+x, c+x)
        def f(c, x):
            out = c + x
            return out, out
            
        init = nb.tensor(0, dtype=nb.DType.int32)
        xs = nb.tensor([1, 2, 3, 4], dtype=nb.DType.int32)
        
        final_carry, ys = scan(f, init, xs)
        
        expected_ys = np.array([1, 3, 6, 10], dtype=np.int32)
        expected_carry = 10
        
        np.testing.assert_array_equal(ys.to_numpy(), expected_ys)
        np.testing.assert_array_equal(final_carry.to_numpy(), expected_carry)
        
        if HAS_JAX:
            def jax_f(c, x):
                out = c + x
                return out, out
            
            jax_init = jnp.array(0, dtype=jnp.int32)
            jax_xs = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
            
            jax_final_carry, jax_ys = jax.lax.scan(jax_f, jax_init, jax_xs)
            
            np.testing.assert_array_equal(ys.to_numpy(), np.array(jax_ys))
            np.testing.assert_array_equal(final_carry.to_numpy(), np.array(jax_final_carry))

    def test_scan_pytree(self):
        # Test with pytree carry and inputs
        def f(carry, x):
            s, c = carry
            val = x['val']
            new_s = s + val
            new_c = c + 1
            return (new_s, new_c), new_s
            
        init = (nb.tensor(0), nb.tensor(0))
        xs = {'val': nb.tensor([1, 2, 3])}
        
        final_carry, ys = scan(f, init, xs)
        
        expected_ys = np.array([1, 3, 6])
        expected_carry_s = 6
        expected_carry_c = 3
        
        np.testing.assert_array_equal(ys.to_numpy(), expected_ys)
        np.testing.assert_array_equal(final_carry[0].to_numpy(), expected_carry_s)
        np.testing.assert_array_equal(final_carry[1].to_numpy(), expected_carry_c)
        
        if HAS_JAX:
            def jax_f(carry, x):
                s, c = carry
                val = x['val']
                new_s = s + val
                new_c = c + 1
                return (new_s, new_c), new_s
            
            jax_init = (jnp.array(0), jnp.array(0))
            jax_xs = {'val': jnp.array([1, 2, 3])}
            
            jax_final_carry, jax_ys = jax.lax.scan(jax_f, jax_init, jax_xs)
            
            np.testing.assert_array_equal(ys.to_numpy(), np.array(jax_ys))
            np.testing.assert_array_equal(final_carry[0].to_numpy(), np.array(jax_final_carry[0]))
            np.testing.assert_array_equal(final_carry[1].to_numpy(), np.array(jax_final_carry[1]))

    def test_scan_no_xs(self):
        # Test scan with xs=None, using length
        def f(c, x):
            # x should be None
            return c + 1, c
            
        init = nb.tensor(0)
        length = 5
        
        final_carry, ys = scan(f, init, None, length=length)
        
        expected_ys = np.arange(5)
        expected_carry = 5
        
        np.testing.assert_array_equal(ys.to_numpy(), expected_ys)
        np.testing.assert_array_equal(final_carry.to_numpy(), expected_carry)

    def test_scan_reverse(self):
        # Test reverse scan
        def f(c, x):
            return c + x, c + x
            
        init = nb.tensor(0, dtype=nb.DType.int32)
        xs = nb.tensor([1, 2, 3, 4], dtype=nb.DType.int32)
        
        # Reverse scan: processes 4, 3, 2, 1
        # c=0, x=4 -> c=4, y=4
        # c=4, x=3 -> c=7, y=7
        # c=7, x=2 -> c=9, y=9
        # c=9, x=1 -> c=10, y=10
        
        final_carry, ys = scan(f, init, xs, reverse=True)
        
        expected_ys = np.array([10, 9, 7, 4], dtype=np.int32)
        expected_carry = 10
        
        np.testing.assert_array_equal(ys.to_numpy(), expected_ys)
        np.testing.assert_array_equal(final_carry.to_numpy(), expected_carry)
        
        if HAS_JAX:
            def jax_f(c, x):
                out = c + x
                return out, out
            
            jax_init = jnp.array(0, dtype=jnp.int32)
            jax_xs = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
            
            jax_final_carry, jax_ys = jax.lax.scan(jax_f, jax_init, jax_xs, reverse=True)
            
            np.testing.assert_array_equal(ys.to_numpy(), np.array(jax_ys))
            np.testing.assert_array_equal(final_carry.to_numpy(), np.array(jax_final_carry))

    def test_scan_nested_vmap(self):
        # scan(vmap(f))
        # This should work naturally as f is just a function
        
        def f(c, x):
            # c: (batch,), x: (batch,)
            return c + x, c * x
            
        # We want to scan over time, but process a batch at each step
        # init: (batch,)
        # xs: (time, batch)
        
        batch_size = 3
        time_steps = 4
        
        init = nb.ones((batch_size,))
        xs = nb.ones((time_steps, batch_size))
        
        def step_fn(c, x):
            # c: scalar, x: scalar (per batch element)
            return c + x, c * x
            
        vmapped_step = nb.vmap(step_fn)
        
        final_carry, ys = scan(vmapped_step, init, xs)
        
        expected_ys = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], dtype=np.float32)
        expected_carry = np.array([5, 5, 5], dtype=np.float32)
        
        np.testing.assert_allclose(ys.to_numpy(), expected_ys)
        np.testing.assert_allclose(final_carry.to_numpy(), expected_carry)

    def test_vmap_of_scan(self):
        # vmap(lambda x: scan(f, init, x))
        # This tests if scan can be vmapped over
        
        def f(c, x):
            return c + x, c + x
            
        def run_scan(x_seq):
            init = nb.tensor(0.0)
            return scan(f, init, x_seq)[1] # return ys
            
        # xs: (batch, time)
        batch_size = 3
        time_steps = 4
        xs = nb.ones((batch_size, time_steps))
        
        # vmap over batch dimension (0)
        # run_scan expects (time,)
        
        ys = nb.vmap(run_scan)(xs)
        
        # Expected: (batch, time)
        # Each batch is cumsum of ones -> [1, 2, 3, 4]
        expected_ys = np.tile(np.array([1, 2, 3, 4], dtype=np.float32), (batch_size, 1))
        
        np.testing.assert_allclose(ys.to_numpy(), expected_ys)

    def test_scan_complex_pytree(self):
        # Deeply nested pytree
        def f(c, x):
            # c: {'a': [v1, v2], 'b': v3}
            # x: (x1, {'x2': x2})
            v1, v2 = c['a']
            v3 = c['b']
            x1, x2_dict = x
            x2 = x2_dict['x2']
            
            new_v1 = v1 + x1
            new_v2 = v2 * x2
            new_v3 = v3 + 1
            
            new_c = {'a': [new_v1, new_v2], 'b': new_v3}
            y = (new_v1, new_v2)
            return new_c, y
            
        init = {'a': [nb.tensor(0), nb.tensor(1)], 'b': nb.tensor(0)}
        xs = (nb.tensor([1, 2]), {'x2': nb.tensor([2, 3])})
        
        final_carry, ys = scan(f, init, xs)
        
        # t=0: x1=1, x2=2. v1=0+1=1, v2=1*2=2, v3=0+1=1
        # t=1: x1=2, x2=3. v1=1+2=3, v2=2*3=6, v3=1+1=2
        
        expected_v1 = np.array([1, 3])
        expected_v2 = np.array([2, 6])
        expected_v3 = 2
        
        np.testing.assert_array_equal(ys[0].to_numpy(), expected_v1)
        np.testing.assert_array_equal(ys[1].to_numpy(), expected_v2)
        np.testing.assert_array_equal(final_carry['b'].to_numpy(), expected_v3)

    def test_scan_bool_carry(self):
        # Test boolean carry (e.g. any/all logic)
        def f(c, x):
            # c: bool, x: bool
            # new_c = c or x
            new_c = nb.maximum(c, x) # logical OR for 0/1
            return new_c, new_c
            
        init = nb.tensor(0, dtype=nb.DType.bool) # False
        xs = nb.tensor([0, 0, 1, 0], dtype=nb.DType.bool) # F, F, T, F
        
        final_carry, ys = scan(f, init, xs)
        
        expected_ys = np.array([0, 0, 1, 1], dtype=bool)
        expected_carry = True
        
    def test_rl_simulation(self):
        # Simulate a simple 1D point mass: p_new = p + v, v_new = v + a
        # State: (position, velocity)
        # Input: acceleration (action)
        # Output: (position, reward) where reward = -p^2 (keep close to 0)
        
        def step(state, action):
            p, v = state
            a = action
            
            v_new = v + a
            p_new = p + v_new
            
            reward = -(p_new * p_new)
            
            return (p_new, v_new), (p_new, reward)
            
        # Initial state: p=10, v=0
        init_state = (nb.tensor(10.0), nb.tensor(0.0))
        
        # Actions: apply -1 acceleration for 10 steps
        actions = nb.ones((10,)) * -1.0
        
        final_state, (positions, rewards) = scan(step, init_state, actions)
        
        # Manual verification
        p, v = 10.0, 0.0
        expected_ps = []
        expected_rs = []
        for _ in range(10):
            v = v - 1.0
            p = p + v
            expected_ps.append(p)
            expected_rs.append(-(p*p))
            
        expected_ps = np.array(expected_ps, dtype=np.float32)
        expected_rs = np.array(expected_rs, dtype=np.float32)
        
        np.testing.assert_allclose(positions.to_numpy(), expected_ps)
        np.testing.assert_allclose(rewards.to_numpy(), expected_rs)
        np.testing.assert_allclose(final_state[0].to_numpy(), expected_ps[-1])

    def test_batched_rl_simulation(self):
        # Same RL simulation but vmapped over a batch of agents
        # Each agent has different initial state and different actions
        
        def step(state, action):
            p, v = state
            a = action
            v_new = v + a
            p_new = p + v_new
            reward = -(p_new * p_new)
            return (p_new, v_new), (p_new, reward)

        def run_sim(init_p, init_v, actions):
            init_state = (init_p, init_v)
            return scan(step, init_state, actions)

        batch_size = 5
        seq_len = 10
        
        # Batch of initial states
        init_ps = nb.tensor(np.arange(batch_size, dtype=np.float32)) # 0, 1, 2, 3, 4
        init_vs = nb.zeros((batch_size,))
        
        # Batch of action sequences: (batch, time) -> (batch, 10)
        # Agent i applies acceleration -i
        actions = nb.tensor(np.linspace(0, -4, batch_size, dtype=np.float32)).reshape((batch_size, 1))
        actions = actions * nb.ones((1, seq_len)) # Broadcast to (batch, seq_len)
        
        # vmap run_sim over batch dimension (0)
        # run_sim expects (scalar_p, scalar_v, vector_actions)
        # We pass (vector_p, vector_v, matrix_actions)
        
        # vmap signature: (p, v, actions) -> ((final_p, final_v), (positions, rewards))
        # Input dims: (B,), (B,), (B, T)
        # Output dims: ((B,), (B,)), ((B, T), (B, T))
        
        (final_ps, final_vs), (all_positions, all_rewards) = nb.vmap(run_sim)(init_ps, init_vs, actions)
        
        # Verify shape
        self.assertEqual(final_ps.shape, (batch_size,))
        self.assertEqual(all_positions.shape, (batch_size, seq_len))
        
        # Verify values for agent 2 (init_p=2, a=-2)
        # p=2, v=0. a=-2.
        # t=0: v=-2, p=0.
        # t=1: v=-4, p=-4.
        agent_idx = 2
        agent_actions = actions.to_numpy()[agent_idx]
        agent_p = init_ps.to_numpy()[agent_idx]
        agent_v = init_vs.to_numpy()[agent_idx]
        
        p, v = agent_p, agent_v
        for a in agent_actions:
            v = v + a
            p = p + v
            
        np.testing.assert_allclose(final_ps.to_numpy()[agent_idx], p)

    def test_rnn_forward(self):
        # Simple RNN: h_t = tanh(W_h @ h_{t-1} + W_x @ x_t)
        
        hidden_dim = 4
        input_dim = 2
        seq_len = 5
        
        W_h = nb.tensor(np.eye(hidden_dim) * 0.5, dtype=nb.DType.float32)
        W_x = nb.tensor(np.ones((hidden_dim, input_dim)), dtype=nb.DType.float32)
        
        def rnn_cell(h, x):
            # h: (hidden,), x: (input,)
            # We need matrix-vector multiplication. 
            # Nabla matmul might expect (M, K) @ (K, N) or (M, K) @ (K).
            # Let's assume we use matmul for (hidden, hidden) @ (hidden) -> (hidden)
            
            # h_next = tanh(W_h @ h + W_x @ x)
            wh_h = W_h @ h
            wx_x = W_x @ x
            pre_act = wh_h + wx_x
            h_next = nb.tanh(pre_act)
            return h_next, h_next

        init_h = nb.zeros((hidden_dim,))
        xs = nb.ones((seq_len, input_dim))
        
        final_h, h_seq = scan(rnn_cell, init_h, xs)
        
        # Manual check
        h = np.zeros(hidden_dim, dtype=np.float32)
        np_Wh = W_h.to_numpy()
        np_Wx = W_x.to_numpy()
        np_xs = xs.to_numpy()
        
        expected_hs = []
        for i in range(seq_len):
            x = np_xs[i]
            pre = np_Wh @ h + np_Wx @ x
            h = np.tanh(pre)
            expected_hs.append(h)
            
        expected_hs = np.array(expected_hs)
        
        np.testing.assert_allclose(h_seq.to_numpy(), expected_hs, rtol=1e-5)

    def test_batched_rnn(self):
        # vmap(scan(rnn))
        # Run RNN over a batch of sequences
        
        hidden_dim = 4
        input_dim = 2
        seq_len = 5
        batch_size = 3
        
        W_h = nb.tensor(np.eye(hidden_dim) * 0.5, dtype=nb.DType.float32)
        W_x = nb.tensor(np.ones((hidden_dim, input_dim)), dtype=nb.DType.float32)
        
        def rnn_cell(h, x):
            wh_h = W_h @ h
            wx_x = W_x @ x
            pre_act = wh_h + wx_x
            h_next = nb.tanh(pre_act)
            return h_next, h_next
            
        def run_rnn(init_h, seq_x):
            return scan(rnn_cell, init_h, seq_x)
            
        # Inputs
        init_hs = nb.zeros((batch_size, hidden_dim))
        xs = nb.ones((batch_size, seq_len, input_dim))
        
        # vmap over batch dim 0
        final_hs, h_seqs = nb.vmap(run_rnn)(init_hs, xs)
        
        self.assertEqual(final_hs.shape, (batch_size, hidden_dim))
        self.assertEqual(h_seqs.shape, (batch_size, seq_len, hidden_dim))
        
        # Since all inputs are identical, all outputs should be identical
        first_seq = h_seqs.to_numpy()[0]
        for i in range(1, batch_size):
            np.testing.assert_allclose(h_seqs.to_numpy()[i], first_seq)

if __name__ == '__main__':
    unittest.main()
