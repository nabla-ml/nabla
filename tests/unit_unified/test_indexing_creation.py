
# ===----------------------------------------------------------------------=== #
# Unified Test: Creation, Indexing, Control Flow
# ===----------------------------------------------------------------------=== #

import pytest
import numpy as np
import nabla as nb
import jax.numpy as jnp
from functools import partial

from .common import (
    Operation, OpConfig, standard_get_args, run_test_with_consistency_check,
    get_test_data_for_shapes
)

# HELPER: Custom Args
def get_creation_args(config: OpConfig):
    return ((), config.params), ((), config.params)

def get_indexing_args(config: OpConfig):
    shapes = config.primal_shapes
    if not shapes: return standard_get_args(config)
    
    x_nb, x_jax = get_test_data_for_shapes((shapes[0],), config)
    x_nb, x_jax = x_nb[0], x_jax[0]
    
    limit = shapes[0][config.params.get("axis", 0)]
    idx_shape = shapes[1]
    idx_np = np.random.randint(0, limit, size=idx_shape)
    
    idx_nb = nb.constant(idx_np, dtype=nb.DType.int64)
    idx_jax = jnp.array(idx_np, dtype="int64")
    
    inputs_nb = [x_nb, idx_nb]
    inputs_jax = [x_jax, idx_jax]
    
    if len(shapes) > 2: # Scatter updates
        u_nb, u_jax = get_test_data_for_shapes((shapes[2],), config)
        inputs_nb.append(u_nb[0])
        inputs_jax.append(u_jax[0])
        
    return (tuple(inputs_nb), config.params), (tuple(inputs_jax), config.params)

def get_where_args(config: OpConfig):
    shapes = config.primal_shapes
    c_np = np.random.choice([True, False], size=shapes[0])
    c_nb = nb.constant(c_np, dtype=nb.DType.bool) # Explicit bool dtype
    c_jax = jnp.array(c_np)
    
    data_nb, data_jax = get_test_data_for_shapes(shapes[1:], config)
    
    return ((c_nb, *data_nb), config.params), ((c_jax, *data_jax), config.params)

# REGISTRY
OPS = {}

# Creation
OPS["zeros"] = Operation("zeros", "CREATION", nb.zeros, jnp.zeros, [OpConfig("R2", params={"shape":(2,3)})], get_creation_args)
OPS["ones"] = Operation("ones", "CREATION", nb.ones, jnp.ones, [OpConfig("R2", params={"shape":(2,3)})], get_creation_args)
OPS["arange"] = Operation("arange", "CREATION", nb.arange, jnp.arange, [OpConfig("Range", params={"start": 0, "stop":5})], get_creation_args)

# Indexing
OPS["gather"] = Operation("gather", "INDEX", nb.gather, lambda x,i,axis: jnp.take(x,i,axis=axis), 
                          [OpConfig("Simp", primal_shapes=((4,4),(2,)), params={"axis":0})], get_indexing_args)
OPS["scatter"] = Operation("scatter", "INDEX", nb.scatter, 
                           lambda x,i,u,axis: x.at[i].set(u) if axis==0 else x,
                           [OpConfig("Simp", primal_shapes=((4,),(1,),(1,)), params={"axis":0})], get_indexing_args)

# Control
OPS["where"] = Operation("where", "CONTROL", nb.where, jnp.where, 
                         [OpConfig("Simp", primal_shapes=((2,2),(2,2),(2,2)))], get_where_args)


@pytest.mark.parametrize("op_name", OPS.keys())
def test_misc_base(op_name):
    op = OPS[op_name]
    config = op.configs[0]
    (a_nb, k_nb), (a_jax, k_jax) = op.get_args(config)
    
    nb_fn = partial(op.nabla_fn, **k_nb)
    jax_fn = partial(op.jax_fn, **k_jax)
    
    run_test_with_consistency_check(f"{op_name}_Base", lambda: nb_fn(*a_nb), lambda: jax_fn(*a_jax))
