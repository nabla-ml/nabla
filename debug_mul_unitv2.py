import nabla as nb
import jax.numpy as jnp
from nabla.core.sharding.spec import DeviceMesh, ShardingSpec

shape=(4,4)
num_elements=16
jax_base=jnp.arange(num_elements,dtype='float32')
x_jax=(jax_base+1).reshape(shape)*0.1
y_jax=(jax_base+2).reshape(shape)*0.1
x=nb.Tensor.from_dlpack(x_jax)
y=nb.Tensor.from_dlpack(y_jax)

mesh=DeviceMesh('test_mesh',(2,2),('x','y'))
spec=ShardingSpec(mesh, (('x',), ()))

x_s=x.with_sharding(spec.mesh, spec.dim_specs)

res=nb.mul(x_s, y)

print('op:', res._impl.output_refs.op.name)
print('arg0 sharding:', res._impl.output_refs.op_args[0].sharding)
print('arg1 sharding:', res._impl.output_refs.op_args[1].sharding)

val = res.numpy()
print('val[0,0]=', val[0,0])
print('expected[0,0]=', (x_jax*y_jax)[0,0])
print('x[0,0]=', x_jax[0,0], 'y[0,0]=', y_jax[0,0])
