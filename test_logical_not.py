import nabla as nb

# Test the logical_not operation with our fixed boolean data generation
print("Testing logical_not operation...")

# Test scalar
scalar_true = nb.array(True)
scalar_false = nb.array(False)
print(f"nb.logical_not(True) = {nb.logical_not(scalar_true)}")
print(f"nb.logical_not(False) = {nb.logical_not(scalar_false)}")

# Test with the new boolean data generation method
print("\nTesting boolean data generation:")
shape = (4,)
bool_data = nb.equal(nb.arange(shape) % 2, nb.array(0))
print(f"Boolean pattern: {bool_data}")
print(f"logical_not of pattern: {nb.logical_not(bool_data)}")

# Test VJP (this should fail gracefully since logical ops aren't differentiable)
print("\nTesting VJP (should fail for logical operations):")
try:

    def f(x):
        return nb.logical_not(x)

    result, vjp_fn = nb.vjp(f, scalar_true)
    print(f"VJP result: {result}")
    grad = vjp_fn(nb.array(True))
    print(f"VJP gradient: {grad}")
except Exception as e:
    print(f"VJP failed as expected: {e}")
