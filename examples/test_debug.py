import nabla as nb


@nb.sjit
def foo(x):
    """Simple function to test JIT compilation."""
    print(f"[DEBUG] Inside foo, x.impl = {x.impl is not None}")
    result = x * x
    print(f"[DEBUG] Inside foo, result.impl = {result.impl is not None}")
    return result


print("[DEBUG] Creating input array")
a = nb.arange((2, 3))
print(f"[DEBUG] Input a.impl = {a.impl is not None}")

print("[DEBUG] Calling foo")
b = foo(a)
print(f"[DEBUG] Result b.impl = {b.impl is not None}")

if b.impl is not None:
    print(f"[DEBUG] Result value: {b}")
else:
    print("[DEBUG] Result impl is None - this is the bug!")
