#!/usr/bin/env python3

import nabla


def simple_add(args):
    return [args[0] + args[1]]


if __name__ == "__main__":
    # Test simple vmap
    a = nabla.arange((3, 4), nabla.DType.float32)  # shape (3, 4)
    b = nabla.arange((4,), nabla.DType.float32)  # shape (4,)

    print("a.shape:", a.shape)
    print("b.shape:", b.shape)

    # This should vectorize over the first axis of a, and broadcast b
    vmapped_add = nabla.vmap(simple_add, [0, None])
    result = vmapped_add([a, b])

    print("result.shape:", result[0].shape)
    print("result:", result[0])
