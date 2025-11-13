import torch
import time


def mlp(args):
    """MLP forward pass using function with all weights passed as arguments.

    Args layout: [input, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6, w7, b7, w8, b8, w_out, b_out]
    """
    input_tensor = args[0]
    w1 = args[1]
    b1 = args[2]
    w2 = args[3]
    b2 = args[4]
    w3 = args[5]
    b3 = args[6]
    w4 = args[7]
    b4 = args[8]
    w5 = args[9]
    b5 = args[10]
    w6 = args[11]
    b6 = args[12]
    w7 = args[13]
    b7 = args[14]
    w8 = args[15]
    b8 = args[16]
    w_out = args[17]
    b_out = args[18]

    # Forward pass
    h1 = torch.relu(input_tensor @ w1 + b1)
    h2 = torch.relu(h1 @ w2 + b2)
    h3 = torch.relu(h2 @ w3 + b3)
    h4 = torch.relu(h3 @ w4 + b4)
    h5 = torch.relu(h4 @ w5 + b5)
    h6 = torch.relu(h5 @ w6 + b6)
    h7 = torch.relu(h6 @ w7 + b7)
    h8 = torch.relu(h7 @ w8 + b8)
    output = h8 @ w_out + b_out

    return output


def main():
    # Create all weights ONCE before the loop
    print("Creating weights...")
    t_weight_start = time.perf_counter()

    w1 = torch.randn(512, 2048) * 0.02
    b1 = torch.randn(2048) * 0.01
    w2 = torch.randn(2048, 4096) * 0.02
    b2 = torch.randn(4096) * 0.01
    w3 = torch.randn(4096, 4096) * 0.02
    b3 = torch.randn(4096) * 0.01
    w4 = torch.randn(4096, 4096) * 0.02
    b4 = torch.randn(4096) * 0.01
    w5 = torch.randn(4096, 2048) * 0.02
    b5 = torch.randn(2048) * 0.01
    w6 = torch.randn(2048, 1024) * 0.02
    b6 = torch.randn(1024) * 0.01
    w7 = torch.randn(1024, 512) * 0.02
    b7 = torch.randn(512) * 0.01
    w8 = torch.randn(512, 256) * 0.02
    b8 = torch.randn(256) * 0.01
    w_out = torch.randn(256, 10) * 0.02
    b_out = torch.randn(10) * 0.01

    t_weight_end = time.perf_counter()
    print(f"Weights created in {(t_weight_end - t_weight_start) * 1000:.3f}ms")
    print("Starting training loop...")
    print("")

    # Run stress test
    for it in range(20000):
        t_iter_start = time.perf_counter()

        # Create input tensor filled with iteration number
        input_tensor = torch.full((4, 512), float(it))

        # Pack all arguments: input + all weights and biases
        all_args = [
            input_tensor,
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            w4,
            b4,
            w5,
            b5,
            w6,
            b6,
            w7,
            b7,
            w8,
            b8,
            w_out,
            b_out,
        ]

        # Forward pass
        output = mlp(all_args)

        t_iter_end = time.perf_counter()
        iter_time_ms = (t_iter_end - t_iter_start) * 1000

        # Print progress
        if it % 100 == 0:
            print(f"Iteration {it} | Time: {iter_time_ms:.3f}ms")


if __name__ == "__main__":
    main()
