"""
Basic Endia Operations
=====================

This example showcases basic operations in Endia, including array creation, arithmetic,
and mathematical functions.
"""

import matplotlib.pyplot as plt

import endia as nb

# Create arrays
x = nd.arange(-5, 5, 0.1)
y1 = nd.sin(x)
y2 = nd.cos(x)

# Convert to NumPy for plotting
x_np = x.numpy()
y1_np = y1.numpy()
y2_np = y2.numpy()

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x_np, y1_np, label="sin(x)")
plt.plot(x_np, y2_np, label="cos(x)")
plt.title("Sine and Cosine functions with Endia")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Display the plot (this is detected by sphinx-gallery)
plt.show()
