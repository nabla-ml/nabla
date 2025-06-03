"""
Basic Nabla Operations
=====================

This example showcases basic operations in Nabla, including array creation, arithmetic,
and mathematical functions.
"""

import matplotlib.pyplot as plt

import nabla as nb

# Create arrays
x = nb.arange(-5, 5, 0.1)
y1 = nb.sin(x)
y2 = nb.cos(x)

# Convert to NumPy for plotting
x_np = x.numpy()
y1_np = y1.numpy()
y2_np = y2.numpy()

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x_np, y1_np, label="sin(x)")
plt.plot(x_np, y2_np, label="cos(x)")
plt.title("Sine and Cosine functions with Nabla")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Display the plot (this is detected by sphinx-gallery)
plt.show()
