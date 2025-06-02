"""
Basic Nabla Operations
======================

This example demonstrates fundamental operations in Nabla, including array creation,
mathematical functions, and visualization. Nabla provides a NumPy-like interface
with automatic differentiation capabilities.
"""
import nabla as nb
import matplotlib.pyplot as plt
import numpy as np

print("Nabla Basic Operations Example")
print("=" * 40)

###############################################################################
# Array Creation and Basic Properties
# ------------------------------------
# Let's start by creating arrays using various methods

# Create an array from numpy data for plotting
x_np = np.linspace(-2 * np.pi, 2 * np.pi, 100)
x = nb.array(x_np)
print(f"Created array with shape: {x.shape}")
print(f"Array dtype: {x.dtype}")

###############################################################################
# Mathematical Functions
# -----------------------
# Apply various mathematical functions

y_sin = nb.sin(x)
y_cos = nb.cos(x)
y_exp = nb.exp(-x**2 / 4)  # Gaussian-like function
y_tanh = nb.tanh(x)

# Convert to NumPy for plotting
x_np = x.to_numpy()
y_sin_np = y_sin.to_numpy()
y_cos_np = y_cos.to_numpy()
y_exp_np = y_exp.to_numpy()
y_tanh_np = y_tanh.to_numpy()

###############################################################################
# Create Beautiful Plots
# -----------------------
# Visualize the mathematical functions

# Set up the plotting style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Nabla Mathematical Functions Showcase', fontsize=16, fontweight='bold')

# Plot 1: Trigonometric functions
ax1.plot(x_np, y_sin_np, label='sin(x)', color='blue', linewidth=2)
ax1.plot(x_np, y_cos_np, label='cos(x)', color='red', linewidth=2)
ax1.set_title('Trigonometric Functions', fontweight='bold')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.axhline(y=0, color='black', linewidth=0.5)
ax1.axvline(x=0, color='black', linewidth=0.5)

# Plot 2: Exponential function
ax2.plot(x_np, y_exp_np, label='exp(-x²/4)', color='green', linewidth=2)
ax2.set_title('Gaussian-like Function', fontweight='bold')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.axvline(x=0, color='black', linewidth=0.5)

# Plot 3: Hyperbolic tangent
ax3.plot(x_np, y_tanh_np, label='tanh(x)', color='purple', linewidth=2)
ax3.set_title('Hyperbolic Tangent', fontweight='bold')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.axhline(y=0, color='black', linewidth=0.5)
ax3.axvline(x=0, color='black', linewidth=0.5)

# Plot 4: Combined view
ax4.plot(x_np, y_sin_np, label='sin(x)', alpha=0.7, linewidth=2)
ax4.plot(x_np, y_cos_np, label='cos(x)', alpha=0.7, linewidth=2)
ax4.plot(x_np, y_exp_np, label='exp(-x²/4)', alpha=0.7, linewidth=2)
ax4.plot(x_np, y_tanh_np, label='tanh(x)', alpha=0.7, linewidth=2)
ax4.set_title('All Functions Combined', fontweight='bold')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.grid(True, alpha=0.3)
ax4.legend()
ax4.axhline(y=0, color='black', linewidth=0.5)
ax4.axvline(x=0, color='black', linewidth=0.5)

plt.tight_layout()
plt.show()

###############################################################################
# Array Operations Example
# -------------------------
# Demonstrate element-wise operations and broadcasting

print("\nArray Operations:")
print("-" * 20)

# Create sample arrays
a = nb.array([1, 2, 3, 4, 5])
b = nb.array([2, 3, 4, 5, 6])

# Element-wise operations
addition = a + b
multiplication = a * b
power = a ** 2

print(f"a = {a.to_numpy()}")
print(f"b = {b.to_numpy()}")
print(f"a + b = {addition.to_numpy()}")
print(f"a * b = {multiplication.to_numpy()}")
print(f"a ** 2 = {power.to_numpy()}")

# Broadcasting example
scalar = 10
broadcast_result = a + scalar
print(f"a + {scalar} = {broadcast_result.to_numpy()}")

###############################################################################
# Matrix Operations
# ------------------
# Show linear algebra capabilities

print("\nMatrix Operations:")
print("-" * 20)

# Create matrices
matrix_a = nb.array([[1, 2], [3, 4]])
matrix_b = nb.array([[5, 6], [7, 8]])

# Matrix multiplication
matrix_product = nb.matmul(matrix_a, matrix_b)

print("Matrix A:")
print(matrix_a.to_numpy())
print("\nMatrix B:")
print(matrix_b.to_numpy())
print("\nA @ B =")
print(matrix_product.to_numpy())

print("\nNabla operations completed successfully! 🎉")