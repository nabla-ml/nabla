# Convolution & Pooling

## `conv2d`

```python
def conv2d(x: "'Tensor'", filter: "'Tensor'", *, stride: 'int | tuple[int, int]' = (1, 1), dilation: 'int | tuple[int, int]' = (1, 1), padding: 'int | tuple[int, int] | tuple[int, int, int, int]' = (0, 0, 0, 0), groups: 'int' = 1, bias: "'Tensor | None'" = None, input_layout: 'Any' = None, filter_layout: 'Any' = None) -> "'Tensor'":
```
Apply a 2D convolution over an input tensor.

Operates on tensors in NHWC layout (batch, height, width, channels).
Filters are expected in HWIO layout (kernel_h, kernel_w, in_channels, out_channels).
Supports autograd (VJP and JVP).

**Parameters**

- **`x`** – Input tensor of shape ``(N, H, W, C_in)``.
- **`filter`** – Convolution kernel of shape ``(K_h, K_w, C_in, C_out)``.
- **`stride`** – Convolution stride as ``(s_h, s_w)`` or a single int.
Default: ``(1, 1)``.
- **`dilation`** – Kernel dilation as ``(d_h, d_w)`` or a single int.
Currently only ``(1, 1)`` is supported. Default: ``(1, 1)``.
- **`padding`** – Padding as an int, ``(pad_h, pad_w)``, or a 4-tuple
``(top, bottom, left, right)``. Default: ``(0, 0, 0, 0)``.
- **`groups`** – Number of blocked connections from input channels to output
channels. Currently only ``1`` is supported. Default: ``1``.
- **`bias`** – Optional bias tensor of shape ``(C_out,)``.

**Returns**

Output tensor of shape ``(N, H_out, W_out, C_out)``.


---
## `conv2d_transpose`

```python
def conv2d_transpose(x: "'Tensor'", filter: "'Tensor'", *, stride: 'int | tuple[int, int]' = (1, 1), dilation: 'int | tuple[int, int]' = (1, 1), padding: 'int | tuple[int, int] | tuple[int, int, int, int]' = (0, 0, 0, 0), output_paddings: 'int | tuple[int, int]' = (0, 0), bias: "'Tensor | None'" = None, input_layout: 'Any' = None, filter_layout: 'Any' = None) -> "'Tensor'":
```
Apply a 2D transposed convolution (fractionally-strided convolution).

Also known as a ``deconvolution`` or ``upsample`` layer. Produces the
gradient of :func:`conv2d` with respect to its input, enabling
encoder–decoder architectures. Fully differentiable.

**Parameters**

- **`x`** – Input tensor of shape ``(N, H, W, C_in)``.
- **`filter`** – Kernel of shape ``(K_h, K_w, C_out, C_in)`` (note the
transposed channel order compared to :func:`conv2d`).
- **`stride`** – Stride of the transposed convolution. Default: ``(1, 1)``.
- **`dilation`** – Kernel dilation. Currently only ``(1, 1)`` is supported.
Default: ``(1, 1)``.
- **`padding`** – Amount of implicit zero-padding removed from the output.
Same format as :func:`conv2d`. Default: ``(0, 0, 0, 0)``.
- **`output_paddings`** – Extra rows/columns added to the output shape.
Currently only ``(0, 0)`` is supported. Default: ``(0, 0)``.
- **`bias`** – Optional bias tensor of shape ``(C_out,)``.

**Returns**

Output tensor of shape ``(N, H_out, W_out, C_out)``.


---
## `avg_pool2d`

```python
def avg_pool2d(x: "'Tensor'", *, kernel_size: 'int | tuple[int, int]', stride: 'int | tuple[int, int] | None' = None, padding: 'int | tuple[int, int] | tuple[int, int, int, int]' = 0, dilation: 'int | tuple[int, int]' = (1, 1)) -> "'Tensor'":
```
Apply 2D average pooling over an NHWC input tensor.

Divides the input into non-overlapping (or strided) windows of size
*kernel_size* and computes the mean of each window.

**Parameters**

- **`x`** – Input tensor of shape ``(N, H, W, C)``.
- **`kernel_size`** – Size of the pooling window as ``(k_h, k_w)`` or a
single int.
- **`stride`** – Stride of the pooling window. Defaults to *kernel_size*
(non-overlapping).
- **`padding`** – Implicit zero-padding added to the input before pooling.
Same format as :func:`~nabla.conv2d`. Default: ``0``.
- **`dilation`** – Currently only ``(1, 1)`` is supported. Default: ``(1, 1)``.

**Returns**

Pooled tensor of shape ``(N, H_out, W_out, C)``.


---
## `max_pool2d`

```python
def max_pool2d(x: "'Tensor'", *, kernel_size: 'int | tuple[int, int]', stride: 'int | tuple[int, int] | None' = None, padding: 'int | tuple[int, int] | tuple[int, int, int, int]' = 0, dilation: 'int | tuple[int, int]' = (1, 1)) -> "'Tensor'":
```
Apply 2D max pooling over an NHWC input tensor.

Divides the input into windows of size *kernel_size* and computes the
maximum of each window.

**Parameters**

- **`x`** – Input tensor of shape ``(N, H, W, C)``.
- **`kernel_size`** – Size of the pooling window as ``(k_h, k_w)`` or a
single int.
- **`stride`** – Stride of the pooling window. Defaults to *kernel_size*
(non-overlapping).
- **`padding`** – Implicit zero-padding added to the input before pooling.
Same format as :func:`~nabla.conv2d`. Default: ``0``.
- **`dilation`** – Currently only ``(1, 1)`` is supported. Default: ``(1, 1)``.

**Returns**

Pooled tensor of shape ``(N, H_out, W_out, C)``.


---
