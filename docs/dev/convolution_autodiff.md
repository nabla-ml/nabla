# Convolution AD Notes (Phase 1)

This note documents the mathematical implementation used for Nabla `conv2d` in [nabla/ops/convolution.py](../../nabla/ops/convolution.py), including VJP and JVP rules in NHWC/RSCF form.

## 1) Forward Definition (`conv2d`)

Let:

- input: `X ∈ ℝ^{N×H×W×C_in}` (NHWC)
- filter: `W ∈ ℝ^{K_h×K_w×C_in×C_out}` (RSCF/HWIO for groups=1)
- optional bias: `b ∈ ℝ^{C_out}`
- stride: `(s_h, s_w)`, dilation: `(d_h, d_w)`
- padding: `(p_t, p_b, p_l, p_r)`

Output size:

- `H_out = floor((H + p_t + p_b - d_h (K_h-1) - 1) / s_h + 1)`
- `W_out = floor((W + p_l + p_r - d_w (K_w-1) - 1) / s_w + 1)`

Elementwise:

`Y[n,h,w,c_o] = Σ_{k_h,k_w,c_i} X_pad[n, h*s_h + k_h*d_h, w*s_w + k_w*d_w, c_i] * W[k_h,k_w,c_i,c_o] + b[c_o]`

where `X_pad` is zero-padded in spatial dimensions.

## 2) VJP for `conv2d`

Given cotangent `G = ∂L/∂Y`:

### 2.1 Input gradient

`∂L/∂X` is the transposed-convolution of `G` with `W` under the same stride/dilation/padding convention:

`dX = conv2d_transpose(G, W; stride, dilation, padding, output_paddings)`

`output_paddings` are solved from shape consistency:

- `base_h = (H_out - 1)s_h - p_t - p_b + d_h(K_h - 1) + 1`
- `base_w = (W_out - 1)s_w - p_l - p_r + d_w(K_w - 1) + 1`
- `out_pad_h = H - base_h`, `out_pad_w = W - base_w`

If either is negative, compute with `max(.,0)` and crop back to input shape.

### 2.2 Filter gradient

By chain rule,

`dW[k_h,k_w,c_i,c_o] = Σ_{n,h,w} X_pad[n, h*s_h + k_h*d_h, w*s_w + k_w*d_w, c_i] * G[n,h,w,c_o]`

Implementation uses the standard conv trick by permuting dimensions and reusing `conv2d`:

- `X_perm = permute(X_pad, (C_in, H, W, N))`
- `G_perm = permute(G, (H_out, W_out, N, C_out))`
- `dW_perm = conv2d(X_perm, G_perm; stride=dilation, dilation=stride, padding=0)`
- `dW = permute(dW_perm, (K_h, K_w, C_in, C_out))`

### 2.3 Bias gradient

`db[c_o] = Σ_{n,h,w} G[n,h,w,c_o]`

i.e. `reduce_sum(G, axis=[0,1,2])`.

## 3) JVP for `conv2d`

Using linearization of a bilinear map in `(X, W)`:

`dY = conv2d(dX, W; θ) + conv2d(X, dW; θ) + dB`

where `θ = (stride, dilation, padding, groups, layout kwargs)` is held fixed.

This is exactly what the implementation computes.

## 4) Why this is correct (brief proof sketch)

Convolution is affine in each argument and bilinear in `(X, W)` once hyperparameters are fixed.

- VJP follows from adjointness of correlation/convolution and linearity of summation.
- JVP follows from first-order expansion:

`conv(X + εdX, W + εdW) = conv(X,W) + ε[conv(dX,W) + conv(X,dW)] + O(ε²)`

and bias contributes additively as `+ ε dB`.

Therefore:

- reverse-mode returns the unique linear adjoint map wrt `X, W, b`,
- forward-mode returns the directional derivative `J·v`.

## 5) Backend constraints in this phase

Current MAX backend behavior in this environment requires:

- `conv2d`: `dilation=(1,1)`
- `conv2d`: grouped mode not yet usable without prepacked filters (`groups=1` enforced)
- `conv2d_transpose`: `output_paddings=(0,0)` only

Nabla validates these explicitly and raises clear frontend errors.
