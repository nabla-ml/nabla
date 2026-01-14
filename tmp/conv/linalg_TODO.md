# Conv2D and Conv2DTranspose plan (parity with custom PyTorch path)

This document outlines what to implement and what exists already to build `conv2d` and `conv2d_transpose` in Nabla with behavior aligned to the PyTorch-based approach used in `tmp.py`.

## High-level plan

- Expose `conv2d` and `conv2d_transpose` ops with PyTorch-like semantics (stride, padding, dilation, groups, output_padding, optional bias).
- Provide both eager and graph (maxpr) executions:
  - `conv2d`: graph uses experimental MAX `conv2d` (with layout conversions); eager uses `unfold + matmul`.
  - `conv2d_transpose`: implement via `fold/unfold + matmul` in both modes (no MAX transposed conv available).
- Keep gradients fully Nabla-internal via composition (`unfold`, `fold`, `matmul`, `permute`, `reshape`, `sum`).
- Tests: mirror `tmp.py` configurations and compare to PyTorch for shapes and numerics, including edge cases.

## Availability check (what’s already in repo)

- Present and suitable
  - `unfold`, `fold` (in `nabla/ops/special.py`):
    - Eager and graph via custom kernels
    - Support `stride`, `dilation`, symmetric zero-`padding`
    - VJP/JVP implemented
  - `matmul` (in `nabla/ops/linalg.py`):
    - Batched and 2D matmul supported, autodiff in place
  - `reshape`, `permute`, `tensor_slice`, `sum`, `split`, `concatenate` (in `nabla/ops/view.py` and `nabla/ops/reduce.py`)
    - Slicing JIT limitation only for negative-step `[::-1]` (not required for our plan)
  - Experimental MAX `conv2d` (in `experimental/nabla/compiler/graph/ops/convolution.mojo`):
    - Expects NHWC input and RSCF filter `(H, W, C_in/groups, C_out)`
    - Supports stride, dilation, per-side padding `(pt, pb, pl, pr)`, and groups

- Absent or not required now
  - MAX `conv_transpose2d`: not present; we implement via `fold/unfold + matmul`
  - General-purpose padding op with modes: not present and not needed for conv because `unfold/fold` handle zero-padding

## API decisions (aligned with PyTorch)

- `conv2d`
  - Inputs: `x` `(N, C_in, H, W)`, `weight` `(C_out, C_in/groups, K_h, K_w)`, optional `bias` `(C_out,)`
  - Params: `stride` (int/tuple), `dilation` (int/tuple), `padding` (int/tuple symmetric), `groups` (int)
  - Output: `(N, C_out, H_out, W_out)`
  - Backward: handled by composition

- `conv2d_transpose`
  - Inputs: `x` `(N, C_out, H_out, W_out)`, `weight` `(C_out, C_in/groups, K_h, K_w)`
  - Params: `stride`, `dilation`, `padding`, `output_padding` (int/tuple), `groups`
  - Output: `(N, C_in, H_in_eff, W_in_eff)` using the standard formula
  - Backward: handled by composition

- Layout handling (graph mode)
  - `conv2d`: permute x NCHW→NHWC, map weight `(C_out, C_in_g, K_h, K_w)` → `(K_h, K_w, C_in_g, C_out)`, call MAX `conv2d`, permute back NHWC→NCHW
  - `conv2d_transpose`: implemented via `fold/unfold` in native layout

## Implementation paths (no code here, just plan)

- `conv2d` eager
  1. `x_cols = unfold(x, kernel=(K_h, K_w), stride, dilation, padding)` → `(N, C_in*K_h*K_w, L)`
  2. Reshape to `(N, groups, C_in_g*K*K, L)`
  3. Reshape weights to `(groups, C_out_g, C_in_g*K*K)`
  4. Batched matmul → `(N, groups, C_out_g, L)` → merge groups to `(N, C_out, L)` → reshape to `(N, C_out, H_out, W_out)`
  5. Add bias via broadcast if provided

- `conv2d` graph
  - Use MAX `conv2d` with layout conversions and per-side padding derived from symmetric `(pH, pW)`

- `conv2d_transpose` eager + graph
  1. Treat as adjoint via `fold`:
  2. For each group, compute columns from `(N, C_out_g, L)` by multiplying with weights to get `(N, C_in_g*K*K, L)`
  3. `fold(columns, output_size=(H_in_eff, W_in_eff), kernel=(K_h, K_w), stride, dilation, padding)`

## Gradients

- By composition: `unfold/fold` provide correct VJP rules; `matmul` has VJP; `sum/reshape/permute` covered.
- Bias gradient: `sum` over `(N, H, W)`.
- Optional later: custom VJPs for perf and memory.

## Padding: is our `pad` op suitable?

- `nabla.ops.view.pad` is an inverse-slice utility used in the VJP of slicing (place a smaller tensor into a larger zero tensor). It is not a general padding op.
- For convolution, we do not need a general padding op: `unfold/fold` already accept `padding=(pH, pW)` and handle zero-padding consistently in eager and graph.
- If future work requires PyTorch-like `F.pad` with modes (`constant`, `reflect`, `replicate`, `circular`), that would be a new op.

## Test plan (to implement)

- Parameter sweeps: `stride > 1`, `padding > 0`, `dilation > 1`, and combinations
- Groups: grouped, depthwise, depthwise multiplier
- Asymmetry: `stride=(sH,sW)`, `dilation=(dH,dW)`, `padding=(pH,pW)`
- Edge shapes: output `1x1` via stride or dilation, stride > kernel size
- dtypes: `float64` and `float32`
- Bias on/off
- Multi-pass accumulation
- Stress case: all params + groups + batch
- Compare outputs and all grads to PyTorch within tolerances per dtype

## Optional follow-ups (after parity)

- Add asymmetric per-side padding to the high-level API and map to MAX; emulate in eager if needed
- Custom kernels for `conv`/`unfold`/`fold` for performance
- Custom VJPs for `conv2d`/`conv2d_transpose` to reduce intermediates
