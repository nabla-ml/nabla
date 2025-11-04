from runtime.asyncrt import DeviceContextPtr
from tensor_internal import InputTensor, OutputTensor
from algorithm import elementwise
from layout import LayoutTensor, RuntimeLayout
from utils.index import IndexList

import compiler

@compiler.register("fold_custom")
struct FoldCustom:
    @staticmethod
    fn execute[
        stride_h: Int,
        stride_w: Int,
        dilation_h: Int,
        dilation_w: Int,
        padding_h: Int,
        padding_w: Int,
        target: StaticString,
    ](
        output: OutputTensor,
        input: InputTensor,
        output_size: InputTensor,
        kernel_size: InputTensor,
        ctx: DeviceContextPtr,
    ) raises:
        """Folds array of sliding local blocks into a single output tensor.

        This is adapted from the Modular Mojo kernel library implementation.
        The core logic iterates over each output position and accumulates
        contributions from all overlapping sliding blocks.

        Parameters:
            stride_h: Stride height of the sliding blocks.
            stride_w: Stride width of the sliding blocks.
            dilation_h: Dilation height of the sliding blocks.
            dilation_w: Dilation width of the sliding blocks.
            padding_h: Padding height to be added on both sides.
            padding_w: Padding width to be added on both sides.
            target: The target architecture to compile for.

        Args:
            output: Output tensor to write to, shape [N, C, H, W].
            input: Input tensor to fold, shape [N, C x kernel size, num_blocks].
            output_size: Spatial shape of the output tensor (H, W).
            kernel_size: Size of the sliding blocks (kH, kW).
            ctx: DeviceContextPtr.
        """
        
        constrained[stride_h > 0 and stride_w > 0, "Stride must be positive"]()
        constrained[
            dilation_h > 0 and dilation_w > 0, "Dilation must be positive"
        ]()
        constrained[
            padding_h >= 0 and padding_w >= 0, "Padding must be non-negative"
        ]()

        # Extract dimensions
        var N = output.shape()[0]
        var C = output.shape()[1]
        var H = output.shape()[2]
        var W = output.shape()[3]
        
        # Get kernel size from input tensor
        var kernel_h = Int(kernel_size.load[1](IndexList[1](0))[0])
        var kernel_w = Int(kernel_size.load[1](IndexList[1](1))[0])
        
        # Calculate number of blocks
        var height_col = (
            H + 2 * padding_h - dilation_h * (kernel_h - 1) - 1
        ) // stride_h + 1
        
        var width_col = (
            W + 2 * padding_w - dilation_w * (kernel_w - 1) - 1
        ) // stride_w + 1

        # Create pointers for direct memory access
        var input_ptr = input.unsafe_ptr()
        var output_ptr = output.unsafe_ptr()
        
        var input_shape = input.shape()
        
        # Calculate strides for row-major layout  
        var input_stride_2 = 1
        var input_stride_1 = input_shape[2]
        var input_stride_0 = input_shape[1] * input_shape[2]
        
        var output_stride_3 = 1
        var output_stride_2 = W
        var output_stride_1 = H * W
        var output_stride_0 = C * H * W

        # Use elementwise to process each output position in parallel - matches Modular's implementation
        @always_inline
        @parameter
        @__copy_capture(
            kernel_w,
            kernel_h,
            height_col,
            width_col,
            input_ptr,
            input_stride_0,
            input_stride_1,
            input_stride_2,
            output_ptr,
            output_stride_0,
            output_stride_1,
            output_stride_2,
            output_stride_3,
        )
        fn fold_fn[
            width: Int, rank_: Int, alignment: Int = 1
        ](idx_arg: IndexList[rank_]):
            constrained[rank_ == 4, "fold_fn: rank must be 4"]()
            var idx = rebind[IndexList[4]](idx_arg)
            
            var batch = idx[0]
            var channel = idx[1]
            var h_out = idx[2]
            var w_out = idx[3]
            
            var output_val = Scalar[output.dtype](0)
            
            # The span of the kernel in the output tensor.
            var kernel_span_w = (kernel_w - 1) * dilation_w + 1
            var kernel_span_h = (kernel_h - 1) * dilation_h + 1

            # Given the position in the output tensor (h_out, w_out), compute the
            # start and end of the kernel patches that might overlap with this position.
            var h_start = max(0, (h_out + padding_h - kernel_span_h) // stride_h + 1)
            var w_start = max(0, (w_out + padding_w - kernel_span_w) // stride_w + 1)
            var h_end = min((h_out + padding_h) // stride_h + 1, height_col)
            var w_end = min((w_out + padding_w) // stride_w + 1, width_col)

            for h in range(h_start, h_end):
                for w in range(w_start, w_end):
                    # Compute the relative position of current output position in the kernel patch.
                    var h_offset = h_out - (h * stride_h - padding_h)
                    var w_offset = w_out - (w * stride_w - padding_w)

                    # Check if the current position is covered by the patch.
                    if h_offset % dilation_h == 0 and w_offset % dilation_w == 0:
                        h_offset = h_offset // dilation_h
                        w_offset = w_offset // dilation_w

                        var channel_offset = channel * kernel_h * kernel_w
                        var kernel_offset = h_offset * kernel_w + w_offset
                        var patch_offset = h * width_col + w

                        # Load and accumulate via direct pointer access
                        var input_idx = batch * input_stride_0 + (channel_offset + kernel_offset) * input_stride_1 + patch_offset * input_stride_2
                        output_val += input_ptr[input_idx].cast[output.dtype]()
            
            # Store via direct pointer access
            var output_idx = batch * output_stride_0 + channel * output_stride_1 + h_out * output_stride_2 + w_out * output_stride_3
            output_ptr[output_idx] = output_val

        var dispatch_shape = IndexList[4](N, C, H, W)
        elementwise[
            func=fold_fn,
            simd_width=1,
            target=target,
            _trace_description="fold_fn",
        ](dispatch_shape, ctx)