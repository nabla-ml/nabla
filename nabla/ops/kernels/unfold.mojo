from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from algorithm import elementwise
from layout import LayoutTensor, RuntimeLayout
from utils.index import IndexList
import compiler


@compiler.register("unfold_custom")
struct UnfoldCustom:
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
        kernel_size: InputTensor,
        ctx: DeviceContextPtr,
    ) raises:
        """Extracts sliding local blocks from a batched input tensor.

        This is the inverse operation of fold. It extracts patches from the input
        tensor and arranges them as columns in the output.

        Parameters:
            stride_h: Stride height of the sliding blocks.
            stride_w: Stride width of the sliding blocks.
            dilation_h: Dilation height of the sliding blocks.
            dilation_w: Dilation width of the sliding blocks.
            padding_h: Padding height to be added on both sides.
            padding_w: Padding width to be added on both sides.
            target: The target architecture to compile for.

        Args:
            output: Output tensor, shape [N, C x kernel_h x kernel_w, L].
            input: Input tensor to unfold, shape [N, C, H, W].
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
        var N = input.shape()[0]
        var C = input.shape()[1]
        var H = input.shape()[2]
        var W = input.shape()[3]
        
        # Get kernel size
        var kernel_h = Int(kernel_size.load[1](IndexList[1](0))[0])
        var kernel_w = Int(kernel_size.load[1](IndexList[1](1))[0])
        
        # Calculate number of blocks
        var height_col = (
            H + 2 * padding_h - dilation_h * (kernel_h - 1) - 1
        ) // stride_h + 1
        
        var width_col = (
            W + 2 * padding_w - dilation_w * (kernel_w - 1) - 1
        ) // stride_w + 1
        
        var L = height_col * width_col
        
        # Create pointers for direct memory access
        var input_ptr = input.unsafe_ptr()
        var output_ptr = output.unsafe_ptr()
        
        # Calculate strides for row-major layout
        var input_stride_3 = 1
        var input_stride_2 = W
        var input_stride_1 = H * W
        var input_stride_0 = C * H * W
        
        var output_stride_2 = 1
        var output_stride_1 = L
        var output_stride_0 = C * kernel_h * kernel_w * L
        
        # Use elementwise to process each output position in parallel
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
            input_stride_3,
            output_ptr,
            output_stride_0,
            output_stride_1,
            output_stride_2,
            C,
            H,
            W,
        )
        fn unfold_fn[
            width: Int, rank_: Int, alignment: Int = 1
        ](idx_arg: IndexList[rank_]):
            constrained[rank_ == 3, "unfold_fn: rank must be 3"]()
            var idx = rebind[IndexList[3]](idx_arg)
            
            var batch = idx[0]
            var c_out = idx[1]  # output channel (flattened kernel position)
            var block_idx = idx[2]  # which block/patch
            
            # Decode block index to spatial position
            var h_col = block_idx // width_col
            var w_col = block_idx % width_col
            
            # Decode output channel to (channel, kernel_h_offset, kernel_w_offset)
            var channel = c_out // (kernel_h * kernel_w)
            var kernel_offset = c_out % (kernel_h * kernel_w)
            var kh = kernel_offset // kernel_w
            var kw = kernel_offset % kernel_w
            
            # Calculate input position
            var h_in = h_col * stride_h - padding_h + kh * dilation_h
            var w_in = w_col * stride_w - padding_w + kw * dilation_w
            
            # Check if position is within bounds (handle padding)
            var value = Scalar[output.dtype](0)
            if h_in >= 0 and h_in < H and w_in >= 0 and w_in < W:
                # Load from input
                var input_idx = batch * input_stride_0 + channel * input_stride_1 + h_in * input_stride_2 + w_in * input_stride_3
                value = input_ptr[input_idx].cast[output.dtype]()
            
            # Store to output
            var output_idx = batch * output_stride_0 + c_out * output_stride_1 + block_idx * output_stride_2
            output_ptr[output_idx] = value

        var dispatch_shape = IndexList[3](N, C * kernel_h * kernel_w, L)
        elementwise[
            func=unfold_fn,
            simd_width=1,
            target=target,
            _trace_description="unfold_fn",
        ](dispatch_shape, ctx)
