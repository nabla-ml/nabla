from gpu.host import DeviceContext
from gpu.id import block_idx, thread_idx
from sys import has_accelerator

# Vector data type and size
alias float_dtype = DType.float32
alias vector_size = 1000

# fn get_any_buffer() -> 


def main():
    @parameter
    # if not has_accelerator():
    #     print("No compatible GPU found")
    # else:
    # Get the context for the attached GPU
    ctx = DeviceContext()

    # Create HostBuffers for input vectors
    lhs_host_buffer = ctx.enqueue_create_host_buffer[DType.uint8](
        vector_size
    )
    var ptr = lhs_host_buffer.unsafe_ptr()
    # rhs_host_buffer = ctx.enqueue_create_host_buffer[float_dtype](
    #     vector_size
    # )
    # ctx.synchronize()

    # # Initialize the input vectors
    # for i in range(vector_size):
    #     lhs_host_buffer[i] = Float32(i)
    #     rhs_host_buffer[i] = Float32(i * 0.5)

    # print("LHS buffer: ", lhs_host_buffer)
    # print("RHS buffer: ", rhs_host_buffer)
