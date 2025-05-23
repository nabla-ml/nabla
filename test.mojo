from memory import ArcPointer, UnsafePointer
from nabla.compiler.driver import DeviceMemory, accelerator, cpu
import nabla 


def main():
    # var cpu_device = cpu()
    # var gpu_device = accelerator()
    # var num_bytes = 1024
    # alias type = DType.float32

    # # allocate memory on cpu
    # var ptr = ArcPointer(DeviceMemory(num_bytes, cpu_device))
    # for i in range(1024 // 8):
    #     ptr[].unsafe_ptr().bitcast[SIMD[type, 1]]()[i] = Float32(i)

    # # move memory to gpu
    # var ptr_gpu = ArcPointer(ptr[].copy_to(gpu_device))

    # # move memory back to cpu
    # var ptr_cpu = ArcPointer(ptr_gpu[].copy_to(cpu_device))
    # for i in range(1024 // 8):
    #     print(ptr_cpu[].unsafe_ptr().bitcast[SIMD[type, 1]]()[i])

    var a = nabla.zeros((2, 3))
    print(nabla.sin(a) + 1)

    # var ptr = UnsafePointer[Scalar[DType.float32]].alloc(16)
    # for i in range(4):
    #     ptr[i] = Float32(i)

    # var none_ptr = ptr.bitcast[NoneType]()

    # var ptr_dtype = none_ptr.bitcast[Scalar[DType.float32]]()
    # for i in range(4):
    #     print(ptr_dtype[i])

