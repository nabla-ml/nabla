from memory import ArcPointer, UnsafePointer
from nabla.compiler.driver import DeviceMemory, accelerator, cpu
from nabla.compiler.driver._status import Status
from nabla.compiler.tensor import TensorSpec
import nabla 
from time import perf_counter

def main():
    var cpu_device = cpu()
    # var gpu_device = accelerator()
    var num_bytes = 10000000
    alias type = DType.float32

    # allocate memory on cpu
    var start = perf_counter()
    for _ in range(10000):
        # var x = DeviceMemory(num_bytes, cpu_device)
        var spec = TensorSpec(DType.uint8, num_bytes)
        var status = Status(cpu_device._lib.value())
        var x = cpu_device._lib.value().create_device_memory_fn(
            UnsafePointer[TensorSpec](to=spec),
            cpu_device._cdev._ptr,
            status.impl,
        )
        for _ in range(10000):
            x.bitcast[SIMD[type, 1]]()[0] = Float32(1.0)
    var end = perf_counter()
    print("Allocating memory on CPU with DeviceMemory took: ", end - start, " seconds")


    start = perf_counter()
    for _ in range(10000):
        var x = UnsafePointer[Scalar[DType.uint8]].alloc(num_bytes).bitcast[NoneType]()
        for _ in range(10000):
            x.bitcast[SIMD[type, 1]]()[0] = Float32(1.0)
        x.free()
    end = perf_counter()
    print("Allocating memory on CPU with UnsafePointer took: ", end - start, " seconds")
    # for i in range(1024 // 8):
    #     ptr[].unsafe_ptr().bitcast[SIMD[type, 1]]()[i] = Float32(i)

    # # move memory to gpu
    # var ptr_gpu = ArcPointer(ptr[].copy_to(gpu_device))

    # # move memory back to cpu
    # var ptr_cpu = ArcPointer(ptr_gpu[].copy_to(cpu_device))
    # for i in range(1024 // 8):
    #     print(ptr_cpu[].unsafe_ptr().bitcast[SIMD[type, 1]]()[i])

    # var a = nabla.zeros((2, 3))
    # print(nabla.sin(a) + 1)

    # var ptr = UnsafePointer[Scalar[DType.float32]].alloc(16)
    # for i in range(4):
    #     ptr[i] = Float32(i)

    # var none_ptr = ptr.bitcast[NoneType]()

    # var ptr_dtype = none_ptr.bitcast[Scalar[DType.float32]]()
    # for i in range(4):
    #     print(ptr_dtype[i])

