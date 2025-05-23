from memory import ArcPointer
from nabla.compiler.driver import DeviceMemory, accelerator, cpu

def main():
    var cpu_device = cpu()
    var gpu_device = accelerator()
    var num_bytes = 1024

    # allocate memory on cpu
    var ptr = ArcPointer(DeviceMemory(num_bytes, cpu_device))
    for i in range(1024 // 8):
        ptr[].unsafe_ptr().bitcast[Scalar[DType.float32]]()[i] = Float32(i)

    # move memory to gpu
    var ptr_gpu = ArcPointer(ptr[].copy_to(gpu_device))

    # move memory back to cpu
    var ptr_cpu = ArcPointer(ptr_gpu[].copy_to(cpu_device))
    for i in range(1024 // 8):
        print(ptr_cpu[].unsafe_ptr().bitcast[Scalar[DType.float32]]()[i])
    

    

    