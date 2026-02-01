
from max.driver import CPU
from max.graph import DeviceRef

cpu_driver = CPU()
cpu_graph = DeviceRef.CPU()

print(f"Driver CPU: {cpu_driver}, type: {type(cpu_driver)}")
print(f"Graph CPU: {cpu_graph}, type: {type(cpu_graph)}")
print(f"Equal: {cpu_driver == cpu_graph}")

# Try conversion
conv = DeviceRef.from_device(cpu_driver)
print(f"Converted: {conv}, type: {type(conv)}")
print(f"Equal after conversion: {conv == cpu_graph}")
