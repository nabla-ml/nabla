## distributed_broadcast()
max.graph.ops.distributed_broadcast(input, signal_buffers)

Broadcast tensor from source GPU to all GPUs.

This op is a collective operation which broadcasts a tensor from the source GPU (where the input tensor resides) to all participating GPUs. Each GPU receives a copy of the input tensor.

Parameters:

input (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray) – Input tensor to broadcast. The device where this tensor resides becomes the root/source of the broadcast.
signal_buffers (Iterable[BufferValue | HasBufferValue]) – Device buffer values used for synchronization. The number of signal buffers determines the number of participating GPUs.
Returns:

List of output tensors, one per device. Each output tensor has the same shape and dtype as the input tensor.

Raises:

ValueError – If input tensor device is not found in signal buffer devices, if devices are not unique, or if there are fewer than 2 signal buffers.

Return type:

list[TensorValue]




## transfer_to()
max.graph.ops.transfer_to(x, device)

Device-to-Device transfer operation.

This op transfers the input tensor from its current device over to another. A device represents a computation unit, like CPU, GPU, etc. This op is useful for instance when working with accelerators, like GPU, where for instance one may need to move data from GPU to GPU, or from one GPU to CPU.

Parameters:

x (Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue) – The input tensor to transfer.
device (Device | DeviceRef) – The device to transfer to.
Returns:

A tensor transferred to device specified.

Return type:

TensorValue


## allgather()
max.graph.ops.allgather(inputs, signal_buffers, axis=0)

Collective allgather operation.

This op is a collective op which takes in tensors from different devices and outputs tensors on different devices. In particular, this operation will gather the inputs across different devices and concatenates them along the specified dimension. The result is then broadcasted back to the same devices that the inputs came from.

Parameters:

inputs (Iterable[Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray]) – The input tensors to gather.
signal_buffers (Iterable[BufferValue | HasBufferValue]) – Device buffer values used for synchronization.
axis (int) – Dimension to concatenate the input tensors. Defaults to 0.
Returns:

An iterable outputs which all hold the gathered output. Each output tensor contains the concatenation of all inputs along the specified dimension.

Return type:

list[TensorValue]


## shard_and_stack()
max.graph.ops.shard_and_stack(inputs, devices, axis=0)

Shards a list of input tensors along a specified axis, producing multiple outputs.

This operation takes multiple input tensors, splits each along the specified axis into len(devices) chunks, and returns one output tensor per device. Each output contains the chunks at the corresponding index stacked from all inputs along a new dimension 0.

This is useful for distributing model weights across multiple devices in tensor parallel configurations.

For example, with 2 inputs A and B, axis=0, and 2 devices:

Input A shape [10, D], Input B shape [10, D]
Output 0: stack([A[0:5], B[0:5]]) -> shape [2, 5, D] on devices[0]
Output 1: stack([A[5:10], B[5:10]]) -> shape [2, 5, D] on devices[1]
With axis=1 and 2 devices:

Input A shape [D, 10], Input B shape [D, 10]
Output 0: stack([A[:, 0:5], B[:, 0:5]]) -> shape [2, D, 5] on devices[0]
Output 1: stack([A[:, 5:10], B[:, 5:10]]) -> shape [2, D, 5] on devices[1]
Parameters:

inputs (Sequence[Value[TensorType] | TensorValue | Shape | Dim | HasTensorValue | int | float | integer[Any] | floating[Any] | DLPackArray]) – A list of symbolic tensors to shard. All tensors must have the same shape, dtype, and device.
devices (Sequence[Device | DeviceRef]) – Target devices for each output tensor. The number of devices determines the number of splits. Each output tensor will be placed on the corresponding device. This enables direct host-to-device transfer without intermediate CPU storage.
axis (int) – The axis along which to split each input tensor. Defaults to 0. Supports negative indexing (e.g., -1 for last axis).
Returns:

A list of len(devices) tensors, each with shape [num_inputs, D0, …, Daxis//len(devices), …, Dn-1] where the input shape is [D0, …, Daxis, …, Dn-1]. Output i contains the stacked chunks at position i from all input tensors, placed on devices[i].

Raises:

ValueError – If inputs list is empty, if devices list is empty, if input tensors don’t have matching shapes, if the dimension size at the axis is not evenly divisible by len(devices), or if axis is out of bounds.

Return type:

list[TensorValue]