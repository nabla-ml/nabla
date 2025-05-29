# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Shorthands for utitly functions from MAX"""

from max.driver import Device

def Accelerator(id: int = 0) -> Device:
    """
    Create an Accelerator device instance with the specified GPU ID.

    Args:
        id: GPU ID (default is 0)

    Returns:
        An instance of the Accelerator class for the specified GPU.
    """
    from max.driver import Accelerator
    return Accelerator(id=id)

def CPU() -> Device:
    """
    Create a CPU device instance.

    Returns:
        An instance of the CPU class.
    """
    from max.driver import CPU
    return CPU()

def device(device_name: str) -> Device:
    """
    Get a device instance based on the provided device name.

    Args:
        device_name: Name of the device (e.g., "cpu", "cuda", "mps")

    Returns:
        An instance of the corresponding Device class.
    """
    from max.driver import CPU, Accelerator

    # the name can sth like "gpu:0" or "gpu:1" or "cpu", so we need to extrac the id form this string wif gpu is part of it and apply it to the device like: Accelerator(id=0) Accelerator(id=1) or CPU()
    if device_name.startswith("gpu"):
        # Extract the GPU ID from the string
        gpu_id = int(device_name.split(":")[1]) if ":" in device_name else 0
        return Accelerator(id=gpu_id)
    elif device_name == "cpu":
        return CPU()
    else:
        raise ValueError(f"Unsupported device: {device_name}. Use 'cpu' or 'gpu:<id>' format.")