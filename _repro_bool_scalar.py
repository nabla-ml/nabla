import numpy as np
import nabla as nb
from max import driver

try:
    arr = np.array(True)
    print(f"Array shape: {arr.shape}, dtype: {arr.dtype}")
    # This is what fails according to the traceback
    # buffer = driver.Buffer.from_dlpack(arr) 
    # print("Buffer created successfully")
    
    # Test if explicit shape works
    arr = np.array(True)
    buf = driver.Buffer.from_dlpack(arr.reshape(1))
    print("Normal 1D buffer created")
    from max.dtype import DType
    buf0 = buf.view(DType.bool, ())
    print("Explicit shape () worked!")
except Exception as e:
    import traceback
    traceback.print_exc()
