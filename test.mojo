from sys.ffi import DLHandle, c_char, _find_dylib, _get_global_or_null

from memory.unsafe_pointer import *
from collections import Optional, OptionalReg
from os import abort
from pathlib import Path

from gpu.host import DeviceContext
from gpu.host import DeviceFunction as AcceleratorFunction
from gpu.host import Dim, FuncAttribute
from gpu.host.compile import get_gpu_target

from buffer.dimlist import DimList
from buffer import NDBuffer

from layout import IntTuple, Layout, LayoutTensor, RuntimeLayout

from runtime.asyncrt import DeviceContextPtr

from memory import ArcPointer, UnsafePointer, OwnedPointer, memcpy
from runtime.asyncrt import DeviceContextPtr  
from memory import ArcPointer, UnsafePointer
from memory.unsafe import bitcast
from python import Python, PythonObject

from collections import InlineArray, Optional, List, Dict
from collections.string import StaticString, StringSlice
from collections.dict import _DictEntryIter, _DictKeyIter

from sys import alignof, external_call, CompilationTarget, sizeof
from sys.param_env import is_defined

from max._mlir.builtin_attributes import StringAttr
# from max._mlir.ir import Identifier, NamedAttribute

from os.atomic import Atomic
from os import abort, PathLike

from utils import Variant, IndexList
from utils._serialize import _serialize