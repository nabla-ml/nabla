Mojo struct

UnsafePointer
@register_passable(trivial)
struct UnsafePointer[mut: Bool, //, type: AnyType, origin: Origin[mut=mut], *, address_space: AddressSpace = AddressSpace.GENERIC]

UnsafePointer represents an indirect reference to one or more values of type T consecutively in memory, and can refer to uninitialized memory.

Because it supports referring to uninitialized memory, it provides unsafe methods for initializing and destroying instances of T, as well as methods for accessing the values once they are initialized. You should instead use safer pointers when possible.

Differences from LegacyUnsafePointer:

UnsafePointer fixes the unsafe implicit mutability and origin casting issues of LegacyUnsafePointer.
UnsafePointer has an inferred mutability parameter.
UnsafePointer does not have a defaulted origin parameter, this must be explicitly specified or unbound.
Important things to know:

This pointer is unsafe and nullable. No bounds checks; reading before writing is undefined.
It does not own existing memory. When memory is heap-allocated with alloc(), you must call .free().
For simple read/write access, use (ptr + i)[] or ptr[i] where i is the offset size.
For SIMD operations on numeric data, use UnsafePointer[Scalar[DType.xxx]] with load[dtype=DType.xxx]() and store[dtype=DType.xxx]().
Key APIs:

free(): Frees memory previously allocated by alloc(). Do not call on pointers that were not allocated by alloc().
+ i / - i: Pointer arithmetic. Returns a new pointer shifted by i elements. No bounds checking.
[] or [i]: Dereference to a reference of the pointee (or at offset i). Only valid if the memory at that location is initialized.
load(): Loads width elements starting at offset (default 0) as SIMD[dtype, width] from UnsafePointer[Scalar[dtype]]. Pass alignment when data is not naturally aligned.
store(): Stores val: SIMD[dtype, width] at offset into UnsafePointer[Scalar[dtype]]. Requires a mutable pointer.
destroy_pointee() / take_pointee(): Explicitly end the lifetime of the current pointee, or move it out, taking ownership.
init_pointee_move() / init_pointee_move_from() / init_pointee_copy() Initialize a pointee that is currently uninitialized, by moving an existing value, moving from another pointee, or by copying an existing value. Use these to manage lifecycles when working with uninitialized memory.
For more information see Unsafe pointers in the Mojo Manual. For a comparison with other pointer types, see Intro to pointers.

Examples:

Element-wise store and load (width = 1):

var ptr = alloc[Float32](4)
for i in range(4):
    ptr.store(i, Float32(i))
var v = ptr.load(2)
print(v[0])  # => 2.0
ptr.free()

Vectorized store and load (width = 4):

var ptr = alloc[Int32](8)
var vec = SIMD[DType.int32, 4](1, 2, 3, 4)
ptr.store(0, vec)
var out = ptr.load[width=4](0)
print(out)  # => [1, 2, 3, 4]
ptr.free()

Pointer arithmetic and dereference:

var ptr = alloc[Int32](3)
(ptr + 0)[] = 10  # offset by 0 elements, then dereference to write
(ptr + 1)[] = 20  # offset +1 element, then dereference to write
ptr[2] = 30  # equivalent offset/dereference with brackets (via __getitem__)
var second = ptr[1]  # reads the element at index 1
print(second, ptr[2])  # => 20 30
ptr.free()

Point to a value on the stack:

var foo: Int = 123
var ptr = UnsafePointer(to=foo)
print(ptr[])  # => 123
# Don't call `free()` because the value was not heap-allocated
# Mojo will destroy it when the `foo` lifetime ends

Parameters
​mut (Bool): Whether the origin is mutable.
​type (AnyType): The type the pointer points to.
​origin (Origin): The origin of the memory being addressed.
​address_space (AddressSpace): The address space associated with the UnsafePointer allocated memory.
Fields
​address (__mlir_type.!kgen.pointer<:trait<@std::@builtin::@anytype::@AnyType> *"type", #lit.struct.extract<:!lit.struct<@std::@builtin::@int::@Int> #lit.struct.extract<:!lit.struct<@std::@memory::@pointer::@AddressSpace> address_space, "_value">, "_mlir_value">>``): The underlying pointer.
Implemented traits
AnyType, Boolable, Comparable, Copyable, Defaultable, DevicePassable, Equatable, ImplicitlyCopyable, ImplicitlyDestructible, Intable, Movable, Stringable, Writable

comptime members
__copyinit__is_trivial
comptime __copyinit__is_trivial = True

__del__is_trivial
comptime __del__is_trivial = True

__moveinit__is_trivial
comptime __moveinit__is_trivial = True

device_type
comptime device_type = UnsafePointer[type, origin, address_space=address_space]

DeviceBuffer dtypes are remapped to UnsafePointer when passed to accelerator devices.

Methods
__init__
__init__() -> Self

Create a null pointer.

__init__(*, unsafe_from_address: Int) -> Self

Create a pointer from a raw address.

Safety: Creating a pointer from a raw address is inherently unsafe as the caller must ensure the address is valid before writing to it, and that the memory is initialized before reading from it. The caller must also ensure the pointer's origin and mutability is valid for the address, failure to to do may result in undefined behavior.

Args:

​unsafe_from_address (Int): The raw address to create a pointer from.
__init__(*, ref [origin, address_space] to: type) -> Self

Constructs a Pointer from a reference to a value.

Args:

​to (type): The value to construct a pointer to.
@implicit
__init__[disambig2: Int = 0](other: UnsafePointer[type, origin, address_space=address_space]) -> UnsafePointer[type, origin_of((muttoimm origin._mlir_origin)), address_space=address_space]

Implicitly casts a mutable pointer to immutable.

Parameters:

​disambig2 (Int): Ignored. Works around name mangling conflict.
Args:

​other (UnsafePointer): The mutable pointer to cast from.
Returns:

UnsafePointer

@implicit
__init__[disambig: Int = 0](other: UnsafePointer[type, origin, address_space=address_space]) -> UnsafePointer[type, MutAnyOrigin, address_space=address_space]

Implicitly casts a mutable pointer to MutAnyOrigin.

Parameters:

​disambig (Int): Ignored. Works around name mangling conflict.
Args:

​other (UnsafePointer): The mutable pointer to cast from.
Returns:

UnsafePointer

@implicit
__init__(other: UnsafePointer[type, origin, address_space=address_space]) -> UnsafePointer[type, ImmutAnyOrigin, address_space=address_space]

Implicitly casts a pointer to ImmutAnyOrigin.

Args:

​other (UnsafePointer): The pointer to cast from.
Returns:

UnsafePointer

__init__[T: ImplicitlyDestructible, //](*, ref [origin] unchecked_downcast_value: PythonObject) -> UnsafePointer[T, origin]

Downcast a PythonObject known to contain a Mojo object to a pointer.

This operation is only valid if the provided Python object contains an initialized Mojo object of matching type.

Parameters:

​T (ImplicitlyDestructible): Pointee type that can be destroyed implicitly (without deinitializer arguments).
Args:

​unchecked_downcast_value (PythonObject): The Python object to downcast from.
Returns:

UnsafePointer

@implicit
__init__(other: LegacyUnsafePointer[type, address_space=address_space, origin=origin]) -> Self

Cast a LegacyUnsafePointer to an UnsafePointer.

Notes: This constructor will be removed in a future version of Mojo when LegacyUnsafePointer is removed.

Args:

​other (LegacyUnsafePointer): The LegacyUnsafePointer to cast from.
Returns:

Self: An UnsafePointer with the same type, mutability, origin and address space as the original LegacyUnsafePointer.

@implicit
__init__(other: LegacyUnsafePointer[type, address_space=address_space, origin=origin]) -> UnsafePointer[type, origin_of((muttoimm origin._mlir_origin)), address_space=address_space]

Cast a LegacyUnsafePointer to an immutable UnsafePointer.

Notes: This constructor will be removed in a future version of Mojo when LegacyUnsafePointer is removed.

Args:

​other (LegacyUnsafePointer): The LegacyUnsafePointer to cast from.
Returns:

UnsafePointer: An UnsafePointer with the same type, origin and address space as the original LegacyUnsafePointer but immutable.

__bool__
__bool__(self) -> Bool

Return true if the pointer is non-null.

Returns:

Bool: Whether the pointer is null.

__getitem__
__getitem__(self) -> ref [origin, address_space] type

Return a reference to the underlying data.

Safety: The pointer must not be null and must point to initialized memory.

Returns:

ref: A reference to the value.

__getitem__[I: Indexer, //](self, offset: I) -> ref [origin, address_space] type

Return a reference to the underlying data, offset by the given index.

Parameters:

​I (Indexer): A type that can be used as an index.
Args:

​offset (I): The offset index.
Returns:

ref: An offset reference.

__lt__
__lt__(self, rhs: UnsafePointer[type, origin, address_space=address_space]) -> Bool

Returns True if this pointer represents a lower address than rhs.

Args:

​rhs (UnsafePointer): The value of the other pointer.
Returns:

Bool: True if this pointer represents a lower address and False otherwise.

__lt__(self, rhs: Self) -> Bool

Returns True if this pointer represents a lower address than rhs.

Args:

​rhs (Self): The value of the other pointer.
Returns:

Bool: True if this pointer represents a lower address and False otherwise.

__le__
__le__(self, rhs: UnsafePointer[type, origin, address_space=address_space]) -> Bool

Returns True if this pointer represents a lower than or equal address than rhs.

Args:

​rhs (UnsafePointer): The value of the other pointer.
Returns:

Bool: True if this pointer represents a lower address and False otherwise.

__le__(self, rhs: Self) -> Bool

Returns True if this pointer represents a lower than or equal address than rhs.

Args:

​rhs (Self): The value of the other pointer.
Returns:

Bool: True if this pointer represents a lower address and False otherwise.

__eq__
__eq__(self, rhs: UnsafePointer[type, origin, address_space=address_space]) -> Bool

Returns True if the two pointers are equal.

Args:

​rhs (UnsafePointer): The value of the other pointer.
Returns:

Bool: True if the two pointers are equal and False otherwise.

__eq__(self, rhs: Self) -> Bool

Returns True if the two pointers are equal.

Args:

​rhs (Self): The value of the other pointer.
Returns:

Bool: True if the two pointers are equal and False otherwise.

__ne__
__ne__(self, rhs: UnsafePointer[type, origin, address_space=address_space]) -> Bool

Returns True if the two pointers are not equal.

Args:

​rhs (UnsafePointer): The value of the other pointer.
Returns:

Bool: True if the two pointers are not equal and False otherwise.

__ne__(self, rhs: Self) -> Bool

Returns True if the two pointers are not equal.

Args:

​rhs (Self): The value of the other pointer.
Returns:

Bool: True if the two pointers are not equal and False otherwise.

__gt__
__gt__(self, rhs: UnsafePointer[type, origin, address_space=address_space]) -> Bool

Returns True if this pointer represents a higher address than rhs.

Args:

​rhs (UnsafePointer): The value of the other pointer.
Returns:

Bool: True if this pointer represents a higher than or equal address and False otherwise.

__gt__(self, rhs: Self) -> Bool

Returns True if this pointer represents a higher address than rhs.

Args:

​rhs (Self): The value of the other pointer.
Returns:

Bool: True if this pointer represents a higher than or equal address and False otherwise.

__ge__
__ge__(self, rhs: UnsafePointer[type, origin, address_space=address_space]) -> Bool

Returns True if this pointer represents a higher than or equal address than rhs.

Args:

​rhs (UnsafePointer): The value of the other pointer.
Returns:

Bool: True if this pointer represents a higher than or equal address and False otherwise.

__ge__(self, rhs: Self) -> Bool

Returns True if this pointer represents a higher than or equal address than rhs.

Args:

​rhs (Self): The value of the other pointer.
Returns:

Bool: True if this pointer represents a higher than or equal address and False otherwise.

__add__
__add__[I: Indexer, //](self, offset: I) -> Self

Return a pointer at an offset from the current one.

Parameters:

​I (Indexer): A type that can be used as an index.
Args:

​offset (I): The offset index.
Returns:

Self: An offset pointer.

__sub__
__sub__[I: Indexer, //](self, offset: I) -> Self

Return a pointer at an offset from the current one.

Parameters:

​I (Indexer): A type that can be used as an index.
Args:

​offset (I): The offset index.
Returns:

Self: An offset pointer.

__iadd__
__iadd__[I: Indexer, //](mut self, offset: I)

Add an offset to this pointer.

Parameters:

​I (Indexer): A type that can be used as an index.
Args:

​offset (I): The offset index.
__isub__
__isub__[I: Indexer, //](mut self, offset: I)

Subtract an offset from this pointer.

Parameters:

​I (Indexer): A type that can be used as an index.
Args:

​offset (I): The offset index.
as_legacy_pointer
as_legacy_pointer(self) -> LegacyUnsafePointer[type, address_space=address_space, origin=origin]

Explicitly cast this pointer to a LegacyUnsafePointer.

Notes: This function will eventually be deprecated and removed in a future version of Mojo.

Returns:

LegacyUnsafePointer: A LegacyUnsafePointer with the same type, mutability, origin, and address space as the original pointer.

offset
offset[I: Indexer, //](self, idx: I) -> Self

Returns a new pointer shifted by the specified offset.

Parameters:

​I (Indexer): A type that can be used as an index.
Args:

​idx (I): The offset of the new pointer.
Returns:

Self: The offset pointer.

__merge_with__
__merge_with__[other_type: AnyStruct[UnsafePointer[type, origin, address_space=address_space]]](self) -> UnsafePointer[type, origin_of((mutcast origin._mlir_origin), (mutcast origin._mlir_origin)), address_space=address_space]

Returns a pointer merged with the specified other_type.

Parameters:

​other_type (AnyStruct): The type of the pointer to merge with.
Returns:

UnsafePointer: A pointer merged with the specified other_type.

__int__
__int__(self) -> Int

Returns the pointer address as an integer.

Returns:

Int: The address of the pointer as an Int.

__str__
__str__(self) -> String

Gets a string representation of the pointer.

Returns:

String: The string representation of the pointer.

write_to
write_to(self, mut writer: T)

Formats this pointer address to the provided Writer.

Args:

​writer (T): The object to write to.
get_type_name
static get_type_name() -> String

Gets this type name, for use in error messages when handing arguments to kernels. TODO: This will go away soon, when we get better error messages for kernel calls.

Returns:

String: This name of the type.

get_device_type_name
static get_device_type_name() -> String

Gets device_type's name.

Returns:

String: The device type's name.

swap_pointees
swap_pointees[U: Movable, //](self: UnsafePointer[U, origin], other: UnsafePointer[U, origin])

Swap the values at the pointers.

This function assumes that self and other may overlap in memory. If that is not the case, or when references are available, you should use builtin.swap instead.

Safety:

self and other must both point to valid, initialized instances of T.
Parameters:

​U (Movable): The type the pointers point to, which must be Movable.
Args:

​other (UnsafePointer): The other pointer to swap with.
as_noalias_ptr
as_noalias_ptr(self) -> Self

Cast the pointer to a new pointer that is known not to locally alias any other pointer. In other words, the pointer transitively does not comptime any other memory value declared in the local function context.

This information is relayed to the optimizer. If the pointer does locally alias another memory value, the behaviour is undefined.

Returns:

Self: A noalias pointer.

load
load[dtype: DType, //, width: Int = 1, *, alignment: Int = align_of[dtype](), volatile: Bool = False, invariant: Bool = _default_invariant[mut]()](self: UnsafePointer[Scalar[dtype], origin, address_space=address_space]) -> SIMD[dtype, width]

Loads width elements from the value the pointer points to.

Use alignment to specify minimal known alignment in bytes; pass a smaller value (such as 1) if loading from packed/unaligned memory. The volatile/invariant flags control reordering and common-subexpression elimination semantics for special cases.

Example:

var p = alloc[Int32](8)
p.store(0, SIMD[DType.int32, 4](1, 2, 3, 4))
var v = p.load[width=4]()
print(v)  # => [1, 2, 3, 4]
p.free()

Constraints:

The width and alignment must be positive integer values.

Parameters:

​dtype (DType): The data type of the SIMD vector.
​width (Int): The number of elements to load.
​alignment (Int): The minimal alignment (bytes) of the address.
​volatile (Bool): Whether the operation is volatile.
​invariant (Bool): Whether the load is from invariant memory.
Returns:

SIMD: The loaded SIMD vector.

load[dtype: DType, //, width: Int = 1, *, alignment: Int = align_of[dtype](), volatile: Bool = False, invariant: Bool = _default_invariant[mut]()](self: UnsafePointer[Scalar[dtype], origin, address_space=address_space], offset: Scalar[dtype]) -> SIMD[dtype, width]

Loads the value the pointer points to with the given offset.

Constraints:

The width and alignment must be positive integer values. The offset must be integer.

Parameters:

​dtype (DType): The data type of SIMD vector elements.
​width (Int): The size of the SIMD vector.
​alignment (Int): The minimal alignment of the address.
​volatile (Bool): Whether the operation is volatile or not.
​invariant (Bool): Whether the memory is load invariant.
Args:

​offset (Scalar): The offset to load from.
Returns:

SIMD: The loaded value.

load[I: Indexer, dtype: DType, //, width: Int = 1, *, alignment: Int = align_of[dtype](), volatile: Bool = False, invariant: Bool = _default_invariant[mut]()](self: UnsafePointer[Scalar[dtype], origin, address_space=address_space], offset: I) -> SIMD[dtype, width]

Loads the value the pointer points to with the given offset.

Constraints:

The width and alignment must be positive integer values.

Parameters:

​I (Indexer): A type that can be used as an index.
​dtype (DType): The data type of SIMD vector elements.
​width (Int): The size of the SIMD vector.
​alignment (Int): The minimal alignment of the address.
​volatile (Bool): Whether the operation is volatile or not.
​invariant (Bool): Whether the memory is load invariant.
Args:

​offset (I): The offset to load from.
Returns:

SIMD: The loaded value.

store
store[I: Indexer, dtype: DType, //, width: Int = 1, *, alignment: Int = align_of[dtype](), volatile: Bool = False](self: UnsafePointer[Scalar[dtype], origin, address_space=address_space], offset: I, val: SIMD[dtype, width])

Stores a single element value at the given offset.

Constraints:

The width and alignment must be positive integer values. The offset must be integer.

Parameters:

​I (Indexer): A type that can be used as an index.
​dtype (DType): The data type of SIMD vector elements.
​width (Int): The size of the SIMD vector.
​alignment (Int): The minimal alignment of the address.
​volatile (Bool): Whether the operation is volatile or not.
Args:

​offset (I): The offset to store to.
​val (SIMD): The value to store.
store[dtype: DType, offset_type: DType, //, width: Int = 1, *, alignment: Int = align_of[dtype](), volatile: Bool = False](self: UnsafePointer[Scalar[dtype], origin, address_space=address_space], offset: Scalar[offset_type], val: SIMD[dtype, width])

Stores a single element value at the given offset.

Constraints:

The width and alignment must be positive integer values.

Parameters:

​dtype (DType): The data type of SIMD vector elements.
​offset_type (DType): The data type of the offset value.
​width (Int): The size of the SIMD vector.
​alignment (Int): The minimal alignment of the address.
​volatile (Bool): Whether the operation is volatile or not.
Args:

​offset (Scalar): The offset to store to.
​val (SIMD): The value to store.
store[dtype: DType, //, width: Int = 1, *, alignment: Int = align_of[dtype](), volatile: Bool = False](self: UnsafePointer[Scalar[dtype], origin, address_space=address_space], val: SIMD[dtype, width])

Stores a single element value val at element offset 0.

Specify alignment when writing to packed/unaligned memory. Requires a mutable pointer. For writing at an element offset, use the overloads that accept an index or scalar offset.

Example:

var p = alloc[Float32](4)
var vec = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
p.store(vec)
var out = p.load[width=4]()
print(out)  # => [1.0, 2.0, 3.0, 4.0]
p.free()

Constraints:

The width and alignment must be positive integer values.

Parameters:

​dtype (DType): The data type of SIMD vector elements.
​width (Int): The number of elements to store.
​alignment (Int): The minimal alignment (bytes) of the address.
​volatile (Bool): Whether the operation is volatile.
Args:

​val (SIMD): The SIMD value to store.
strided_load
strided_load[dtype: DType, T: Intable, //, width: Int](self: UnsafePointer[Scalar[dtype], origin, address_space=address_space], stride: T) -> SIMD[dtype, width]

Performs a strided load of the SIMD vector.

Parameters:

​dtype (DType): DType of returned SIMD value.
​T (Intable): The Intable type of the stride.
​width (Int): The SIMD width.
Args:

​stride (T): The stride between loads.
Returns:

SIMD: A vector which is stride loaded.

strided_store
strided_store[dtype: DType, T: Intable, //, width: Int = 1](self: UnsafePointer[Scalar[dtype], origin, address_space=address_space], val: SIMD[dtype, width], stride: T)

Performs a strided store of the SIMD vector.

Parameters:

​dtype (DType): DType of val, the SIMD value to store.
​T (Intable): The Intable type of the stride.
​width (Int): The SIMD width.
Args:

​val (SIMD): The SIMD value to store.
​stride (T): The stride between stores.
gather
gather[dtype: DType, //, *, width: Int = 1, alignment: Int = align_of[dtype]()](self: UnsafePointer[Scalar[dtype], origin, address_space=address_space], offset: SIMD[dtype, width], mask: SIMD[DType.bool, width] = SIMD[DType.bool, width](True), default: SIMD[dtype, width] = 0) -> SIMD[dtype, width]

Gathers a SIMD vector from offsets of the current pointer.

This method loads from memory addresses calculated by appropriately shifting the current pointer according to the offset SIMD vector, or takes from the default SIMD vector, depending on the values of the mask SIMD vector.

If a mask element is True, the respective result element is given by the current pointer and the offset SIMD vector; otherwise, the result element is taken from the default SIMD vector.

Constraints:

The offset type must be an integral type. The alignment must be a power of two integer value.

Parameters:

​dtype (DType): DType of the return SIMD.
​width (Int): The SIMD width.
​alignment (Int): The minimal alignment of the address.
Args:

​offset (SIMD): The SIMD vector of offsets to gather from.
​mask (SIMD): The SIMD vector of boolean values, indicating for each element whether to load from memory or to take from the default SIMD vector.
​default (SIMD): The SIMD vector providing default values to be taken where the mask SIMD vector is False.
Returns:

SIMD: The SIMD vector containing the gathered values.

scatter
scatter[dtype: DType, //, *, width: Int = 1, alignment: Int = align_of[dtype]()](self: UnsafePointer[Scalar[dtype], origin, address_space=address_space], offset: SIMD[dtype, width], val: SIMD[dtype, width], mask: SIMD[DType.bool, width] = SIMD[DType.bool, width](True))

Scatters a SIMD vector into offsets of the current pointer.

This method stores at memory addresses calculated by appropriately shifting the current pointer according to the offset SIMD vector, depending on the values of the mask SIMD vector.

If a mask element is True, the respective element in the val SIMD vector is stored at the memory address defined by the current pointer and the offset SIMD vector; otherwise, no action is taken for that element in val.

If the same offset is targeted multiple times, the values are stored in the order they appear in the val SIMD vector, from the first to the last element.

Constraints:

The offset type must be an integral type. The alignment must be a power of two integer value.

Parameters:

​dtype (DType): DType of value, the result SIMD buffer.
​width (Int): The SIMD width.
​alignment (Int): The minimal alignment of the address.
Args:

​offset (SIMD): The SIMD vector of offsets to scatter into.
​val (SIMD): The SIMD vector containing the values to be scattered.
​mask (SIMD): The SIMD vector of boolean values, indicating for each element whether to store at memory or not.
free
free(self: UnsafePointer[type, origin])

Free the memory referenced by the pointer.

bitcast
bitcast[T: AnyType](self) -> UnsafePointer[T, origin, address_space=address_space]

Bitcasts an UnsafePointer to a different type.

Parameters:

​T (AnyType): The target type.
Returns:

UnsafePointer: A new pointer object with the specified type and the same address, mutability, and origin as the original pointer.

mut_cast
mut_cast[target_mut: Bool](self) -> UnsafePointer[type, origin_of((mutcast origin._mlir_origin)), address_space=address_space]

Changes the mutability of a pointer.

This is a safe way to change the mutability of a pointer with an unbounded mutability. This function will emit a compile time error if you try to cast an immutable pointer to mutable.

Parameters:

​target_mut (Bool): Mutability of the destination pointer.
Returns:

UnsafePointer: A pointer with the same type, origin and address space as the original pointer, but with the newly specified mutability.

unsafe_mut_cast
unsafe_mut_cast[target_mut: Bool](self) -> UnsafePointer[type, origin_of((mutcast origin._mlir_origin)), address_space=address_space]

Changes the mutability of a pointer.

If you are unconditionally casting the mutability to False, use as_immutable instead. If you are casting to mutable or a parameterized mutability, prefer using the safe mut_cast method instead.

Safety: Casting the mutability of a pointer is inherently very unsafe. Improper usage can lead to undefined behavior. Consider restricting types to their proper mutability at the function signature level. For example, taking an MutUnsafePointer[T, ...] as an argument over an unbound UnsafePointer[T, ...] is preferred.

Parameters:

​target_mut (Bool): Mutability of the destination pointer.
Returns:

UnsafePointer: A pointer with the same type, origin and address space as the original pointer, but with the newly specified mutability.

unsafe_origin_cast
unsafe_origin_cast[target_origin: Origin[mut=mut]](self) -> UnsafePointer[type, target_origin, address_space=address_space]

Changes the origin of a pointer.

If you are unconditionally casting the origin to an AnyOrigin, use as_any_origin instead.

Safety: Casting the origin of a pointer is inherently very unsafe. Improper usage can lead to undefined behavior or unexpected variable destruction. Considering parameterizing the origin at the function level to avoid unnecessary casts.

Parameters:

​target_origin (Origin): Origin of the destination pointer.
Returns:

UnsafePointer: A pointer with the same type, mutability and address space as the original pointer, but with the newly specified origin.

as_immutable
as_immutable(self) -> UnsafePointer[type, origin_of((muttoimm origin._mlir_origin)), address_space=address_space]

Changes the mutability of a pointer to immutable.

Unlike unsafe_mut_cast, this function is always safe to use as casting from (im)mutable to immutable is always safe.

Returns:

UnsafePointer: A pointer with the mutability set to immutable.

as_any_origin
as_any_origin(self) -> UnsafePointer[type, ImmutAnyOrigin, address_space=address_space]

Casts the origin of an immutable pointer to ImmutAnyOrigin.

It is usually preferred to maintain concrete origin values instead of using ImmutAnyOrigin. However, if it is needed, keep in mind that ImmutAnyOrigin can alias any memory value, so Mojo's ASAP destruction will not apply during the lifetime of the pointer.

Returns:

UnsafePointer: A pointer with the origin set to ImmutAnyOrigin.

as_any_origin(self) -> UnsafePointer[type, MutAnyOrigin, address_space=address_space]

Casts the origin of a mutable pointer to MutAnyOrigin.

This requires the pointer to already be mutable as casting mutability is inherently very unsafe.

It is usually preferred to maintain concrete origin values instead of using MutAnyOrigin. However, if it is needed, keep in mind that MutAnyOrigin can alias any memory value, so Mojo's ASAP destruction will not apply during the lifetime of the pointer.

Returns:

UnsafePointer: A pointer with the origin set to MutAnyOrigin.

address_space_cast
address_space_cast[target_address_space: AddressSpace = address_space](self) -> UnsafePointer[type, origin, address_space=target_address_space]

Casts this pointer to a different address space.

Parameters:

​target_address_space (AddressSpace): The address space of the result.
Returns:

UnsafePointer: A new pointer object with the same type and the same address, as the original pointer and the new address space.

destroy_pointee
destroy_pointee[T: ImplicitlyDestructible, //](self: UnsafePointer[T, origin])

Destroy the pointed-to value.

The pointer must not be null, and the pointer memory location is assumed to contain a valid initialized instance of type. This is equivalent to _ = self.take_pointee() but doesn't require Movable and is more efficient because it doesn't invoke __moveinit__.

Parameters:

​T (ImplicitlyDestructible): Pointee type that can be destroyed implicitly (without deinitializer arguments).
destroy_pointee_with
destroy_pointee_with(self: UnsafePointer[type, origin], destroy_func: fn(var type) -> None)

Destroy the pointed-to value using a user-provided destructor function.

This can be used to destroy non-ImplicitlyDestructible values in-place without moving.

Args:

​destroy_func (fn(var type) -> None): A function that takes ownership of the pointee value for the purpose of deinitializing it.
take_pointee
take_pointee[T: Movable, //](self: UnsafePointer[T, origin]) -> T

Move the value at the pointer out, leaving it uninitialized.

The pointer must not be null, and the pointer memory location is assumed to contain a valid initialized instance of T.

This performs a consuming move, ending the origin of the value stored in this pointer memory location. Subsequent reads of this pointer are not valid. If a new valid value is stored using init_pointee_move(), then reading from this pointer becomes valid again.

Parameters:

​T (Movable): The type the pointer points to, which must be Movable.
Returns:

T: The value at the pointer.

init_pointee_move
init_pointee_move[T: Movable, //](self: UnsafePointer[T, origin], var value: T)

Emplace a new value into the pointer location, moving from value.

The pointer memory location is assumed to contain uninitialized data, and consequently the current contents of this pointer are not destructed before writing value. Similarly, ownership of value is logically transferred into the pointer location.

When compared to init_pointee_copy, this avoids an extra copy on the caller side when the value is an owned rvalue.

Parameters:

​T (Movable): The type the pointer points to, which must be Movable.
Args:

​value (T): The value to emplace.
init_pointee_copy
init_pointee_copy[T: Copyable, //](self: UnsafePointer[T, origin], value: T)

Emplace a copy of value into the pointer location.

The pointer memory location is assumed to contain uninitialized data, and consequently the current contents of this pointer are not destructed before writing value. Similarly, ownership of value is logically transferred into the pointer location.

When compared to init_pointee_move, this avoids an extra move on the callee side when the value must be copied.

Parameters:

​T (Copyable): The type the pointer points to, which must be Copyable.
Args:

​value (T): The value to emplace.
init_pointee_move_from
init_pointee_move_from[T: Movable, //](self: UnsafePointer[T, origin], src: UnsafePointer[T, origin])

Moves the value src points to into the memory location pointed to by self.

The self pointer memory location is assumed to contain uninitialized data prior to this assignment, and consequently the current contents of this pointer are not destructed before writing the value from the src pointer.

Ownership of the value is logically transferred from src into self's pointer location.

After this call, the src pointee value should be treated as uninitialized data. Subsequent reads of or destructor calls on the src pointee value are invalid, unless and until a new valid value has been moved into the src pointer's memory location using an init_pointee_*() operation.

This transfers the value out of src and into self using at most one __moveinit__() call.

Example:

var a_ptr = alloc[String](1)
var b_ptr = alloc[String](2)

# Initialize A pointee
a_ptr.init_pointee_move("foo")

# Perform the move
b_ptr.init_pointee_move_from(a_ptr)

# Clean up
b_ptr.destroy_pointee()
a_ptr.free()
b_ptr.free()

Safety:

self and src must be non-null
src must contain a valid, initialized instance of T
The pointee contents of self should be uninitialized. If self was previously written with a valid value, that value will be be overwritten and its destructor will NOT be run.
Parameters:

​T (Movable): The type the pointer points to, which must be Movable.
Args:

​src (UnsafePointer): Source pointer that the value will be moved from.