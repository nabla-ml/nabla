Mojo struct

Variant
struct Variant[*Ts: AnyType]

A union that can hold a runtime-variant value from a set of predefined types.

Variant is a discriminated union type, similar to std::variant in C++ or enum in Rust. It can store exactly one value that can be any of the specified types, determined at runtime.

The key feature is that the actual type stored in a Variant is determined at runtime, not compile time. This allows you to change what type a variant holds during program execution. Memory-wise, a variant only uses the space needed for the largest possible type plus a small discriminant field to track which type is currently active.

Tips:

use isa[T]() to check what type a variant is
use unsafe_take[T]() to take a value from the variant
use [T] to get a value out of a variant
This currently does an extra copy/move until we have origins
It also temporarily requires the value to be mutable
use set[T](var new_value: T) to reset the variant to a new value
use is_type_supported[T] to check if the variant permits the type T
Note: Currently, variant operations require the variant to be mutable (mut), even for read operations.

Example:

from utils import Variant
import random

comptime IntOrString = Variant[Int, String]

fn to_string(mut x: IntOrString) -> String:
    if x.isa[String]():
        return x[String]
    return String(x[Int])

var an_int = IntOrString(4)
var a_string = IntOrString("I'm a string!")
var who_knows = IntOrString(0)
# Randomly change who_knows to a string
random.seed()
if random.random_ui64(0, 1):
    who_knows.set[String]("I'm also a string!")

print(a_string[String])      # => I'm a string!
print(an_int[Int])           # => 4
print(to_string(who_knows))  # Either 0 or "I'm also a string!"

if who_knows.isa[String]():
    print("It's a String!")

Example usage for error handling:

comptime Result = Variant[String, Error]

fn process_data(data: String) -> Result:
    if len(data) == 0:
        return Result(Error("Empty data"))
    return Result(String("Processed: ", data))

var result = process_data("Hello")
if result.isa[String]():
    print("Success:", result[String])
else:
    print("Error:", result[Error])

Example usage in a List to create a heterogeneous list:

comptime MixedType = Variant[Int, Float64, String, Bool]

var mixed_list = List[MixedType]()
mixed_list.append(MixedType(42))
mixed_list.append(MixedType(3.14))
mixed_list.append(MixedType("hello"))
mixed_list.append(MixedType(True))

for item in mixed_list:
    if item.isa[String]():
        print("String:", item[String])
    elif item.isa[Int]():
        print("Integer:", item[Int])
    elif item.isa[Float64]():
        print("Float:", item[Float64])
    elif item.isa[Bool]():
        print("Boolean:", item[Bool])

Parameters
​*Ts (AnyType): The possible types that this variant can hold. All types must implement Copyable.
Implemented traits
AnyType, Copyable, ImplicitlyCopyable, ImplicitlyDestructible, Movable

comptime members
__copyinit__is_trivial
comptime __copyinit__is_trivial = _all_trivial_copyinit[Ts]()

__del__is_trivial
comptime __del__is_trivial = _all_trivial_del[Ts]()

__moveinit__is_trivial
comptime __moveinit__is_trivial = _all_trivial_moveinit[Ts]()

Methods
__init__
__init__(out self, *, unsafe_uninitialized: Tuple[])

Unsafely create an uninitialized Variant.

Args:

​unsafe_uninitialized (Tuple): Marker argument indicating this initializer is unsafe.
@implicit
__init__[T: Movable](out self, var value: T)

Create a variant with one of the types.

Parameters:

​T (Movable): The type to initialize the variant to. Generally this should be able to be inferred from the call type, eg. Variant[Int, String](4).
Args:

​value (T): The value to initialize the variant with.
__copyinit__
__copyinit__(out self, other: Self)

Creates a deep copy of an existing variant.

Args:

​other (Self): The variant to copy from.
__moveinit__
__moveinit__(out self, deinit other: Self)

Move initializer for the variant.

Args:

​other (Self): The variant to move.
__del__
__del__(deinit self)

Destroy the variant.

__getitem__
__getitem__[T: AnyType](ref self) -> ref [self] T

Get the value out of the variant as a type-checked type.

This explicitly check that your value is of that type! If you haven't verified the type correctness at runtime, the program will abort!

For now this has the limitations that it - requires the variant value to be mutable

Parameters:

​T (AnyType): The type of the value to get out.
Returns:

ref: A reference to the internal data.

take
take[T: Movable](deinit self) -> T

Take the current value of the variant with the provided type.

The caller takes ownership of the underlying value.

This explicitly check that your value is of that type! If you haven't verified the type correctness at runtime, the program will abort!

Parameters:

​T (Movable): The type to take out.
Returns:

T: The underlying data to be taken out as an owned value.

unsafe_take
unsafe_take[T: Movable](mut self) -> T

Unsafely take the current value of the variant with the provided type.

The caller takes ownership of the underlying value.

This doesn't explicitly check that your value is of that type! If you haven't verified the type correctness at runtime, you'll get a type that looks like your type, but has potentially unsafe and garbage member data.

Parameters:

​T (Movable): The type to take out.
Returns:

T: The underlying data to be taken out as an owned value.

replace
replace[Tin: Movable & ImplicitlyDestructible, Tout: Movable](mut self, var value: Tin) -> Tout

Replace the current value of the variant with the provided type.

The caller takes ownership of the underlying value.

This explicitly check that your value is of that type! If you haven't verified the type correctness at runtime, the program will abort!

Parameters:

​Tin (Movable & ImplicitlyDestructible): The type to put in.
​Tout (Movable): The type to take out.
Args:

​value (Tin): The value to put in.
Returns:

Tout: The underlying data to be taken out as an owned value.

unsafe_replace
unsafe_replace[Tin: Movable, Tout: Movable](mut self, var value: Tin) -> Tout

Unsafely replace the current value of the variant with the provided type.

The caller takes ownership of the underlying value.

This doesn't explicitly check that your value is of that type! If you haven't verified the type correctness at runtime, you'll get a type that looks like your type, but has potentially unsafe and garbage member data.

Parameters:

​Tin (Movable): The type to put in.
​Tout (Movable): The type to take out.
Args:

​value (Tin): The value to put in.
Returns:

Tout: The underlying data to be taken out as an owned value.

set
set[T: Movable](mut self, var value: T)

Set the variant value.

This will call the destructor on the old value, and update the variant's internal type and data to the new value.

Parameters:

​T (Movable): The new variant type. Must be one of the Variant's type arguments.
Args:

​value (T): The new value to set the variant to.
isa
isa[T: AnyType](self) -> Bool

Check if the variant contains the required type.

Parameters:

​T (AnyType): The type to check.
Returns:

Bool: True if the variant contains the requested type.

unsafe_get
unsafe_get[T: AnyType](ref self) -> ref [self] T

Get the value out of the variant as a type-checked type.

This doesn't explicitly check that your value is of that type! If you haven't verified the type correctness at runtime, you'll get a type that looks like your type, but has potentially unsafe and garbage member data.

For now this has the limitations that it - requires the variant value to be mutable

Parameters:

​T (AnyType): The type of the value to get out.
Returns:

ref: The internal data represented as a Pointer[T].

is_type_supported
static is_type_supported[T: AnyType]() -> Bool

Check if a type can be used by the Variant.

Example:

from utils import Variant

def takes_variant(mut arg: Variant):
    if arg.is_type_supported[Float64]():
        arg = Float64(1.5)

def main():
    var x = Variant[Int, Float64](1)
    takes_variant(x)
    if x.isa[Float64]():
        print(x[Float64]) # 1.5

For example, the Variant[Int, Bool] permits Int and Bool.

Parameters:

​T (AnyType): The type of the value to check support for.
Returns:

Bool: True if type T is supported by the Variant.

destroy_with
destroy_with[T: AnyType](deinit self, destroy_func: fn(var T) -> None)

Destroy a value contained in this Variant in-place using a caller provided destructor function.

This method can be used to destroy linear types in a Variant in-place, without requiring that they be Movable.

This method will abort if this variant does not current contain an element of the specified type T.

Parameters:

​T (AnyType): The element type the variant is expected to currently contain, and which will be destroyed by destroy_func.
Args:

​destroy_func (fn(var T) -> None): Caller-provided destructor function for destroying an instance of T.