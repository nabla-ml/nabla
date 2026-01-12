Mojo struct

Dict
struct Dict[K: KeyElement, V: Copyable & ImplicitlyDestructible, H: Hasher = default_hasher]

A container that stores key-value pairs.

The Dict type is Mojo's primary associative collection, similar to Python's dict (dictionary). Unlike a List, which stores elements by index, a Dict stores values associated with unique keys, which enables fast lookups, insertions, and deletions.

You can create a Dict in several ways:

# Empty dictionary
var empty_dict = Dict[String, Int]()

# Dictionary literal syntax
var scores = {"Alice": 95, "Bob": 87, "Charlie": 92}

# Pre-allocated capacity (must be power of 2, >= 8)
var large_dict = Dict[String, Int](power_of_two_initial_capacity=64)

# From separate key and value lists
var keys = ["red", "green", "blue"]
var values = [255, 128, 64]
var colors = Dict[String, Int]()
for key, value in zip(keys, values):
    colors[String(key)] = value # cast list iterator to key-type

Be aware of the following characteristics:

Type safety: Both keys and values must be homogeneous types, determined at compile time. This is more restrictive than Python dictionaries but provides better performance:

var string_to_int = {"count": 42}     # Dict[String, Int]
var int_to_string = {1: "one"}        # Dict[Int, String]
var mixed = {"key": 1, 2: "val"}      # Error! Keys must be same type

However, you can get around this by defining your dictionary key and/or value type as Variant. This is a discriminated union type, meaning it can store any number of different types that can vary at runtime.

Value semantics: A Dict is value semantic by default. Copying a Dict creates a deep copy of all key-value pairs. To avoid accidental copies, Dict is not implicitly copyable—you must explicitly copy it using the .copy() method.

var dict1 = {"a": 1, "b": 2}
# var dict2 = dict1  # Error: Dict is not implicitly copyable
var dict2 = dict1.copy()  # Deep copy
dict2["c"] = 3
print(dict1.__str__())   # => {"a": 1, "b": 2}
print(dict2.__str__())   # => {"a": 1, "b": 2, "c": 3}

This is different from Python, where assignment creates a reference to the same dictionary. For more information, read about value semantics.

Iteration uses immutable references: When iterating over keys, values, or items, you get immutable references unless you specify ref or var:

var inventory = {"apples": 10, "bananas": 5}

# Default behavior creates immutable (read-only) references
for value in inventory.values():
    value += 1  # error: expression must be mutable

# Using `ref` gets mutable (read-write) references
for ref value in inventory.values():
    value += 1  # Modify inventory values in-place
print(inventory.__str__())  # => {"apples": 11, "bananas": 6}

# Using `var` gets an owned copy of the value
for var key in inventory.keys():
    inventory[key] += 1  # Modify inventory values in-place
print(inventory.__str__())  # => {"apples": 12, "bananas": 7}

Note that indexing into a Dict with a key that's a reference to the key owned by the Dict produces a confusing error related to argument exclusivity. Using var key in the previous example creates an owned copy of the key, avoiding the error.

KeyError handling: Directly accessing values with the [] operator will raise DictKeyError if the key is not found:

var phonebook = {"Alice": "555-0101", "Bob": "555-0102"}
print(phonebook["Charlie"])  # => DictKeyError

For safe access, you should instead use get():

var phonebook = {"Alice": "555-0101", "Bob": "555-0102"}
var phone = phonebook.get("Charlie")
print(phone.__str__()) if phone else print('phone not found')

Examples:

var phonebook = {"Alice": "555-0101", "Bob": "555-0102"}

# Add/update entries
phonebook["Charlie"] = "555-0103"    # Add new entry
phonebook["Alice"] = "555-0199"      # Update existing entry

# Access directly (unsafe and raises DictKeyError if key not found)
print(phonebook["Alice"])            # => 555-0199

# Access safely
var phone = phonebook.get("David")   # Returns Optional type
print(phone.or_else("phone not found!"))

# Access safely with default value
phone = phonebook.get("David", "555-0000")
print(phone.__str__())               # => '555-0000'

# Check for keys
if "Bob" in phonebook:
    print("Found Bob")

# Remove (pop) entries
print(phonebook.pop("Charlie"))         # Remove and return: "555-0103"
print(phonebook.pop("Unknown", "N/A"))  # Pop with default

# Iterate over a dictionary
for key in phonebook.keys():
    print("Key:", key)

for value in phonebook.values():
    print("Value:", value)

for item in phonebook.items():
    print(item.key, "=>", item.value)

for var key in phonebook:
    print(key, "=>", phonebook[key])

# Number of key-value pairs
print('len:', len(phonebook))        # => len: 2

# Dictionary operations
var backup = phonebook.copy()        # Explicit copy
phonebook.clear()                    # Remove all entries

# Merge dictionaries
var more_numbers = {"David": "555-0104", "Eve": "555-0105"}
backup.update(more_numbers)          # Merge in-place
var combined = backup | more_numbers # Create new merged dict
print(combined.__str__())

Parameters
​K (KeyElement): The type of keys stored in the dictionary.
​V (Copyable & ImplicitlyDestructible): The type of values stored in the dictionary.
​H (Hasher): The type of hasher used to hash the keys.
Implemented traits
AnyType, Boolable, Copyable, Defaultable, ImplicitlyDestructible, Iterable, Movable, Representable, Sized, Stringable, Writable

comptime members
__copyinit__is_trivial
comptime __copyinit__is_trivial = False

__del__is_trivial
comptime __del__is_trivial = False

__moveinit__is_trivial
comptime __moveinit__is_trivial = True

EMPTY
comptime EMPTY = -1

Marker for an empty slot in the hash table.

IteratorType
comptime IteratorType[iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]] = _DictKeyIter[K, V, H, iterable_origin]

The iterator type for this dictionary.

Parameters
​iterable_mut (Bool): Whether the iterable is mutable.
​iterable_origin (Origin): The origin of the iterable.
REMOVED
comptime REMOVED = -2

Marker for a removed slot in the hash table.

Methods
__init__
__init__(out self)

Initialize an empty dictiontary.

__init__(out self, *, power_of_two_initial_capacity: Int)

Initialize an empty dictiontary with a pre-reserved initial capacity.

Examples:

var x = Dict[Int, Int](power_of_two_initial_capacity = 1024)
# Insert (2/3 of 1024) entries without reallocation.

Args:

​power_of_two_initial_capacity (Int): At least 8, has to be a power of two.
__init__(out self, var keys: List[K], var values: List[V], __dict_literal__: Tuple[])

Constructs a dictionary from the given keys and values.

Args:

​keys (List): The list of keys to build the dictionary with.
​values (List): The corresponding values to pair with the keys.
​dict_literal (Tuple): Tell Mojo to use this method for dict literals.
__copyinit__
__copyinit__(out self, existing: Self)

Copy an existing dictiontary.

Args:

​existing (Self): The existing dict.
__bool__
__bool__(self) -> Bool

Check if the dictionary is empty or not.

Returns:

Bool: False if the dictionary is empty, True if there is at least one element.

__getitem__
__getitem__(ref self, ref key: K) -> ref [origin_of($1._entries._value.value)] V

Retrieve a value out of the dictionary.

Args:

​key (K): The key to retrieve.
Returns:

ref: The value associated with the key, if it's present.

Raises:

DictKeyError if the key isn't present.

__setitem__
__setitem__(mut self, var key: K, var value: V)

Set a value in the dictionary by key.

Args:

​key (K): The key to associate with the specified value.
​value (V): The data to store in the dictionary.
__contains__
__contains__(self, key: K) -> Bool

Check if a given key is in the dictionary or not.

Args:

​key (K): The key to check.
Returns:

Bool: True if the key exists in the dictionary, False otherwise.

__or__
__or__(self, other: Self) -> Self

Merge self with other and return the result as a new dict.

Args:

​other (Self): The dictionary to merge with.
Returns:

Self: The result of the merge.

__ior__
__ior__(mut self, other: Self)

Merge self with other in place.

Args:

​other (Self): The dictionary to merge with.
fromkeys
static fromkeys(keys: List[K], value: V) -> Self

Create a new dictionary with keys from list and values set to value.

Args:

​keys (List): The keys to set.
​value (V): The value to set.
Returns:

Self: The new dictionary.

static fromkeys(keys: List[K], value: Optional[V] = None) -> Dict[K, Optional[V], H]

Create a new dictionary with keys from list and values set to value.

Args:

​keys (List): The keys to set.
​value (Optional): The value to set.
Returns:

Dict: The new dictionary.

__iter__
__iter__(ref self) -> _DictKeyIter[K, V, H, self_is_origin]

Iterate over the dict's keys as immutable references.

Returns:

_DictKeyIter: An iterator of immutable references to the dictionary keys.

__reversed__
__reversed__(ref self) -> _DictKeyIter[K, V, H, self_is_origin, False]

Iterate backwards over the dict keys, returning immutable references.

Returns:

_DictKeyIter: A reversed iterator of immutable references to the dict keys.

__len__
__len__(self) -> Int

The number of elements currently stored in the dictionary.

Returns:

Int: The number of elements currently stored in the dictionary.

__repr__
__repr__(self) -> String

Returns a string representation of a Dict.

Returns:

String: A string representation of the Dict.

__str__
__str__(self) -> String

Returns a string representation of a Dict.

Examples:

var my_dict = Dict[Int, Float64]()
my_dict[1] = 1.1
my_dict[2] = 2.2
dict_as_string = String(my_dict)
print(dict_as_string)
# prints "{1: 1.1, 2: 2.2}"

Returns:

String: A string representation of the Dict.

write_to
write_to(self, mut writer: T)

Write my_list.__str__() to a Writer.

Constraints:

K must conform to Representable. V must conform to Representable.

Args:

​writer (T): The object to write to.
find
find(self, key: K) -> Optional[V]

Find a value in the dictionary by key.

Args:

​key (K): The key to search for in the dictionary.
Returns:

Optional: An optional value containing a copy of the value if it was present, otherwise an empty Optional.

get
get(self, key: K) -> Optional[V]

Get a value from the dictionary by key.

Args:

​key (K): The key to search for in the dictionary.
Returns:

Optional: An optional value containing a copy of the value if it was present, otherwise an empty Optional.

get(self, key: K, var default: V) -> V

Get a value from the dictionary by key.

Args:

​key (K): The key to search for in the dictionary.
​default (V): Default value to return.
Returns:

V: A copy of the value if it was present, otherwise default.

pop
pop(mut self, key: K, var default: V) -> V

Remove a value from the dictionary by key.

Args:

​key (K): The key to remove from the dictionary.
​default (V): A default value to return if the key was not found instead of raising.
Returns:

V: The value associated with the key, if it was in the dictionary. If it wasn't, return the provided default value instead.

pop(mut self, ref key: K) -> V

Remove a value from the dictionary by key.

Args:

​key (K): The key to remove from the dictionary.
Returns:

V: The value associated with the key, if it was in the dictionary. Raises otherwise.

Raises:

DictKeyError if the key was not present in the dictionary.

popitem
popitem(mut self) -> DictEntry[K, V, H]

Remove and return a (key, value) pair from the dictionary.

Notes: Pairs are returned in LIFO order. popitem() is useful to destructively iterate over a dictionary, as often used in set algorithms. If the dictionary is empty, calling popitem() raises a EmptyDictError.

Returns:

DictEntry: Last dictionary item

Raises:

EmptyDictError if the dictionary is empty.

keys
keys(ref self) -> _DictKeyIter[K, V, H, self_is_origin]

Iterate over the dict's keys as immutable references.

Returns:

_DictKeyIter: An iterator of immutable references to the dictionary keys.

values
values(ref self) -> _DictValueIter[K, V, H, self_is_origin]

Iterate over the dict's values as references.

Returns:

_DictValueIter: An iterator of references to the dictionary values.

items
items(ref self) -> _DictEntryIter[K, V, H, self_is_origin]

Iterate over the dict's entries as immutable references.

Examples:

var my_dict = Dict[String, Int]()
my_dict["a"] = 1
my_dict["b"] = 2

for e in my_dict.items():
    print(e.key, e.value)

Notes: These can't yet be unpacked like Python dict items, but you can access the key and value as attributes.

Returns:

_DictEntryIter: An iterator of immutable references to the dictionary entries.

take_items
take_items(mut self) -> _TakeDictEntryIter[K, V, H, self]

Iterate over the dict's entries and move them out of the dictionary effectively draining the dictionary.

Examples:

var my_dict = Dict[String, Int]()
my_dict["a"] = 1
my_dict["b"] = 2

for entry in my_dict.take_items():
    print(entry.key, entry.value)

print(len(my_dict))
# prints 0

Returns:

_TakeDictEntryIter: An iterator of mutable references to the dictionary entries that moves them out of the dictionary.

update
update(mut self, other: Self, /)

Update the dictionary with the key/value pairs from other, overwriting existing keys.

Notes: The argument must be positional only.

Args:

​other (Self): The dictionary to update from.
clear
clear(mut self)

Remove all elements from the dictionary.

setdefault
setdefault(mut self, key: K, var default: V) -> ref [origin_of(*[0,0]._entries._value.value)] V

Get a value from the dictionary by key, or set it to a default if it doesn't exist.

Args:

​key (K): The key to search for in the dictionary.
​default (V): The default value to set if the key is not present.
Returns:

ref: The value associated with the key, or the default value if it wasn't present.