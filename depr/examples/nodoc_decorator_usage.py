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

"""
Example demonstrating how to use the @nodoc decorator in your Nabla code.

This file shows various ways to mark functions and classes as non-documentable.
"""

from nabla.utils.docs import nodoc

# === PUBLIC API (will be documented) ===


def public_function(x, y):
    """
    This is a public function that will appear in the documentation.

    Args:
        x: First input
        y: Second input

    Returns:
        The result of the operation
    """
    return _internal_helper(x, y)


class PublicClass:
    """
    This is a public class that will appear in the documentation.

    This class demonstrates the proper use of @nodoc for internal methods.
    """

    def __init__(self, value):
        """Initialize the class with a value."""
        self.value = value
        self._internal_state = self._setup_internal_state()

    def public_method(self):
        """This method will appear in the documentation."""
        return self._internal_processing()

    @nodoc
    def _internal_processing(self):
        """This method won't appear in docs due to @nodoc decorator."""
        return self.value * 2

    def _setup_internal_state(self):
        """This method won't appear in docs (starts with underscore)."""
        return {"initialized": True}


# === INTERNAL API (will NOT be documented) ===


@nodoc
def _internal_helper(x, y):
    """
    This is an internal helper function that won't appear in documentation.

    Even though it has a detailed docstring, the @nodoc decorator prevents
    it from being included in the generated documentation.
    """
    return x + y


@nodoc
class InternalUtilityClass:
    """
    This entire class is marked as internal and won't appear in documentation.

    This is useful for implementation details that users shouldn't need to know about.
    """

    def __init__(self):
        pass

    def utility_method(self):
        """Even public-looking methods in @nodoc classes are excluded."""
        pass


# === MIXED EXAMPLE ===


class MixedVisibilityClass:
    """
    This class demonstrates mixing public and internal methods.

    The class itself will be documented, but some methods will be excluded.
    """

    def public_api_method(self):
        """This method will be documented."""
        return self._call_internal_method()

    @nodoc
    def experimental_method(self):
        """
        This method is experimental and shouldn't be in public docs yet.

        Use @nodoc for methods that are:
        - Experimental or unstable
        - Internal implementation details
        - Debug/testing utilities
        - Deprecated functionality
        """
        pass

    def _call_internal_method(self):
        """This won't be documented (starts with underscore)."""
        return "internal result"


# === DECORATOR VARIATIONS ===

# You can also use the aliases
from nabla.utils.docs import no_doc, skip_doc


@no_doc
def another_internal_function():
    """Alternative decorator name."""
    pass


@skip_doc
def yet_another_internal_function():
    """Another alternative decorator name."""
    pass


# === EXAMPLE USAGE IN YOUR MODULES ===

"""
Here's how you might use @nodoc in your actual Nabla modules:

# In nabla/core/tensor.py
class Tensor:
    '''Main tensor class - will be documented.'''
    
    def __init__(self, data):
        '''Public constructor - will be documented.'''
        self.data = data
    
    def sum(self):
        '''Public method - will be documented.'''
        return self._internal_sum_impl()
    
    @nodoc
    def _internal_sum_impl(self):
        '''Implementation detail - won't be documented.'''
        # Complex implementation here
        pass
    
    @nodoc
    def _debug_info(self):
        '''Debug helper - won't be documented.'''
        return {"shape": self.shape, "dtype": self.dtype}

# In nabla/ops/binary.py
def add(x, y):
    '''Public add function - will be documented.'''
    return _optimized_add_kernel(x, y)

@nodoc
def _optimized_add_kernel(x, y):
    '''Internal kernel implementation - won't be documented.'''
    # Low-level implementation
    pass

@nodoc  
class _BinaryOpMetadata:
    '''Internal metadata class - won't be documented.'''
    pass
"""
