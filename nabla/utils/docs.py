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
Documentation utilities for Nabla.

This module provides decorators and utilities for controlling documentation generation.
"""

from functools import wraps
from typing import Any, Callable, TypeVar, Union

# Type variables for generic decorator support
F = TypeVar('F', bound=Callable[..., Any])
C = TypeVar('C', bound=type)


def nodoc(obj: Union[F, C]) -> Union[F, C]:
    """
    Decorator to mark functions, methods, or classes as non-documentable.
    
    This decorator adds a special attribute to the decorated object that can be
    checked by documentation generation tools to exclude it from generated docs.
    
    Args:
        obj: The function, method, or class to mark as non-documentable.
        
    Returns:
        The original object with the `_nabla_nodoc` attribute set to True.
        
    Examples:
        >>> @nodoc
        ... def internal_helper():
        ...     '''This function won't appear in generated docs.'''
        ...     pass
        
        >>> @nodoc
        ... class InternalClass:
        ...     '''This class won't appear in generated docs.'''
        ...     pass
    """
    # Mark the object as non-documentable
    obj._nabla_nodoc = True
    return obj


def is_documentable(obj: Any) -> bool:
    """
    Check if an object should be included in documentation.
    
    Args:
        obj: The object to check.
        
    Returns:
        False if the object is marked with @nodoc, True otherwise.
    """
    return not getattr(obj, '_nabla_nodoc', False)


def should_document(obj: Any, name: str) -> bool:
    """
    Determine if an object should be documented based on various criteria.
    
    This function checks:
    1. If the object is marked with @nodoc decorator
    2. If the name starts with underscore (private/internal)
    3. Other standard exclusion criteria
    
    Args:
        obj: The object to check.
        name: The name of the object.
        
    Returns:
        True if the object should be documented, False otherwise.
    """
    # Skip if marked with @nodoc
    if not is_documentable(obj):
        return False
    
    # Skip private/internal items (starting with _)
    if name.startswith('_'):
        return False
    
    # Skip if it's a module and doesn't have proper documentation
    if hasattr(obj, '__module__') and not hasattr(obj, '__doc__'):
        return False
    
    return True


# Alias for convenience
no_doc = nodoc
skip_doc = nodoc
