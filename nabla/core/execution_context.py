# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or beautiful, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Thread-safe execution context for model caching."""

import threading
from collections.abc import Callable
from typing import Optional

from max.engine import Model


class ThreadSafeExecutionContext:
    """Thread-safe wrapper around the global execution context dictionary."""

    def __init__(self) -> None:
        self._cache: dict[int, Model] = {}
        self._lock = threading.RLock()  # Using RLock to allow recursive locking

    def get(self, key: int) -> Optional[Model]:
        """Get a model from the cache. Returns None if not found."""
        with self._lock:
            return self._cache.get(key)

    def set(self, key: int, model: Model) -> None:
        """Set a model in the cache."""
        with self._lock:
            self._cache[key] = model

    def get_or_create(self, key: int, factory: Callable[[], Model]) -> Model:
        """
        Get a model from cache, or create it using the factory function if not found.
        This is thread-safe and ensures only one thread creates the model for a given key.
        """
        # First, try to get without holding lock for long
        with self._lock:
            if key in self._cache:
                return self._cache[key]

        # Model not found, need to create it
        # Use double-checked locking pattern
        with self._lock:
            # Check again in case another thread created it while we were waiting
            if key in self._cache:
                return self._cache[key]

            # Create the model (this might take time, but we need to hold the lock
            # to prevent multiple threads from creating the same model)
            model = factory()
            self._cache[key] = model
            return model

    def contains(self, key: int) -> bool:
        """Check if a key exists in the cache."""
        with self._lock:
            return key in self._cache

    def clear(self) -> None:
        """Clear the entire cache. Useful for testing or memory management."""
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        """Get the current size of the cache."""
        with self._lock:
            return len(self._cache)


# Global model cache with thread safety
global_execution_context = ThreadSafeExecutionContext()
