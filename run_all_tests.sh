#!/bin/bash
set -e

echo "Running all tests via pytest..."
# Run all tests in the verified directories
python3 -m pytest tests/unit/sharding/ tests/integration/with_sharding/ tests/integration/no_sharding/ -v

echo "All tests passed!"
