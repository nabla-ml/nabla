#!/bin/bash
export PYTHONPATH=$(pwd)
echo "Running all tests individually..."
find tests -name "test_*.py" | while read -r test_file; do
    # Convert path to module name: remove .py, replace / with .
    module_name=$(echo "$test_file" | sed 's/\.py$//' | sed 's/\//./g')
    echo "Running module: $module_name"
    python3 -m "$module_name" -v
    if [ $? -ne 0 ]; then
        echo "Tests failed in $module_name"
        exit 1
    fi
done
echo "All tests passed!"
