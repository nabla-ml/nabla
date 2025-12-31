#!/bin/bash
export PYTHONPATH=$(pwd)
python3 -m tests.unit.custom_ops.test_custom_sharding -v
