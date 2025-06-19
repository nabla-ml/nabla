# Binary Operations Test Suite

This test suite validates Nabla's binary operations against JAX across **11 operations × 19 transformations × 10 rank combinations = 2,090 total tests**.

## File Structure

```bash
tests/unit/
├── test_binary_ops.py          # Main test suite
├── test_utils.py               # Reusable test utilities
├── test_errors.py              # Error handling and formatting
└── README.md                   # This file
```

## Basic Commands

### 1. Test Single Operation (Default Ranks)

```bash
# Test one operation with default ranks (2,2) - fastest
python tests/unit/test_binary_ops.py add

# Or with pytest
pytest tests/unit/test_binary_ops.py -k "add" --tb=short --maxfail=3 -v
```

### 2. Test Single Operation (Specific Ranks)

```bash
# Test one operation with all rank combinations
python tests/unit/test_binary_ops.py add --all-ranks

# Or test specific rank combinations with pytest
pytest tests/unit/test_binary_ops.py -k "add and rank_combination0" -v  # scalar + scalar
pytest tests/unit/test_binary_ops.py -k "add and rank_combination4" -v  # scalar + vector
```

### 3. Test All Operations (Default Ranks)

```bash
# Test all operations with default ranks - quick smoke test
pytest tests/unit/test_binary_ops.py::test_all_operations_default_ranks -v

# Or with original runner
python tests/unit/test_binary_ops.py all
```

### 4. Test All Operations (All Ranks)

```bash
# Full comprehensive test - takes ~10 minutes
pytest tests/unit/test_binary_ops.py -m benchmark -v -s

# Or with original runner
python tests/unit/test_binary_ops.py all --all-ranks
```

## Rank Combinations

- **0**: Scalar `()`
- **1**: Vector `(4,)`  
- **2**: Matrix `(2,3)`
- **3**: Tensor `(2,2,3)`

**Combinations tested**:

- Same ranks: (0,0), (1,1), (2,2), (3,3)
- Broadcasting: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)

## Operations Tested

`add`, `mul`, `sub`, `div`, `floordiv`, `pow`, `greater_equal`, `equal`, `not_equal`, `maximum`, `minimum`

## Progress Monitoring

```bash
# See live output during long tests
pytest -v -s --tb=short

# Stop after few failures for debugging
pytest --maxfail=5 --tb=short
```
