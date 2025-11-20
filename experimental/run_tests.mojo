"""
Test runner for nabla tests.
Run this file to execute all tests.
"""

from tests.test_exports import test_exports
from tests.test_refactor import test_refactor
from tests.test_module import test_module
from tests.test_motree import test_motree
from tests.test_jit import test_jit 


fn main() raises:
    print("=" * 60)
    print("Running Nabla Tests")
    print("=" * 60)
    print()

    # Test 1: Exports
    print("\n[1/5] Running test_exports...")
    print("-" * 60)
    test_exports()

    # Test 2: Refactor verification
    print("\n[2/5] Running test_refactor...")
    print("-" * 60)
    test_refactor()

    # Test 3: Module/MLP
    print("\n[3/5] Running test_module...")
    print("-" * 60)
    test_module()

    # Test 4: MoTree
    print("\n[4/5] Running test_motree...")
    print("-" * 60)
    test_motree()

    # Test 5: JIT compilation (this is intensive!)
    print("\n[5/5] Running test_jit...")
    print("-" * 60)
    test_jit()

    print()
    print("=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
