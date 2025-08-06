"""
Standardized error messages and error handling for test suite.

This module provides consistent error messaging across all binary operation tests,
making it easier to understand test failures and debug issues.
"""

from enum import Enum


class ErrorType(Enum):
    """Standard error types for test results"""

    SUCCESS = "success"
    NABLA_ONLY_FAILED = "nabla_only_failed"
    JAX_ONLY_FAILED = "jax_only_failed"
    RESULTS_MISMATCH = "results_mismatch"
    COMPARISON_FAILED = "comparison_failed"
    BOTH_FAILED_CONSISTENTLY = "both_failed_consistently"
    TUPLE_LENGTH_MISMATCH = "tuple_length_mismatch"
    TUPLE_ITEM_MISMATCH = "tuple_item_mismatch"


def format_error_message(
    test_name: str, error_type: ErrorType, details: str | None = None
) -> str:
    """
    Generate consistent, informative error messages for test results.

    Args:
        test_name: Name of the test being run
        error_type: Type of error/result
        details: Additional details about the error

    Returns:
        Formatted error message string
    """
    base_msg = f"{test_name}"

    if error_type == ErrorType.SUCCESS:
        return f"✓ {base_msg}"

    elif error_type == ErrorType.NABLA_ONLY_FAILED:
        return f"✗ {base_msg}: Nabla failed, JAX succeeded - {details}"

    elif error_type == ErrorType.JAX_ONLY_FAILED:
        return f"✗ {base_msg}: JAX failed, Nabla succeeded - {details}"

    elif error_type == ErrorType.RESULTS_MISMATCH:
        return f"✗ {base_msg}: Results don't match - {details}"

    elif error_type == ErrorType.COMPARISON_FAILED:
        return f"✗ {base_msg}: Comparison failed - {details}"

    elif error_type == ErrorType.BOTH_FAILED_CONSISTENTLY:
        return f"✓ {base_msg} (both frameworks failed consistently)"

    elif error_type == ErrorType.TUPLE_LENGTH_MISMATCH:
        return f"✗ {base_msg}: Tuple length mismatch - {details}"

    elif error_type == ErrorType.TUPLE_ITEM_MISMATCH:
        return f"✗ {base_msg}: Tuple item mismatch - {details}"

    else:
        return f"? {base_msg}: Unknown error type - {details}"


def enhanced_error_message(
    test_name: str,
    operation_name: str,
    rank_combo: tuple,
    transformation: str,
    error_type: ErrorType,
    details: str | None = None,
) -> str:
    """
    Generate enhanced error messages with operation context.

    Args:
        test_name: Base test name
        operation_name: Name of the binary operation (add, mul, etc.)
        rank_combo: Tuple of (rank_x, rank_y)
        transformation: Name of the transformation being tested
        error_type: Type of error/result
        details: Additional error details

    Returns:
        Enhanced error message with full context
    """
    context = f"[{operation_name}] ranks{rank_combo} {transformation}"
    enhanced_name = f"{context} - {test_name}"

    return format_error_message(enhanced_name, error_type, details)


def categorize_error(error_message: str) -> str:
    """
    Categorize errors to help with debugging.

    Args:
        error_message: Raw error message from exception

    Returns:
        Error category string
    """
    error_lower = error_message.lower()

    if "broadcast" in error_lower or "shape" in error_lower:
        return "Broadcasting/Shape Error"
    elif "memory" in error_lower or "segmentation" in error_lower:
        return "Memory Error"
    elif "compilation" in error_lower or "jit" in error_lower:
        return "JIT Compilation Error"
    elif "gradient" in error_lower or "vjp" in error_lower or "jvp" in error_lower:
        return "Autodiff Error"
    elif "vmap" in error_lower:
        return "Vectorization Error"
    elif "device" in error_lower or "gpu" in error_lower:
        return "Device/GPU Error"
    elif "type" in error_lower or "dtype" in error_lower:
        return "Type Error"
    else:
        return "Other Error"


class ErrorSummary:
    """Collect and summarize test errors for reporting"""

    def __init__(self):
        self.errors = []
        self.passed = 0
        self.failed = 0

    def add_result(
        self,
        test_name: str,
        operation: str,
        rank_combo: tuple,
        transformation: str,
        success: bool,
        error_details: str | None = None,
    ):
        """Add a test result to the summary"""
        if success:
            self.passed += 1
        else:
            self.failed += 1
            self.errors.append(
                {
                    "test_name": test_name,
                    "operation": operation,
                    "rank_combo": rank_combo,
                    "transformation": transformation,
                    "error_details": error_details,
                    "error_category": categorize_error(error_details)
                    if error_details
                    else "Unknown",
                }
            )

    def get_summary(self) -> str:
        """Get formatted summary of all test results"""
        total = self.passed + self.failed
        success_rate = (self.passed / total * 100) if total > 0 else 0

        summary = [
            "Test Results Summary:",
            f"  Passed: {self.passed}",
            f"  Failed: {self.failed}",
            f"  Total:  {total}",
            f"  Success Rate: {success_rate:.1f}%",
        ]

        if self.errors:
            summary.append("\nFailed Tests by Category:")

            # Group errors by category
            by_category = {}
            for error in self.errors:
                category = error["error_category"]
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(error)

            for category, errors in by_category.items():
                summary.append(f"  {category}: {len(errors)} failures")
                for error in errors[:3]:  # Show first 3 examples
                    summary.append(
                        f"    - {error['operation']} ranks{error['rank_combo']} {error['transformation']}"
                    )
                if len(errors) > 3:
                    summary.append(f"    ... and {len(errors) - 3} more")

        return "\n".join(summary)
