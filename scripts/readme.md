Automated release script for nabla-ml with date-based versioning.

Version format: YY.MMDD (e.g., 25.0529 for May 29, 2025)

Usage:
    python scripts/release.py [OPTIONS]

Options:
    --dry-run          : Preview changes without executing
    --skip-tests       : Skip running tests
    --skip-upload      : Skip PyPI upload (build only)

Examples:
    python scripts/release.py                    # Release 25.0529
    python scripts/release.py --dry-run          # Preview only
    python scripts/release.py --skip-tests       # Skip test run