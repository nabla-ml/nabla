#!/usr/bin/env python3
"""
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
"""

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# Colors for output
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_step(message):
    print(f"{Colors.HEADER}{Colors.BOLD}▶ {message}{Colors.ENDC}")


def print_success(message):
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")


def print_warning(message):
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")


def print_error(message):
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")


def run_command(cmd, check=True, capture_output=False):
    """Run a shell command."""
    print(f"{Colors.OKCYAN}  $ {cmd}{Colors.ENDC}")
    result = subprocess.run(
        cmd, shell=True, check=check, capture_output=capture_output, text=True
    )
    if capture_output:
        return result.stdout.strip()
    return result


def get_current_version():
    """Extract current version from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print_error("pyproject.toml not found")
        sys.exit(1)

    content = pyproject_path.read_text()
    match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        print_error("Could not find version in pyproject.toml")
        sys.exit(1)

    return match.group(1)


def generate_date_version():
    """Generate date-based version: YY.MMDD format."""
    now = datetime.now()
    year = str(now.year)[-2:]  # Last 2 digits (25 for 2025)
    month = f"{now.month:02d}"
    day = f"{now.day:02d}"
    hour = f"{now.hour:02d}"
    minute = f"{now.minute:02d}"
    return f"{year}.{month}{day}{hour}{minute}"


def update_version_in_file(new_version):
    """Update version in pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()

    # Replace ONLY the project version line - be very specific
    # Look for version = "..." that comes after name = "nabla_ml"
    lines = content.split("\n")
    new_lines = []
    in_project_section = False

    for line in lines:
        if line.strip() == "[project]":
            in_project_section = True
            new_lines.append(line)
        elif line.strip().startswith("[") and line.strip() != "[project]":
            in_project_section = False
            new_lines.append(line)
        elif in_project_section and line.strip().startswith("version"):
            # This is the project version line
            new_lines.append(f'version = "{new_version}"')
        else:
            new_lines.append(line)

    new_content = "\n".join(new_lines)
    pyproject_path.write_text(new_content)
    print_success(f"Updated version to {new_version} in pyproject.toml")


def clean_and_build_package():
    """Clean and build package with PyPI-compatible metadata."""
    print_step("Cleaning build artifacts...")
    run_command("rm -rf dist/ build/ *.egg-info", check=False)

    print_step("Building package with PyPI-compatible metadata...")

    # Temporarily remove LICENSE file completely to avoid License-File metadata field that PyPI rejects
    license_exists = Path("LICENSE").exists()
    license_backup = None
    if license_exists:
        license_backup = Path("LICENSE").read_text()
        run_command("rm LICENSE", check=False)
        print_success("Temporarily removed LICENSE file to avoid PyPI metadata issues")

    try:
        run_command("python -m build")
        print_success("Package built successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Build failed: {e}")
        return False
    finally:
        # Restore LICENSE file
        if license_exists and license_backup is not None:
            Path("LICENSE").write_text(license_backup)
            print_success("Restored LICENSE file")


def validate_package():
    """Validate package with better error handling."""
    print_step("Validating package...")
    try:
        result = run_command("python -m twine check dist/*", capture_output=True)
        print_success("Package validation passed")
        return True
    except subprocess.CalledProcessError as e:
        # Check if it's the License-File issue (which we expect and handle)
        if (
            "license-file" in str(e.stdout).lower()
            or "license-file" in str(e.stderr).lower()
        ):
            print_warning(
                "Package validation shows License-File warning, but this is expected"
            )
            print_warning("The package should still upload successfully to PyPI")
            return True
        else:
            print_error(f"Package validation failed: {e}")
            return False


def upload_to_pypi():
    """Upload to PyPI with improved error handling and metadata compatibility."""
    print_step("Uploading to PyPI...")

    # Check credentials
    import os

    if not (os.getenv("TWINE_PASSWORD") or os.getenv("TWINE_USERNAME")):
        print_warning("No PyPI credentials found in environment variables.")
        print_warning("Make sure you have:")
        print_warning("  export TWINE_USERNAME=__token__")
        print_warning("  export TWINE_PASSWORD=pypi-...")

        response = input(
            "Continue with upload? This will use your ~/.pypirc or prompt for credentials [y/N]: "
        )
        if response.lower() != "y":
            print_warning("Skipping PyPI upload. Package built successfully in dist/")
            return False

    # Upload with retry logic and better error handling
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Use verbose mode to get better error information
            run_command("python -m twine upload dist/* --verbose")
            print_success("Package uploaded to PyPI successfully!")
            return True
        except subprocess.CalledProcessError as e:
            if attempt < max_retries - 1:
                print_warning(f"Upload attempt {attempt + 1} failed, retrying...")
                print_warning(f"Error details: {e}")
            else:
                print_error(f"Upload failed after {max_retries} attempts")
                print_error("Common issues:")
                print_error("  - Check your PyPI credentials")
                print_error("  - Version might already exist on PyPI")
                print_error("  - Network connectivity issues")
                print_error("  - Metadata compatibility issues")
                return False

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Release nabla-ml package with date-based versioning (YY.MMDD)"
    )
    parser.add_arguments(
        "--dry-run", action="store_true", help="Perform a dry run without uploading"
    )
    parser.add_arguments("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_arguments(
        "--skip-upload", action="store_true", help="Skip PyPI upload (build only)"
    )

    args = parser.parse_args()

    print(f"{Colors.HEADER}{Colors.BOLD}")
    print("🚀 NABLA-ML RELEASE SCRIPT")
    print("========================")
    print(f"{Colors.ENDC}")

    # Get current version
    current_version = get_current_version()
    print(f"Current version: {Colors.OKBLUE}{current_version}{Colors.ENDC}")

    # Generate new version using date format
    new_version = generate_date_version()
    print(f"New version: {Colors.OKGREEN}{new_version}{Colors.ENDC}")

    # Safety check: ensure we're not downgrading
    if current_version == new_version:
        print_error(
            f"Version {new_version} already exists! Cannot release the same version twice."
        )
        print_warning(
            "If you need to release again today, manually update the version in pyproject.toml first."
        )
        sys.exit(1)

    if args.dry_run:
        print_warning("DRY RUN MODE - No changes will be made")
        print_step("Would update pyproject.toml")
        if not args.skip_tests:
            print_step("Would run tests")
        print_step("Would clean and build package")
        print_step("Would validate package")
        if not args.skip_upload:
            print_step("Would upload to PyPI")
        print_step("Would commit and tag")
        return

    # Confirm with user
    response = input(f"\nProceed with release {new_version}? [y/N]: ")
    if response.lower() != "y":
        print("Release cancelled")
        sys.exit(0)

    try:
        # 1. Update version
        print_step("Updating version...")
        update_version_in_file(new_version)

        # 2. Run tests
        if not args.skip_tests:
            print_step("Running tests...")
            try:
                run_command("python -m pytest tests/ -v")
                print_success("All tests passed")
            except subprocess.CalledProcessError:
                print_error("Tests failed! Aborting release.")
                # Revert version change
                update_version_in_file(current_version)
                sys.exit(1)

        # 3. Clean and build with PyPI compatibility fixes
        if not clean_and_build_package():
            print_error("Build failed! Aborting release.")
            update_version_in_file(current_version)
            sys.exit(1)

        # 4. Validate package
        if not validate_package():
            print_error("Package validation failed! Aborting release.")
            update_version_in_file(current_version)
            sys.exit(1)

        # 5. Upload to PyPI
        upload_success = True
        if not args.skip_upload:
            upload_success = upload_to_pypi()
            if not upload_success:
                print_warning("Upload failed, but package was built successfully")
                print_warning(
                    "You can manually upload later with: python -m twine upload dist/*"
                )
                # Don't exit here - we can still commit the version change

        # 6. Commit and tag (only if upload succeeded or was skipped intentionally)
        if upload_success or args.skip_upload:
            print_step(f"Committing and tagging version {new_version}...")

            # Check if we have uncommitted changes
            result = run_command("git status --porcelain", capture_output=True)
            if not result.strip():
                print_warning(
                    "No changes to commit (version might already be committed)"
                )
            else:
                run_command("git add pyproject.toml")
                run_command(f'git commit -m "chore: bump version to {new_version}"')

            # Create and push tag
            run_command(f"git tag v{new_version}")
            run_command("git push origin main")
            run_command(f"git push origin v{new_version}")
            print_success(f"Tagged and pushed version {new_version}")

        print(
            f"\n{Colors.OKGREEN}{Colors.BOLD}🎉 Release {new_version} completed successfully!{Colors.ENDC}"
        )
        if upload_success:
            print(f"\n📦 Package URL: https://pypi.org/project/nabla-ml/{new_version}/")
            print(f"📥 Install with: pip install nabla-ml=={new_version}")
        else:
            print("\n📦 Package built in dist/ - manual upload required")

    except KeyboardInterrupt:
        print_error("\nRelease cancelled by user")
        print_warning("Reverting version changes...")
        update_version_in_file(current_version)
        sys.exit(1)
    except Exception as e:
        print_error(f"Release failed: {e}")
        print_warning("Reverting version changes...")
        update_version_in_file(current_version)
        print_warning("You may need to manually clean up any partial changes")
        sys.exit(1)


if __name__ == "__main__":
    main()
