# Scripts Directory

This directory contains essential scripts for maintaining the Nabla project.

## API Documentation

### Generate API Documentation

To update the API documentation with the latest code changes:

```bash
# Generate API documentation only
python scripts/generate_api_docs.py

# Generate API docs AND build complete documentation
python scripts/update_docs.py
```

The `generate_api_docs.py` script:

- Introspects the Nabla codebase
- Generates static Markdown files for all API components
- Organizes functions by category (array, trafos, creation, unary, binary, etc.)
- Extracts real function signatures and docstrings
- Creates cross-references between related functions

The generated files in `docs/api/` should be committed to the repository so that GitHub Actions can build the documentation without installing Nabla.

## Release Management

### Create a Release

```bash
# For a full release (tests + build + upload + git commit/tag):
python scripts/release.py

# Or skip tests if you want to go faster:
python scripts/release.py --skip-tests

# Or just build without uploading:
python scripts/release.py --skip-upload
```

## Available Scripts

| Script | Purpose |
|--------|---------|
| `generate_api_docs.py` | Generate static API documentation from source code |
| `update_docs.py` | Generate API docs + build complete documentation |
| `release.py` | Automated release management (build, test, upload, tag) |
| `setup_pypi.sh` | PyPI package setup script |

## Documentation Workflow

1. **Make code changes** to Nabla
2. **Update documentation**: `python scripts/update_docs.py`
3. **Review generated docs** in `docs/_build/html/index.html`
4. **Commit API files**: `git add docs/api/` and commit
5. **Push to trigger** GitHub Pages deployment

The build system (`docs/build.sh`) automatically calls `generate_api_docs.py` for local builds when Nabla is installed.