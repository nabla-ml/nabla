# Nabla Documentation

This directory contains the source files for the [Nabla API documentation](https://www.nablaml.com).

## Building Locally

### Prerequisites

Make sure you have the development dependencies installed:

```bash
pip install -r requirements-dev.txt
pip install -e .
```

### Build

From the **project root**:

```bash
make docs
```

This runs `docs/build.sh`, which:
1. Cleans any previous build
2. Generates API reference pages from `docs/structure.json`
3. Builds HTML via Sphinx
4. Generates a sitemap and fixes HTML quirks

### Preview

Serve the built docs locally:

```bash
make docs-serve
```

Then open [http://localhost:8000](http://localhost:8000).

## How It Works

| File | Purpose |
|---|---|
| `structure.json` | Defines every API item (functions, classes, modules) to document |
| `scripts/build_from_json.py` | Reads `structure.json` → generates Markdown files in `api/` |
| `conf.py` | Sphinx configuration (theme, extensions, mocking) |
| `index.md` | Documentation landing page |
| `build.sh` | End-to-end build script |

## Adding New API Items

To document a new function or class:

1. **Edit `structure.json`** — add an entry to the appropriate section:
   ```json
   { "name": "my_function", "type": "function", "path": "nabla.my_function" }
   ```
   For classes with methods to show:
   ```json
   { "name": "MyClass", "type": "class", "path": "nabla.MyClass", "show_methods": true }
   ```

2. **Rebuild** — `make docs`

The build script auto-generates the Markdown pages from `structure.json`, so you don't need to manually create page files.