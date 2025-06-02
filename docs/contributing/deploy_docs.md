# Deploying Documentation

This guide explains how to deploy the Nabla documentation using GitHub Pages.

## Automatic Deployment with GitHub Actions

The Nabla documentation is automatically built and deployed to GitHub Pages whenever changes are pushed to the `main` or `nabla-python` branch. This process is handled by a GitHub Actions workflow defined in `.github/workflows/docs.yml`.

### How It Works

1. When changes are pushed to the `main` branch, the GitHub Actions workflow is triggered.
2. The workflow checks out the repository code, sets up Python, and installs the required dependencies.
3. It builds the documentation using Sphinx.
4. The built documentation is then deployed to GitHub Pages.

### Viewing the Deployed Documentation

Once deployed, the documentation is available at:

```
https://nabla-ml.github.io/nabla/
```

## Manual Build and Deployment

If you need to build and test the documentation locally, follow these steps:

### Prerequisites

1. Install the documentation requirements:

```bash
pip install -r docs/requirements.txt
```

2. Install Nabla in development mode:

```bash
pip install -e .
```

### Building the Documentation

To build the documentation:

```bash
cd docs
sphinx-build -b html . _build/html
```

The built documentation will be in the `docs/_build/html` directory. You can view it by opening `docs/_build/html/index.html` in your web browser.

### Troubleshooting Build Errors

If you encounter build errors:

1. Check for syntax errors in docstrings or markdown files.
2. Ensure all required dependencies are installed.
3. Look for missing or incorrect cross-references between documents.

### Manually Triggering Deployment

You can manually trigger a documentation build and deployment from GitHub:

1. Go to the repository on GitHub.
2. Navigate to "Actions" → "Build and Deploy Documentation".
3. Click "Run workflow" and select the branch from which to build (typically `main`).

## Documentation Structure

The documentation is structured as follows:

- `docs/index.md`: Main landing page
- `docs/getting_started.md`: Initial setup and quick start guide
- `docs/tutorials/`: Tutorial guides
- `docs/api/`: API reference documentation
- `docs/examples/`: Example code snippets and usage patterns

## Adding New Documentation

When adding new documentation pages:

1. Create your `.md` file in the appropriate directory.
2. Update the appropriate table of contents by adding your page to the relevant `{toctree}` directive.
3. Add cross-references to your page using Sphinx's linking syntax.

## Documentation Best Practices

1. Use clear, concise language.
2. Structure content with appropriate headings and bullet points.
3. Include code examples for key functionality.
4. Add cross-references to related documentation.
5. Update documentation when making code changes.
