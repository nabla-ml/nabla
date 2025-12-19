# Documentation Scripts

This directory contains scripts for maintaining and optimizing the Nabla documentation, including automated SEO optimization for CI/CD workflows.

## CI/CD Integration

### Configure SEO for Environment

```bash
python docs/scripts/configure_seo.py [html_dir]
```

Environment-aware SEO configuration that:
- Sets proper domain from `DOCS_BASE_URL` environment variable
- Validates SEO consistency across all files
- Used automatically in GitHub Actions workflows

## API Documentation

### Generate API Documentation

```bash
python docs/scripts/generate_api_docs.py
```

Automatically generates structured API documentation from the codebase.

### Generate Sitemap

```bash
python docs/scripts/generate_sitemap.py
```

Creates `sitemap.xml` for search engine optimization with environment-aware domain configuration.

## SEO and Quality Assurance

### SEO Audit

```bash
python docs/scripts/seo_audit.py
```

Performs comprehensive SEO analysis of the documentation site.

### Check Indexing Status

```bash
python docs/scripts/check_indexing.py
```

Verifies Google indexing status and structured data validity.

### Fix SEO Issues

```bash
python docs/scripts/fix_seo_issues.py
```

Automatically fixes common SEO problems in HTML files.

### Fix Duplicate Viewport Tags

```bash
python docs/scripts/fix_duplicate_viewport.py
```

Removes duplicate viewport meta tags from HTML files.

### SEO Improvements

```bash
python docs/scripts/seo_improvements.py
```

Applies advanced SEO optimizations to documentation.

### Final SEO Report

```bash
python docs/scripts/final_seo_report.py
```

Generates a comprehensive SEO compliance report.

## Usage

These scripts are typically run as part of the documentation build process or for maintenance tasks. Most SEO scripts should be run after building the HTML documentation with `bash docs/build.sh`.
