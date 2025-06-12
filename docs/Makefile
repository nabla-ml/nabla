# Makefile for Nabla documentation

# You can set these variables from the command line.
SPHINXOPTS    = --keep-going
SPHINXBUILD  = sphinx-build
SOURCEDIR    = .
BUILDDIR     = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help clean html livehtml install-deps generate-api

# Clean build directory
clean:
	rm -rf $(BUILDDIR)/*
	rm -rf api/generated/*

# Install documentation dependencies
install-deps:
	pip install -r requirements.txt

# Generate API documentation stubs (requires nabla to be installed)
generate-api:
	python ../scripts/generate_api_docs.py

# Build HTML documentation
html: 
	$(SPHINXBUILD) -b html $(SPHINXOPTS) "$(SOURCEDIR)" "$(BUILDDIR)/html"
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

# Build HTML with live reload for development
livehtml:
	sphinx-autobuild -b html $(SPHINXOPTS) "$(SOURCEDIR)" "$(BUILDDIR)/html" --watch ../nabla

# Build for CI/CD (minimal, no autosummary generation)
html-ci:
	$(SPHINXBUILD) -b html $(SPHINXOPTS) -D autosummary_generate=False "$(SOURCEDIR)" "$(BUILDDIR)/html"

# Full build with API generation (for local development)
html-full: generate-api html

# Check for broken links
linkcheck:
	$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)"

.PHONY: clean html livehtml html-ci html-full linkcheck
