```bash
# For a full release (tests + build + upload + git commit/tag):
python scripts/release.py

# Or skip tests if you want to go faster:
python scripts/release.py --skip-tests

# Or just build without uploading:
python scripts/release.py --skip-upload
```