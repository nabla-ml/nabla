# Documentation Generation Workflow

This document explains the automated process for generating the Nabla API reference documentation. The system is designed to convert Python docstrings from the `nabla` source code into a structured, navigable set of Markdown files, which can then be used by static site generators like Sphinx or MkDocs.

The core principle is to separate the **structure** of the documentation (what goes where) from the **content** (the docstrings themselves).

## Key Components

The generation process relies on two key files:

1.  **`docs/structure.json`**: The blueprint. This file is the single source of truth for the *hierarchy* of the documentation. It defines every page, section, and API item that should be included, and in what order.
2.  **`docs/scripts/build_from_json.py`**: The engine. This Python script reads the `structure.json` blueprint, introspects the live `nabla` library code to pull out docstrings, formats everything into Markdown, and writes the final `.md` files to the `docs/api/` directory.

---

## How It Works

The `build_from_json.py` script executes the following steps:

1.  **Parse `structure.json`**: The script starts by reading the entire JSON file to understand the desired documentation structure.

2.  **Traverse the Hierarchy**: It iterates through the defined `modules`. The script is designed to handle two types of modules:
    *   **Directory Modules** (e.g., `ops`, `nn`): If a module contains a `"subsections"` key, the script creates a corresponding directory (e.g., `docs/api/ops/`) and an `index.md` file for that directory's table of contents. It then generates a separate Markdown file for each subsection.
    *   **Single-Page Modules** (e.g., `tensor`): If a module does not have subsections and instead has an `"items"` key directly, the script generates a single top-level Markdown file for it (e.g., `docs/api/tensor.md`).

3.  **Dynamic Introspection**: For each API item listed in the JSON (like `nabla.ops.binary.add`), the script uses Python's `importlib` and `inspect` modules to:
    *   Dynamically import the function or class from the live `nabla` library.
    *   Extract its call signature (e.g., `(x: Tensor, y: Tensor) -> Tensor`).
    *   Extract its raw, unprocessed docstring.

4.  **Hybrid Docstring Parsing**: The script uses a two-pronged approach to parse the docstring content:
    *   **Standard Sections**: For sections like `Parameters`, `Returns`, and general descriptions, it uses the `docstring-parser` library.
    *   **Examples Section**: To ensure doctests (`>>> ...`) and their output are rendered correctly and in order, the script uses a **robust manual parser**. This parser specifically looks for the `Examples` header and its underline (`--------`), then intelligently processes the following lines to create perfectly formatted Markdown code blocks. This was implemented to fix bugs where the output appeared before the code.

5.  **Markdown Generation**: The parsed signature and docstring data are formatted into clean Markdown, including headers, code blocks with Python syntax highlighting, bullet points for parameters, and horizontal rules to separate items.

6.  **File Output**: The final Markdown content is written to the appropriate file within the `docs/api/` directory. The script will create any necessary directories and files, overwriting old ones to ensure the docs are always up-to-date.

---

## How to Run the Generator

1.  Make sure you are in your project's virtual environment with all dependencies installed.
2.  From the **project root directory**, run the script:

    ```bash
    python docs/scripts/build_from_json.py
    ```

3.  The script will print its progress to the console and report any errors, such as a function path not being found. The generated files will appear in `docs/api/`.

---

## How to Adapt and Maintain

### Adding a New Function or Class

This is the most common task.

1.  **Find the right location** in the `docs/structure.json` file. For example, if you've added a new binary operation, find the `"ops_binary"` subsection.
2.  **Add a new JSON object** to the `"items"` array for your function. Make sure the `"path"` points to the correct, full import path of the object.

    ```json
    // Inside the "items" array for "ops_binary"
    {
      "name": "your_new_function",
      "type": "function",
      "path": "nabla.ops.binary.your_new_function"
    }
    ```

3.  **Re-run the generator script.** Your new function will now appear in the documentation.

### Reorganizing the Documentation

To change the order of items, create new sections, or move a function from one page to another, **you only need to edit `docs/structure.json`**. The script will automatically generate the new file structure on its next run.

---

### Docstring Formatting Conventions

The parser is specifically tuned for **NumPy-style docstrings**. To ensure your docstrings are parsed correctly, please follow this format, especially for the `Examples` section.

```python
"""A brief one-line summary of the function.

A more detailed multi-line description of what the function does, its
features, and any important notes for the user.

Parameters
----------
arg1 : type
    Description of the first argument.
arg2 : type, optional
    Description of the second argument. Default is `None`.

Returns
-------
return_type
    Description of the returned value.

Examples
--------
A brief description of the first example.

>>> import nabla as nb
>>> x = nb.tensor([1, 2, 3])
>>> your_function(x)
Tensor([2, 4, 6], dtype=int32)

Another example, perhaps showing a different use case.

>>> y = nb.tensor([10, 20, 30])
>>> your_function(x, arg2=y)
Tensor([11, 22, 33], dtype=int32)
"""
```

**Key Points:**
- The section headers (`Parameters`, `Returns`, `Examples`) are followed by an underline of hyphens (`---`).
- Each example code block starts with `>>>`. The output should follow directly on the next line(s).
- Blank lines are used to separate examples.