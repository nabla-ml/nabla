#!/usr/bin/env python3
"""
Generate complete API documentation for Nabla.

This script introspects the Nabla codebase and generates static Markdown files
for all API components, eliminating the need for autodoc during Sphinx builds.
Supports @nodoc decorator to exclude items from documentation.
"""

import ast
import sys
from pathlib import Path
from typing import Any, Optional


class APIDocGenerator:
    """Generate API documentation from Python source code."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.nabla_path = project_root / "nabla"
        self.docs_api_path = project_root / "docs" / "api"

        # Add nabla to Python path for imports
        sys.path.insert(0, str(project_root))

    def is_decorated_with_nodoc(self, node: ast.AST) -> bool:
        """Check if a function or class has the @nodoc decorator."""
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            return False

        for decorator in node.decorator_list:
            # Handle simple decorator names like @nodoc
            if (
                isinstance(decorator, ast.Name)
                and decorator.id
                in (
                    "nodoc",
                    "no_doc",
                    "skip_doc",
                )
                or isinstance(decorator, ast.Attribute)
                and decorator.attr
                in (
                    "nodoc",
                    "no_doc",
                    "skip_doc",
                )
            ):
                return True
        return False

    def should_document_runtime(self, obj: Any, name: str) -> bool:
        """Check if an object should be documented using runtime inspection."""
        try:
            # Import the docs utility to check the runtime decorator
            from nabla.utils.docs import should_document

            return should_document(obj, name)
        except ImportError:
            # Fallback to basic filtering if utils.docs is not available
            return not name.startswith("_")

    def extract_docstring(self, node: ast.AST) -> Optional[str]:
        """Extract docstring from an AST node."""
        if (
            isinstance(node, ast.FunctionDef | ast.ClassDef | ast.AsyncFunctionDef)
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            return node.body[0].value.value
        return None

    def extract_function_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature from AST node."""
        args = []

        # Regular arguments
        for arg in node.args.args:
            annotation = ""
            if arg.annotation:
                annotation = f": {ast.unparse(arg.annotation)}"
            args.append(f"{arg.arg}{annotation}")

        # Handle defaults
        defaults = node.args.defaults
        if defaults:
            default_offset = len(args) - len(defaults)
            for i, default in enumerate(defaults):
                idx = default_offset + i
                if idx < len(args):
                    args[idx] += f" = {ast.unparse(default)}"

        # *args
        if node.args.vararg:
            annotation = ""
            if node.args.vararg.annotation:
                annotation = f": {ast.unparse(node.args.vararg.annotation)}"
            args.append(f"*{node.args.vararg.arg}{annotation}")

        # **kwargs
        if node.args.kwarg:
            annotation = ""
            if node.args.kwarg.annotation:
                annotation = f": {ast.unparse(node.args.kwarg.annotation)}"
            args.append(f"**{node.args.kwarg.arg}{annotation}")

        # Return annotation
        return_annotation = ""
        if node.returns:
            return_annotation = f" -> {ast.unparse(node.returns)}"

        return f"({', '.join(args)}){return_annotation}"

    def parse_python_file(self, file_path: Path) -> dict[str, Any]:
        """Parse a Python file and extract API information."""
        try:
            with Path(file_path).open(encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
        except Exception as e:
            print(f"Warning: Could not parse {file_path}: {e}")
            return {}

        api_info = {
            "module_docstring": None,
            "functions": [],
            "classes": [],
            "constants": [],
            "imports": [],
        }

        # Extract module docstring
        if (
            tree.body
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)
        ):
            api_info["module_docstring"] = tree.body[0].value.value

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not node.name.startswith("_"):  # Skip private functions
                    func_info = {
                        "name": node.name,
                        "signature": self.extract_function_signature(node),
                        "docstring": self.extract_docstring(node),
                        "is_async": isinstance(node, ast.AsyncFunctionDef),
                    }
                    api_info["functions"].append(func_info)

            elif isinstance(node, ast.ClassDef):
                if not node.name.startswith("_"):  # Skip private classes
                    methods = []
                    for item in node.body:
                        if isinstance(
                            item, ast.FunctionDef
                        ) and not item.name.startswith("_"):
                            method_info = {
                                "name": item.name,
                                "signature": self.extract_function_signature(item),
                                "docstring": self.extract_docstring(item),
                            }
                            methods.append(method_info)

                    class_info = {
                        "name": node.name,
                        "docstring": self.extract_docstring(node),
                        "methods": methods,
                        "bases": [ast.unparse(base) for base in node.bases],
                    }
                    api_info["classes"].append(class_info)

            elif isinstance(node, ast.Assign):
                # Extract constants (uppercase variables)
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        api_info["constants"].append(
                            {
                                "name": target.id,
                                "value": ast.unparse(node.value)
                                if hasattr(ast, "unparse")
                                else "N/A",
                            }
                        )

        return api_info

    def get_module_exports(self, module_path: Path) -> list[str]:
        """Get __all__ exports from a module if available."""
        try:
            with Path(module_path).open(encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "__all__"
                ) and isinstance(node.value, ast.List):
                    return [
                        elt.value
                        for elt in node.value.elts
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    ]
        except Exception:
            pass
        return []

    def format_docstring(self, docstring: Optional[str]) -> str:
        """Format a docstring for Markdown."""
        if not docstring:
            return "*No description available.*"

        # Clean up the docstring
        lines = docstring.strip().split("\n")
        # Remove common leading whitespace
        if len(lines) > 1:
            common_indent = min(
                len(line) - len(line.lstrip()) for line in lines[1:] if line.strip()
            )
            lines = [lines[0]] + [
                line[common_indent:] if len(line) > common_indent else line
                for line in lines[1:]
            ]

        return "\n".join(lines)

    def generate_function_docs(self, functions: list[dict]) -> str:
        """Generate documentation for functions."""
        if not functions:
            return ""

        docs = []
        for func in functions:
            name = func["name"]
            signature = func["signature"]
            docstring = self.format_docstring(func["docstring"])

            async_prefix = "async " if func.get("is_async", False) else ""

            docs.append(f"""
### {name}

```python
{async_prefix}def {name}{signature}
```

{docstring}
""")

        return "\n".join(docs)

    def generate_class_docs(self, classes: list[dict]) -> str:
        """Generate documentation for classes."""
        if not classes:
            return ""

        docs = []
        for cls in classes:
            name = cls["name"]
            docstring = self.format_docstring(cls["docstring"])
            bases = cls.get("bases", [])

            inheritance = f"({', '.join(bases)})" if bases else ""

            docs.append(f"""
### {name}

```python
class {name}{inheritance}
```

{docstring}
""")

            # Add methods
            if cls["methods"]:
                docs.append("#### Methods\n")
                for method in cls["methods"]:
                    method_name = method["name"]
                    method_sig = method["signature"]
                    method_doc = self.format_docstring(method["docstring"])

                    docs.append(f"""
##### {method_name}

```python
def {method_name}{method_sig}
```

{method_doc}
""")

        return "\n".join(docs)

    def generate_module_doc(self, module_name: str, api_info: dict[str, Any]) -> str:
        """Generate complete documentation for a module."""
        title = module_name.replace("_", " ").title()

        doc = f"""# {title}

{self.format_docstring(api_info.get("module_docstring"))}
"""

        # Add functions
        if api_info["functions"]:
            doc += "\n## Functions\n"
            doc += self.generate_function_docs(api_info["functions"])

        # Add classes
        if api_info["classes"]:
            doc += "\n## Classes\n"
            doc += self.generate_class_docs(api_info["classes"])

        # Add constants
        if api_info["constants"]:
            doc += "\n## Constants\n"
            for const in api_info["constants"]:
                doc += f"\n### {const['name']}\n\n```python\n{const['name']} = {const['value']}\n```\n"

        return doc

    def introspect_nabla_ops(self) -> dict[str, list[str]]:
        """Introspect Nabla to get all operations organized by category."""
        ops_categories = {
            "array": [],
            "trafos": [],
            "binary": [],
            "unary": [],
            "creation": [],
            "reduction": [],
            "linalg": [],
            "manipulation": [],
            "core": [],
        }

        # Scan ops modules
        ops_path = self.nabla_path / "ops"
        if ops_path.exists():
            for py_file in ops_path.glob("*.py"):
                if py_file.name == "__init__.py":
                    continue

                module_name = py_file.stem
                exports = self.get_module_exports(py_file)

                # Map module names to categories
                category_map = {
                    "binary": "binary",
                    "unary": "unary",
                    "creation": "creation",
                    "reduce": "reduction",
                    "linalg": "linalg",
                    "view": "manipulation",
                }

                category = category_map.get(module_name, "core")
                ops_categories[category].extend(exports)

        # Core transformations (separate from Array class)
        core_trafos = [
            "jit",
            "djit",
            "vjp",
            "jvp",
            "vmap",
            "grad",
            "jacrev",
            "jacfwd",
            "xpr",
        ]
        ops_categories["trafos"].extend(core_trafos)

        # Array class
        ops_categories["array"].append("Array")

        return ops_categories

    def generate_api_reference_docs(self):
        """Generate all API reference documentation."""
        print("Generating API documentation...")

        # Create API docs directory
        self.docs_api_path.mkdir(exist_ok=True)

        # Get operations by category
        ops_by_category = self.introspect_nabla_ops()

        # Generate documentation for each category
        category_docs = {
            "array": "Array Class",
            "trafos": "Function Transformations",
            "creation": "Array Creation",
            "unary": "Unary Operations",
            "binary": "Binary Operations",
            "reduction": "Reduction Operations",
            "linalg": "Linear Algebra",
            "manipulation": "Array Manipulation",
        }

        for category, title in category_docs.items():
            ops = ops_by_category.get(category, [])
            self.generate_category_doc(category, title, ops)

        # Update main API index
        self.generate_api_index()

        print("API documentation generated successfully!")

    def generate_category_doc(self, category: str, title: str, operations: list[str]):
        """Generate documentation for a specific category."""
        doc = f"""# {title}

{self.get_category_description(category)}

```{{toctree}}
:maxdepth: 1
:caption: Functions

"""

        # Sort operations alphabetically
        operations = sorted(set(operations))

        for op in operations:
            if op:  # Skip empty strings
                # Create individual function page
                self.generate_function_page(op, category)

                # Add to toctree
                doc += f"{category}_{op}\n"

        doc += "```\n\n"

        # Add quick reference section
        doc += "## Quick Reference\n\n"

        for op in operations:
            if op:
                func_info = self.introspect_function(op)
                doc += f"""### {{doc}}`{op} <{category}_{op}>`

```python
{func_info["signature"]}
```

{self.get_operation_description(op)}

"""

        # Write category overview file
        output_file = self.docs_api_path / f"{category}.md"
        with output_file.open("w", encoding="utf-8") as f:
            f.write(doc)

        print(f"Generated {output_file}")

    def get_category_description(self, category: str) -> str:
        """Get description for a category."""
        descriptions = {
            "array": "The core Array class with its properties, methods, and operator overloading.",
            "trafos": "Function transformations for compilation, vectorization, and automatic differentiation.",
            "creation": "Functions for creating new arrays with various initialization patterns.",
            "unary": "Element-wise unary operations that operate on a single array.",
            "binary": "Element-wise binary operations that operate on two arrays.",
            "reduction": "Operations that reduce arrays along specified dimensions.",
            "linalg": "Linear algebra operations including matrix multiplication and decompositions.",
            "manipulation": "Operations for reshaping, indexing, and manipulating array structure.",
            "core": "Core components of Nabla.",
        }
        return descriptions.get(category, "Nabla operations.")

    def generate_function_page(self, function_name: str, category: str):
        """Generate a dedicated page for a single function."""
        # Try to find the actual function implementation and extract docstring
        func_info = self.introspect_function(function_name)

        # Generate the function page
        doc = f"""# {function_name}

## Signature

```python
{func_info["signature"]}
```

## Description

{func_info["docstring"]}

"""

        # Add parameters documentation if available
        if func_info.get("parameters"):
            doc += f"""## Parameters

{func_info["parameters"]}

"""

        # Add returns documentation if available
        if func_info.get("returns"):
            doc += f"""## Returns

{func_info["returns"]}

"""

        # Add examples if available, otherwise generate basic ones for common functions
        if func_info.get("examples"):
            doc += f"""## Examples

{func_info["examples"]}

"""
        else:
            # Generate basic examples for common functions
            example = self.generate_basic_example(function_name, category)
            if example:
                doc += f"""## Examples

```python
import nabla as nb

{example}
```

"""

        # Add notes if available
        if func_info.get("notes"):
            doc += f"""## Notes

{func_info["notes"]}

"""

        # Add related functions section
        related = self.get_related_functions(function_name, category)
        if related:
            doc += f"""## See Also

{related}

"""

        # Write individual function file
        output_file = self.docs_api_path / f"{category}_{function_name}.md"
        with output_file.open("w", encoding="utf-8") as f:
            f.write(doc)

        print(f"Generated {output_file}")

    def generate_basic_example(self, function_name: str, category: str) -> str:
        """Generate basic usage examples for common functions."""
        examples = {
            "sin": """# Basic trigonometric function
x = nb.array([0, np.pi/2, np.pi])
result = nb.sin(x)
print(result)  # [0, 1, 0] (approximately)""",
            "cos": """# Basic trigonometric function
x = nb.array([0, np.pi/2, np.pi])
result = nb.cos(x)
print(result)  # [1, 0, -1] (approximately)""",
            "exp": """# Exponential function
x = nb.array([0, 1, 2])
result = nb.exp(x)
print(result)  # [1, e, e^2] (approximately)""",
            "log": """# Natural logarithm
x = nb.array([1, np.e, np.e**2])
result = nb.log(x)
print(result)  # [0, 1, 2] (approximately)""",
            "add": """# Element-wise addition
a = nb.array([1, 2, 3])
b = nb.array([4, 5, 6])
result = nb.add(a, b)
print(result)  # [5, 7, 9]""",
            "mul": """# Element-wise multiplication
a = nb.array([1, 2, 3])
b = nb.array([4, 5, 6])
result = nb.mul(a, b)
print(result)  # [4, 10, 18]""",
            "sum": """# Sum along axes
x = nb.array([[1, 2], [3, 4]])
result = nb.sum(x)  # Sum all elements
print(result)  # 10

result_axis0 = nb.sum(x, axis=0)  # Sum along rows
print(result_axis0)  # [4, 6]""",
            "matmul": """# Matrix multiplication
A = nb.array([[1, 2], [3, 4]])
B = nb.array([[5, 6], [7, 8]])
result = nb.matmul(A, B)
print(result)  # [[19, 22], [43, 50]]""",
            "reshape": """# Change array shape
x = nb.array([1, 2, 3, 4, 5, 6])
result = nb.reshape(x, (2, 3))
print(result)  # [[1, 2, 3], [4, 5, 6]]""",
            "transpose": """# Transpose array
x = nb.array([[1, 2, 3], [4, 5, 6]])
result = nb.transpose(x)
print(result)  # [[1, 4], [2, 5], [3, 6]]""",
            "ones": """# Create array of ones
result = nb.ones((2, 3))
print(result)  # [[1, 1, 1], [1, 1, 1]]""",
            "zeros": """# Create array of zeros
result = nb.zeros((2, 3))
print(result)  # [[0, 0, 0], [0, 0, 0]]""",
            "array": """# Create array from data
data = [[1, 2, 3], [4, 5, 6]]
result = nb.array(data)
print(result)  # [[1, 2, 3], [4, 5, 6]]""",
            "vjp": """# Vector-Jacobian product for reverse-mode AD
def f(x):
    return nb.sum(x ** 2)

x = nb.array([1.0, 2.0, 3.0])
output, vjp_fn = nb.vjp(f, x)
gradients = vjp_fn(nb.ones_like(output))
print(gradients)  # [2.0, 4.0, 6.0]""",
            "jit": """# Just-in-time compilation
@nb.jit
def fast_function(x):
    return nb.sum(x ** 2) + nb.mean(x)

x = nb.randn((1000,))
result = fast_function(x)  # Compiled on first call""",
            "vmap": """# Vectorize function over batch dimension
def dot_product(a, b):
    return nb.sum(a * b)

# Vectorize over first dimension
batch_dot = nb.vmap(dot_product, in_axes=(0, 0))

a_batch = nb.randn((10, 5))  # 10 vectors of length 5
b_batch = nb.randn((10, 5))
results = batch_dot(a_batch, b_batch)  # 10 dot products""",
        }

        return examples.get(function_name, "")

    def get_related_functions(self, function_name: str, category: str) -> str:
        """Get related functions for cross-references."""
        related_map = {
            "sin": "- {doc}`cos <unary_cos>` - Cosine function\n- {doc}`exp <unary_exp>` - Exponential function",
            "cos": "- {doc}`sin <unary_sin>` - Sine function\n- {doc}`exp <unary_exp>` - Exponential function",
            "exp": "- {doc}`log <unary_log>` - Natural logarithm\n- {doc}`sin <unary_sin>`, {doc}`cos <unary_cos>` - Trigonometric functions",
            "log": "- {doc}`exp <unary_exp>` - Exponential function",
            "add": "- {doc}`sub <binary_sub>` - Subtraction\n- {doc}`mul <binary_mul>` - Multiplication\n- {doc}`div <binary_div>` - Division",
            "sub": "- {doc}`add <binary_add>` - Addition\n- {doc}`mul <binary_mul>` - Multiplication\n- {doc}`div <binary_div>` - Division",
            "mul": "- {doc}`add <binary_add>` - Addition\n- {doc}`div <binary_div>` - Division\n- {doc}`pow <binary_pow>` - Exponentiation",
            "div": "- {doc}`mul <binary_mul>` - Multiplication\n- {doc}`add <binary_add>` - Addition\n- {doc}`sub <binary_sub>` - Subtraction",
            "vjp": "- {doc}`jvp <core_jvp>` - Jacobian-vector product\n- {doc}`grad <core_grad>` - Automatic differentiation",
            "jvp": "- {doc}`vjp <core_vjp>` - Vector-Jacobian product\n- {doc}`grad <core_grad>` - Automatic differentiation",
            "jit": "- {doc}`vmap <core_vmap>` - Vectorization\n- {doc}`grad <core_grad>` - Automatic differentiation",
            "vmap": "- {doc}`jit <core_jit>` - Just-in-time compilation\n- {doc}`vjp <core_vjp>`, {doc}`jvp <core_jvp>` - Automatic differentiation",
        }

        return related_map.get(function_name, "")

    def introspect_function(self, function_name: str) -> dict[str, str]:
        """Introspect a function to get its real signature and docstring."""
        # Default fallback
        func_info = {
            "signature": f"nabla.{function_name}(...)",
            "docstring": f"*Documentation for `{function_name}` is being generated. Please check the source code for implementation details.*",
            "examples": None,
            "parameters": None,
            "returns": None,
            "notes": None,
        }

        try:
            # Try to import nabla and get the actual function
            import nabla

            if hasattr(nabla, function_name):
                func = getattr(nabla, function_name)

                # Get signature
                try:
                    import inspect

                    sig = inspect.signature(func)
                    func_info["signature"] = f"nabla.{function_name}{sig}"
                except Exception:
                    pass

                # Get docstring
                if func.__doc__:
                    full_docstring = func.__doc__.strip()

                    # Parse docstring sections
                    lines = full_docstring.split("\n")

                    # Main description (everything before first section)
                    main_desc_lines = []
                    current_section = None
                    current_content = []

                    for line in lines:
                        line = line.strip()

                        # Check for common docstring sections
                        if line.lower().startswith(
                            ("parameters:", "args:", "arguments:")
                        ):
                            if main_desc_lines or current_content:
                                if current_section is None:
                                    main_desc_lines.extend(current_content)
                                else:
                                    func_info[current_section] = "\n".join(
                                        current_content
                                    )
                            current_section = "parameters"
                            current_content = []
                        elif line.lower().startswith(("returns:", "return:")):
                            if main_desc_lines or current_content:
                                if current_section is None:
                                    main_desc_lines.extend(current_content)
                                else:
                                    func_info[current_section] = "\n".join(
                                        current_content
                                    )
                            current_section = "returns"
                            current_content = []
                        elif line.lower().startswith(("examples:", "example:")):
                            if main_desc_lines or current_content:
                                if current_section is None:
                                    main_desc_lines.extend(current_content)
                                else:
                                    func_info[current_section] = "\n".join(
                                        current_content
                                    )
                            current_section = "examples"
                            current_content = []
                        elif line.lower().startswith(("notes:", "note:")):
                            if main_desc_lines or current_content:
                                if current_section is None:
                                    main_desc_lines.extend(current_content)
                                else:
                                    func_info[current_section] = "\n".join(
                                        current_content
                                    )
                            current_section = "notes"
                            current_content = []
                        else:
                            if current_section is None:
                                main_desc_lines.append(line)
                            else:
                                current_content.append(line)

                    # Handle remaining content
                    if current_content:
                        if current_section:
                            func_info[current_section] = "\n".join(current_content)
                        else:
                            main_desc_lines.extend(current_content)

                    # Set main description
                    if main_desc_lines:
                        func_info["docstring"] = "\n".join(main_desc_lines)
                    else:
                        func_info["docstring"] = full_docstring

            # Try to find the function in source files for better documentation
            else:
                # Search in ops modules
                ops_path = self.nabla_path / "ops"
                for py_file in ops_path.glob("*.py"):
                    if py_file.name == "__init__.py":
                        continue

                    api_info = self.parse_python_file(py_file)
                    for func in api_info["functions"]:
                        if func["name"] == function_name and func["docstring"]:
                            func_info["docstring"] = func["docstring"]
                            func_info["signature"] = (
                                f"nabla.{function_name}{func['signature']}"
                            )
                            break

        except Exception as e:
            print(f"Warning: Could not introspect function {function_name}: {e}")
            # Use source code parsing as fallback
            self._try_source_parsing(function_name, func_info)

        return func_info

    def _try_source_parsing(self, function_name: str, func_info: dict[str, str]):
        """Fallback: try to extract info from source code parsing."""
        try:
            # Search in ops modules
            ops_path = self.nabla_path / "ops"
            for py_file in ops_path.glob("*.py"):
                if py_file.name == "__init__.py":
                    continue

                api_info = self.parse_python_file(py_file)

                # Look for the function
                for func in api_info["functions"]:
                    if func["name"] == function_name:
                        if func["docstring"]:
                            func_info["docstring"] = func["docstring"]
                        func_info["signature"] = (
                            f"nabla.{function_name}{func['signature']}"
                        )
                        return

                # Look for classes that might contain the operation
                for cls in api_info["classes"]:
                    if cls["name"].lower().replace("op", "") == function_name.lower():
                        if cls["docstring"]:
                            func_info["docstring"] = cls["docstring"]
                        return

        except Exception as e:
            print(f"Warning: Source parsing failed for {function_name}: {e}")

    def get_operation_description(self, op: str) -> str:
        """Get description for a specific operation."""
        # Enhanced descriptions for common operations
        descriptions = {
            # Core operations
            "Array": "The fundamental array type in Nabla.",
            "jit": "Just-in-time compilation for performance optimization.",
            "vjp": "Vector-Jacobian product for reverse-mode automatic differentiation.",
            "jvp": "Jacobian-vector product for forward-mode automatic differentiation.",
            "vmap": "Vectorization transformation for batching operations.",
            "grad": "Automatic differentiation to compute gradients.",
            "xpr": "Create expression graphs for deferred execution.",
            # Creation operations
            "array": "Create a new array from data.",
            "zeros": "Create an array filled with zeros.",
            "ones": "Create an array filled with ones.",
            "randn": "Create an array with random values from normal distribution.",
            "arange": "Create an array with evenly spaced values.",
            "zeros_like": "Create a zero array with the same shape as input.",
            "ones_like": "Create an array of ones with the same shape as input.",
            # Unary operations
            "sin": "Element-wise sine function.",
            "cos": "Element-wise cosine function.",
            "exp": "Element-wise exponential function.",
            "log": "Element-wise natural logarithm.",
            "relu": "Element-wise rectified linear unit activation.",
            "negate": "Element-wise negation.",
            "cast": "Cast array elements to a different data type.",
            # Binary operations
            "add": "Element-wise addition of two arrays.",
            "sub": "Element-wise subtraction of two arrays.",
            "mul": "Element-wise multiplication of two arrays.",
            "div": "Element-wise division of two arrays.",
            "pow": "Element-wise exponentiation.",
            "maximum": "Element-wise maximum of two arrays.",
            "minimum": "Element-wise minimum of two arrays.",
            # Reduction operations
            "sum": "Sum of array elements over given axes.",
            "mean": "Arithmetic mean of array elements over given axes.",
            "max": "Maximum of array elements over given axes.",
            "min": "Minimum of array elements over given axes.",
            # Linear algebra
            "matmul": "Matrix multiplication.",
            "dot": "Dot product of two arrays.",
            # Manipulation
            "reshape": "Change the shape of an array without changing its data.",
            "transpose": "Permute the dimensions of an array.",
            "squeeze": "Remove single-dimensional entries from array shape.",
            "unsqueeze": "Add single-dimensional entries to array shape.",
        }

        return descriptions.get(op, f"Nabla operation: `{op}`")

    def generate_api_index(self):
        """Generate the main API index file."""
        index_content = """# API Reference

This page contains the complete API reference for Nabla, organized by functionality.

```{toctree}
:maxdepth: 2
:caption: API Documentation

array
trafos
creation
unary
binary
reduction
linalg
manipulation
```

## Quick Reference

### Core Components

- {doc}`array` - The fundamental Array class with properties and methods
- {doc}`trafos` - Function transformations (jit, vmap, grad, etc.)

### Array Operations

- {doc}`creation` - Array creation functions
- {doc}`unary` - Element-wise unary operations (sin, cos, exp, etc.)
- {doc}`binary` - Element-wise binary operations (add, multiply, etc.)
- {doc}`reduction` - Reduction operations (sum, mean, etc.)
- {doc}`linalg` - Linear algebra operations
- {doc}`manipulation` - Array view and manipulation operations

## Overview

Nabla provides a comprehensive set of APIs for array operations, function transformations, and automatic differentiation:

- **Array Class**: The fundamental `Array` class with its properties, methods, and operator overloading
- **Function Transformations**: Tools like `jit`, `vmap`, `grad`, `vjp`, and `jvp` for compilation and differentiation
- **Array Operations**: Creation, manipulation, and mathematical operations on arrays
"""

        index_file = self.docs_api_path / "index.md"
        with index_file.open("w", encoding="utf-8") as f:
            f.write(index_content)

        print(f"Generated {index_file}")


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent.parent  # docs/scripts/ -> docs/ -> project/
    generator = APIDocGenerator(project_root)
    generator.generate_api_reference_docs()


if __name__ == "__main__":
    main()
