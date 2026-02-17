# PyTree Utilities

## `tree_flatten`

```python
def tree_flatten(tree: 'Any', is_leaf: 'Callable[[Any], bool] | None' = None) -> 'tuple[list[Any], PyTreeDef]':
```
Flatten a pytree into leaves and structure.


---
## `tree_unflatten`

```python
def tree_unflatten(treedef: 'PyTreeDef', leaves: 'list[Any]') -> 'Any':
```
Reconstruct a pytree from structure info and leaves.


---
## `tree_map`

```python
def tree_map(fn: 'Callable[..., Any]', tree: 'Any', *rest: 'Any', is_leaf: 'Callable[[Any], bool] | None' = None) -> 'Any':
```
Apply a function to every leaf of a pytree.


---
## `tree_leaves`

```python
def tree_leaves(tree: 'Any', is_leaf: 'Callable[[Any], bool] | None' = None) -> 'list[Any]':
```
Get all leaves from a pytree (optimized version - doesn't build treedef).


---
## `tree_structure`

```python
def tree_structure(tree: 'Any', is_leaf: 'Callable[[Any], bool] | None' = None) -> 'PyTreeDef':
```
Get structure info from a pytree.


---
## `PyTreeDef`

```python
class PyTreeDef(kind: 'int', meta: 'Any', children: 'tuple[PyTreeDef, ...]', num_leaves: 'int') -> 'None':
```
Immutable definition of a pytree's structure.


---
## `register_pytree_node`

```python
def register_pytree_node(cls: 'type', flatten_fn: 'Callable[[Any], tuple[list[Any], Any]]', unflatten_fn: 'Callable[[Any, list[Any]], Any]') -> 'None':
```
Register a custom class as a pytree container node.


---
