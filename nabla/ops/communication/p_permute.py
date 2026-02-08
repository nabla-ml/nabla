# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from max.graph import TensorValue, ops

from .base import CollectiveOperation

if TYPE_CHECKING:
    from ...core.sharding.spec import DeviceMesh


class PPermuteOp(CollectiveOperation):
    """Point-to-point permutation collective.

    Each device sends its value to exactly one other device according to a
    permutation table.
    """

    @property
    def name(self) -> str:
        return "ppermute"

    def compute_physical_shape(
        self, args: list, kwargs: dict, output_sharding: Any = None
    ) -> tuple[list[tuple[int, ...]], list[Any], list[Any]]:
        return self._compute_local_preserved_shapes(args, kwargs)

    def vjp_rule(self, primals: list, cotangents: list, outputs: list, kwargs: dict) -> list:
        """VJP for ppermute: permute back with inverse table."""
        perm = kwargs.get("permutation")
        inv_perm = [(dst, src) for src, dst in perm]
        from .p_permute import ppermute

        return [ppermute(cotangents[0], inv_perm)]

    def infer_sharding_spec(self, args: Any, mesh: DeviceMesh, kwargs: dict) -> Any:
        """Infer sharding for PPermute (Adaptation Layer)."""
        input_tensor = args[0]
        input_sharding = input_tensor.sharding
        # PPermute preserves sharding spec (it just moves data between devices).
        return input_sharding, [input_sharding], False

    def execute(self, args: list, kwargs: dict) -> Any:
        """Point-to-point permutation (Physical)."""
        from ...core import GRAPH, Tensor

        sharded_tensor: Tensor = args[0]
        permutation = kwargs.get("permutation")

        if permutation is None:
            raise ValueError("PPermuteOp requires a 'permutation' argument.")

        # 1. Derive Metadata
        mesh = self._derive_mesh(sharded_tensor, kwargs)

        # 2. Validation & Early Exit
        if not sharded_tensor.sharding:
            return (sharded_tensor.values, None, None)

        # 2. Execution Context
        with GRAPH.graph:
            values = sharded_tensor.values

            # Ported logic from kernel
            result_graph_values = self._ppermute_logic(values, permutation, mesh=mesh)

        # 3. Output Spec (Preserve input spec)
        output_spec = sharded_tensor.sharding

        return (result_graph_values, output_spec, mesh)

    def _ppermute_logic(
        self,
        shard_graph_values: list[TensorValue],
        permutation: list[tuple],
        mesh: DeviceMesh = None,
    ) -> list[TensorValue]:
        """Permute values between devices according to permutation."""
        num_devices = len(shard_graph_values)

        dest_to_src = {}
        for src, dst in permutation:
            if dst in dest_to_src:
                raise ValueError(
                    f"Destination {dst} appears multiple times in permutation"
                )
            dest_to_src[dst] = src

        results = []

        for dst in range(num_devices):
            if dst in dest_to_src:
                src = dest_to_src[dst]
                val = shard_graph_values[src]

                if mesh and mesh.is_distributed:
                    val = ops.transfer_to(val, mesh.device_refs[dst])

                results.append(val)
            else:

                template = shard_graph_values[0]
                zero_val = ops.constant(0, template.type.dtype, template.type.device)
                zero_val = ops.broadcast_to(zero_val, template.type.shape)
                results.append(zero_val)

        return results


_ppermute_op = PPermuteOp()


def ppermute(sharded_tensor, permutation: list[tuple]):
    """Point-to-point permutation collective."""
    return _ppermute_op([sharded_tensor], {"permutation": permutation})[0]
