# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""Tests for finetune checkpoint save/load roundtrip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import nabla as nb

from .conftest import make_rng


class TestCheckpointRoundtrip:
    def test_save_and_load_lora_checkpoint(self, tmp_path: Path):
        rng = make_rng(90)
        w = nb.Tensor.from_dlpack(rng.normal(size=(10, 6)).astype(np.float32))
        adapter = nb.nn.finetune.init_lora_adapter(w, rank=3, init_std=0.01)
        opt = nb.nn.optim.adamw_init(adapter)

        # Do a fake update to populate optimizer state
        grads = {
            "A": nb.Tensor.from_dlpack(rng.normal(size=(10, 3)).astype(np.float32)),
            "B": nb.Tensor.from_dlpack(rng.normal(size=(3, 6)).astype(np.float32)),
        }
        adapter, opt = nb.nn.optim.adamw_update(adapter, grads, opt, lr=1e-2)

        ckpt = tmp_path / "ckpt"
        nb.nn.finetune.save_finetune_checkpoint(
            ckpt,
            lora_params=adapter,
            optimizer_state=opt,
            metadata={"tag": "test"},
        )

        # Load into fresh templates
        template = nb.nn.finetune.init_lora_adapter(w, rank=3, init_std=0.01)
        opt_template = nb.nn.optim.adamw_init(template)
        loaded_lora, loaded_opt, meta = nb.nn.finetune.load_finetune_checkpoint(
            ckpt, lora_template=template, optimizer_template=opt_template
        )

        nb.testing.assert_allclose(loaded_lora["A"], adapter["A"], rtol=1e-5, atol=1e-5)
        nb.testing.assert_allclose(loaded_lora["B"], adapter["B"], rtol=1e-5, atol=1e-5)
        assert loaded_opt is not None
        assert loaded_opt["step"] == opt["step"]
        assert meta["user_metadata"]["tag"] == "test"

    def test_checkpoint_without_optimizer(self, tmp_path: Path):
        rng = make_rng(91)
        w = nb.Tensor.from_dlpack(rng.normal(size=(8, 4)).astype(np.float32))
        adapter = nb.nn.finetune.init_lora_adapter(w, rank=2)

        ckpt = tmp_path / "ckpt_no_opt"
        nb.nn.finetune.save_finetune_checkpoint(
            ckpt, lora_params=adapter, metadata={"epoch": 1}
        )

        template = nb.nn.finetune.init_lora_adapter(w, rank=2)
        loaded_lora, loaded_opt, meta = nb.nn.finetune.load_finetune_checkpoint(
            ckpt, lora_template=template
        )

        nb.testing.assert_allclose(loaded_lora["A"], adapter["A"], rtol=1e-5, atol=1e-5)
        nb.testing.assert_allclose(loaded_lora["B"], adapter["B"], rtol=1e-5, atol=1e-5)
        assert loaded_opt is None
        assert meta["user_metadata"]["epoch"] == 1

    def test_missing_checkpoint_raises(self, tmp_path: Path):
        rng = make_rng(92)
        w = nb.Tensor.from_dlpack(rng.normal(size=(4, 4)).astype(np.float32))
        template = nb.nn.finetune.init_lora_adapter(w, rank=2)

        with pytest.raises(FileNotFoundError):
            nb.nn.finetune.load_finetune_checkpoint(
                tmp_path / "nonexistent", lora_template=template
            )
