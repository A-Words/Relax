# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Frozen-reference checksum, optimizer and sidecar tests."""

import ast
import hashlib
import json
import os
import sys
import types
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from relax.backends.megatron.reference_integrity import (
    REFERENCE_LOADER_MODE,
    DPOReferenceIdentity,
    canonical_tensor_sha256,
    read_reference_identity,
    reference_identity_path,
    resolve_dpo_reference_checkpoint,
    write_reference_identity,
)
from relax.engine.sft.runtime import is_preference_mode
from relax.utils.training import tensor_backper


def test_megatron_resume_detection_ignores_fresh_output_directory(tmp_path):
    pytest.importorskip("megatron.training.checkpointing")
    from relax.backends.megatron.checkpoint import is_megatron_checkpoint

    output = tmp_path / "run"
    output.mkdir()
    (output / "transformer_config.json").write_text("{}", encoding="utf-8")
    assert not is_megatron_checkpoint(output)
    (output / "latest_checkpointed_iteration.txt").write_text("1", encoding="utf-8")
    assert is_megatron_checkpoint(output)
    assert is_megatron_checkpoint(tmp_path / "iter_0000001")


def test_canonical_tensor_digest_is_order_stable_and_byte_sensitive():
    first = canonical_tensor_sha256([("b", torch.tensor([2.0])), ("a", torch.tensor([1.0]))])
    reordered = canonical_tensor_sha256([("a", torch.tensor([1.0])), ("b", torch.tensor([2.0]))])
    changed = canonical_tensor_sha256([("a", torch.tensor([1.0])), ("b", torch.tensor([3.0]))])
    assert first == reordered
    assert first != changed
    assert first != canonical_tensor_sha256([("a", torch.tensor([1], dtype=torch.int64)), ("b", torch.tensor([2.0]))])


def test_reference_identity_sidecar_is_required_and_rejects_schema_damage(tmp_path):
    path = tmp_path / "relax_dpo_reference.json"
    with pytest.raises(FileNotFoundError):
        read_reference_identity(path)
    identity = DPOReferenceIdentity(1, "repo", "revision", "loader", "a" * 64)
    write_reference_identity(path, identity)
    assert read_reference_identity(path) == identity
    payload = identity.to_dict()
    payload["schema_version"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        read_reference_identity(path)


def test_reference_identity_reads_legacy_probe_fields_and_omits_them_on_write(tmp_path):
    path = tmp_path / "relax_dpo_reference.json"
    payload = {
        "schema_version": 1,
        "repository": "repo",
        "revision": "revision",
        "loader_mode": "loader",
        "parameter_sha256": "a" * 64,
        "probe_sha256": "b" * 64,
        "probe_manifest": {"tokens": [[1, 2], [1, 3]], "loss_masks": [[0, 1], [0, 1]]},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    identity = read_reference_identity(path)
    assert identity == DPOReferenceIdentity(1, "repo", "revision", "loader", "a" * 64)
    write_reference_identity(path, identity)
    assert json.loads(path.read_text()) == {
        key: value for key, value in payload.items() if key not in {"probe_sha256", "probe_manifest"}
    }


def _write_local_download_metadata(checkpoint, filename, revision):
    source = checkpoint / filename
    metadata = checkpoint / ".cache" / "huggingface" / "download" / f"{filename}.metadata"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    content = source.read_bytes()
    if source.suffix == ".safetensors":
        etag = hashlib.sha256(content).hexdigest()
    else:
        etag = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    metadata.write_text(f"{revision}\n{etag}\n{source.stat().st_mtime}\n", encoding="utf-8")


def test_resolve_dpo_reference_checkpoint_uses_the_pinned_configured_local_snapshot(monkeypatch, tmp_path):
    revision = "a" * 40
    snapshot = tmp_path / f"Qwen3-0.6B-{revision}"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")
    _write_local_download_metadata(snapshot, "config.json", revision)
    _write_local_download_metadata(snapshot, "model.safetensors", revision)
    observed = {}

    def snapshot_download(**kwargs):
        observed.update(kwargs)
        return str(snapshot)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=snapshot_download),
    )
    assert resolve_dpo_reference_checkpoint("org/model", revision, str(snapshot)) == str(snapshot.resolve())
    assert observed == {
        "repo_id": "org/model",
        "revision": revision,
        "local_dir": str(snapshot.resolve()),
        "local_files_only": True,
    }


def test_resolve_dpo_reference_checkpoint_rejects_missing_local_snapshot(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    def snapshot_download(**_kwargs):
        raise OSError("not cached")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=snapshot_download),
    )
    with pytest.raises(RuntimeError, match="hf download org/model --revision"):
        resolve_dpo_reference_checkpoint("org/model", "a" * 40, str(checkpoint))


def test_resolve_dpo_reference_checkpoint_rejects_different_resolved_directory(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    other = tmp_path / "other"
    checkpoint.mkdir()
    other.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (other / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=lambda **_kwargs: str(other)),
    )
    with pytest.raises(RuntimeError, match="different from --hf-checkpoint"):
        resolve_dpo_reference_checkpoint("org/model", "a" * 40, str(checkpoint))


def test_resolve_dpo_reference_checkpoint_rejects_missing_pinned_local_metadata(monkeypatch, tmp_path):
    revision = "a" * 40
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=lambda **_kwargs: str(checkpoint)),
    )
    with pytest.raises(RuntimeError, match="missing valid Hugging Face local-dir metadata"):
        resolve_dpo_reference_checkpoint("org/model", revision, str(checkpoint))


def test_resolve_dpo_reference_checkpoint_rejects_mismatched_file_metadata(monkeypatch, tmp_path):
    revision = "a" * 40
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    _write_local_download_metadata(checkpoint, "config.json", "c" * 40)
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    _write_local_download_metadata(checkpoint, "model.safetensors", revision)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=lambda **_kwargs: str(checkpoint)),
    )
    with pytest.raises(RuntimeError, match="metadata does not match the pinned revision"):
        resolve_dpo_reference_checkpoint("org/model", revision, str(checkpoint))


def test_resolve_dpo_reference_checkpoint_rejects_replaced_file_with_restored_mtime(monkeypatch, tmp_path):
    revision = "a" * 40
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = checkpoint / "config.json"
    config.write_text("{}", encoding="utf-8")
    _write_local_download_metadata(checkpoint, "config.json", revision)
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"weights")
    original_stat = weights.stat()
    _write_local_download_metadata(checkpoint, "model.safetensors", revision)
    weights.write_bytes(b"changed")
    os.utime(weights, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=lambda **_kwargs: str(checkpoint)),
    )
    with pytest.raises(RuntimeError, match="contents do not match its Hugging Face ETag"):
        resolve_dpo_reference_checkpoint("org/model", revision, str(checkpoint))


def test_resolve_dpo_reference_checkpoint_rejects_snapshot_without_supported_weights(monkeypatch, tmp_path):
    revision = "a" * 40
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    _write_local_download_metadata(checkpoint, "config.json", revision)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=lambda **_kwargs: str(checkpoint)),
    )
    with pytest.raises(RuntimeError, match="no supported model weights or index"):
        resolve_dpo_reference_checkpoint("org/model", revision, str(checkpoint))


def test_resolve_dpo_reference_checkpoint_accepts_complete_safetensors_index(monkeypatch, tmp_path):
    revision = "a" * 40
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    index_name = "model.safetensors.index.json"
    shard_names = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    weight_map = {f"layer.{index}.weight": shard for index, shard in enumerate(shard_names)}
    (checkpoint / index_name).write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    for shard_name in shard_names:
        (checkpoint / shard_name).write_bytes(shard_name.encode())
    for filename in ("config.json", index_name, *shard_names):
        _write_local_download_metadata(checkpoint, filename, revision)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=lambda **_kwargs: str(checkpoint)),
    )
    assert resolve_dpo_reference_checkpoint("org/model", revision, str(checkpoint)) == str(checkpoint.resolve())


def test_resolve_dpo_reference_checkpoint_rejects_missing_indexed_shard_and_metadata(monkeypatch, tmp_path):
    revision = "a" * 40
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    index_name = "model.safetensors.index.json"
    shard_name = "model-00001-of-00001.safetensors"
    (checkpoint / index_name).write_text(json.dumps({"weight_map": {"model.weight": shard_name}}), encoding="utf-8")
    (checkpoint / shard_name).write_bytes(b"weights")
    for filename in ("config.json", index_name, shard_name):
        _write_local_download_metadata(checkpoint, filename, revision)
    (checkpoint / shard_name).unlink()
    (checkpoint / ".cache" / "huggingface" / "download" / f"{shard_name}.metadata").unlink()

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=lambda **_kwargs: str(checkpoint)),
    )
    with pytest.raises(RuntimeError, match="weight index points to a missing shard"):
        resolve_dpo_reference_checkpoint("org/model", revision, str(checkpoint))


@pytest.fixture
def reference_actor_methods():
    """Load the real lifecycle methods without importing the GPU actor
    stack."""
    source = Path(__file__).resolve().parents[3] / "relax/backends/megatron/actor.py"
    tree = ast.parse(source.read_text())
    actor = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MegatronTrainRayActor")
    method_names = {
        "_switch_model",
        "_is_standard_dpo",
        "_assert_dpo_reference_identity",
        "_rebuild_dpo_reference",
        "save_model",
    }
    methods = [node for node in actor.body if isinstance(node, ast.FunctionDef) and node.name in method_names]
    for method in methods:
        method.decorator_list = []
    namespace = {
        "DPOReferenceIdentity": DPOReferenceIdentity,
        "REFERENCE_LOADER_MODE": REFERENCE_LOADER_MODE,
        "canonical_tensor_sha256": canonical_tensor_sha256,
        "is_preference_mode": is_preference_mode,
        "reference_identity_path": reference_identity_path,
        "write_reference_identity": write_reference_identity,
        "device_utils": types.SimpleNamespace(maybe_backend_process_on_model_switch=lambda: None),
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
    return type("ReferenceActor", (), {name: namespace[name] for name in method_names}), namespace


@pytest.mark.parametrize("outcome", ["success", "loader_failure", "identity_mismatch"])
def test_reference_rebuild_preserves_actor_and_optimizer(monkeypatch, reference_actor_methods, outcome):
    actor_type, namespace = reference_actor_methods
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    parameter.square().sum().backward()
    optimizer.step()
    actor_value = parameter.detach().clone()
    monkeypatch.setattr(tensor_backper, "_PIN_MEMORY", False)
    monkeypatch.setattr(tensor_backper, "_NON_BLOCKING", False)
    monkeypatch.setattr(tensor_backper.device_module, "synchronize", lambda: None)

    instance = actor_type()
    instance.args = Namespace(
        load="checkpoint",
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
        megatron_to_hf_mode="bridge",
        dpo_reference_repository="repo",
        dpo_reference_revision="revision",
    )
    original_args = vars(instance.args).copy()
    instance.model = [object()]
    instance.optimizer = optimizer
    instance.weights_backuper = tensor_backper.TensorBackuper.create(lambda: [("weight", parameter)], single_tag=None)
    instance.weights_backuper.backup("actor")
    instance._active_model_tag = "actor"
    instance._expected_dpo_reference_identity = None
    if outcome == "identity_mismatch":
        instance._expected_dpo_reference_identity = DPOReferenceIdentity(
            1, "repo", "revision", REFERENCE_LOADER_MODE, "a" * 64
        )
    instance._assert_dp_reference_digest_equal = Mock()

    def load_reference(model, loaded_optimizer, scheduler, **kwargs):
        assert model is instance.model
        assert loaded_optimizer is scheduler is None
        assert (
            instance.args.load,
            instance.args.no_load_optim,
            instance.args.no_load_rng,
            instance.args.finetune,
        ) == ("hf-path", True, True, True)
        parameter.data.fill_(99)
        if outcome == "loader_failure":
            raise RuntimeError("injected loader failure")

    namespace["load_checkpoint"] = load_reference
    namespace["named_params_and_buffers"] = lambda *args, **kwargs: [("weight", parameter)]
    optimizer_state_before = {name: value.clone() for name, value in optimizer.state[parameter].items()}
    errors = {
        "loader_failure": "injected loader failure",
        "identity_mismatch": "frozen-reference identity mismatch",
    }
    with pytest.raises(RuntimeError, match=errors[outcome]) if outcome in errors else nullcontext():
        instance._rebuild_dpo_reference("hf-path")
    torch.testing.assert_close(parameter, actor_value)
    assert instance._active_model_tag == "actor"
    assert vars(instance.args) == original_args
    assert optimizer.state[parameter].keys() == optimizer_state_before.keys()
    for name, expected in optimizer_state_before.items():
        torch.testing.assert_close(optimizer.state[parameter][name], expected, rtol=0, atol=0)
    if outcome in {"loader_failure", "identity_mismatch"}:
        assert "ref" not in instance.weights_backuper.backup_tags
    elif outcome == "success":
        reference = instance.weights_backuper.get("ref")["weight"]
        torch.testing.assert_close(reference, torch.tensor([99.0]))
        parameter.data.add_(1)
        torch.testing.assert_close(reference, torch.tensor([99.0]))
        assert instance._dpo_reference_identity.parameter_sha256 == canonical_tensor_sha256([("weight", reference)])


def test_save_model_persists_reference_identity(tmp_path, reference_actor_methods):
    actor_type, namespace = reference_actor_methods
    instance = actor_type()
    instance.args = Namespace(
        loss_type="sft",
        sft_objective="dpo",
        dpo_reference_free=False,
        debug_rollout_only=False,
        offload_train=False,
        async_save=False,
        save=str(tmp_path),
        save_hf=None,
    )
    instance.role = "actor"
    instance.model = [object()]
    instance.optimizer = instance.opt_param_scheduler = None
    instance._dpo_reference_identity = DPOReferenceIdentity(1, "repo", "revision", REFERENCE_LOADER_MODE, "a" * 64)
    namespace.update(
        dist=types.SimpleNamespace(get_rank=lambda **kwargs: 0, barrier=lambda **kwargs: None),
        get_gloo_group=lambda: None,
        rotate_ckpt=Mock(),
        save=Mock(),
    )
    instance.save_model(7)
    namespace["save"].assert_called_once_with(7, instance.model, None, None, lora_only=False)
    path = reference_identity_path(tmp_path, 7)
    assert read_reference_identity(path) == instance._dpo_reference_identity
    assert set(json.loads(path.read_text())) == {
        "schema_version",
        "repository",
        "revision",
        "loader_mode",
        "parameter_sha256",
    }
