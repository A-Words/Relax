# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pure numerical and batching tests for offline preference training."""

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from relax.utils.training.preference_utils import (
    build_causal_lm_labels,
    dpo_pair_loss,
    pack_preference_pair_indices,
    require_tensor_condition,
    reward_model_pair_loss,
    select_packed_sequence_scores,
)


def test_tensor_condition_uses_async_assert_without_python_bool_on_cuda(monkeypatch):
    class _CudaCondition:
        device = SimpleNamespace(type="cuda")

        def __bool__(self):
            raise AssertionError("CUDA conditions must not be converted to Python bool")

    calls = []
    monkeypatch.setattr(torch, "_assert_async", lambda condition, message: calls.append((condition, message)))
    condition = _CudaCondition()

    require_tensor_condition(condition, "finite")

    assert calls == [(condition, "finite")]


def test_build_causal_lm_labels_uses_next_token_mask():
    tokens = torch.tensor([10, 11, 12, 13, 14])
    raw_mask = torch.tensor([0, 0, 1, 1, 1])

    labels = build_causal_lm_labels(tokens, raw_mask)

    assert labels.tolist() == [-100, 12, 13, 14, -100]


@pytest.mark.parametrize("beta", [0.01, 0.1, 1.0])
def test_dpo_pair_loss_matches_independent_reference_and_gradient(beta: float):
    policy_chosen = torch.tensor([-2.0, -0.25, 4.0], dtype=torch.float32, requires_grad=True)
    policy_rejected = torch.tensor([-3.0, 0.75, -1.0], dtype=torch.float32, requires_grad=True)
    ref_chosen = torch.tensor([-2.5, 0.5, 1.0], dtype=torch.float32)
    ref_rejected = torch.tensor([-2.0, -0.5, -2.0], dtype=torch.float32)

    actual = dpo_pair_loss(
        policy_chosen,
        policy_rejected,
        reference_chosen=ref_chosen,
        reference_rejected=ref_rejected,
        beta=beta,
    )
    expected = -F.logsigmoid(beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)))
    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)

    actual.sum().backward()
    actual_grad = policy_chosen.grad.detach().clone()
    policy_chosen.grad = None
    expected.sum().backward()
    assert torch.allclose(actual_grad, policy_chosen.grad, rtol=1e-6, atol=1e-6)


def test_dpo_pair_loss_reference_free_matches_independent_reference():
    chosen = torch.tensor([-1.0, 2.0], requires_grad=True)
    rejected = torch.tensor([0.5, -2.0], requires_grad=True)

    actual = dpo_pair_loss(chosen, rejected, beta=0.1, reference_free=True)
    expected = -F.logsigmoid(0.1 * (chosen - rejected))

    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_dpo_pair_loss_rejects_missing_reference_and_non_finite_values():
    finite = torch.tensor([0.0])
    with pytest.raises(ValueError, match="reference log-probabilities"):
        dpo_pair_loss(finite, finite)
    with pytest.raises(ValueError, match="finite"):
        dpo_pair_loss(torch.tensor([float("nan")]), finite, reference_free=True)


def test_reward_model_pair_loss_matches_independent_reference_and_gradient():
    chosen = torch.tensor([1.0, -1.0, 0.0], requires_grad=True)
    rejected = torch.tensor([0.0, 2.0, 0.0], requires_grad=True)

    actual = reward_model_pair_loss(chosen, rejected)
    expected = -F.logsigmoid(chosen - rejected)
    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)

    actual.sum().backward()
    actual_grad = chosen.grad.detach().clone()
    chosen.grad = None
    expected.sum().backward()
    assert torch.allclose(actual_grad, chosen.grad, rtol=1e-6, atol=1e-6)


def test_reward_model_pair_loss_uses_shared_tensor_condition(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "relax.utils.training.preference_utils.require_tensor_condition",
        lambda condition, message: calls.append((condition, message)),
    )

    reward_model_pair_loss(torch.tensor([1.0]), torch.tensor([0.0]))

    assert len(calls) == 1
    assert "finite" in calls[0][1]


def test_reward_model_pair_loss_rejects_non_finite_cpu_margin():
    with pytest.raises(ValueError, match="finite"):
        reward_model_pair_loss(torch.tensor([float("inf")]), torch.tensor([0.0]))


def test_select_packed_sequence_scores_preserves_pair_order_and_gradient():
    flat_logits = torch.arange(10, dtype=torch.float32, requires_grad=True)
    logits = flat_logits.reshape(1, 10, 1)

    scores = select_packed_sequence_scores(logits, [3, 2, 4], [2, 1, 3])

    assert scores.tolist() == [2.0, 4.0, 8.0]
    scores.sum().backward()
    expected = torch.zeros(10)
    expected[[2, 4, 8]] = 1
    assert torch.equal(flat_logits.grad, expected)


def test_select_packed_sequence_scores_rejects_invalid_position():
    with pytest.raises(ValueError, match="outside"):
        select_packed_sequence_scores(torch.zeros(1, 4, 1), [4], [4])


def test_pair_packer_is_deterministic_complete_and_capacity_safe():
    costs = [2, 4, 4, 5, 5]
    pair_ids = ["a", "b", "c", "d", "e"]

    bins = pack_preference_pair_indices(costs, pair_ids, capacity=10)

    assert sorted(index for group in bins for index in group) == list(range(len(costs)))
    assert all(sum(costs[index] for index in group) <= 10 for group in bins)
    assert bins == pack_preference_pair_indices(costs, pair_ids, capacity=10)


def test_pair_packer_reports_oversize_pair():
    with pytest.raises(ValueError, match="oversize.*pair-a.*11"):
        pack_preference_pair_indices([11], ["pair-a"], capacity=10)


def test_pinned_seqlen_sampler_consumes_pair_costs_and_keeps_equal_dp_groups():
    """Exercise the Docker-pinned TransferQueue sampler, not a local stand-
    in."""
    transfer_queue = pytest.importorskip("transfer_queue")
    sampler_type = transfer_queue.SeqlenBalancedSampler
    source_path = Path(inspect.getsourcefile(sampler_type) or "")
    normalized_source = source_path.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(normalized_source).hexdigest() == (
        "dc6c2db50df4b9448d4845ccacef67a400517db892b5cd55de2e22f6baf6888b"
    )

    class PairPartition:
        def __init__(self, pair_rows):
            self.requested_indexes = None
            self.metadata = {
                index: {"total_lengths": len(row["chosen_tokens"]) + len(row["rejected_tokens"])}
                for index, row in enumerate(pair_rows)
            }

        def get_custom_meta(self, indexes):
            self.requested_indexes = list(indexes)
            return {index: self.metadata[index] for index in indexes}

    rows = [
        {"pair_id": 100, "chosen_tokens": list(range(60)), "rejected_tokens": list(range(40))},
        {"pair_id": 101, "chosen_tokens": list(range(50)), "rejected_tokens": list(range(40))},
        {"pair_id": 102, "chosen_tokens": list(range(6)), "rejected_tokens": list(range(4))},
        {"pair_id": 103, "chosen_tokens": [0], "rejected_tokens": [1]},
    ]
    partition = PairPartition(rows)
    sampler = sampler_type(n_samples_per_prompt=1, dp_size=2)
    assignments = []
    for rank in range(2):
        sampled, consumed = sampler.sample(
            [0, 1, 2, 3],
            batch_size=2,
            task_name="dpo",
            partition_id="train_0",
            dp_rank=rank,
            batch_index=0,
            partition=partition,
        )
        assert sampled == consumed
        assert len(sampled) == 2
        assignments.append(sampled)
        for index in sampled:
            assert rows[index]["pair_id"] == 100 + index
            assert set(rows[index]) == {"pair_id", "chosen_tokens", "rejected_tokens"}

    assert partition.requested_indexes == [0, 1, 2, 3]
    assert sorted(index for rank_rows in assignments for index in rank_rows) == [0, 1, 2, 3]
    rank_costs = [sum(partition.metadata[index]["total_lengths"] for index in rank_rows) for rank_rows in assignments]
    assert sorted(rank_costs) == [100, 102]
