import sys
import types
import unittest
from unittest.mock import AsyncMock

import torch


if "lagent.utils" not in sys.modules:
    lagent_module = types.ModuleType("lagent")
    lagent_utils_module = types.ModuleType("lagent.utils")
    lagent_rate_limiter_module = types.ModuleType("lagent.utils.rate_limiter")
    lagent_utils_module.create_object = lambda config: config
    lagent_rate_limiter_module.get_shared_async_token_bucket = lambda *args, **kwargs: None
    lagent_module.utils = lagent_utils_module
    sys.modules.setdefault("lagent", lagent_module)
    sys.modules.setdefault("lagent.utils", lagent_utils_module)
    sys.modules.setdefault("lagent.utils.rate_limiter", lagent_rate_limiter_module)


from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.agent_in_sandbox_loop import AgentInSandboxLoop
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import AgentRolloutItem, RolloutStatus
from xtuner.v1.rl.distillation.rollout_teacher_client import RolloutTeacherClient
from xtuner.v1.rl.trainer.controller import TrainingController


def _state(*, data_source: str = "agent") -> RolloutState:
    return RolloutState(
        rollout_id=1,
        group_id=1,
        message=[],
        status=Status.COMPLETED,
        reward={"score": 1.0},
        extra_fields={"origin_data_source": data_source},
    )


class TestAgenticTeacherScoring(unittest.TestCase):
    def setUp(self) -> None:
        self.client = RolloutTeacherClient.__new__(RolloutTeacherClient)
        self.client.name = "teacher"

    def test_prepare_scoring_input_keeps_masked_turns_in_suffix(self):
        state = _state()
        state.input_ids = [10, 11, 20, 30, 31, 40]
        state.labels = [-100, -100, 20, -100, -100, 40]

        scoring_input = self.client._prepare_scoring_input(state)

        self.assertEqual(scoring_input, ([10, 11], [20, 30, 31, 40]))
        self.assertEqual(state.status, Status.COMPLETED)

    def test_prepare_scoring_input_rejects_trace_without_trainable_labels(self):
        state = _state()
        state.input_ids = [10, 11, 12]
        state.labels = [-100, -100, -100]

        self.assertIsNone(self.client._prepare_scoring_input(state))
        self.assertEqual(state.status, Status.FAILED)
        self.assertIn("at least one trainable label", state.error_msg)


class TestAgenticRolloutCollection(unittest.IsolatedAsyncioTestCase):
    async def test_scores_each_flattened_trace_with_its_routed_teacher(self):
        first = _state(data_source="first")
        second = _state(data_source="second")
        first_teacher = AsyncMock()
        first_teacher.compute_logprobs = AsyncMock(side_effect=lambda state: state)
        second_teacher = AsyncMock()
        second_teacher.compute_logprobs = AsyncMock(side_effect=lambda state: state)

        loop = AgentInSandboxLoop.__new__(AgentInSandboxLoop)
        loop.generate_group = AsyncMock(return_value=[first, second])
        loop.teacher_clients = {"teacher-a": first_teacher, "teacher-b": second_teacher}
        loop.data_source_teacher_map = {"first": "teacher-a", "second": "teacher-b"}

        result = await loop.collect_rollout_group([_state()])

        self.assertEqual(result, [first, second])
        first_teacher.compute_logprobs.assert_awaited_once_with(first)
        second_teacher.compute_logprobs.assert_awaited_once_with(second)

    async def test_agent_rollout_preserves_origin_data_source(self):
        loop = AgentInSandboxLoop.__new__(AgentInSandboxLoop)
        loop.mode = "train"
        state = _state()
        item = AgentRolloutItem(
            id="sample",
            data_source="agent-source",
            instruction="instruction.md",
            status=RolloutStatus.FAILED,
        )

        result = await loop._build_rollout_states(state, item)

        self.assertEqual(result[0].extra_fields["origin_data_source"], "agent-source")


class TestMixedPositionIdPacking(unittest.TestCase):
    def test_text_position_ids_expand_when_packed_with_vl_sample(self):
        text_ctx = SequenceContext.from_input_ids((torch.tensor([[1, 2]]),), device="cpu")
        vl_ctx = SequenceContext.from_input_ids((torch.tensor([[3, 4]]),), device="cpu")
        vl_ctx.position_ids = vl_ctx.position_ids.unsqueeze(0).expand(3, -1, -1)
        data_batches = [
            {
                "seq_ctx": text_ctx,
                "shifted_labels": torch.tensor([[1, 2]]),
                "advantage": [1.0, 1.0],
                "rollout_logprobs": None,
            },
            {
                "seq_ctx": vl_ctx,
                "shifted_labels": torch.tensor([[3, 4]]),
                "advantage": [1.0, 1.0],
                "rollout_logprobs": None,
            },
        ]

        packed = TrainingController(workers=[])._packing(data_batches, pack_max_length=4, language_cfg=None)

        self.assertEqual(tuple(text_ctx.position_ids.shape), (3, 1, 2))
        self.assertEqual(tuple(packed[0]["seq_ctx"].position_ids.shape), (3, 1, 4))


if __name__ == "__main__":
    unittest.main()
