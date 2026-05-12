"""Unit tests for :class:`Sampler.sample_prompt` and Phase 1 sampler changes.

Covers the new trajectory-level entry point, legacy ``sample_from_dataloader``
byte-compatibility, and the ``prompt_repeat_k`` deprecation warning.
"""

import unittest
import warnings
from unittest.mock import MagicMock, patch

from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.rl.agent_loop_manager.sampler import Sampler, SamplerConfig
from xtuner.v1.rl.agent_loop_manager.trajectory_scheduler import PromptRequest
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig


_DETERMINISTIC_FLAG = "xtuner.v1.rl.agent_loop_manager.sampler.XTUNER_DETERMINISTIC"


def _build_dataloader(items):
    dl = MagicMock()
    dl.__iter__.side_effect = lambda: iter(items)
    dl.set_epoch = MagicMock()
    return dl


def _build_sampler(items=None, prompt_repeat_k=1):
    if items is None:
        items = [
            [RolloutState(message=[{"role": "user", "content": f"q{i}"}])]
            for i in range(10)
        ]
    dl = _build_dataloader(items)
    replay_buffer = AsyncReplayBufferConfig().build()
    return Sampler(dataloader=dl, prompt_repeat_k=prompt_repeat_k, replay_buffer=replay_buffer)


class TestSamplePrompt(unittest.IsolatedAsyncioTestCase):
    async def test_returns_prompt_request(self):
        sampler = _build_sampler()
        req = await sampler.sample_prompt(task_name="gsm8k")
        self.assertIsInstance(req, PromptRequest)
        self.assertEqual(req.task_name, "gsm8k")
        self.assertIsNotNone(req.prompt.message_uid)
        self.assertEqual(req.prompt_uid, req.prompt.message_uid)

    async def test_prompt_uid_is_int_even_without_dataloader_uid(self):
        # Dataloader items have message_uid=None by default.
        sampler = _build_sampler()
        req = await sampler.sample_prompt(task_name="t")
        self.assertIsInstance(req.prompt_uid, int)

    async def test_consecutive_calls_yield_different_uids(self):
        sampler = _build_sampler()
        req1 = await sampler.sample_prompt(task_name="t")
        req2 = await sampler.sample_prompt(task_name="t")
        self.assertNotEqual(req1.prompt_uid, req2.prompt_uid)

    async def test_deepcopy_does_not_mutate_dataloader_item(self):
        original_message = [{"role": "user", "content": "q0"}]
        shared_item = RolloutState(message=original_message)
        sampler = _build_sampler(items=[[shared_item]])
        req = await sampler.sample_prompt(task_name="t")
        req.prompt.message.append({"role": "leaked"})
        # Underlying dataloader item should not see the mutation.
        self.assertEqual(len(original_message), 1)

    async def test_non_deterministic_does_not_stamp_dataloader_message_uid(self):
        # In non-deterministic mode the sampler must leave the dataloader's
        # original message_uid untouched; the fallback uuid goes only onto
        # the deep-copied PromptRequest.prompt.
        shared_item = RolloutState(message=[{"role": "user", "content": "q"}])
        sampler = _build_sampler(items=[[shared_item]])
        with patch(_DETERMINISTIC_FLAG, False):
            req = await sampler.sample_prompt(task_name="t")
        self.assertIsNone(shared_item.message_uid)
        self.assertIsNotNone(req.prompt.message_uid)

    async def test_advances_consumed_samples(self):
        sampler = _build_sampler()
        self.assertEqual(sampler._consumed_samples, 0)
        await sampler.sample_prompt(task_name="t")
        self.assertEqual(sampler._consumed_samples, 1)
        await sampler.sample_prompt(task_name="t")
        self.assertEqual(sampler._consumed_samples, 2)


class TestSamplePromptDeterministic(unittest.IsolatedAsyncioTestCase):
    async def test_deterministic_message_uid_matches_counter(self):
        with patch(_DETERMINISTIC_FLAG, True):
            sampler = _build_sampler()
            req1 = await sampler.sample_prompt(task_name="t")
            req2 = await sampler.sample_prompt(task_name="t")
            req3 = await sampler.sample_prompt(task_name="t")
        self.assertEqual(req1.prompt_uid, 0)
        self.assertEqual(req2.prompt_uid, 1)
        self.assertEqual(req3.prompt_uid, 2)


class TestLegacySampleFromDataloader(unittest.TestCase):
    def test_deterministic_group_uids_match_legacy_scheme(self):
        with patch(_DETERMINISTIC_FLAG, True):
            sampler = _build_sampler(prompt_repeat_k=4)
            g0 = sampler.sample_from_dataloader()
            g1 = sampler.sample_from_dataloader()
        self.assertEqual([item.message_uid for item in g0], [0, 0, 0, 0])
        self.assertEqual([item.uid for item in g0], [0, 1, 2, 3])
        self.assertEqual([item.session_uid for item in g0], [0, 1, 2, 3])
        self.assertEqual([item.message_uid for item in g1], [1, 1, 1, 1])
        self.assertEqual([item.uid for item in g1], [4, 5, 6, 7])
        self.assertEqual([item.session_uid for item in g1], [4, 5, 6, 7])

    def test_deterministic_group_size_one(self):
        with patch(_DETERMINISTIC_FLAG, True):
            sampler = _build_sampler(prompt_repeat_k=1)
            g0 = sampler.sample_from_dataloader()
            g1 = sampler.sample_from_dataloader()
        self.assertEqual(len(g0), 1)
        self.assertEqual(g0[0].message_uid, 0)
        self.assertEqual(g0[0].uid, 0)
        self.assertEqual(g1[0].message_uid, 1)
        self.assertEqual(g1[0].uid, 1)

    def test_non_deterministic_uids_are_unique_per_item(self):
        with patch(_DETERMINISTIC_FLAG, False):
            sampler = _build_sampler(prompt_repeat_k=3)
            g = sampler.sample_from_dataloader()
        self.assertEqual(len({item.uid for item in g}), 3)

    def test_consumed_samples_increments_per_group(self):
        with patch(_DETERMINISTIC_FLAG, True):
            sampler = _build_sampler(prompt_repeat_k=4)
            self.assertEqual(sampler._consumed_samples, 0)
            sampler.sample_from_dataloader()
            self.assertEqual(sampler._consumed_samples, 1)
            sampler.sample_from_dataloader()
            self.assertEqual(sampler._consumed_samples, 2)


class TestSamplerConfigDeprecationWarning(unittest.TestCase):
    def _build_env(self):
        dl_cfg = MagicMock()
        dl_cfg.build.return_value = _build_dataloader(
            [[RolloutState(message=[])] for _ in range(3)]
        )
        tokenizer = MagicMock()
        replay_buffer = AsyncReplayBufferConfig().build()
        return dl_cfg, tokenizer, replay_buffer

    def test_no_warning_at_default_repeat_k(self):
        dl_cfg, tokenizer, replay_buffer = self._build_env()
        cfg = SamplerConfig.model_construct(dataloader_cfg=dl_cfg)
        self.assertEqual(cfg.prompt_repeat_k, 1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            cfg.build(tokenizer, replay_buffer)
        sampler_warnings = [
            w for w in caught
            if issubclass(w.category, DeprecationWarning)
            and "prompt_repeat_k" in str(w.message)
        ]
        self.assertEqual(len(sampler_warnings), 0)

    def test_warning_when_repeat_k_above_one(self):
        dl_cfg, tokenizer, replay_buffer = self._build_env()
        cfg = SamplerConfig.model_construct(dataloader_cfg=dl_cfg, prompt_repeat_k=4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            cfg.build(tokenizer, replay_buffer)
        sampler_warnings = [
            w for w in caught
            if issubclass(w.category, DeprecationWarning)
            and "prompt_repeat_k" in str(w.message)
        ]
        self.assertEqual(len(sampler_warnings), 1)


if __name__ == "__main__":
    unittest.main()
