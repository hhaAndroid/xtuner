import asyncio
import json

from xtuner.v1.rl.rollout.session_server import FMT_OPENAI, SessionServer


def test_lmdeploy_request_is_not_adapted():
    """SGLang compatibility translation must never leak into LMDeploy."""
    request = {
        "top_k": 0,
        "return_logprob": True,
        "return_token_ids": True,
        "return_routed_experts": True,
        "input_ids": [1, 2, 3],
        "messages": [],
    }

    for backend in (None, "lmdeploy"):
        server = object.__new__(SessionServer)
        server.rollout_backend = backend
        worker_request = request.copy()

        translated = server._adapt_worker_request(worker_request)

        assert translated is worker_request
        assert translated == request


def test_lmdeploy_routed_experts_keep_original_shared_store_namespace(monkeypatch):
    calls = []

    class FakeGetMethod:
        async def remote(self, key):
            assert key == "lmdeploy-shared-store-key"
            return [[[7]]]

    class FakeActor:
        get = FakeGetMethod()

    def fake_get_actor(name, namespace):
        calls.append((name, namespace))
        return FakeActor()

    monkeypatch.setattr("xtuner.v1.rl.rollout.session_server.ray.get_actor", fake_get_actor)
    server = object.__new__(SessionServer)
    server.rollout_backend = "lmdeploy"
    server._shared_store_actor = None

    decoded = asyncio.run(server._decode_routed_experts("lmdeploy-shared-store-key"))

    assert calls == [("shared_store", "lmdeploy")]
    assert decoded.tolist() == [[[7]]]


def test_sglang_request_translation():
    server = object.__new__(SessionServer)
    server.rollout_backend = "sglang"

    request = {
        "top_k": 0,
        "return_logprob": True,
        "return_token_ids": True,
        "input_ids": [1, 2, 3],
    }
    translated = server._adapt_worker_request(request)

    assert translated["top_k"] == -1
    assert translated["logprobs"] is True
    assert "return_logprob" not in translated
    # These fields are part of XTuner's SGLang sglext protocol. They must stay
    # on the request so the streaming response can carry exact training ids.
    assert translated["return_token_ids"] is True
    assert translated["input_ids"] == [1, 2, 3]


def test_parse_lmdeploy_extension_stream_unchanged():
    events = [
        {
            "id": "chatcmpl-test",
            "model": "test-model",
            "choices": [
                {
                    "delta": {"role": "assistant", "content": "answer"},
                    "output_ids": [10],
                    "output_token_logprobs": [[-0.25, 10]],
                    "finish_reason": None,
                }
            ],
        },
        {
            "choices": [
                {
                    "delta": {"content": "!"},
                    "output_ids": [11],
                    "output_token_logprobs": [[-0.5, 11]],
                    "routed_experts": "lmdeploy-shared-store-key",
                    "finish_reason": "stop",
                }
            ]
        },
    ]
    raw = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"

    parsed = SessionServer._parse_stream_response(raw.encode(), FMT_OPENAI)

    assert parsed is not None
    choice = parsed["choices"][0]
    assert choice["message"]["content"] == "answer!"
    assert choice["output_ids"] == [10, 11]
    assert choice["output_token_logprobs"] == [[-0.25, 10], [-0.5, 11]]
    assert choice["routed_experts"] == "lmdeploy-shared-store-key"


def test_parse_sglang_standard_stream_logprobs():
    events = [
        {
            "id": "chatcmpl-test",
            "model": "test-model",
            "choices": [
                {
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        },
        {
            "choices": [
                {
                    "delta": {"content": "answer"},
                    "logprobs": {
                        "content": [
                            {"token": "answer", "bytes": list(b"answer"), "logprob": -0.25, "top_logprobs": []}
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {},
                    "logprobs": {
                        "content": [
                            {"token": "<eos>", "bytes": [], "logprob": -0.5, "top_logprobs": []}
                        ]
                    },
                    "finish_reason": "stop",
                }
            ]
        },
    ]
    raw = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"

    parsed = SessionServer._parse_stream_response(raw.encode(), FMT_OPENAI)

    assert parsed is not None
    assert parsed["choices"][0]["message"]["content"] == "answer"
    assert parsed["choices"][0]["_standard_output_logprobs"] == [-0.25, -0.5]
