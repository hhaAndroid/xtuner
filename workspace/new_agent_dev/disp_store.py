#!/usr/bin/env python3
import inspect
import os
import time

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

import ray

_STORE_NAME = "rollout_trace_store"
_STORE_NAMESPACE = "xtuner_rollout"
INTERVAL = 2
LIMIT = 10
SHOW_TRACE = True
TRACE_TOKEN_PREVIEW = 20
RESOLVE_ROUTED_EXPERTS = False


def summarize_routed_expert(value):
    if value is None:
        return None
    if isinstance(value, ray.ObjectRef):
        if not RESOLVE_ROUTED_EXPERTS:
            return f"ObjectRef({value.hex()[:12]}...)"
        value = ray.get(value)
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return f"{type(value).__name__}(shape={tuple(value.shape)}, dtype={value.dtype})"
    return repr(value)


def show_raw_search(store, session_id: str, prompt_text: str):
    matched_key, nodes = ray.get(store.search.remote(session_id, prompt_text, filter_none=True))
    print(f"    raw_search matched={matched_key == prompt_text} matched_len={len(matched_key)} nodes={len(nodes)}")
    for node_idx, node in enumerate(nodes):
        value = node.value
        if value is None:
            print(f"      node#{node_idx} value=None")
            continue

        # text = value.text.replace("\n", "\\n")
        text = value.text
        # if len(text) > 160:
        #     text = text[:160] + "..."
        print(f"      node#{node_idx} text_len={len(value.text)} text={text}")
        print(f"        length={value.length}")
        print(f"        token_ids_len={len(value.token_ids)} token_ids[:{TRACE_TOKEN_PREVIEW}]={value.token_ids[:TRACE_TOKEN_PREVIEW]}")
        print(f"        labels_len={len(value.labels)} labels[:{TRACE_TOKEN_PREVIEW}]={value.labels[:TRACE_TOKEN_PREVIEW]}")
        print(f"        logprobs_len={len(value.logprobs)} logprobs[:{TRACE_TOKEN_PREVIEW}]={value.logprobs[:TRACE_TOKEN_PREVIEW]}")
        print(f"        expert_key={summarize_routed_expert(value.expert_key)}")


def show_session(store, session_id: str, limit: int):
    keys = ray.get(store.keys.remote(session_id))
    print(f"\n[session={session_id}] keys={len(keys)}")
    for key in keys:
        print('--------------------------------\n')
        print(f"{len(key)}")
        print('--------------------------------\n')
    # shown_keys = keys[-limit:]
    # for i, key in enumerate(shown_keys, start=max(0, len(keys) - limit)):
    #     text = key.replace("\n", "\\n")
    #     if len(text) > 200:
    #         text = text[:200] + "..."
    #     print(f"  #{i} len={len(key)} {text}")
    #     if SHOW_TRACE:
    #         show_raw_search(store, session_id, key)
    

    objects = ray.get(store.get_objects.remote(keys))
    print(f"objects={len(objects)} {objects}")
    for i, obj in enumerate(objects):
        print(f"  #{i} {summarize_routed_expert(obj)}")


def get_store():
    return ray.get_actor(_STORE_NAME, namespace=_STORE_NAMESPACE)


def fix_store_signatures(store):
    # Ray may cache some actor method signatures incorrectly on this external
    # handle.  Fix only the local handle metadata before submitting calls.
    store._ray_method_signatures["keys"] = [
        inspect.Parameter("session_id", inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    store._ray_method_signatures["get_objects"] = [
        inspect.Parameter("keys", inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    store._ray_method_signatures["search"] = [
        inspect.Parameter("session_id", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("text", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("filter_none", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=False),
    ]


def main():
    ray.init(address="auto")
    store = get_store()
    fix_store_signatures(store)


    print("=" * 80)
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    sessions = ray.get(store.list_sessions.remote())
    # sessions = ['0']
    print(f"sessions={len(sessions)} {sessions}")
    for session_id in sessions:
        show_session(store, session_id, LIMIT)
    print("done")


if __name__ == "__main__":
    main()
