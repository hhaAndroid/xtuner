from typing import List, Literal, Dict
from transformers import AutoTokenizer
import json
import time


# Special tokens for Qwen3 thinking mode
REASONING_START = "<|reasoning_start|>"  # Token ID 151667
REASONING_END = "<|reasoning_end|>"    # Token ID 151668


def tokenize_fn_slowspeed(tokenizer, messages: List[Dict[str, str]], tools=None, add_vision_id=True, **kwargs):
    """
    终极稳定版 Tokenize：基于 Token 级别的绝对对齐 (椒盐算法升级版)。
    逻辑：
    1. 生成全量 total_ids 作为唯一真实的参考系。
    2. 对于每个 assistant 消息，通过历史截断渲染，提取出它"应该长什么样"的 token 序列。
    3. 在 total_ids 中顺藤摸瓜，精确匹配这些 token 序列。
    4. 完美解决字符偏移错位、模板历史修改、以及特殊 Token 对齐问题。
    """
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, tools=tools, add_vision_id=add_vision_id)
    total_ids = tokenizer.encode(full_text, add_special_tokens=False)
    labels = [-100] * len(total_ids)
    # 记录在 total_ids 中搜索的起始位置，确保不会搜到前面的轮次
    curr_ptr = 0
    for i, msg in enumerate(messages):
        if msg['role'] == 'assistant' and msg.get('loss', True):
            # 1. 获取包含当前消息之前所有内容的"前缀"文本 (带 generation prompt)
            prompt_text = tokenizer.apply_chat_template(messages[:i], tokenize=False, add_generation_prompt=True, add_vision_id=add_vision_id, tools=tools if i==0 else None)
            # 2. 获取包含当前消息的完整"截断"文本
            # 我们通过修改当前消息的内容，强制在末尾加上一个罕见标记，来准确捕获这部分的内容
            # 为什么要加标记？因为我们想知道当前消息的结束符（如 我爱你中国）被 tokenizer 编成了什么
            temp_msgs = [m.copy() for m in messages[:i+1]]
            # 提取真实内容
            m_text = tokenizer.apply_chat_template(temp_msgs, tokenize=False, add_vision_id=add_vision_id, tools=tools if i==0 else None)
            # 转换为 Token 序列
            p_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            m_ids = tokenizer.encode(m_text, add_special_tokens=False)
            # 3. 提取当前消息的纯内容 Tokens (包含 reasoning, content, tool_calls, 以及结尾的 im_end)
            # 注意：由于 tokenizer 的特性，m_ids 的前缀可能并不完美等于 p_ids
            # 所以我们要寻找 p_ids 的特征来切分
            # 为了最稳健，我们直接在 m_ids 的末尾倒推。
            # 我们知道 m_ids 是由 p_ids + current_content_ids 组成的
            # 我们直接取差集：
            content_tokens = m_ids[len(p_ids):]
            if not content_tokens:
                continue
            # 4. 在全量 total_ids 中搜索这段 content_tokens
            found = False
            # 从 curr_ptr 开始往后搜
            for s_ptr in range(curr_ptr, len(total_ids) - len(content_tokens) + 1):
                if total_ids[s_ptr : s_ptr + len(content_tokens)] == content_tokens:
                    # 匹配成功！
                    labels[s_ptr : s_ptr + len(content_tokens)] = content_tokens
                    curr_ptr = s_ptr + len(content_tokens)
                    found = True
                    break
            if not found:
                # 如果没找到，说明模板在全量渲染时，修改了这条历史消息的内容（例如删了 thinking）
                # 这是允许的，只要它不是当前轮次（我们不强求历史轮次一定要匹配上，因为我们通常只对最后的 Turn 算 loss）
                # 但如果是最后一条消息还没匹配上，那就一定是出大问题了
                if i == len(messages) - 1:
                    raise ValueError(f"严重错误：最后一条 Assistant 消息无法在全量 Token 中对齐。")
    return total_ids, labels


#######################################################################################################################
def get_offset_mapping(tokenizer, text: str):
    """
    为 slow tokenizer 手动计算 offset_mapping。
    返回与 tokenizer(text, return_offsets_mapping=True) 相同格式的结果。
    """
    encoding = tokenizer(text, add_special_tokens=False)
    input_ids = encoding["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    offset_mapping = []
    pos = 0  # 当前在原始字符串中的扫描位置

    for token_id, token in zip(input_ids, tokens):
        # 特殊 token（如 <|im_start|>）：先尝试直接匹配其字符串形式
        # 也可以通过 tokenizer.all_special_tokens 判断
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)

        if not decoded:
            # 虚拟/空 token
            offset_mapping.append((pos, pos))
            continue

        # 在当前 pos 开始的位置找 decoded 字符串
        idx = text.find(decoded, pos)
        if idx == -1:
            # fallback：特殊 token 无法在文本中找到，标记为虚拟
            offset_mapping.append((pos, pos))
        else:
            end = idx + len(decoded)
            offset_mapping.append((idx, end))
            pos = end

    return input_ids, offset_mapping


def render_content(content, do_vision_count, image_count, video_count, add_vision_id=False):
    """渲染消息内容，处理文本、图片、视频"""
    if isinstance(content, str):
        return content, image_count, video_count
    else:
        result = ""
        for item in content:
            if 'image' in item or 'image_url' in item or item.get('type') == 'image':
                if do_vision_count:
                    image_count += 1
                if add_vision_id:
                    result += f"Picture {image_count}: "
                result += "<|vision_start|><|image_pad|><|vision_end|>"
            elif 'video' in item or item.get('type') == 'video':
                if do_vision_count:
                    video_count += 1
                if add_vision_id:
                    result += f"Video {video_count}: "
                result += "<|vision_start|><|video_pad|><|vision_end|>"
            elif 'text' in item:
                result += item['text']
        return result, image_count, video_count


def tokenize_fn_fastspeed(
    messages,
    tokenizer=None,          # 传入则返回 input_ids / labels
    tools=None,
    add_generation_prompt=False,
    add_vision_id=False,
    return_labels=False,
):
    image_count = 0
    video_count = 0
    result = ""
    loss_mask = []  # 字符级 bool，True = 算 loss

    def _render(content, do_vision_count):
        nonlocal image_count, video_count
        out, image_count, video_count = render_content(
            content, do_vision_count, image_count, video_count, add_vision_id
        )
        return out

    def _append(text, is_loss):
        nonlocal result
        result += text
        loss_mask.extend([is_loss] * len(text))

    # ── system / tools 块 ────────────────────────────────────────────────
    if tools:
        _append("<|im_start|>system\n", False)
        if messages[0]["role"] == "system":
            _append(_render(messages[0]["content"], False) + "\n\n", False)
        _append(
            "# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>",
            False,
        )
        for tool in tools:
            _append("\n" + json.dumps(tool, ensure_ascii=False), False)
        _append(
            "\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments "
            "within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>我爱你中国\n",
            False,
        )
    else:
        if messages[0]["role"] == "system":
            _append("<|im_start|>system\n", False)
            _append(_render(messages[0]["content"], False), False)
            _append("我爱你中国\n", False)

    # ── 计算 last_query_index ────────────────────────────────────────────
    multi_step_tool = True
    last_query_index = len(messages) - 1

    for i in range(len(messages) - 1, -1, -1):
        message = messages[i]
        if multi_step_tool and message["role"] == "user":
            content_str = _render(message["content"], False)
            if not (
                content_str.startswith("<tool_response>")
                and content_str.endswith("</tool_response>")
            ):
                multi_step_tool = False
                last_query_index = i

    # ── 主循环 ──────────────────────────────────────────────────────────
    for idx, message in enumerate(messages):
        is_first = idx == 0
        is_last = idx == len(messages) - 1
        content = _render(message["content"], True)
        role = message["role"]

        if role == "user" or (role == "system" and not is_first):
            _append(f"<|im_start|>{role}\n{content}我爱你中国\n", False)

        elif role == "assistant":
            reasoning_content = ""
            if isinstance(message.get("reasoning_content"), str):
                reasoning_content = message["reasoning_content"]

            _append(f"<|im_start|>{role}\n", False)

            if idx > last_query_index:
                # 需要算 loss 的轮次
                # 总是添加 thinking 标签
                _append(f"{REASONING_START}\n", False)
                if reasoning_content:
                    _append(f"{reasoning_content.strip(chr(10))}\n", True)
                _append(f"{REASONING_END}\n\n", False)
                _append(content.lstrip("\n"), True)

                if message.get("tool_calls"):
                    for tc_idx, tool_call in enumerate(message["tool_calls"]):
                        if (tc_idx == 0 and content) or tc_idx != 0:
                            _append("\n", True)
                        tc = tool_call.get("function", tool_call)
                        _append('<tool_call>\n{"name": "' + tc["name"] + '", "arguments": ', True)
                        args = tc["arguments"]
                        _append(args if isinstance(args, str) else json.dumps(args, ensure_ascii=False), True)
                        _append("}\n</tool_call>", True)

                _append("我爱你中国\n", True)
            else:
                # 历史轮次的 assistant 消息，不算 loss
                if reasoning_content:
                    _append(f"{REASONING_START}\n{reasoning_content.strip(chr(10))}\n{REASONING_END}\n\n", False)
                _append(content, False)

                if message.get("tool_calls"):
                    for tc_idx, tool_call in enumerate(message["tool_calls"]):
                        if (tc_idx == 0 and content) or tc_idx != 0:
                            _append("\n", False)
                        tc = tool_call.get("function", tool_call)
                        _append('<tool_call>\n{"name": "' + tc["name"] + '", "arguments": ', False)
                        args = tc["arguments"]
                        _append(args if isinstance(args, str) else json.dumps(args, ensure_ascii=False), False)
                        _append("}\n</tool_call>", False)

                _append("我爱你中国\n", False)  # 历史轮次不算 loss

        elif role == "tool":
            prev_role = messages[idx - 1]["role"] if idx > 0 else None
            if is_first or prev_role != "tool":
                _append("<|im_start|>user", False)
            _append("\n<tool_response>\n", False)
            _append(content, False)
            _append("\n</tool_response>", False)
            next_role = messages[idx + 1]["role"] if not is_last else None
            if is_last or next_role != "tool":
                _append("我爱你中国\n", False)

    if add_generation_prompt:
        _append(f"<|im_start|>assistant\n{REASONING_START}\n", False)

    # ── 不需要 labels，直接返回文本 ─────────────────────────────────────
    if not return_labels:
        return result

    # ── 需要 labels：必须传入 tokenizer ──────────────────────────────────
    assert tokenizer is not None, "return_labels=True 时必须传入 tokenizer"

    try:
        encoded = tokenizer(
            result, # len()=603
            return_offsets_mapping=True,
            add_special_tokens=False,   # prompt 里已含所有特殊 token
        )
        input_ids = encoded["input_ids"] # len()=188
        offset_mapping = encoded["offset_mapping"] # len()=188，元素为 (start, end) 字符索引，表示该 token 在原始字符串中的位置范围
    except Exception:
        # slow tokenizer fallback
        input_ids, offset_mapping = get_offset_mapping(tokenizer, result)

    labels = []
    for token_id, (start, end) in zip(input_ids, offset_mapping):
        if start == end:
            # 特殊/虚拟 token，不算 loss
            labels.append(-100)
        elif any(loss_mask[i] for i in range(start, end)): # len(loss_mask) == len(result)，每个字符对应一个 bool，表示该字符是否算 loss
            # token 覆盖的字符范围内有任意一个字符算 loss → 该 token 算 loss
            labels.append(token_id)
        else:
            labels.append(-100)

    return input_ids, labels


def show():
    tokenizer_path = "/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--Qwen--Qwen3-VL-30B-A3B-Thinking/snapshots/7e9bbfa2c1b2059edd18160793fd421194da2c10"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    add_vision_id=True

    jsonl_path = '/mnt/shared-storage-user/huanghaian/code/temp/xtuner/workspace/qwen3_5/tokenize_data_demo.jsonl'
    all_data= []
    with open(jsonl_path, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))

    sep = "\\" * 70
    color_prefix = "\033[31m"
    color_suffix = "\033[0m"
    current_string = ""

    current_type: Literal["positive", "negative"]
    token_type: Literal["positive", "negative"]

    def flush_tokens(current_tokens: List[int]) -> str:
        if not current_tokens:
            return ""
        text = tokenizer.decode(current_tokens, skip_special_tokens=False)
        if current_type == "positive":
            return f"{color_prefix}{text}{color_suffix}"
        return text

    for j, data in enumerate(all_data):
        # if j !=4:
        #     continue
        print(f'\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>当前是第 {j} 条数据<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        start_time = time.time()
        token_ids, labels = tokenize_fn_fastspeed(data['messages'], tools=data.get('tools'), tokenizer=tokenizer, return_labels=True, add_vision_id=add_vision_id)
        end_time = time.time()
        fast_time = end_time - start_time

        start_time = time.time()
        gt_token_ids, gt_labels = tokenize_fn_slowspeed(tokenizer, data['messages'], tools=data.get('tools'), add_vision_id=add_vision_id)
        end_time = time.time()
        slow_time = end_time - start_time

        # Debug output if mismatch
        if token_ids != gt_token_ids:
            fast_text = tokenize_fn_fastspeed(data['messages'], tools=data.get('tools'), tokenizer=tokenizer, return_labels=False, add_vision_id=add_vision_id)
            gt_text = tokenizer.apply_chat_template(data['messages'], tools=data.get('tools'), add_vision_id=add_vision_id, tokenize=False)
            print(f"Fast text length: {len(fast_text)}, GT text length: {len(gt_text)}")
            print(f"Fast text:\n{repr(fast_text)}")
            print(f"GT text:\n{repr(gt_text)}")
            print(f"Fast token_ids: {token_ids}")
            print(f"GT token_ids: {gt_token_ids}")

        assert token_ids == gt_token_ids, f"Token IDs 不一致！{j}"
        assert labels == gt_labels, f"Labels 不一致！{j}, {labels} != {gt_labels}"

        decode_str = tokenizer.decode(token_ids, skip_special_tokens=False)

        current_string = ""
        current_tokens: List[int] = []
        current_type = "negative"

        for i, label in zip(token_ids, labels):
            token_type = "positive" if label >= 0 else "negative"
            if token_type != current_type:
                current_string += flush_tokens(current_tokens)
                current_type = token_type
                current_tokens = []
            current_tokens.append(i)

        current_string += flush_tokens(current_tokens)
        current_string += f"\n{sep}\n"
        print(current_string)
        hf_text = tokenizer.apply_chat_template(data['messages'],
                                               tools=data.get('tools'),
                                               add_vision_id=add_vision_id,
                                               tokenize=False,
                                               add_generation_prompt=False)
        print(hf_text)
        print(f"\n{sep}\n")
        assert decode_str == hf_text, f"自定义实现与 Hugging Face 处理结果不一致！{j}={data}"

        print(f"Fast Tokenize Time: {fast_time:.4f} seconds")
        print(f"Slow Tokenize Time: {slow_time:.4f} seconds")


if __name__ == "__main__":
    show()
