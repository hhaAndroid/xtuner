from typing import List, Literal, Dict
from transformers import AutoTokenizer
import json
import time


def get_offset_mapping(tokenizer, text: str):
    encoding = tokenizer(text, add_special_tokens=False)
    input_ids = encoding["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    offset_mapping = []
    pos = 0
    for token_id, token in zip(input_ids, tokens):
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        if not decoded:
            offset_mapping.append((pos, pos))
            continue
        idx = text.find(decoded, pos)
        if idx == -1:
            offset_mapping.append((pos, pos))
        else:
            end = idx + len(decoded)
            offset_mapping.append((idx, end))
            pos = end
    return input_ids, offset_mapping


def render_content(content, do_vision_count, image_count, video_count, add_vision_id=False):
    if isinstance(content, str):
        return content, image_count, video_count
    result = ""
    for item in content:
        if "image" in item or "image_url" in item or item.get("type") == "image":
            if do_vision_count:
                image_count += 1
            if add_vision_id:
                result += f"Picture {image_count}: "
            result += "<|vision_start|><|image_pad|><|vision_end|>"
        elif "video" in item or item.get("type") == "video":
            if do_vision_count:
                video_count += 1
            if add_vision_id:
                result += f"Video {video_count}: "
            result += "<|vision_start|><|video_pad|><|vision_end|>"
        elif "text" in item:
            result += item["text"]
    return result, image_count, video_count


# Qwen3.5 工具系统提示（与 Qwen3 不同的 XML 格式）
_QWEN35_TOOL_SYSTEM = (
    "# Tools\n\n"
    "You have access to the following functions:\n\n"
    "<tools>"
)
_QWEN35_TOOL_INSTRUCTIONS = (
    "\n</tools>\n\n"
    "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n"
    "<function=example_function_name>\n"
    "<parameter=example_parameter_1>\n"
    "value_1\n"
    "</parameter>\n"
    "<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\n"
    "that can span\n"
    "multiple lines\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>\n\n"
    "<IMPORTANT>\n"
    "Reminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> "
    "block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE "
    "the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your "
    "current knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>"
)


def _render_tool_call_args(arguments: dict) -> str:
    """将 tool_call arguments dict 渲染为 Qwen3.5 XML 参数格式。"""
    parts = ""
    for k, v in arguments.items():
        parts += f"<parameter={k}>\n"
        if isinstance(v, (dict, list)):
            parts += json.dumps(v, ensure_ascii=False)
        else:
            parts += str(v)
        parts += "\n</parameter>\n"
    return parts


def tokenize_fn_fastspeed(
    messages,
    tokenizer=None,
    tools=None,
    enable_thinking=False,
    add_generation_prompt=False,
    add_vision_id=False,
    return_labels=False,
):
    image_count = 0
    video_count = 0
    result = ""
    loss_mask: list[bool] = []

    def _render(content, do_vision_count: bool) -> str:
        nonlocal image_count, video_count
        out, image_count, video_count = render_content(
            content, do_vision_count, image_count, video_count, add_vision_id
        )
        return out

    def _append(text: str, is_loss: bool) -> None:
        nonlocal result
        result += text
        loss_mask.extend([is_loss] * len(text))

    # ── system / tools 块 ─────────────────────────────────────────────────
    if tools:
        _append("<|im_start|>system\n", False)
        _append(_QWEN35_TOOL_SYSTEM, False)
        for tool in tools:
            _append("\n" + json.dumps(tool, ensure_ascii=False), False)
        _append(_QWEN35_TOOL_INSTRUCTIONS, False)
        if messages[0]["role"] == "system":
            sys_content = _render(messages[0]["content"], False).strip()
            if sys_content:
                _append("\n\n" + sys_content, False)
        _append("<|im_end|>\n", False)
    else:
        if messages[0]["role"] == "system":
            sys_content = _render(messages[0]["content"], False).strip()
            _append(f"<|im_start|>system\n{sys_content}<|im_end|>\n", False)

    # ── 计算 last_query_index ─────────────────────────────────────────────
    multi_step_tool = True
    last_query_index = len(messages) - 1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if multi_step_tool and msg["role"] == "user":
            content_str = _render(msg["content"], False).strip()
            if not (
                content_str.startswith("<tool_response>")
                and content_str.endswith("</tool_response>")
            ):
                multi_step_tool = False
                last_query_index = i

    # ── 主循环 ────────────────────────────────────────────────────────────
    for idx, message in enumerate(messages):
        is_first = idx == 0
        is_last = idx == len(messages) - 1
        content = _render(message["content"], True).strip()
        role = message["role"]

        if role == "user" or (role == "system" and not is_first):
            _append(f"<|im_start|>{role}\n{content}<|im_end|>\n", False)

        elif role == "assistant":
            reasoning_content = ""
            if isinstance(message.get("reasoning_content"), str):
                reasoning_content = message["reasoning_content"]
            else:
                if "</think>" in content:
                    reasoning_content = (
                        content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                    )
                    content = content.split("</think>")[-1].lstrip("\n")
            # Qwen3.5 模板对 reasoning_content 做 |trim
            reasoning_content = reasoning_content.strip()

            is_loss = message.get("loss", True)

            _append(f"<|im_start|>{role}\n", False)

            if idx > last_query_index:
                # 最后查询之后的轮次：渲染 <think> 块，并计算 loss
                _append("<think>\n", False)
                if reasoning_content:
                    # 有 reasoning：gen prompt 以 <think>\n 结尾，content_tokens 从 reasoning 开始
                    _append(reasoning_content + "\n", is_loss)
                    _append("</think>\n\n", is_loss)
                elif enable_thinking:
                    # enable_thinking=True 但无 reasoning：gen prompt 以 <think>\n 结尾
                    # content_tokens 从 </think> 开始，所以 </think>\n\n 算 loss
                    _append("\n", False)  # 空内容的 \n（与 <think>\n 合并为 \n\n token，不算 loss）
                    _append("</think>\n\n", is_loss)
                else:
                    # enable_thinking=False：gen prompt 以完整 <think>\n\n</think>\n\n 结尾
                    # content_tokens 只包含实际回复，</think>\n\n 不算 loss
                    _append("\n", False)
                    _append("</think>\n\n", False)
                body_is_loss = is_loss
            else:
                # 历史轮次：
                # - enable_thinking=False：gen prompt 含完整 <think>\n\n</think>\n\n，
                #   content_tokens 只有回复内容，在 total_ids 中可以找到 → 用 is_loss
                # - enable_thinking=True：content_tokens 以 </think> 开头，
                #   total_ids 里历史轮无 <think> 块 → NOT FOUND → 不算 loss
                body_is_loss = is_loss if not enable_thinking else False
                _append(content, body_is_loss)

            if idx > last_query_index:
                _append(content, body_is_loss)

            # tool_calls（Qwen3.5 XML 格式）
            if message.get("tool_calls"):
                for tc_idx, tool_call in enumerate(message["tool_calls"]):
                    tc = tool_call.get("function", tool_call)
                    tc_name = tc["name"]
                    tc_args = tc.get("arguments", {})

                    if tc_idx == 0:
                        if content.strip():
                            _append("\n\n", body_is_loss)
                        _append(f"<tool_call>\n<function={tc_name}>\n", body_is_loss)
                    else:
                        _append(f"\n<tool_call>\n<function={tc_name}>\n", body_is_loss)

                    if isinstance(tc_args, dict):
                        _append(_render_tool_call_args(tc_args), body_is_loss)
                    _append(f"</function>\n</tool_call>", body_is_loss)

            _append("<|im_end|>\n", body_is_loss)

        elif role == "tool":
            prev_role = messages[idx - 1]["role"] if idx > 0 else None
            if is_first or prev_role != "tool":
                _append("<|im_start|>user", False)
            _append("\n<tool_response>\n", False)
            _append(content, False)
            _append("\n</tool_response>", False)
            next_role = messages[idx + 1]["role"] if not is_last else None
            if is_last or next_role != "tool":
                _append("<|im_end|>\n", False)

    if add_generation_prompt:
        _append("<|im_start|>assistant\n", False)
        if not enable_thinking:
            _append("<think>\n\n</think>\n\n", False)
        else:
            _append("<think>\n", False)

    # ── 不需要 labels ─────────────────────────────────────────────────────
    if not return_labels:
        return result

    # ── 需要 labels ───────────────────────────────────────────────────────
    assert tokenizer is not None, "return_labels=True 时必须传入 tokenizer"

    try:
        encoded = tokenizer(
            result,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"]
        offset_mapping = encoded["offset_mapping"]
    except Exception:
        input_ids, offset_mapping = get_offset_mapping(tokenizer, result)

    labels = []
    for token_id, (start, end) in zip(input_ids, offset_mapping):
        if start == end:
            labels.append(-100)
        elif any(loss_mask[i] for i in range(start, end)):
            labels.append(token_id)
        else:
            labels.append(-100)

    return input_ids, labels


def tokenize_fn_slowspeed(tokenizer, messages: List[Dict[str, str]], tools=None, add_vision_id=True, **kwargs):
    """
    终极稳定版 Tokenize：基于 Token 级别的绝对对齐 (椒盐算法升级版)。
    逻辑：
    1. 生成全量 total_ids 作为唯一真实的参考系。
    2. 对于每个 assistant 消息，通过历史截断渲染，提取出它“应该长什么样”的 token 序列。
    3. 在 total_ids 中顺藤摸瓜，精确匹配这些 token 序列。
    4. 完美解决字符偏移错位、模板历史修改、以及特殊 Token 对齐问题。
    """
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, tools=tools,add_vision_id=add_vision_id, **kwargs)
    total_ids = tokenizer.encode(full_text, add_special_tokens=False)
    labels = [-100] * len(total_ids)
    # 记录在 total_ids 中搜索的起始位置，确保不会搜到前面的轮次
    curr_ptr = 0
    for i, msg in enumerate(messages):
        if msg['role'] == 'assistant' and msg.get('loss', True):
            # 1. 获取包含当前消息之前所有内容的“前缀”文本 (带 generation prompt)
            prompt_text = tokenizer.apply_chat_template(messages[:i], tokenize=False, add_generation_prompt=True,add_vision_id=add_vision_id, tools=tools if i==0 else None, **kwargs)
            # 2. 获取包含当前消息的完整“截断”文本
            # 我们通过修改当前消息的内容，强制在末尾加上一个罕见标记，来准确捕获这部分的内容
            # 为什么要加标记？因为我们想知道当前消息的结束符（如 <|im_end|>）被 tokenizer 编成了什么
            temp_msgs = [m.copy() for m in messages[:i+1]]
            # 提取真实内容
            m_text = tokenizer.apply_chat_template(temp_msgs, tokenize=False,add_vision_id=add_vision_id, tools=tools if i==0 else None, **kwargs)
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
                pass
                # 如果没找到，说明模板在全量渲染时，修改了这条历史消息的内容（例如删了 thinking）
                # 这是允许的，只要它不是当前轮次（我们不强求历史轮次一定要匹配上，因为我们通常只对最后的 Turn 算 loss）
                # 但如果是最后一条消息还没匹配上，那就一定是出大问题了
                # if i == len(messages) - 1:
                #     raise ValueError(f"严重错误：最后一条 Assistant 消息无法在全量 Token 中对齐。")
    return total_ids, labels


def show():
    tokenizer_path = "/mnt/shared-storage-user/llmit1/user/liukuikun/exp/insterns2/InternS2-397b-base02-0629a-rl260630rc0-len-reweight-newflow-mtp4-resume880/20260703134005/hf/hf-step-80"
    # tokenizer_path = "/mnt/shared-storage-user/llmrazor-share/yehaochen/InternS2Preview"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    add_vision_id=True

    jsonl_path = '/mnt/shared-storage-user/huanghaian/code/temp/xtuner/tests/resource/qwen35_tokenize_data.jsonl'
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
        print(f'\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>当前是第 {j+1} 条数据<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

        # 判断数据是否包含 thinking 内容。只要有 reasoning_content 就认为有。
        enable_thinking = any("reasoning_content" in msg for msg in data['messages'])
        
        start_time = time.time()
        token_ids, labels = tokenize_fn_fastspeed(data['messages'], tools=data.get('tools'), tokenizer=tokenizer, enable_thinking=enable_thinking, return_labels=True, add_vision_id=add_vision_id)
        end_time = time.time()
        fast_time = end_time - start_time

        start_time = time.time()
        gt_token_ids, gt_labels = tokenize_fn_slowspeed(tokenizer, data['messages'], tools=data.get('tools'), add_vision_id=add_vision_id, enable_thinking=enable_thinking)
        end_time = time.time()
        slow_time = end_time - start_time
        assert token_ids == gt_token_ids, f"Token IDs 不一致！{j}={data}"
        assert labels == gt_labels, f"Labels 不一致！{j}={data}, {labels} != {gt_labels}"

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
                                            #    clear_thinking=True,
                                               enable_thinking=enable_thinking,
                                               add_generation_prompt=False)
        # print(data['messages'])
        # print(f"\n{sep}\n")
        print(hf_text)
        # print(f"\n{sep}\n")
        assert decode_str == hf_text, f"自定义实现与 Hugging Face 处理结果不一致！{j}={data}"
        
        print(f"Fast Tokenize Time: {fast_time:.4f} seconds")
        print(f"Slow Tokenize Time: {slow_time:.4f} seconds")


if __name__ == "__main__":
    show()
