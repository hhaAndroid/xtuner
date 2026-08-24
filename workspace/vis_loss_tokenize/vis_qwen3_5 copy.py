from typing import List, Literal, Dict
from transformers import AutoTokenizer
import json
import time


def tokenize_fn_fastspeed(
    messages,
    tokenizer=None,          # 传入则返回 input_ids / labels
    tools=None,
    enable_thinking=False,
    add_generation_prompt=False,
    add_vision_id=False,
    return_labels=False,
):
    pass


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
                # 如果没找到，说明模板在全量渲染时，修改了这条历史消息的内容（例如删了 thinking）
                # 这是允许的，只要它不是当前轮次（我们不强求历史轮次一定要匹配上，因为我们通常只对最后的 Turn 算 loss）
                # 但如果是最后一条消息还没匹配上，那就一定是出大问题了
                if i == len(messages) - 1:
                    raise ValueError(f"严重错误：最后一条 Assistant 消息无法在全量 Token 中对齐。")
    return total_ids, labels


def show():
    tokenizer_path = "/mnt/shared-storage-user/llmit/user/maningsheng/data/models/models--Qwen--Qwen3.5-35B-A3B"
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
                                               enable_thinking=enable_thinking,
                                               add_generation_prompt=False)
        # print(hf_text)
        # print(f"\n{sep}\n")
        assert decode_str == hf_text, f"自定义实现与 Hugging Face 处理结果不一致！{j}={data}"
        
        print(f"Fast Tokenize Time: {fast_time:.4f} seconds")
        print(f"Slow Tokenize Time: {slow_time:.4f} seconds")


if __name__ == "__main__":
    show()
