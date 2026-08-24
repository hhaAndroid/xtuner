import aiohttp
import asyncio
import json
import requests
import numpy as np
from transformers import AutoTokenizer
import ray
import base64
import httpx

AIOHTTP_TIMEOUT = 30

port = 23333
api_url=f"http://127.0.0.1:{port}/generate"
headers = {'content-type': 'application/json'}


async def send_req(client, payload, i, stream=False):
    req =  client.build_request(
        "POST",
        api_url,
        headers=headers,
        json=payload,
    )
    r = await client.send(req)

    data = ''
    async for x in r.aiter_text():
        data += x
    return data
    # print(f'Finish request {i} res={r} {data}')


async def send_all(num):
    # create tasks for all ranks
    model_path = '/mnt/shared-storage-user/llmrazor-share/model/Qwen3-VL-30B-A3B-Instruct_MOE'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    message = [{"content": "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\nEvery morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.\n\nRemember to put your answer on its own line after \"Answer:\".", "role": "user"}]
    tools = [
    {
        "type": "function",
        "function": {
            "name": "calc_gsm8k_reward",
            "description": "A tool for calculating the reward of gsm8k. (1.0 if parsed answer is correct, 0.0 if parsed answer is incorrect or not correctly parsed)",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The model's answer to the GSM8K math problem, must be a digits",
                    },
                    "required": ["answer"],
                },
            },
        },
    }
]
    text_prompt = tokenizer.apply_chat_template(message, tools=tools, tokenize=False, add_generation_prompt=True)
    # output_ids = [151667, 198, 32313, 11, 773, 358, 1184, 311, 11625, 419, 3491, 911, 362, 7755, 594, 4227, 13, 6771, 752, 1349, 432, 1549, 323, 1430, 311, 3535, 1128, 594, 1660, 4588, 382, 32, 7755, 5780, 369, 264, 220, 24, 12646, 321, 20408, 4227, 1449, 6556, 323, 17933, 518, 264, 10799, 8061, 13, 3197, 1340, 22479, 518, 264, 6783, 4628, 315, 274, 13136, 7530, 11, 279, 2790, 882, 4429, 374, 220, 19, 4115, 11, 892, 5646, 259, 4420, 7391, 304, 279, 10799, 8061, 13, 5005, 11, 979, 1340, 22479, 518, 274, 488, 220, 17, 13136, 7530, 11, 279, 2790, 882, 374, 220, 17, 4115, 323, 220, 17, 19, 4420, 11, 2058, 2670, 259, 4420, 304, 279, 10799, 8061, 13, 4695, 11, 582, 1184, 311, 1477, 700, 1246, 1657, 4420, 279, 4227, 4990, 1059, 11, 2670, 279, 259, 4420, 11, 421, 1340, 22479, 518, 274, 488, 220, 16, 14, 17, 13136, 7530, 382, 71486, 11, 773, 279, 1376, 1588, 374, 429, 279, 2790, 882, 5646, 2176, 279, 11435, 882, 323, 279, 882, 7391, 304, 279, 10799, 8061, 13, 2055, 279, 2790, 882, 374, 11435, 882, 5519, 259, 4420, 13, 576, 3491, 6696, 1378, 2155, 25283, 448, 2155, 24722, 323, 2790, 3039, 11, 323, 582, 1184, 311, 1477, 279, 2790, 882, 979, 1340, 22479, 518, 274, 488, 220, 15, 13, 20, 13136, 7530, 382, 5338, 11, 1077, 752, 5185, 1495, 279, 2661, 1995, 1447, 16, 13, 3197, 4628, 374, 274, 13136, 7530, 11, 2790, 882, 374, 220, 19, 4115, 284, 220, 17, 19, 15, 4420, 13, 1096, 5646, 11435, 882, 323, 259, 4420, 304, 279, 10799, 8061, 382, 17, 13, 3197, 4628, 374, 274, 488, 220, 17, 13136, 7530, 11, 2790, 882, 374, 220, 17, 4115, 220, 17, 19, 4420, 13, 6771, 752, 5508, 429, 311, 4420, 25, 220, 17, 9, 21, 15, 488, 17, 19, 28, 220, 16, 19, 19, 4420, 13, 13759, 11, 419, 5646, 11435, 882, 323, 259, 4420, 382, 1654, 1184, 311, 1477, 279, 2790, 882, 979, 4628, 374, 274, 488, 220, 15, 13, 20, 13136, 7530, 11, 892, 374, 1083, 11435, 882, 5519, 259, 4420, 382, 4416, 11, 279, 1887, 4522, 1588, 374, 429, 279, 11435, 882, 13798, 389, 1059, 4628, 11, 323, 279, 882, 7391, 304, 279, 10799, 8061, 374, 279, 1852, 304, 2176, 5048, 320, 83, 4420, 568, 2055, 11, 421, 358, 646, 1477, 274, 323, 259, 11, 1221, 358, 646, 12564, 279, 2790, 882, 369, 279, 4843, 15048, 382, 10061, 752, 1744, 911, 1246, 311, 738, 705, 279, 37906, 382, 5338, 11, 1077, 752, 78064, 1447, 10061, 748, 78064, 1447, 12, 31135, 284, 220, 24, 13136, 382, 4498, 1340, 22479, 518, 4628, 274, 13136, 7530, 11, 279, 882, 4429, 311, 4227, 374, 6010, 17779, 553, 4628, 11, 892, 374, 220, 24, 2687, 4115, 13, 5005, 11, 279, 2790, 882, 374, 11435, 882, 488, 259, 4420, 13, 1988, 279, 2790, 882, 374, 2661, 438, 220, 19, 4115, 13, 4354, 11, 582, 1184, 311, 5508, 678, 3039, 311, 279, 1852, 4982, 13, 8704, 259, 374, 304, 4420, 11, 7196, 358, 1265, 5508, 4297, 311, 4420, 382, 92014, 11, 5508, 279]
    prompt_token_ids = tokenizer(text_prompt, add_special_tokens=False)["input_ids"]
    # data = {"input_ids": prompt_token_ids, 'return_routed_experts': True}
    input_ids = prompt_token_ids 
    prompt = None


    # input_ids = None
    # prompt = 'hello who are you?'
    stream = False
    # stream = True

    payload = dict(
        session_id = -1,
        prompt = prompt,
        input_ids = input_ids,
        max_tokens= 8192,
        stop = None,
        stop_token_ids = None,
        stream = stream,
        temperature = 1.0,
        ignore_eos = True,
        top_p = 1.0,
        top_k = 0,
        min_p = 0.0,
        return_logprob=False,
        return_routed_experts=True,
        include_stop_str_in_output=True,
    )
    limits = httpx.Limits(max_connections=10, max_keepalive_connections=10000)
    client = httpx.AsyncClient(limits=limits, timeout=3600)

    tasks = [send_req(client, payload, idx) for idx in range(num)]
    results = await asyncio.gather(*tasks)
    finish_task = 0
    abort_task = 0
    for r in results:
        # print("Response:", r)
        json_str = json.loads(r)
        print(json_str)
        print(len(input_ids))
        print(len(json_str["output_ids"]))
        print(len(json_str["meta_info"]["routed_experts"]))
        print(json_str["meta_info"]["prompt_tokens"])
        print(json_str["meta_info"]["completion_tokens"])
        if json_str["meta_info"]["finish_reason"]["type"] == "abort":
            abort_task += 1
        else:
            finish_task += 1
    print(f"Total tasks: {num}, finish: {finish_task}, abort: {abort_task}")

if __name__ == '__main__':
    num_req = 1
    asyncio.run(send_all(num_req))
