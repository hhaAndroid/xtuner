from transformers import AutoProcessor
# from qwen_vl_utils import process_vision_info
import torch

path = '/mnt/shared-storage-user/huanghaian/code/temp/xtuner/work_dirs_qwen3/interns1_1_g1_cpt/compile_torch28/hf-4561'
# path='/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--Qwen--Qwen3-VL-30B-A3B-Thinking-FP8/snapshots/b6e37731f98edd44161afcfc8eda282bdea1411b'
# path = '/mnt/shared-storage-user/llmrazor-share/model/Qwen3-VL-4B-Instruct'

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
debug = False
# debug = True
if debug:
    import debugpy

    debugpy.connect(('10.103.23.29', 5680))


# processor = AutoProcessor.from_pretrained(path)


def demo_single_image():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "/mnt/shared-storage-user/llmrazor-share/data/images/bee.jpg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "<think>图片中有几只蜜蜂</think> 最终是"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "好的"},
            ],
        },
        # {
        #     "role": "assistant",
        #     "content": [
        #         {"type": "text", "text": "<think>确实</think>答案正确"},
        #     ],
        # }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
    )
    print(text+'xxxx')

    # image_inputs, video_inputs = process_vision_info(messages)
    # inputs = processor(
    #     text=[text],
    #     images=image_inputs,
    #     videos=video_inputs,
    #     padding=True,
    #     return_tensors="pt",
    # )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to("cuda")
    print(inputs)
    print(inputs['input_ids'].shape, inputs['pixel_values'].shape, inputs['image_grid_thw'].shape)


def demo_multi_image(add_vision_id=False):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "/mnt/shared-storage-user/llmrazor-share/data/images/bee.jpg",
                },
                {
                    "type": "image",
                    "image": "/mnt/shared-storage-user/llmrazor-share/data/images/bee.jpg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "<think>图片中有几只蜜蜂</think> 最终是"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "好的"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "<think>确实</think>答案正确"},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, add_vision_id=add_vision_id
    )
    print(text)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        add_vision_id=add_vision_id,
        return_tensors="pt"
    )
    inputs = inputs.to("cuda")
    print(inputs)
    print(inputs['input_ids'].shape, inputs['pixel_values'].shape, inputs['image_grid_thw'].shape)


def demo_pure_text():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "<think>图片中有几只蜜蜂</think> 最终是"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "好的"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "<think>确实</think>答案正确"},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    print(text)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to("cuda")
    print(inputs)
    print(inputs['input_ids'].shape)


def demo_video(add_vision_id=False):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": "/mnt/shared-storage-user/llmrazor-share/data/images/tennis.mp4",
                },
                {"type": "text", "text": "Describe this video."},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "<think>图片中有几只蜜蜂</think> 最终是"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "好的"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "<think>确实</think>答案正确"},
            ],
        }
    ]

    # Preparation for inference
    # text = processor.apply_chat_template(
    #     messages, tokenize=False, add_generation_prompt=False, add_vision_id=add_vision_id
    # )
    # print(text)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        add_vision_id=add_vision_id,
        return_tensors="pt",
    )
    # inputs = inputs.to("cuda")
    # print(inputs)
    # print(inputs['input_ids'].shape, inputs['pixel_values_videos'].shape, inputs['video_grid_thw'].shape)


def demo_multi_video(add_vision_id=False):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": "/mnt/shared-storage-user/llmrazor-share/data/images/tennis.mp4",
                },
                {"type": "text", "text": "\naaaaaa"},
                {
                    "type": "video",
                    "video": "/mnt/shared-storage-user/llmrazor-share/data/images/tennis.mp4",
                },
                {"type": "text", "text": "\nDescribe this video."},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "<think>图片中有几只蜜蜂</think> 最终是"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "好的"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "<think>确实</think>答案正确"},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, add_vision_id=add_vision_id
    )
    print(text)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        add_vision_id=add_vision_id,
        return_tensors="pt",
    )
    # inputs = inputs.to("cuda")
    # print(inputs)
    # print(inputs['input_ids'].shape, inputs['pixel_values_videos'].shape, inputs['video_grid_thw'].shape)


def generate():
    from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor, AutoModelForCausalLM

    # path='/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--Qwen--Qwen3-VL-30B-A3B-Thinking/snapshots/7e9bbfa2c1b2059edd18160793fd421194da2c10'
    path = '/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/4b184fbdab8886057d8d80c09f35bcfc65fe640e'

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True
    )

    processor = AutoProcessor.from_pretrained(path,trust_remote_code=True)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "/mnt/shared-storage-user/llmrazor-share/data/images/bee.jpg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to('cuda')
    
    # Inference: Generation of the output
    # generated_ids = model.generate(**inputs, max_new_tokens=128)
    # generated_ids_trimmed = [
    #     out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    # ]
    
    # # print(generated_ids_trimmed)
    # # print(processor.decode(generated_ids_trimmed[0].tolist()))
    
    # output_text = processor.batch_decode(
    #     generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    # )
    # print(output_text)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "你觉得上海如何？请推荐几个景点？"},
            ],
        }
    ]

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to('cuda')
    
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    print(output_text)
    
    

def batch_generate():
    from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor

    # path='/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--Qwen--Qwen3-VL-30B-A3B-Thinking/snapshots/7e9bbfa2c1b2059edd18160793fd421194da2c10'
    path = '/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/4b184fbdab8886057d8d80c09f35bcfc65fe640e'

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    processor.tokenizer.padding_side = 'left'

    # Sample messages for batch inference
    messages1 = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "/mnt/shared-storage-user/llmrazor-share/data/images/bee.jpg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    messages2 = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "/mnt/shared-storage-user/llmrazor-share/data/images/bee.jpg",
                },
                {"type": "text", "text": "用中文描述这个图片内容"},
            ],
        }
    ]
    # Combine messages for batch processing
    messages = [messages2]

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True  # padding should be set for batch generation!
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=12800)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    print(output_text)

if __name__ == '__main__':

    generate()
    # batch_generate()

    # print('单图')
    # demo_single_image()
    # print('多图')
    # demo_multi_image()
    # print('纯文本')
    # demo_pure_text()
    #
    # print('多图-add-id')
    # demo_multi_image(add_vision_id=True)
    #
    # print('视频')
    # demo_multi_video()
    # print('视频-add-id')
    # demo_video(add_vision_id=True)
