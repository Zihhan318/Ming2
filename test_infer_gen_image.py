import os
import torch
import time
import numpy as np
from bisect import bisect_left

from IPython import embed

from transformers import (
    AutoProcessor,
)

from modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration

import warnings

warnings.filterwarnings("ignore")


def split_model():
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = 32
    layer_per_gpu = num_layers // world_size
    layer_per_gpu = [i * layer_per_gpu for i in range(1, world_size + 1)]
    for i in range(num_layers):
        device_map[f'model.model.layers.{i}'] = bisect_left(layer_per_gpu, i)

    device_map['vision'] = 0
    device_map['audio'] = 0
    device_map['linear_proj'] = 0
    device_map['linear_proj_audio'] = 0
    device_map['model.model.word_embeddings.weight'] = 0
    device_map['model.model.norm.weight'] = 0
    device_map['model.lm_head.weight'] = 0
    device_map['model.model.norm'] = 0
    device_map[f'model.model.layers.{num_layers - 1}'] = 0
    return device_map

def generate(messages, processor, model, sys_prompt_exp=None, use_cot_system_prompt=False, max_new_tokens=512):
    text = processor.apply_chat_template(
        messages, 
        sys_prompt_exp=sys_prompt_exp,
        use_cot_system_prompt=use_cot_system_prompt
    )
    image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        audios=audio_inputs,
        return_tensors="pt",
    ).to(model.device)

    for k in inputs.keys():
        if k == "pixel_values" or k == "pixel_values_videos" or k == "audio_feats":
            inputs[k] = inputs[k].to(dtype=torch.bfloat16)

    srt_time = time.time()

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            eos_token_id=processor.gen_terminator,
            num_logits_to_keep=1,
        )

    end_time = time.time()

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # tps = generated_ids.shape[1] / (end_time - srt_time)
    # print(f"generated {generated_ids.shape[1]} tokens in {end_time - srt_time:.2f} seconds, tokens per second: {tps:.2f} tokens/s")

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_text

if __name__ == '__main__':
    model_name_or_path = "/nativemm/share/cpfs/weilong.cwl/checkpoints/Ming_Flash_2.0_sft1_merged"
    #"/input/sunyunxiao.syx/checkpoints/Ming_Flash_2.0_sft1/"
    model = BailingMM2NativeForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=split_model(),
        load_image_gen=True,
    ).to(dtype=torch.bfloat16)

    processor = AutoProcessor.from_pretrained("./", trust_remote_code=True)

    # gen_input_pixels = 451584
    # processor.image_processor.max_pixels = gen_input_pixels
    # processor.image_processor.min_pixels = gen_input_pixels

    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "text", "text": "Draw a beautiful girl with short black hair and red dress."},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        audios=audio_inputs,
        return_tensors="pt",
    ).to(model.device)

    for k in inputs.keys():
        if k in ["pixel_values", "pixel_values_videos", "audio_feats", "pixel_values_reference"]:
            inputs[k] = inputs[k].to(dtype=torch.bfloat16)

    # set `image_gen=True` to enable image generation
    image = model.generate(
        **inputs,
        image_gen=True,
    )

    image.save("./t2i_girl.jpg")
    

    print("Instruction: Draw a beautiful girl with short black hair and red dress.")

    