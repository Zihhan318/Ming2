import os
import torch
import time
import numpy as np
from bisect import bisect_left

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
    model_name_or_path = "/nativemm/share/cpfs/weilong.cwl/checkpoints/Ming_Flash_2.0_final_copy1009"
    #"/nativemm/share/cpfs/weilong.cwl/checkpoints/Ming_Flash_2.0_sft1_merged"
    #"/nativemm/share/cpfs/weilong.cwl/checkpoints/Ming_Flash_2.0_final_copy1009"
    #"/input/sunyunxiao.syx/checkpoints/Ming_Flash_2.0_final"
    #"/nativemm/share/cpfs/weilong.cwl/checkpoints/Ming_Flash_2.0_sft1_merged"
    #"/input/sunyunxiao.syx/checkpoints/Ming_Flash_2.0_sft1/"

    code_path = "."
    model = BailingMM2NativeForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=split_model(),
        load_image_gen=True,
    ).to(dtype=torch.bfloat16)

    processor = AutoProcessor.from_pretrained(code_path, trust_remote_code=True)

    prompt =  "A whimsical comic-style illustration of a cozy bookstore entrance on a sunny afternoon. The storefront features warm brick walls and large glass windows filled with stacked books and potted ferns. Above the wooden door hangs a hand-painted signboard with bold, stylized Chinese characters reading “理解与生成统一” accented with curling vines and tiny stars. Sunlight casts playful shadows on the cobblestone path leading to the door, where a vintage lantern in a sunbeam add charm. The linework is clean, colors vibrant yet soft, evoking a friendly, storybook atmosphere. No people or vehicles are present, emphasizing quiet serenity."
    #"A photorealistic daytime scene of an elevated highway under a clear blue sky, bathed in bright natural sunlight. The asphalt road glistens slightly, flanked by concrete barriers and distant rolling hills covered in lush greenery. Overhead, a few wispy cirrus clouds drift lazily. A prominent green road sign with white Chinese characters reads “理解与生成” mounted on a sturdy metal pole at the roadside. The composition emphasizes depth and perspective, with the highway curving gently into the horizon. No vehicles or people are present, evoking a serene, contemplative atmosphere. The lighting is crisp, casting soft shadows that enhance the realism of the concrete textures and road markings."
    #"A realistic NASA-style deep space probe or modular space station drifts silently in the vastness of deep space, with a large, detailed rocky or ice-covered planet partially visible in the background. Intense, direct light from a single off-camera star casts extremely sharp and well-defined shadows across the probe’s complex hull, which features metallic textures, solar panels, and various functional modules. The same light illuminates the star-facing side of the planet, creating strong light-dark contrast on its surface (potentially showing craters, canyons, or ice caps). The planet’s terminator line is clearly visible, showing a dramatic transition from bright illumination to deep shadow. The overall scene is dominated by a cool color palette: the deep space background is ink-black, dotted with a few precise distant stars; the shadowed parts of the planet and probe appear in deep blues or cool grays. Emphasize the vacuum of space and the hard quality of light, creating a quiet, lonely, yet highly technological dark sci-fi atmosphere. Photorealistic render, focusing on material details (like brushed or matte metal, wrinkles on insulation layers), precise light and shadow effects, high resolution, cinematic quality."
    #"A sunlit elevated highway under clear blue skies, flanked by lush greenery, with a prominent road sign reading “通往理解与生成统一”，no vehicles or people visible."
    #"Sun-drenched cyberpunk metropolis: holographic skyscrapers, floating transports, and a massive neon “蚂蚁 Ming-Omni” sign glowing electric blue against chrome towers, ultra-detailed daytime cityscape."
    #"Sunlit cyberpunk cityscape with holographic skyscrapers, glowing flying cars, and a massive neon sign displaying “蚂蚁 Ming-Omni” in electric blue and pink, casting reflections on metallic streets."

    messages = [
        {
            "role": "HUMAN",
            "content": [          
                {"type": "text", "text": prompt},
            ],
        }
    ]
    image_save_path = "./bookshore.jpg"

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
        image_gen_seed=0,
    )
    
    
    image.save(image_save_path)


    prompt =  "将文字内容 替换成 “理解与生成促进”"
    #"A photorealistic daytime scene of an elevated highway under a clear blue sky, bathed in bright natural sunlight. The asphalt road glistens slightly, flanked by concrete barriers and distant rolling hills covered in lush greenery. Overhead, a few wispy cirrus clouds drift lazily. A prominent green road sign with white Chinese characters reads “理解与生成” mounted on a sturdy metal pole at the roadside. The composition emphasizes depth and perspective, with the highway curving gently into the horizon. No vehicles or people are present, evoking a serene, contemplative atmosphere. The lighting is crisp, casting soft shadows that enhance the realism of the concrete textures and road markings."
    #"A realistic NASA-style deep space probe or modular space station drifts silently in the vastness of deep space, with a large, detailed rocky or ice-covered planet partially visible in the background. Intense, direct light from a single off-camera star casts extremely sharp and well-defined shadows across the probe’s complex hull, which features metallic textures, solar panels, and various functional modules. The same light illuminates the star-facing side of the planet, creating strong light-dark contrast on its surface (potentially showing craters, canyons, or ice caps). The planet’s terminator line is clearly visible, showing a dramatic transition from bright illumination to deep shadow. The overall scene is dominated by a cool color palette: the deep space background is ink-black, dotted with a few precise distant stars; the shadowed parts of the planet and probe appear in deep blues or cool grays. Emphasize the vacuum of space and the hard quality of light, creating a quiet, lonely, yet highly technological dark sci-fi atmosphere. Photorealistic render, focusing on material details (like brushed or matte metal, wrinkles on insulation layers), precise light and shadow effects, high resolution, cinematic quality."
    #"A sunlit elevated highway under clear blue skies, flanked by lush greenery, with a prominent road sign reading “通往理解与生成统一”，no vehicles or people visible."
    #"Sun-drenched cyberpunk metropolis: holographic skyscrapers, floating transports, and a massive neon “蚂蚁 Ming-Omni” sign glowing electric blue against chrome towers, ultra-detailed daytime cityscape."
    #"Sunlit cyberpunk cityscape with holographic skyscrapers, glowing flying cars, and a massive neon sign displaying “蚂蚁 Ming-Omni” in electric blue and pink, casting reflections on metallic streets."

    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "image", "image": "./bookshore.jpg"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    image_save_path = "./bookshore_edit.jpg"

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
        image_gen_seed=11,
    )
    
    
    image.save(image_save_path)
   
    

    