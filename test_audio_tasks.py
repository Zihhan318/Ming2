import os
import torch
import time
import numpy as np
from bisect import bisect_left

from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig
)

from modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration
import warnings
import argparse

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
    device_map['talker'] = 0
    return device_map


class BailingMMInfer:
    def __init__(self,
                 model_name_or_path,
                 generation_config=None,
                 ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.model, self.tokenizer, self.processor = self.load_model_processor()

        if generation_config is None:
            generation_config = {"num_beams": 1}
        self.generation_config = GenerationConfig.from_dict(generation_config)

    def load_model_processor(self):
        tokenizer = AutoTokenizer.from_pretrained('.', trust_remote_code=True)
        processor = AutoProcessor.from_pretrained('.', trust_remote_code=True)

        model = BailingMM2NativeForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=split_model(),
            load_talker=False,
        ).to(dtype=torch.bfloat16)
        return model, tokenizer, processor

    def generate(self, messages, max_new_tokens=512, sys_prompt_exp=None, use_cot_system_prompt=False, lang=None):
        text = self.processor.apply_chat_template(
            messages,
            sys_prompt_exp=sys_prompt_exp,
            use_cot_system_prompt=use_cot_system_prompt
        )

        image_inputs, video_inputs, audio_inputs = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            audios=audio_inputs,
            audio_kwargs={"use_whisper_encoder": True},
            return_tensors="pt",
        ).to(self.model.device)
        
        if lang is not None:
            language = torch.tensor([self.tokenizer.encode(f'{lang}\t')]).to(inputs['input_ids'].device)
            inputs['input_ids'] = torch.cat([inputs['input_ids'], language], dim=1)
            attention_mask = inputs['attention_mask']
            inputs['attention_mask'] = torch.ones(inputs['input_ids'].shape, dtype=attention_mask.dtype)

        for k in inputs.keys():
            if k == "pixel_values" or k == "pixel_values_videos" or k == "audio_feats":
                inputs[k] = inputs[k].to(dtype=torch.bfloat16)

        srt_time = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                eos_token_id=self.processor.gen_terminator,
                generation_config=self.generation_config,
                num_logits_to_keep=1,
            )

        end_time = time.time()
        print(self.tokenizer.decode(generated_ids[0]))
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # tps = generated_ids.shape[1] / (end_time - srt_time)
        # print(f"generated {generated_ids.shape[1]} tokens in {end_time - srt_time:.2f} seconds, tokens per second: {tps:.2f} tokens/s")

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        prompt = self.tokenizer.decode(inputs['input_ids'][0]).replace('<audioPatch>', '')
        print(f"prompt: {prompt}")
        return output_text
    
    
if __name__ == '__main__':
    # model_name_or_path = '/input/yangmingxue.ymx/ckpts/Ming_Flash_2.0_sft1'
    model_name_or_path = '/pcache-mnt/ro/checkpoint/164002/430380/203999706/Ming-Flash-2.0-20251005-HF/master_0/100/ckpt' # aistudio://12872297/Ming-Flash-2.0-20251005-HF
    model = BailingMMInfer(
        model_name_or_path,
    )

    audio_path = "data/wavs/"
    
    # ASR
    print("Testing ASR...")
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {
                    "type": "text",
                    "text": "Please recognize the language of this speech and transcribe it. Format: oral.",
                },
                {"type": "audio", "audio": os.path.join(audio_path, "BAC009S0915W0292.wav")},
            ],
        },
    ]
    srt_time = time.time()
    output = model.generate(messages=messages, lang="Chinese")
    print(f"debug asr output:{output}")
    print(f"Generate time asr: {(time.time() - srt_time):.2f}s")

    # Speech QA
    print("Testing Speech QA...")
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "audio", "audio": os.path.join(audio_path, "speechQA_sample.wav")},
            ],
        },
    ]
    srt_time = time.time()
    output = model.generate(messages=messages)
    print(f"debug speechqa output:{output}")
    print(f"Generate time speechqa: {(time.time() - srt_time):.2f}s")

    # # Speech QA with asr
    # print("Testing Speech QA...")
    # messages = [
    #     {
    #         "role": "HUMAN",
    #         "content": [
    #             {
    #                 "type": "text",
    #                 # "text": "Please first recognize the language of this speech and transcribe it, then answer the question or follow the instruction in the speech."
    #                 "text": "Transcribe the speech, then answer the question or follow the instruction in the speech."
    #             },
    #             {"type": "audio", "audio": os.path.join(audio_path, "speechQA_sample.wav")},
    #         ],
    #     },
    # ]
    # srt_time = time.time()
    # output = model.generate(messages=messages)
    # print(f"debug speechqa output:{output}")
    # print(f"Generate time speechqa: {(time.time() - srt_time):.2f}s")

    # # Speech QA & TTS
    # if model.model.talker is not None and model.model.talker_vae is not None:
    #     model.model.talker.use_vllm = False
    #     model.model.talker.eval()
    #     model.model.talker_vae.eval()
    #     output_text = '这是一条测试音频。欢迎使用百灵。'
    #     srt_time = time.time()
    #     all_wavs = []
    #     with torch.inference_mode():
    #         for tts_speech, text_list in model.model.talker.omni_audio_generation(
    #                 tts_text=output_text, 
    #                 # prompt_text='诶，你今天有空吗？我们能不能聊聊这件事啊？',
    #                 # prompt_wav_path='data/wavs/prompt_15014.wav',
    #                 prompt_text='感谢你的认可。',
    #                 prompt_wav_path='data/spks/prompt.wav',
    #                 audio_detokenizer=model.model.talker_vae, stream=True
    #         ):
    #             all_wavs.append(tts_speech)
    #     waveform = torch.cat(all_wavs, dim=-1)
    #     import soundfile
    #     soundfile.write(f"out.wav", waveform.T.numpy(), model.model.talker_vae.sr)
    #     print(f"Generate time: {(time.time() - srt_time):.2f}s")
    #     print('save Speech QA to out.wav')


    # AAC
    print("Testing AAC...")
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {
                    "type": "text",
                    "text": "请写一句话描述这段音频。",
                },
                {"type": "audio", "audio": "data/wavs/glass-breaking-151256.mp3"},
            ],
        },
    ]
    srt_time = time.time()
    output = model.generate(messages=messages, lang="Chinese")
    print(f"debug aac output:{output}")
    print(f"Generate time aac: {(time.time() - srt_time):.2f}s")

    # Multi-Audio
    print("Testing Multi-Audio...")
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {
                    "type": "text",
                    "text": "这两条音频分别讲了什么"
                },
                {"type": "audio", "audio": "data/wavs/BAC009S0915W0292.wav"},
                {"type": "audio", "audio": "data/wavs/BAC009S0915W0283.wav"},
            ],
        },
    ]
    srt_time = time.time()
    output = model.generate(messages=messages, lang="Chinese")
    print(f"debug Multi-Audio output:{output}")
    print(f"Generate time: {(time.time() - srt_time):.2f}s")

    # # QA
    # messages = [
    #     {
    #         "role": "HUMAN",
    #         "content": [
    #             {"type": "text", "text": "An electric car runs on electricity via\nA. gasoline\nB. a power station\nC. electrical conductors\nD. fuel\nWhat is the answer to the above multiple choice question? Select one of the following: A, B, C, or D."}
    #         ],
    #     }
    # ]
    # srt_time = time.time()
    # output = model.generate(messages=messages)
    # print(f"debug qa output:{output}")
    # print(f"Generate time speechqa: {(time.time() - srt_time):.2f}s")



    # One audio multi query
    messages = [
         {
            "role": "HUMAN",
            "content": [
            {
                "type": "text",
                "text": "这段音频是什么"
            },
            {
                "type": "audio",
                "audio": "data/wavs/glass-breaking-151256.mp3"
                # 'audio': "data/wavs/BAC009S0915W0283.wav"
            }
            ]
        },
        {
            "role": "ASSISTANT",
            "content": [
            {
                "type": "text",
                "text": "玻璃杯"
            }
            ]
        },
        {
            "role": "HUMAN",
            "content": [
                {'type': "audio", 'audio': "data/wavs/BAC009S0915W0283.wav"},
                {"type": "text", "text": "这两段音频有啥区别"},
                
            ],
        }
    ]
    srt_time = time.time()
    output = model.generate(messages=messages)
    print(f"debug output:{output}")
    print(f"Generate time: {(time.time() - srt_time):.2f}s")
