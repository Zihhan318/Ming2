# Email: wanren.pj@antgroup.com
# Copyright (c) Ant Group. All rights reserved.
from dataclasses import dataclass
from typing import Optional, Tuple, List
import os
import yaml
import re
import json
import torch
import torch.nn as nn
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from contextlib import nullcontext
import threading
import numpy as np
import time
import torch
import uuid

from transformers import Qwen2Config, PreTrainedModel
from transformers import Qwen2Model, AutoTokenizer
from configuration_bailing_talker import BailingTalkerConfig
from transformers.utils import ModelOutput
from talker_tn.talker_tn import TalkerTN
import logging
from talker_module.cfm import CFM, get_epss_timesteps
from talker_module.dit import DiT
from talker_module.aggregator import Aggregator
from transformers import StaticCache
from concurrent.futures import ThreadPoolExecutor

from front.number_en import normalize_numbers
from front.text_segment_cut import cut_text_by_semantic_length, is_chinese
from front.toolkit import tokenize_mixed_text_iterator


class CFMGraphExecutor:
    def __init__(self, config, cfm, aggregator, stop_head):
        self.config = config
        self.cfm = cfm
        self.aggregator = aggregator
        self.stop_head = stop_head
        self.initialized = False
        
        # 占位符
        self.last_hidden_state_placeholder = None
        self.his_lat_placeholder = None
        self.randn_like_placeholder = None
        self.t_placeholder = None
        self.gen_lat_placeholder = None
        self.inputs_embeds_placeholder = None
        self.stop_out_placeholder = None
        self.graph = None
        
    def execute(self, input_tensor, his_lat):
        randn_tensor = torch.randn_like(his_lat)
        t = get_epss_timesteps(
            self.config.steps, 
            device=input_tensor.device, 
            dtype=input_tensor.dtype
        )
        
        # 初始化
        if not self.initialized:
            self._initialize_graph(input_tensor, his_lat, randn_tensor, t)
        
        self.last_hidden_state_placeholder.copy_(input_tensor)
        self.his_lat_placeholder.copy_(his_lat)
        self.randn_like_placeholder.copy_(randn_tensor)
        self.t_placeholder.copy_(t)
        # torch.cuda.current_stream().synchronize()
        
        # 回放
        self.graph.replay()
        
        gen_lat = torch.empty_like(self.gen_lat_placeholder)
        gen_lat.copy_(self.gen_lat_placeholder)
        
        inputs_embeds = torch.empty_like(self.inputs_embeds_placeholder)
        inputs_embeds.copy_(self.inputs_embeds_placeholder)
        
        stop_out = torch.empty_like(self.stop_out_placeholder)
        stop_out.copy_(self.stop_out_placeholder)
        
        # torch.cuda.current_stream().synchronize()
        
        return gen_lat, inputs_embeds, stop_out
    
    def _initialize_graph(self, input_tensor, his_lat, randn_tensor, t):
        self.last_hidden_state_placeholder = torch.empty_like(input_tensor)
        self.his_lat_placeholder = torch.empty_like(his_lat)
        self.randn_like_placeholder = torch.empty_like(randn_tensor)
        self.t_placeholder = get_epss_timesteps(
            self.config.steps, 
            device=input_tensor.device, 
            dtype=input_tensor.dtype
        )
        
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.gen_lat_placeholder = self.cfm.sample(
                self.last_hidden_state_placeholder,
                self.his_lat_placeholder,
                self.randn_like_placeholder,
                self.t_placeholder
            )

            self.inputs_embeds_placeholder = self.aggregator(self.gen_lat_placeholder)
            self.stop_out_placeholder = self.stop_head(
                self.last_hidden_state_placeholder[:, -1, :]
            ).softmax(dim=-1)
        
        self.initialized = True

from queue import Queue
from threading import Lock

class CFMGraphExecutorPool:
    def __init__(self, config, cfm, aggregator, stop_head, pool_size=5):
        self.config = config
        self.cfm = cfm
        self.aggregator = aggregator
        self.stop_head = stop_head
        self.pool_size = pool_size
        self.pool = Queue(maxsize=pool_size)
        self.lock = Lock()  # 确保线程安全
        
        self._initialize_pool()
    
    def _initialize_pool(self):
        for _ in range(self.pool_size):
            executor = CFMGraphExecutor(
                self.config, 
                self.cfm, 
                self.aggregator, 
                self.stop_head
            )
            self.pool.put(executor)
    
    def acquire(self):
        return self.pool.get()

    def release(self, executor):
        if isinstance(executor, CFMGraphExecutor):
            self.pool.put(executor)
    
    def execute(self, input_tensor, his_lat):
        executor = self.acquire()
        try:
            gen_lat, inputs_embeds, stop_out = executor.execute(input_tensor, his_lat)
        finally:
            self.release(executor)
            return gen_lat, inputs_embeds, stop_out
    
    def __len__(self):
        return self.pool.qsize()
    
    def __str__(self):
        return f"CFMGraphExecutorPool(pool_size={self.pool_size}, available={self.__len__()})"


@dataclass
class BailingTalkerOutputWithPast(ModelOutput):
    pass

import queue

class BailingTalker2(PreTrainedModel):
    config_class = BailingTalkerConfig
    base_model_prefix = "model"

    def __init__(self, config: BailingTalkerConfig):
        super().__init__(config)

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(f'{self.config.name_or_path}/llm')
        self.model_config = Qwen2Config.from_pretrained(f'{self.config.name_or_path}/llm')
        self.model = Qwen2Model(self.model_config)
        self.model.config._attn_implementation="sdpa"
    

        self.cfm = CFM(DiT(
            llm_input_dim=self.model.config.hidden_size,
            **config.flowmodel,
        ), steps=config.steps, cfg_strength=config.cfg_strength)

        self.aggregator = Aggregator(
            llm_input_dim=self.model.config.hidden_size,
            **config.aggregator,
        )

        self.stop_head = nn.Linear(self.model.config.hidden_size, 2, bias=True)
        self.patch_size = config.patch_size

        self.normalizer = TalkerTN()

        self.lock = threading.Lock()
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.vae_cache = {}

        self.initialized = None
        self.initial_lock = threading.Lock()
        self.registered_prompt = dict()
        self.max_conc = 8
        self.executor = ThreadPoolExecutor(max_workers=self.max_conc)
        self.sampler_pool = CFMGraphExecutorPool(self.config, self.cfm, self.aggregator, self.stop_head, self.max_conc)
        self.model_graph_pool = queue.Queue()
        self.past_key_values = None
        for _ in range(self.max_conc):
            self.model_graph_pool.put((None, None, None, None, None))

        cur_dir = os.path.abspath(os.path.dirname(__file__))
        self.voice_json_dict = json.load(open(f'{cur_dir}/data/voice_name.json', 'r'))
        for key, value in self.voice_json_dict.items():
            prompt_wav_path = os.path.join(cur_dir, self.voice_json_dict[key]["prompt_wav_path"])
            self.voice_json_dict[key]["prompt_wav_path"] = prompt_wav_path

    def set_multithread_conc(self, max_thread_conc):
        self.max_conc = max_thread_conc
        self.executor = ThreadPoolExecutor(max_workers=self.max_conc)
        self.sampler_pool = CFMGraphExecutorPool(
            self.config, self.cfm, self.aggregator, self.stop_head, max_thread_conc
        )
        self.model_graph_pool = queue.Queue()
        for _ in range(self.max_conc):
            self.model_graph_pool.put((None, None, None, None, None))

        self.initial_graph()

        
    def set_multithread_conc(self, max_thread_conc):
        self.sampler_pool = CFMGraphExecutorPool(self.config, self.cfm, self.aggregator, self.stop_head, max_thread_conc)
        self.model_graph_pool = queue.Queue()
        for _ in range(self.max_conc):
            self.model_graph_pool.put((None, None, None, None, None))

        self.initial_graph()

    def initial_graph(self):
        
        with self.initial_lock:
            if not self.initialized:
                for _ in range(self.max_conc):
                    this_uuid = str(uuid.uuid1())

                    with (self.lock):
                        self.tts_speech_token_dict[this_uuid] = []
                        self.llm_end_dict[this_uuid] = False
                        self.vae_cache[this_uuid] = None

                    text = "初始化编译图"
                    prompt_text_input_part = []
                    prompt_wav_lat = prompt_wav_emb = None
                    future = self.executor.submit(
                        self.llm_job,
                        text,
                        prompt_text_input_part,
                        prompt_wav_lat,
                        prompt_wav_emb,
                        this_uuid
                    )
                    future.result()


                    with self.lock:
                        self.tts_speech_token_dict.pop(this_uuid)
                        self.llm_end_dict.pop(this_uuid)
                        self.vae_cache.pop(this_uuid)

                self.initialized = True

    def set_use_vllm(self, use_vllm: bool, vllm_in_process: bool = False):
        ...

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def generate(
        self,
        talker_text_prefix: torch.LongTensor,
        prompt_wav_lat=None,
        prompt_wav_emb=None,
        min_new_token=10,
    ):
        step = 0

        inputs_embeds = self.model.get_input_embeddings()(talker_text_prefix)
        if prompt_wav_emb is not None:
            inputs_embeds = torch.cat([inputs_embeds, prompt_wav_emb], dim=1)
            his_lat = prompt_wav_lat[:, -self.patch_size:]
        else:
            his_lat = torch.zeros((inputs_embeds.shape[0], self.patch_size, 64),
                                    dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        start_t = time.perf_counter()
        max_cache_len = 2048
        past_key_values, inputs_embeds_placeholder, cache_position_placeholder, outputs_placeholder, model_graph = self.model_graph_pool.get()
        if past_key_values is None:
            past_key_values = StaticCache(config=self.model.config, max_batch_size=1, max_cache_len=max_cache_len, device=self.model.device, dtype=self.model.dtype)
        else:
            past_key_values.reset()

        prefill_len = inputs_embeds.shape[1]

        while step < 1000 and step < max_cache_len - prefill_len:
            
            if step == 0:
                outputs = self.model(
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=True
                )
            else:
                past_seen_tokens = past_key_values.get_seq_length()
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )

                # outputs = self.model(
                #     past_key_values=self.past_key_values,
                #     inputs_embeds=inputs_embeds,
                #     use_cache=True,
                #     cache_position=cache_position,
                # )

                if model_graph is None:
                    model_graph = torch.cuda.CUDAGraph()
                    inputs_embeds_placeholder = torch.empty_like(inputs_embeds)
                    cache_position_placeholder = torch.empty_like(cache_position)
                    with torch.cuda.graph(model_graph):
                        outputs_placeholder = self.model(
                            past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds_placeholder,
                            use_cache=True,
                            cache_position=cache_position_placeholder,
                        )

                inputs_embeds_placeholder.copy_(inputs_embeds)
                cache_position_placeholder.copy_(cache_position)
                
                # 回放
                model_graph.replay()
                
                outputs = outputs_placeholder
                
            llm_end_time = time.perf_counter()
            
            # # 原始实现
            # t = 1/32. * torch.tensor([0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32], device=his_lat.device, dtype=his_lat.dtype)
            # gen_lat  = self.cfm.sample(outputs.last_hidden_state[:, -1:, :], his_lat, torch.randn_like(his_lat), t)
            # inputs_embeds = self.aggregator(gen_lat)
            # stop_out = self.stop_head(outputs.last_hidden_state[:, -1, :]).softmax(dim=-1).cpu()
            
            gen_lat, inputs_embeds, stop_out = self.sampler_pool.execute(outputs.last_hidden_state[:, -1:, :], his_lat)
            
            end_t = time.perf_counter()
            # print(f"step time cost: {llm_end_time - start_t:.3f}s {end_t - llm_end_time:.3f}s")
            start_t = end_t

            
            his_lat = gen_lat

            if step > min_new_token and stop_out.cpu()[0, 1] > 0.5:
                yield gen_lat, True
                break

            yield gen_lat, False
            step += 1
        self.model_graph_pool.put((past_key_values, inputs_embeds_placeholder, cache_position_placeholder, outputs_placeholder, model_graph))

    def omni_audio_generation_func(
        self,
        tts_text,
        prompt_text_input_part=None,
        prompt_wav_lat=None,
        prompt_wav_emb=None,
    ):
        text_input_part = self.tokenizer.encode(tts_text)
        talker_text_prefix = prompt_text_input_part + text_input_part + self.tokenizer.encode("<|endoftext|>")
        talker_text_prefix = torch.tensor([talker_text_prefix], dtype=torch.int64, device=self.device)
        for audio_token in self.generate(
            talker_text_prefix=talker_text_prefix,
            prompt_wav_lat=prompt_wav_lat,
            prompt_wav_emb=prompt_wav_emb,
        ):
            yield audio_token

    def token2wav(
        self,
        audio_detokenizer,
        token,
        cache=None,
        stream=False,
        last_chunk=False,
    ):
        speech, _, new_cache = audio_detokenizer.decoder(
            torch.cat(token, dim=1), cache=cache, streaming=stream, last_chunk=last_chunk)
        return speech[0].detach().float(), new_cache

    def llm_job(
        self,
        text,
        prompt_text_input_part,
        prompt_wav_lat,
        prompt_wav_emb,
        this_uuid,
    ):
        with torch.cuda.stream(torch.cuda.Stream(self.device)):
            for audio_token in self.omni_audio_generation_func(
                tts_text=text,
                prompt_text_input_part=prompt_text_input_part,
                prompt_wav_lat=prompt_wav_lat,
                prompt_wav_emb=prompt_wav_emb,
            ):
                self.tts_speech_token_dict[this_uuid].append(audio_token)

        self.llm_end_dict[this_uuid] = True

    def tts_job(
        self,
        text,
        audio_detokenizer,
        prompt_text_input_part,
        prompt_wav_lat,
        prompt_wav_emb,
        stream):
        this_uuid = str(uuid.uuid1())

        with (self.lock):
            self.tts_speech_token_dict[this_uuid] = []
            self.llm_end_dict[this_uuid] = False
            self.vae_cache[this_uuid] = None

        future = self.executor.submit(
            self.llm_job,
            text,
            prompt_text_input_part,
            prompt_wav_lat,
            prompt_wav_emb,
            this_uuid
        )

        if stream is True:
            token_offset = 0
            while True:
                time.sleep(0.1)
                nxt = len(self.tts_speech_token_dict[this_uuid])
                if nxt > token_offset:
                    this_tts_speech_token = self.tts_speech_token_dict[this_uuid][token_offset:nxt]

                    last_chunk = this_tts_speech_token[-1][-1]
                    this_tts_speech_token = [ii[0] for ii in this_tts_speech_token]
                    this_tts_speech, self.vae_cache[this_uuid] = self.token2wav(
                        audio_detokenizer=audio_detokenizer,
                        token=this_tts_speech_token,
                        cache=self.vae_cache[this_uuid], stream=True, last_chunk=last_chunk
                    )
                    token_offset = nxt
                    yield {"tts_speech": this_tts_speech.cpu()}

                if self.llm_end_dict[this_uuid] is True and token_offset == len(self.tts_speech_token_dict[this_uuid]):
                    break
            future.result()
        else:
            # deal with all tokens
            future.result()
            this_tts_speech_token = self.tts_speech_token_dict[this_uuid]
            this_tts_speech_token = [ii[0] for ii in this_tts_speech_token]
            this_tts_speech, self.vae_cache[this_uuid] = self.token2wav(
                audio_detokenizer=audio_detokenizer,
                token=this_tts_speech_token,
                cache=self.vae_cache[this_uuid],
                stream=False
            )
            
            yield {"tts_speech": this_tts_speech.cpu()}
        
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()
        
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.vae_cache.pop(this_uuid)

    def register_prompt_wav(self, prompt_text, prompt_wav_path, audio_detokenizer):
        prompt_text_input_part = self.tokenizer.encode(prompt_text)
        speech, sample_rate = torchaudio.load(prompt_wav_path, backend='soundfile')
        if sample_rate != 16000:
            speech = torchaudio.transforms.Resample(sample_rate, 16000)(speech)
        prompt_wav_lat = audio_detokenizer.get_latent(speech.to(dtype=torch.bfloat16, device=self.device))  # btd
        if prompt_wav_lat.shape[1] % self.patch_size != 0:
            prompt_wav_lat = prompt_wav_lat[:, :-(prompt_wav_lat.shape[1]%self.patch_size)]
        prompt_wav_lat = prompt_wav_lat.reshape(-1, self.patch_size, prompt_wav_lat.shape[-1])
        prompt_wav_emb = self.aggregator(prompt_wav_lat)
        prompt_wav_lat = prompt_wav_lat.reshape(1, -1, prompt_wav_lat.shape[-1])
        prompt_wav_emb = prompt_wav_emb.reshape(1, -1, prompt_wav_emb.shape[-1])
        self.registered_prompt[prompt_wav_path] = {
            "prompt_text_input_part": prompt_text_input_part,
            "prompt_wav_lat": prompt_wav_lat,
            "prompt_wav_emb": prompt_wav_emb,
        }
        logging.info(f"register_prompt_wav with {prompt_text}, {prompt_wav_path}")


    def get_prompt_emb(self, prompt_text, prompt_wav_path, audio_detokenizer):
        if prompt_wav_path not in self.registered_prompt:
            self.register_prompt_wav(prompt_text, prompt_wav_path, audio_detokenizer)
        registered_prompt_msg = self.registered_prompt[prompt_wav_path]
        return registered_prompt_msg['prompt_text_input_part'], registered_prompt_msg["prompt_wav_lat"], registered_prompt_msg["prompt_wav_emb"]


    def omni_audio_generation(
        self,
        tts_text,
        voice_name="DB30",
        prompt_text=None,
        prompt_wav_path=None,
        max_length=50,
        audio_detokenizer=None,
        stream=False,
        **kwargs,
    ):
        talker_last_time = time.perf_counter()
        self.initial_graph()

        if voice_name in self.voice_json_dict:
            prompt_text = self.voice_json_dict[voice_name]["prompt_text"]
            prompt_wav_path = self.voice_json_dict[voice_name]["prompt_wav_path"]
        prompt_text_input_part, prompt_wav_lat, prompt_wav_emb = self.get_prompt_emb(prompt_text, prompt_wav_path, audio_detokenizer)

        assert (
            max_length > 0
        ), f"max_length must be greater than 0, but here is {max_length}"
        streaming_text = []
        count = 0
        cache_position = {}

        # str2list, for english
        tts_text = tokenize_mixed_text_iterator(tts_text)
        
        for i, ele in enumerate(tts_text):
            if len(ele) == 0:
                continue

            # 正常的分句，不包括英文句点
            if ele[-1] in "！？。!?" and (
                len(streaming_text) >= 12 or count > 0 and len(streaming_text) >= 8
            ):
                streaming_text.append(ele)
                streaming_text = "".join(streaming_text)

                sub_output_dict = cut_text_by_semantic_length(streaming_text, max_length)
                text_list = sub_output_dict["fragments"]
                if not text_list:
                    logging.info(f'{streaming_text}\thas no valid segments')
                    continue
                length = len(text_list[0])

                if len(cache_position) == 0:
                    cache_position.update({count: (0, length)})
                else:
                    start_idx = list(cache_position.values())[-1][0]
                    end_idx = list(cache_position.values())[-1][1]
                    cache_position.update({count: (end_idx, end_idx+length)})
                # sub_position_dict = sub_output_dict["positions"]

                for text_ori in text_list:
                    if not is_chinese(text_ori):
                        text = normalize_numbers(text_ori)
                    else:
                        text = text_ori
                    # original normalization
                    text = self.normalizer.normalize(text)

                    if text[0] == "，":
                        text = text[1:]
                    # 首句流式，其余句子非流式
                    if count == 0:
                        for idx, this_tts_speech_dict in enumerate(
                            self.tts_job(
                                text=text,
                                audio_detokenizer=audio_detokenizer,
                                prompt_text_input_part=prompt_text_input_part,
                                prompt_wav_lat=prompt_wav_lat,
                                prompt_wav_emb=prompt_wav_emb,
                                stream=stream & True,
                            )
                        ):
                            yield this_tts_speech_dict["tts_speech"], (
                                text_ori if idx == 0 else ""
                            ), cache_position[count], None
                    else:
                        for idx, this_tts_speech_dict in enumerate(
                            self.tts_job(
                                text=text,
                                audio_detokenizer=audio_detokenizer,
                                prompt_text_input_part=prompt_text_input_part,
                                prompt_wav_lat=prompt_wav_lat,
                                prompt_wav_emb=prompt_wav_emb,
                                stream=False,
                            )
                        ):
                            yield this_tts_speech_dict["tts_speech"], (
                                text_ori if idx == 0 else ""
                            ), cache_position[count], float(this_tts_speech_dict["tts_speech"].shape[-1]/16000)

                streaming_text = []
                count += 1

            # 单独判断小数点
            elif ele[-1] == "." and \
                (len(streaming_text) >= 12 or count > 0 and len(streaming_text)>=8) and \
                bool(re.search(r'[0-9]',streaming_text[-1][-1])) is False:

                streaming_text.append(ele)
                streaming_text = "".join(streaming_text)

                sub_output_dict = cut_text_by_semantic_length(streaming_text, max_length)
                text_list = sub_output_dict["fragments"]
                if not text_list:
                    logging.info(f'{streaming_text}\thas no valid segments')
                    continue
                length = len(text_list[0])

                if len(cache_position) == 0:
                    cache_position.update({count: (0, length)})
                else:
                    start_idx = list(cache_position.values())[-1][0]
                    end_idx = list(cache_position.values())[-1][1]
                    cache_position.update({count: (end_idx, end_idx+length)})
                # sub_position_dict = sub_output_dict["positions"]

                for text_ori in text_list:
                    if not is_chinese(text_ori):
                        text = normalize_numbers(text_ori)
                    else:
                        text = text_ori
                    # original normalization
                    text = self.normalizer.normalize(text)

                    if text[0] == "，":
                        text = text[1:]
                    # 首句流式，其余句子非流式
                    if count == 0:
                        for idx, this_tts_speech_dict in enumerate(
                            self.tts_job(
                                text=text,
                                audio_detokenizer=audio_detokenizer,
                                prompt_text_input_part=prompt_text_input_part,
                                prompt_wav_lat=prompt_wav_lat,
                                prompt_wav_emb=prompt_wav_emb,
                                stream=stream & True,
                            )
                        ):
                            yield this_tts_speech_dict["tts_speech"], (
                                text_ori if idx == 0 else ""
                            ), cache_position[count], None
                    else:
                        for idx, this_tts_speech_dict in enumerate(
                            self.tts_job(
                                text=text,
                                audio_detokenizer=audio_detokenizer,
                                prompt_text_input_part=prompt_text_input_part,
                                prompt_wav_lat=prompt_wav_lat,
                                prompt_wav_emb=prompt_wav_emb,
                                stream=False,
                            )
                        ):
                            yield this_tts_speech_dict["tts_speech"], (
                                text_ori if idx == 0 else ""
                            ), cache_position[count], float(this_tts_speech_dict["tts_speech"].shape[-1]/16000)

                streaming_text = []
                count += 1

            # 针对换行符的判断
            elif ele[-1] == "\n":
                if len(streaming_text) > 0:
                    # 中文
                    if bool(re.search(r"[\u4e00-\u9fff]", "".join(streaming_text))):
                        if len(streaming_text) > 0 and bool(
                            re.search(r"[\u4e00-\u9fff]", streaming_text[-1][-1])
                        ):
                            ele = "，"
                            streaming_text.append(ele)
                    # 英文
                    else:
                        # 当前单词尾部无符号
                        if len(ele) > 1 and bool(re.search(r"[a-zA-Z]", ele[-2])):
                            ele = ele[:-1] + "."
                        # 当前单词尾部有符号
                        else:
                            ele = ele[:-1]
                        streaming_text.append(ele)
                        
                # 触发分句条件
                if len(streaming_text) >= 12 or count > 0 and len(streaming_text) >= 8:
                    streaming_text = "".join(streaming_text)

                    text_list = cut_text_by_semantic_length(streaming_text, max_length)
                    text_list = text_list["fragments"]
                    if not text_list:
                        logging.info(f'{streaming_text}\thas no valid segments')
                        continue
                    length = len(text_list[0])
                    if len(cache_position) == 0:
                        cache_position.update({count: (0, length)})
                    else:
                        start_idx = list(cache_position.values())[-1][0]
                        end_idx = list(cache_position.values())[-1][1]
                        cache_position.update({count: (end_idx, end_idx+length)})

                    logging.info("针对换行符的判断")
                    for text_ori in text_list:
                        if not is_chinese(text_ori):
                            text = normalize_numbers(text_ori)
                        else:
                            text = text_ori
                        # original normalization
                        text = self.normalizer.normalize(text)
                        
                        if text[0] == "，":
                            text = text[1:]

                        if count == 0:  # 首句流式
                            for idx, this_tts_speech_dict in enumerate(
                                self.tts_job(
                                    text=text,
                                    audio_detokenizer=audio_detokenizer,
                                    prompt_text_input_part=prompt_text_input_part,
                                    prompt_wav_lat=prompt_wav_lat,
                                    prompt_wav_emb=prompt_wav_emb,
                                    stream=stream & True,
                                )
                            ):
                                yield this_tts_speech_dict["tts_speech"], (
                                    text_ori if idx == 0 else ""
                                ), cache_position[count], None
                        else:  # 非流式
                            for idx, this_tts_speech_dict in enumerate(
                                self.tts_job(
                                    text=text,
                                    audio_detokenizer=audio_detokenizer,
                                    prompt_text_input_part=prompt_text_input_part,
                                    prompt_wav_lat=prompt_wav_lat,
                                    prompt_wav_emb=prompt_wav_emb,
                                    stream=False,
                                )
                            ):
                                yield this_tts_speech_dict["tts_speech"], (
                                    text_ori if idx == 0 else ""
                                ), cache_position[count], float(this_tts_speech_dict["tts_speech"].shape[-1]/16000)

                    streaming_text = []
                    count += 1
            else:
                streaming_text.append(ele)

        # for last sentence, if contain meaningful content
        if len(streaming_text) > 0 and re.search(
            r"[a-zA-Z\u4e00-\u9fff1-9]", "".join(streaming_text)
        ):
            streaming_text = "".join(streaming_text)
            text_list = cut_text_by_semantic_length(streaming_text, max_length)
            text_list = text_list["fragments"]
            if text_list:
                length = len(text_list[0])

                if len(cache_position) == 0:
                    cache_position.update({count: (0, length)})
                else:
                    start_idx = list(cache_position.values())[-1][0]
                    end_idx = list(cache_position.values())[-1][1]
                    cache_position.update({count: (end_idx, end_idx+length)})
                
                logging.info("for last sentence")
                for text_ori in text_list:
                    if not is_chinese(text_ori):
                        text = normalize_numbers(text_ori)
                    else:
                        text = text_ori
                    # original normalization
                    text = self.normalizer.normalize(text)

                    if text[0] == "，":
                        text = text[1:]

                    if count == 0:  # 首句流式
                        for idx, this_tts_speech_dict in enumerate(
                            self.tts_job(
                                text=text,
                                audio_detokenizer=audio_detokenizer,
                                prompt_text_input_part=prompt_text_input_part,
                                prompt_wav_lat=prompt_wav_lat,
                                prompt_wav_emb=prompt_wav_emb,
                                stream=stream & True,
                            )
                        ):
                            yield this_tts_speech_dict["tts_speech"], (
                                text_ori if idx == 0 else ""
                            ), cache_position[count], None
                    else:  # 非流式
                        for idx, this_tts_speech_dict in enumerate(
                            self.tts_job(
                                text=text,
                                audio_detokenizer=audio_detokenizer,
                                prompt_text_input_part=prompt_text_input_part,
                                prompt_wav_lat=prompt_wav_lat,
                                prompt_wav_emb=prompt_wav_emb,
                                stream=False,
                            )
                        ):
                            yield this_tts_speech_dict["tts_speech"], (
                                text_ori if idx == 0 else ""
                            ), cache_position[count], float(this_tts_speech_dict["tts_speech"].shape[-1]/16000)
                            