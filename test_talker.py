# Author: wanren
# Email: wanren.pj@antgroup.com
# Date: 2025/9/28
import re
import torch
import numpy as np
import soundfile
from modeling_bailing_talker import BailingTalker2
from AudioVAE.modeling_audio_vae import AudioVAE


def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def thread_task_func(talker, tts_text, vae, output_dir=None):

    is_chinese = contains_chinese(tts_text)
    # support english
    if not is_chinese:
        tts_text = tts_text.split()

    torch.manual_seed(1024)
    np.random.seed(1024)

    def input_wrapper(tts_text):
        # for i in tts_text:
        #     yield i
        return tts_text

    import time
    start_time = time.perf_counter()
    all_wavs = []
    for tts_speech, text_list, _, _ in talker.omni_audio_generation(
            tts_text=input_wrapper(tts_text),
            audio_detokenizer=vae, stream=False):
        all_wavs.append(tts_speech)
        print(f"output: {text_list}")
    waveform = torch.cat(all_wavs, dim=-1)
    end_time = time.perf_counter()
    sample_points = waveform.shape[1]  # 取"采样点数"维度（根据实际波形形状调整，见下方说明）
    sample_rate = vae.sr  # 音频采样率（如44100 Hz = 每秒44100个采样点）

    audio_duration = sample_points / sample_rate
    print(f"inference time cost: {end_time - start_time:.3f}s, duration: {audio_duration:.3f}s, rtf: {(end_time - start_time) / audio_duration:.3f}")
    if output_dir:
        soundfile.write(output_dir, waveform.T.numpy(), vae.sr)    
        print("save audio in ", output_dir)


def test_tts():
    device = f'cuda:0'
    mdl_path = '/heyuan2_12/workspace/wanren.pj/resource/inclusionAI/Ming-Lite-Omni-1.5'
    talker = BailingTalker2.from_pretrained(
        f'{mdl_path}/talker', torch_dtype=torch.bfloat16).eval().to(dtype=torch.bfloat16, device=device)
    talker.use_vllm = False
    vae = AudioVAE.from_pretrained(
        f'{mdl_path}/talker/vae', torch_dtype=torch.bfloat16).eval().to(dtype=torch.bfloat16, device=device)

    
    test_long_text = "13年第一次买车，一辆很老的二手桑塔纳，当时在酒店工作，白班下班时候我舅舅开着刚买回来的车带上我去姥爷家，我爸坐前面我坐在后面，那时候正好我舅舅没有出门，让他带着我练了几天车，车是手动挡的，从起步开始一点点练，那时候胆子小，只在废弃的厂子里练习练习，来回在路上都是我舅舅开，"
    
    test_long_text = '这是一条测试语句。欢迎使用百灵。你可以问我一些问题。'
    thread_num = 8
    # for _ in range(thread_num):
    #     thread_task_func(talker, test_long_text, vae, 'out_tts_init.wav')
    thread_task_func(talker, test_long_text, vae, 'out_tts_init.wav')
    thread_task_func(talker, test_long_text, vae, 'out_tts_init.wav')

    import threading
    tts_text = '这是一条测试语句。欢迎使用百灵。你可以问我一些问题。'

    for _ in range(3):
        thread_tasks = []
        for idx in range(thread_num):
            thread_tasks.append(threading.Thread(target=thread_task_func, args=(talker, tts_text, vae, f'out_tts_{idx}.wav')))
        
        for task in thread_tasks:
            task.start()
        
        for task in thread_tasks:
            task.join()
    
if __name__ == '__main__':
    test_tts()
