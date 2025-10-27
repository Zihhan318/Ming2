import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2Model, Qwen2Config

from .istft import ISTFTHead


class Encoder(nn.Module):
    def __init__(self, encoder_args, input_dim=320, hop_size=320, latent_dim=64):
        super().__init__()
        config = Qwen2Config.from_dict(config_dict=encoder_args)
        self.encoder = Qwen2Model(config)
        self.input_dim = input_dim
        self.hop_size = hop_size
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(input_dim, config.hidden_size, bias=False)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc3 = nn.Linear(config.hidden_size, latent_dim*2)
        self.norm = nn.LayerNorm(config.hidden_size)

    def pad_waveform(self, x):
        length = x.size(1)
        if length % self.input_dim == 0:
            return x

        pad_length = self.input_dim - (length % self.input_dim)

        x = F.pad(x, (0, pad_length, 0, 0), value=0)
        assert x.size(1) % self.input_dim == 0
        return x

    def get_frames(self, x):
        num_frames_total = (x.size(-1) + self.hop_size - 1) // self.hop_size  # 向上取整的帧数
        expected_len = (num_frames_total - 1) * self.hop_size + self.input_dim
        padding_needed = expected_len - x.size(-1)
        waveform = F.pad(x, (0, padding_needed), value=0.0)

        frames = waveform.unfold(dimension=-1, size=self.input_dim, step=self.hop_size)   # [B, T, d]
        return frames

    def forward(self, waveform):
        """

        Args:
            waveform: [B, T]

        Returns:

        """
        x = self.get_frames(waveform)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.encoder(inputs_embeds=x)
        x = x.last_hidden_state
        x = self.fc3(x)
        return x, waveform.unsqueeze(1)


class Decoder(nn.Module):
    def __init__(self, decoder_args, output_dim=320, latent_dim=64, semantic_model=None, is_semantic_model_casual=False):
        super().__init__()
        config = Qwen2Config.from_dict(config_dict=decoder_args)
        self.decoder = Qwen2Model(config)
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(latent_dim, config.hidden_size)

        if semantic_model is not None:
            self.gelu = nn.GELU()
            self.fc2 = nn.Linear(config.hidden_size, semantic_model.audio_emb_dim)
            self.semantic_model = semantic_model
            self.fc3 = nn.Linear(semantic_model.audio_emb_dim, config.hidden_size)
            self.is_semantic_model_casual = is_semantic_model_casual
        else:
            self.semantic_model = None

        self.hop_length = output_dim
        self.head = ISTFTHead(dim=config.hidden_size, n_fft=self.hop_length * 4, hop_length=self.hop_length, padding="same")

    def forward(self, x, only_semantic_emb=False, cache=None, streaming=False, last_chunk=False):
        """

        Args:
            waveform: [B, T, d]
            only_semantic_emb: 是否只返回semantic embedding

        Returns:

        """
        x = self.fc1(x)
        residual = x

        if self.semantic_model is not None:
            x = self.fc2(self.gelu(x))
            mask = 1 if self.is_semantic_model_casual else None
            x, _ = self.semantic_model(whipser_feats=x, encoder_only=True, mask=mask)
            semantic_emb = x
            if only_semantic_emb:
                return semantic_emb
            x = self.fc3(x)
            x += residual
        else:
            semantic_emb = None

        if streaming:
            if cache is None:
                past_key_values = audio_buffer = window_buffer = None
            else:
                past_key_values, audio_buffer, window_buffer = cache
            x = self.decoder(inputs_embeds=x, use_cache=True, past_key_values=past_key_values)
            past_key_values = x.past_key_values
            x, _, audio_buffer, window_buffer = self.head(
                x.last_hidden_state, audio_buffer=audio_buffer, window_buffer=window_buffer,
                streaming=True, last_chunk=last_chunk)
            return x, semantic_emb, (past_key_values, audio_buffer, window_buffer)
        
        x = self.decoder(inputs_embeds=x, use_cache=False, past_key_values=None)
        x, _, audio_buffer, window_buffer = self.head(
            x.last_hidden_state, audio_buffer=None, window_buffer=None, streaming=False, last_chunk=False)
        return x, semantic_emb, None
