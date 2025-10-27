from dataclasses import dataclass
from typing import Optional
from diffusers.models.autoencoders.autoencoder_oobleck import OobleckDiagonalGaussianDistribution
from transformers import PreTrainedModel
import torch
import torch.nn as nn
from transformers.utils import ModelOutput

# from .audio_encoder import WhisperAudioEncoder
from .configuration_audio_vae import AudioVAEconfig
from .vae_modules import Encoder, Decoder


@dataclass
class AudioVAEOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    gt_wav: Optional[torch.FloatTensor] = None
    gen_wav: Optional[torch.FloatTensor] = None
    vq_code: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    vq_loss: Optional[torch.FloatTensor] = None
    mel_loss: Optional[torch.FloatTensor] = None
    stats: Optional = None
    semantic_emb: Optional[torch.FloatTensor] = None
    semantic_emb_gt: Optional[torch.FloatTensor] = None
    kl: Optional[torch.FloatTensor] = None
    latent: Optional[torch.FloatTensor] = None


class AudioVAE(PreTrainedModel):
    config_class = AudioVAEconfig

    def __init__(self, config: AudioVAEconfig):
        super().__init__(config)
        self.pre_initialized_paras = set()
        self.encoder = Encoder(
            encoder_args=config.enc_kwargs['backbone'],
            input_dim=config.enc_kwargs['input_dim'],
            hop_size=config.enc_kwargs.get('hop_size', 320),
            latent_dim=config.enc_kwargs['latent_dim']
        )

        if config.semantic_module_kwargs is not None:
            semantic_model = WhisperAudioEncoder.from_pretrained(config.semantic_module_kwargs['whisper_model_path'])
            self.semantic_emb_dim = config.semantic_module_kwargs['whisper_encoder']['n_state']
            self.set_module_init_state(semantic_model, 'decoder.semantic_model')
            is_semantic_model_casual = config.semantic_module_kwargs['casual']
        else:
            semantic_model = None
            is_semantic_model_casual = False

        self.decoder = Decoder(
            decoder_args=config.dec_kwargs['backbone'],
            output_dim=config.dec_kwargs['output_dim'],
            latent_dim=config.dec_kwargs['latent_dim'],
            semantic_model=semantic_model,
            is_semantic_model_casual=is_semantic_model_casual
        )

        self.sr = 16000
        # if config.semantic_module_kwargs is not None:
        #     self.whisper_encoder = WhisperAudioEncoder.from_pretrained(config.semantic_module_kwargs['whisper_model_path'])
        #     self.whisper_encoder.eval()
        #     self.set_module_init_state(self.whisper_encoder, 'whisper_encoder')
        #     for n, p in self.whisper_encoder.named_parameters():
        #         p.requires_grad = False
        # else:
        #     self.whisper_encoder = None
        #
        # # 只训练语义
        # # for n, p in self.named_parameters():
        # #     p.requires_grad = False
        # # for n, p in self.decoder.fc2.named_parameters():
        # #     p.requires_grad = True
        # # for n, p in self.decoder.semantic_model.named_parameters():
        # #     p.requires_grad = True
        #
        # self.post_init()

    def set_module_init_state(self, module, prefix):
        for submodule in module.modules():
            submodule._is_hf_initialized = True
        for n, p in module.state_dict().items():
            self.pre_initialized_paras.add(f'{prefix}.{n}')

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if self.config.init_method == 'kaiming':
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            else:
                module.weight.data.normal_(mean=0.0, std=std)

            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        audio_feats,
        stats=None,
        is_train=True
    ):
        waveform = audio_feats['waveform']
        h, y = self.encoder(waveform)
        h = h.transpose(1, 2)  # [B, d, T]

        posterior = OobleckDiagonalGaussianDistribution(h)
        z = posterior.sample()  # [B, d/2, T]
        kl = posterior.kl()

        y_, semantic_emb = self.decoder(z.transpose(1, 2))

        if self.whisper_encoder is not None and is_train:
            with torch.no_grad():
                semantic_emb_gt, _ = self.whisper_encoder(**audio_feats)
        else:
            semantic_emb_gt = None

        if semantic_emb is not None and semantic_emb_gt is not None:
            min_len = min(semantic_emb.size(1), semantic_emb_gt.size(1))
            semantic_emb = semantic_emb[:, :min_len, :]
            semantic_emb_gt = semantic_emb_gt[:, :min_len, :]
        min_len = min(y.size(-1), y_.size(-1))
        y = y[:, :, :min_len]
        y_ = y_[:, :, :min_len]

        return AudioVAEOutputWithPast(
            gt_wav=y,
            gen_wav=y_,
            stats=stats,
            kl=kl,
            latent=z,
            semantic_emb=semantic_emb,
            semantic_emb_gt=semantic_emb_gt
        )

    @torch.inference_mode()
    def get_sample_z(self, audio_feats):
        waveform = audio_feats['waveform']
        waveform_length = audio_feats['waveform_length']
        frame_num = torch.ceil(waveform_length/self.config.enc_kwargs['input_dim']).to(torch.int32)
        h, y = self.encoder(waveform)
        h = h.transpose(1, 2)  # [B, d, T]

        posterior = OobleckDiagonalGaussianDistribution(h)
        z = posterior.sample()  # [B, d/2, T]
        z = z.transpose(1, 2)
        return z, frame_num

    def get_latent(self, waveform):
        h, y = self.encoder(waveform)
        h = h.transpose(1, 2)  # [B, d, T]

        posterior = OobleckDiagonalGaussianDistribution(h)
        z = posterior.sample()  # [B, d/2, T]
        return z.transpose(1, 2)

    @torch.inference_mode()
    def get_semantic_emb(self, audio_feats):
        z, frame_num = self.get_sample_z(audio_feats)
        semantic_emb = self.decoder(z.transpose(1, 2), only_semantic_emb=True)
        return semantic_emb, frame_num

    def infer_from_latent(self, z):
        y, _ = self.decoder(z.transpose(1, 2))
        return y
