# Author: wanren
# Email: wanren.pj@antgroup.com
# Date: 2025/7/24
from __future__ import annotations

from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
# from torchdiffeq import odeint


def get_epss_timesteps(n, device, dtype):
    dt = 1 / 32
    predefined_timesteps = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    t = predefined_timesteps.get(n, [])
    if not t:
        return torch.linspace(0, 1, n + 1, device=device, dtype=dtype)
    return dt * torch.tensor(t, device=device, dtype=dtype)


class CFM(nn.Module):
    def __init__(self, model,
                 steps=32, cfg_strength=2.0,
                 sway_sampling_coef=-1, use_epss=True,
                 odeint_kwargs=None,
                 sigma=-1, temperature=1.5):
        super().__init__()
        # transformer
        self.model = model

        self.steps, self.cfg_strength = steps, cfg_strength
        self.sway_sampling_coef, self.use_epss = sway_sampling_coef, use_epss
        # sampling related
        self.odeint_kwargs = {
            # 'atol': 1e-5,
            # 'rtol': 1e-5,
            'method': "euler"  # 'midpoint'
        } if odeint_kwargs is None else odeint_kwargs

        self.sigma = sigma  # 0.25
        self.temperature = temperature

    @torch.no_grad()
    def sample(self, llm_cond, lat_cond, y0, t, spk_emb=None, gen_len=None):
        # self.eval()

        def fn(fn_t, x):
            # predict flow (cond)
            if self.cfg_strength < 1e-5:
                pred = self.model(x, fn_t, llm_cond, lat_cond, spk_emb)
                return pred

            # predict flow (cond and uncond), for classifier-free guidance
            pred_cfg = self.model.forward_with_cfg(x, fn_t, llm_cond, lat_cond, spk_emb)
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * self.cfg_strength

        # noise input
        # if gen_len is None:
        #     y0 = torch.randn_like(lat_cond)
        # else:
        #     y0 = torch.randn((lat_cond.shape[0], gen_len, lat_cond.shape[2]),
        #                      dtype=lat_cond.dtype, device=lat_cond.device)

        t_start = 0

        # if t_start == 0 and self.use_epss:  # use Empirically Pruned Step Sampling for low NFE
        #     # t = get_epss_timesteps(self.steps, device=y0.device, dtype=y0.dtype)
        # else:
        #     t = torch.linspace(t_start, 1, self.steps + 1, device=y0.device, dtype=y0.dtype)
        if self.sway_sampling_coef is not None:
            t = t + self.sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        # trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        trajectory = [y0]
        for t0, t1 in zip(t[:-1], t[1:]):
            dt = t1 - t0
            y0 = y0 + fn(t0, y0) * dt
            # if self.sigma > 0:
            #     y0 = y0 + self.sigma * (self.temperature ** 0.5) * (abs(dt) ** 0.5) * torch.randn_like(y0)
            trajectory.append(y0)

        sampled = trajectory[-1]
        out = sampled

        return out

    def forward(self, llm_cond, lat_cond, lat_tag, loss_weight, spk_emb=None, bat_size=None):
        # mel is x1
        x1 = lat_tag.detach()

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = torch.rand((llm_cond.shape[0],), dtype=llm_cond.dtype, device=llm_cond.device)

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        xt = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # forward
        pred = self.model(xt, time, llm_cond, lat_cond, spk_emb)[:, -lat_tag.shape[1]:]

        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss * loss_weight

        return loss.sum()
