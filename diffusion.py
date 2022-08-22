import torch
import torch.nn.functional as F
from tqdm import trange


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def _extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class DiffusionUtils:
    def __init__(self, betas):
        self._betas = betas
        self._alphas = 1 - betas
        self._alpha_cumprod = torch.cumprod(self._alphas, dim=0)
        self._alphas_cumprod_prev = F.pad(self._alpha_cumprod[:-1], (1, 0), value=1.0)
        self._sqrt_recip_alphas = torch.sqrt(1.0 / self._alphas)
        self._sqrt_alphas_cumprod = torch.sqrt(self._alpha_cumprod)
        self._sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self._alpha_cumprod)
        self._posterior_variance = betas * (1. - self._alphas_cumprod_prev) / (1. - self._alpha_cumprod)
        self._timesteps = len(betas)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alphas_cumprod_t = _extract(self._sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alphas_cumprod_t = _extract(
            self._sqrt_one_minus_alphas_cumprod, t, x0.shape
        )

        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, denoise_model, x0, t, noise=None, loss_type="l2"):
        if noise is None:
            noise = torch.randn_like(x0)

        x_noisy = self.q_sample(x0=x0, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    @torch.no_grad()
    def sample_all(self, model, batch_size, device, img_size, use_tqdm=False):
        x = torch.randn((batch_size, 3, img_size, img_size), device=device)
        yield x
        if use_tqdm:
            it = trange(len(self._betas) - 1, 0, -1)
        else:
            it = range(len(self._betas) - 1, 0, -1)
        for t in it:
            z = torch.randn_like(x, device=device) if t > 1 else torch.zeros_like(x, device=device)
            epsilon_pred = model(x, torch.ones(batch_size, device=device) * t)
            var = self._posterior_variance[t]
            mean = self._sqrt_recip_alphas[t] * (x - self._betas[t] / self._sqrt_one_minus_alphas_cumprod[t] * epsilon_pred)
            x = mean + torch.sqrt(var) * z
            yield x

    @torch.no_grad()
    def sample(self, model, batch_size, device, img_size, use_tqdm=False):
        x = list(self.sample_all(model, batch_size, device, img_size, use_tqdm=use_tqdm))
        return x[-1]

    @property
    def timesteps(self):
        return self._timesteps

    @property
    def betas(self):
        return self._betas

    @property
    def alphas(self):
        return self._alphas

    @property
    def alphas_cumprod(self):
        return self._alpha_cumprod
