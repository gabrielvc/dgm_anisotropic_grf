import yaml
from IPython.display import Audio, display

import torch
from torch import nn
from torch import Tensor
from audio_diffusion_pytorch import KarrasSchedule

from sound_model.module_base import Model
from local_paths import REPO_PATH
from ddrm.functions.svd_replacement import H_functions


class SoundModelWrapper(nn.Module):
    """"""

    def __init__(
        self,
        denoiser_fn: nn.Module,
        sigmas: Tensor,
        n_instruments: int = 4,
        len_chunk: int = 2**18,
        sample_rate: int = 22050,
    ):
        super().__init__()

        self.denoiser = denoiser_fn

        self.len_chunk = len_chunk
        self.n_instruments = n_instruments
        self.sample_rate = sample_rate

        # deduce alphas_cumprod
        sigmas = sigmas.flip(0)
        alphas_cumprod = 1 / (1 + sigmas**2)

        self.sigmas = sigmas
        self.alphas_cumprod = alphas_cumprod

    def forward(self, x: Tensor, t: int) -> Tensor:
        acp_t = self.alphas_cumprod[t]
        scaled_x_t = x / acp_t**0.5
        sigma_t = self.sigmas[t]

        score_t = (self.denoiser(scaled_x_t, sigma=sigma_t) - scaled_x_t) / sigma_t**2
        return -sigma_t * score_t


def load_sound_model(device: str = "cpu"):
    """"""

    # load configs
    with open(REPO_PATH / "configs" / "sound_model.yaml") as f:
        model_config = yaml.safe_load(f)

    # load model
    model: Model = torch.load(
        model_config["model_path"], map_location=device, weights_only=False
    )

    model.eval()
    model.requires_grad_(False)
    denoise_fn = model.model.diffusion.denoise_fn

    n_diffusion_steps = model_config["n_diffusion_steps"]
    sigma_min = model_config["sigma_min"]
    sigma_max = model_config["sigma_max"]
    rho = model_config["rho"]

    n_instruments = model_config["n_instruments"]
    len_chunk = model_config["len_chunk"]
    sample_rate = model_config["sample_rate"]

    sigmas = KarrasSchedule(sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)(
        n_diffusion_steps, device
    )
    wrapped_model = SoundModelWrapper(
        denoise_fn, sigmas, n_instruments, len_chunk, sample_rate
    )

    return wrapped_model, wrapped_model.alphas_cumprod


def display_sound(track, sample_rate=22050, normalize=True):
    """"""
    track = track.detach().cpu()
    display(Audio(track.detach().cpu(), rate=sample_rate, normalize=normalize))
