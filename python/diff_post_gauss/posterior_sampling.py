import torch
from pathlib import Path

torch.set_float32_matmul_precision("medium")

import click
from hydra import initialize, compose
import h5py

from diffgauss.utils import load_diffusion_net
import os

from datasources.gauss_spde import AmbientGaussH5Dataset
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
import matplotlib.pyplot as plt

from posterior_samplers.diffusion_utils import ddim
from posterior_samplers.mgdm import mgdm
from posterior_samplers.dps import dps
from posterior_samplers.ddnm import ddnm_plus
from posterior_samplers.diffpir import diffpir
from posterior_samplers.resample.algo import resample
from posterior_samplers.pgdm import pgdm
from posterior_samplers.reddiff import reddiff
from posterior_samplers.psld import psld
from posterior_samplers.daps import daps
from posterior_samplers.pnp_dm.algo import pnp_dm
from posterior_samplers.mgps import mgps
from ddrm.functions.svd_replacement import Inpainting, SuperResolution


from utils.experiments_tools import get_gpu_memory_consumption
from utils.metrics import LPIPS, PSNR, SSIM
from utils.im_invp_utils import generate_invp, Hsimple
from posterior_samplers.diffusion_utils import load_epsilon_net, EpsilonNetSVD
from utils.im_invp_utils import InverseProblem
from tqdm import tqdm


class VPEpsilonFromDenoiser(torch.nn.Module):

    def __init__(self, base_denoiser, alphas_cumprod, timesteps):
        super().__init__()
        self.base_denoiser = base_denoiser
        self.alphas_cumprod = alphas_cumprod
        self.acp_f8 = alphas_cumprod.to(torch.float64)
        self.timesteps = timesteps
        self.net = None

    def predict_x0(self, x, t):
        if isinstance(t, int):
            t = torch.tensor([t], device=x.device)
        elif torch.is_floating_point(t):
            t = t.round().long()

        alpha_t = self.acp_f8[t]
        eq_sigma = ((1 - alpha_t) / (alpha_t)) ** 0.5
        ve_x = x / (alpha_t**0.5)
        denoised_x_ve = self.base_denoiser(ve_x, eq_sigma)
        return denoised_x_ve.to(x.dtype)

    def forward(self, x, t):
        if isinstance(t, int):
            t = torch.tensor([t], device=x.device)
        elif torch.is_floating_point(t):
            t = t.round().long()

        alpha_t = self.acp_f8[t]
        eq_sigma = ((1 - alpha_t) / (alpha_t)) ** 0.5
        ve_x = x / (alpha_t**0.5)
        denoised_x_ve = self.base_denoiser(ve_x, eq_sigma)
        return ((ve_x - denoised_x_ve) / eq_sigma).to(x.dtype)


def make_inverse_problem(
    inverse_problem_cfg, img_shape
):
    if inverse_problem_cfg.task == "inpainting_iso":
        with h5py.File(inverse_problem_cfg.mask_path, "r") as f:
            if inverse_problem_cfg.mask_type == "unif":
                mask = torch.from_numpy(f["unif"][inverse_problem_cfg.sample_index, inverse_problem_cfg.mask_index][None, None])
                noise = torch.from_numpy(f["noise_unif"][inverse_problem_cfg.sample_index, inverse_problem_cfg.mask_index][None, None])
            elif inverse_problem_cfg.mask_type == "clust":
                mask = torch.from_numpy(f["clust"][inverse_problem_cfg.sample_index, inverse_problem_cfg.mask_index]][None])
                noise = torch.from_numpy(f["noise_clust"][inverse_problem_cfg.sample_index, inverse_problem_cfg.mask_index]][None])
            else:
                raise NotImplementedError("Only unif and cluster implemented")
        with h5py.File(inverse_problem_cfg.origin_path, "r") as f:
            x_orig = torch.from_numpy(f["data"][inverse_problem_cfg.sample_index])
        H_func = Inpainting(
                missing_indices=torch.where(mask.flatten() == 0)[0],
                channels=img_shape[0],
                img_dim=img_shape[-1],
                device=torch.get_default_device(),
            )    
        obs = H_func.H(x_orig + noise)
        obs = obs.to(torch.get_default_device())

        def log_pot(x):
            diff = obs.reshape(1, -1) - H_func.H(x)
            return -0.5 * torch.norm(diff) ** 2 / inverse_problem_cfg.std**2

        inverse_problem = InverseProblem(
            obs=obs,
            H_func=H_func,
            std=inverse_problem_cfg.std,
            log_pot=log_pot,
            task=inverse_problem_cfg.task,
        )
    elif inverse_problem_cfg.task == "cloud_inpainting":

        with h5py.File(inverse_problem_cfg.data_path, "r") as f:
            mask = torch.from_numpy(f["mask"][inverse_problem_cfg.idx][None]).to(torch.get_default_device())
            x_orig = torch.from_numpy(f["data"][inverse_problem_cfg.idx][None]).to(torch.get_default_device())

        H_func = Inpainting(
            missing_indices=torch.where(mask.flatten() == 0)[0],
            channels=img_shape[0],
            img_dim=img_shape[-1],
            device=torch.get_default_device(),
        )
        obs_mean = (x_orig[mask == 1]).mean()
        obs_std = (x_orig[mask == 1]).std()
        x_orig = x_orig - obs_mean
        x_orig = x_orig / obs_std

        obs = H_func.H(x_orig.unsqueeze(0))

        obs = obs.to(torch.get_default_device())

        def log_pot(x):
            diff = obs.reshape(1, -1) - H_func.H(x)
            return -0.5 * (diff**2).sum() / inverse_problem_cfg.std**2

        inverse_problem = InverseProblem(
            obs=obs,
            H_func=H_func,
            std=inverse_problem_cfg.std,
            log_pot=log_pot,
            task=inverse_problem_cfg.task,
        )
    elif inverse_problem_cfg.task == "inpainting":
        with h5py.File(inverse_problem_cfg.data_path, "r") as f:
            x_orig = torch.from_numpy(f["data"][inverse_problem_cfg.data_idx]).float().to(torch.get_default_device())

        mask = torch.rand_like(x_orig) < inverse_problem_cfg.random_ptg

        H_func = Inpainting(
            missing_indices=torch.where(mask.flatten() == 0)[0],
            channels=img_shape[0],
            img_dim=img_shape[-1],
            device=torch.get_default_device(),
        )
        obs = H_func.H(x_orig.unsqueeze(0))
        obs = obs + torch.randn_like(obs)*inverse_problem_cfg.std
        obs = obs.to(torch.get_default_device())

        def log_pot(x):
            diff = obs.reshape(1, -1) - H_func.H(x)
            return -0.5 * torch.norm(diff) ** 2 / inverse_problem_cfg.std**2

        inverse_problem = InverseProblem(
            obs=obs,
            H_func=H_func,
            std=inverse_problem_cfg.std,
            log_pot=log_pot,
            task=inverse_problem_cfg.task,
        )
    elif inverse_problem_cfg.task == "super_resolution":
        with h5py.File(inverse_problem_cfg.data_path, "r") as f:
            x_orig = torch.from_numpy(f["data"][inverse_problem_cfg.data_idx]).float().to(torch.get_default_device())

        H_func = SuperResolution(
            ratio=inverse_problem_cfg.ratio,
            channels=img_shape[0],
            img_dim=img_shape[-1],
            device=torch.get_default_device(),
        )
        
        obs = H_func.H(x_orig)

        obs = obs + inverse_problem_cfg.std * torch.randn_like(obs)
        obs = obs.to(torch.get_default_device())

        def log_pot(x):
            diff = obs.reshape(1, -1) - H_func.H(x)
            return -0.5 * torch.norm(diff) ** 2 / inverse_problem_cfg.std**2

        inverse_problem = InverseProblem(
            obs=obs,
            H_func=H_func,
            std=inverse_problem_cfg.std,
            log_pot=log_pot,
            task=inverse_problem_cfg.task,
        )
    else:
        raise NotImplementedError()
    return H_func, inverse_problem, x_orig

@click.command()
@click.option(
    "--denoiser_preset",
    help="Configuration preset",
    metavar="STR",
    type=str,
    required=True,
)
@click.option(
    "--config_path",
    help="Path to config",
    metavar="STR",
    type=str,
    default="configs/",
    required=True,
)
@click.option(
    "--ckpt_path",
    help="path to model checkpoint",
    metavar="STR",
    type=str,
    required=True,
)
@click.option(
    "--sampler_preset",
    help="Configuration of the sampler",
    metavar="STR",
    type=str,
    required=True,
)
@click.option(
    "--save_folder",
    help="Where to save",
    metavar="STR",
    type=str,
    required=True,
)
@click.option(
    "--seed_offset",
    help="trainer offset",
    metavar="INT",
    type=int,
    default=0,
    required=True,
)
@click.option(
    "--inverse_problem_preset",
    help="inverse problem preset",
    metavar="STR",
    type=str,
    required=True,
)
def cmdline(
    denoiser_preset: str,
    config_path: str,
    ckpt_path: str,
    sampler_preset: str,
    save_folder: str,
    seed_offset: int,
    inverse_problem_preset: str,
    **opts,
):

    save_file_path = os.path.join(
        save_folder,
        denoiser_preset.replace(".yaml", ""),
        inverse_problem_preset.replace(".yaml", ""),
        sampler_preset.replace(".yaml", ""),
        "raw_data",
    )

    Path(save_file_path).mkdir(parents=True, exist_ok=True)
    with initialize(version_base=None, config_path=config_path):
        # config is relative to a module
        diffusion_cfg = compose(config_name=denoiser_preset)

    with initialize(version_base=None, config_path="configs/conditional_sampler"):
        sampler_cfg = compose(config_name=sampler_preset)

    with initialize(version_base=None, config_path="configs/inverse_problems"):
        inverse_problem_cfg = compose(config_name=inverse_problem_preset)

    device = "cuda:0"
    torch.set_default_device(device)

    # Loading diffusion stuff
    img_shape, denoiser = load_diffusion_net(
        diffusion_cfg, ckpt_path, None, is_latent=False
    )
    denoiser = denoiser.ema_models["ema_16-97"]
    denoiser.eval()
    denoiser.requires_grad_(False)

    alphas_cumprod = torch.linspace(0.9999, 0.98, 1000)
    alphas_cumprod = torch.cumprod(alphas_cumprod, 0).clip(1e-10, 1)
    alphas_cumprod = torch.concatenate([torch.tensor([1.0]), alphas_cumprod])
    t_max = torch.where((((1 - alphas_cumprod) / alphas_cumprod) ** 0.5) > 80)[0][
        0
    ].item()
    timesteps = torch.linspace(0, t_max, sampler_cfg.nsteps + 1)
    epsilon_net = VPEpsilonFromDenoiser(
        base_denoiser=denoiser,
        alphas_cumprod=alphas_cumprod.cuda(),
        timesteps=timesteps.long().cuda(),
    )

    # Loading sampler stuff
    sampler = {
        "mgdm": mgdm,
        "dps": dps,
        "pgdm": pgdm,
        "ddnm": ddnm_plus,
        "diffpir": diffpir,
        "reddiff": reddiff,
        "daps": daps,
        "pnp_dm": pnp_dm,
        "resample": resample,
        "psld": psld,
        "mgps": mgps,
    }[sampler_cfg.name]

    H_func, inverse_problem, x_orig = make_inverse_problem(
        inverse_problem_cfg, img_shape
    )
    if sampler_cfg.name == "pgdm":
        epsilon_net = EpsilonNetSVD(
            net=epsilon_net,
            alphas_cumprod=epsilon_net.alphas_cumprod,
            timesteps=epsilon_net.timesteps,
            H_func=H_func,
            shape=img_shape,
        )
    # Running
    if seed_offset == -1:
        seed_offset = torch.randint(low=0, high=2**30, size=(1,)).item()
    torch.manual_seed(seed_offset)
    initial_noise = torch.randn(sampler_cfg.nsamples, *img_shape)
    samples = sampler(
        initial_noise=initial_noise,
        inverse_problem=inverse_problem,
        epsilon_net=epsilon_net,
        **sampler_cfg.parameters,
    )
    torch.save(
        obj={"samples": samples,},
        f=os.path.join(save_file_path, f"{seed_offset}.pt"),
    )


if __name__ == "__main__":
    cmdline()
