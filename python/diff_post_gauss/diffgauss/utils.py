import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
import yaml
from diffgauss.training import EDM2Diffusion 
import torch

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_pil_image(x):
    x = (x - x.min()) / (x.max() - x.min()) * 255
    sample = x.cpu().permute(0, 2, 3, 1)
    if sample.shape[-1] == 1:
        sample = sample[..., 0]
    sample = sample.numpy().astype(np.uint8)
    if sample.shape[0] == 1:
        img_pil = PIL.Image.fromarray(sample[0])
        return img_pil
    else:
        return [PIL.Image.fromarray(s) for s in sample]


def display(x, save_path=None, title=None):
    img_pil = get_pil_image(x)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img_pil)
    if title:
        ax.set_title(title)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.show()
    if save_path is not None:
        fig.savefig(save_path + ".png")


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = PIL.Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def load_diffusion_net(diffusion_cfg, ckpt_path, sampler_config, is_latent: bool = False):

    if not is_latent:
        img_shape=(1, 256, 256)

        effective_batch_size = diffusion_cfg["datasource"]["batch_size"]*torch.cuda.device_count()*diffusion_cfg["trainer"]["accumulate_grad_batches"]
        denoiser = EDM2Diffusion.load_from_checkpoint(
            ckpt_path,
            diffusion_config=diffusion_cfg["diffusion"],
            optim_config=diffusion_cfg["optimizer"],
            denoiser_config=diffusion_cfg["denoiser"],
            gammas=diffusion_cfg["ema"]["gammas"],
            batch_size=effective_batch_size,
        )
    if sampler_config is not None:
        denoiser.configure_predict_step(mode="generate", **sampler_config)
    
    return img_shape, denoiser