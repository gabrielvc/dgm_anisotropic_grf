import matplotlib.pyplot as plt
import h5py
from numpy import random
import numpy as np
import click

def build_image_plot(img, ax=None, fig=None, vmin=-3, vmax=3):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(img, vmin=vmin, vmax=vmax)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    fig.patch.set_visible(False)
    fig.tight_layout()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    return fig, ax


@click.command()
@click.option(
    "--observation_path",
    help="paht to the observation data",
    metavar="STR",
    type=str,
    required=True,
)
@click.option(
    "--observation_idx",
    help="id of observation in observation file",
    metavar="INT",
    type=int,
    required=True,
)
@click.option(
    "--save_path",
    help="Where to save your file",
    metavar="STR",
    type=str,
    required=True,
)
@click.option(
    "--samples_path",
    help="Path to the samples from the posterior samping",
    metavar="STR",
    type=str,
    default="",
    required=True,
)
def cmdline(observation_path, observation_idx, save_path, samples_path):
    idx = observation_idx
    with h5py.File(observation_path, "r") as f:
        mask = f["mask"][idx][None]
        x_orig = f["data"][idx][None]

    obs_mean = (x_orig[mask == 1]).mean()
    obs_std = (x_orig[mask == 1]).std()
    x_orig = x_orig - obs_mean
    x_orig = x_orig / obs_std

    obs = x_orig[mask == 1]

    fig, ax = build_image_plot(x_orig[0])
    fig.savefig(f'{save_path}/true.png')
    fig, ax = build_image_plot((x_orig * mask)[0] - 100*(1 - mask[0]))
    fig.savefig(f'{save_path}/observation.png')

    samples_path = "vae_cloud_samples_0.0001_1000_20_100_v2.h5"
    with h5py.File(samples_path, "r") as f:
        samples = f["data_samples"][idx, :78]
        
    fig, ax = build_image_plot(samples.std(axis=0)[0], vmin=0, vmax=0.3)
    fig.savefig(f'{save_path}/std.png')

    fig, ax = build_image_plot(samples.mean(axis=0)[0])
    fig.savefig(f'{save_path}/mean.png')

    for i in random.randint(0, len(samples), size=10):
        fig, ax = build_image_plot(samples[i, 0])
        fig.savefig(f'{save_path}/sample_{i}.png')


if __name__ == "__main__":
    cmdline()
