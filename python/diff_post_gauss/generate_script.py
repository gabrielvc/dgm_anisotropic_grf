import torch
import os
import torch

#torch.set_float32_matmul_precision("medium")
import os
import lightning as L
import click
from hydra import initialize, compose
from pathlib import Path
from diffgauss.utils import load_diffusion_net
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.callbacks import BasePredictionWriter


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval, trainer_offset: int = 0):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.trainer_offset = trainer_offset

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ) -> None:
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(
            {"images": prediction.cpu(), **{k: v.cpu() for k, v in batch.items()}},
            os.path.join(
                self.output_dir, f"predictions_{self.trainer_offset + trainer.global_rank}_{batch_idx}.pt"
            ),
        )


class NoiseDataset(Dataset):

    def __init__(self, std_max, length, img_shape) -> None:
        super().__init__()
        self.length = length
        self.img_shape = img_shape
        self.std_max = std_max

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {
            "initial_noise": torch.randn(self.img_shape) * self.std_max,
        }


@click.command()
@click.option(
    "--preset",
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
    "--batch_size",
    help="Batch size",
    metavar="INT",
    type=int,
    default=64,
    required=True,
)
@click.option(
    "--n_samples",
    help="Number of samples to generate",
    metavar="INT",
    type=int,
    default=50_000,
    required=True,
)
@click.option(
    "--config_sampler",
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
    "--trainer_offset",
    help="trainer offset",
    metavar="INT",
    type=int,
    default=0,
    required=True,
)
def cmdline(
    preset: str,
    config_path: str,
    ckpt_path: str,
    batch_size: int,
    n_samples: int,
    config_sampler: str,
    save_folder: str,
    trainer_offset: int,
    **opts,
):
    
    save_file_path = os.path.join(
        save_folder,
        preset.replace(".yaml", ""),
        f"{config_sampler.replace(".yaml", "")}",
        "raw_data",
    )

    Path(os.path.join(*save_file_path.split("/"))).mkdir(parents=True, exist_ok=True)
    with initialize(version_base=None, config_path=config_path):
        # config is relative to a module
        diffusion_cfg = compose(config_name=preset)

    with initialize(version_base=None, config_path="configs/unconditional_sampler"):
        sampler_cfg = compose(config_name=config_sampler)

    pred_writer = CustomWriter(output_dir=save_file_path, write_interval="batch", trainer_offset=trainer_offset)
    trainer = L.Trainer(
        accelerator="gpu", strategy="ddp", callbacks=[pred_writer]
    )
    #with trainer.init_module():
    img_shape, denoiser = load_diffusion_net(
        diffusion_cfg, ckpt_path, sampler_cfg, is_latent=False
    )
    dataloader = DataLoader(
        dataset=NoiseDataset(
            std_max=(diffusion_cfg.diffusion["sigma_max"]**2 + 1)**.5,
            length=n_samples,
            img_shape=img_shape,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=3,
    )
    trainer.predict(denoiser, dataloader, return_predictions=False)


if __name__ == "__main__":
    cmdline()
