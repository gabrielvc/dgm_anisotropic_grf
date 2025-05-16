import torch
torch.set_float32_matmul_precision("high")
import datetime
import lightning as L
import click
from hydra import initialize, compose
from latent_vae.vaes import GaussGaussVAE
from latent_vae.utils import get_autoencoder_makers_and_configs, get_datasource


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
    required=True,
)
@click.option(
    "--ckpt_path",
    help="Path to checkpoint",
    metavar="STR",
    type=str,
    default="",
    required=True,
)
def cmdline(preset: str, config_path: str, ckpt_path: str,  **opts):
    now = "{date:%Y-%m-%d_%H:%M:%S}".format(date=datetime.datetime.now())
    with initialize(version_base=None, config_path=config_path):
        # config is relative to a module
        cfg = compose(config_name=preset)
        ds, img_shape = get_datasource(cfg["datasource"])

        encoder_class, decoder_class, encoder_cfg, decoder_cfg = (
            get_autoencoder_makers_and_configs(
                cfg["autoencoder"],
                variance_type=cfg["vae"]["variance_type"],
                img_shape=img_shape,
            )
        )
        logger = L.pytorch.loggers.TensorBoardLogger(
            save_dir=cfg["save_logs"],
            name=cfg["name"],
            version=f"{preset.replace('.yaml', '')}_{now}",
        )
        checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
          **cfg["checkpoint"]
        )
        lr_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval="step")
        trainer = L.Trainer(
            **cfg["trainer"],
            accumulate_grad_batches=cfg["optimizers"]["accumulate_grad"],
            callbacks=[checkpoint_callback, lr_monitor],
            logger=logger,
        )
        latent_shape_res = encoder_cfg["img_resolution"] // (
            2 ** (len(encoder_cfg["channel_mult"]) - 1)
        )
        latent_shape = (
            cfg["autoencoder"]["latent_channels"],
            latent_shape_res,
            latent_shape_res,
        )
        with trainer.init_module():
            vae = GaussGaussVAE(
                encoder_class=encoder_class,
                decoder_class=decoder_class,
                encoder_params=encoder_cfg,
                decoder_params=decoder_cfg,
                prior_mean=torch.zeros((1,) + latent_shape),
                prior_logvar=torch.zeros((1,) + latent_shape),
                optim_config=cfg["optimizers"],
                **cfg["vae"],
            )
        trainer.fit(
            model=vae,
            datamodule=ds,
            ckpt_path=None if ckpt_path == "" else ckpt_path
        )


if __name__ == "__main__":
    cmdline()
