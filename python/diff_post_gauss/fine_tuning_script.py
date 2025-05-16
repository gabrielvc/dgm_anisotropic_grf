import torch
torch.set_float32_matmul_precision('medium')
from datasources.gauss_spde import GaussVEDataModule

import datetime
import lightning as L
import click
from hydra import initialize, compose
from diffgauss.utils import load_diffusion_net


def load_datasource(datasource_cfg, diffusion_cfg):
    return GaussVEDataModule(
    **datasource_cfg, diffusion_cfg=diffusion_cfg, is_latent=datasource_cfg["name"] == "LatentGauss"
    )
        

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
    help="Checkpoit for resume training",
    metavar="STR",
    type=str,
    default="",
    required=True
)
def cmdline(preset: str, config_path: str, ckpt_path, **opts):
    now = "{date:%Y-%m-%d_%H:%M:%S}".format(date=datetime.datetime.now())
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=preset)
        ds = load_datasource(cfg["datasource"], diffusion_cfg=cfg["diffusion"])

        logger = L.pytorch.loggers.TensorBoardLogger(
            save_dir=cfg["save_logs"],
            name=cfg["datasource"]["name"],
            version=f"{preset.replace('.yaml', '')}_{now}",
        )
        checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
            **cfg["checkpoint"]
        )

        lr_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval="step")
        
        device_monitor = L.pytorch.callbacks.DeviceStatsMonitor()
        callbacks=[
            checkpoint_callback,
            lr_monitor,
            device_monitor
            ]
        
        trainer = L.Trainer(
            **cfg["trainer"],
            callbacks=callbacks,
            logger=logger,
        )
        
        
        effective_batch_size = cfg["datasource"]["batch_size"]*torch.cuda.device_count()*cfg["trainer"]["accumulate_grad_batches"]
        print(effective_batch_size)
        with trainer.init_module():
            img_shape, diff = load_diffusion_net(
            cfg, cfg["starting_model_checkpoint"], sampler_config=None, is_latent=False
        )
    trainer.fit(
        model=diff,
        datamodule=ds,
        ckpt_path=ckpt_path if ckpt_path != "" else None
    )

if __name__ == "__main__":
    cmdline()
