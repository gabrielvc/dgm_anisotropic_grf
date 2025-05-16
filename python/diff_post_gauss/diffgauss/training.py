from typing import List, Dict, Any
import torch

from diffgauss.weight_average_callback import KarrasAveragingFunction
from typing import Tuple

torch.set_float32_matmul_precision("medium")

from diffgauss.edm2.training.networks_edm2 import Precond
from diffgauss.samplers import EDM, DDIM, create_karras_sigmas
import lightning as L
import math
from functools import partial


class KarrasLRScheduler:
    """LR Scheduler unction from https://arxiv.org/pdf/2312.02696."""

    def __init__(self, ref_n_samples: int, rampup_n_samples: int, batch_size: int) -> None:
        self.ref_batches = ref_n_samples / batch_size
        self.rampup_batches = rampup_n_samples / batch_size

    @torch.no_grad()
    def __call__(self, batch_idx: int) -> float:
        coeff = 1
        if self.ref_batches > 0:
            coeff /= math.sqrt(max((batch_idx + 1) / self.ref_batches, 1))
        if self.rampup_batches > 0:
            coeff *= min(((batch_idx + 1) / self.rampup_batches), 1)
        return coeff


class AbstractDiffusion(L.LightningModule):
    def __init__(
        self,
        optim_config: Dict[str, Any],
        denoiser_config: Dict[str, Any],
        diffusion_config: Dict[str, Any],
        batch_size: int,
        validation_sigmas: List[float] = [0.1, 0.3, 0.5, 1, 10],
        **kwargs,
    ):
        super().__init__()
        self.optim_config = optim_config
        self.diffusion_config = diffusion_config
        self.validation_sigmas = validation_sigmas
        self.denoiser_config = denoiser_config
        self.denoiser = Precond(**denoiser_config)
        self.automatic_optimization = True
        self.batch_size = batch_size

    def sample(self, noise, mode="DDIM", num_steps: int = 25, **kwargs):
        if mode == "EDM":
            sampler = EDM(tqdm_disable=True)
        elif mode == "DDIM":
            sampler = DDIM(tqdm_disable=True, **kwargs)
        else:
            raise NotImplementedError("Available samplers are EDM and DDIM")
        stds = create_karras_sigmas(
            N=num_steps,
            sigma_max=self.diffusion_config["sigma_max"],
            sigma_min=self.diffusion_config["sigma_min"],
        ).to(self.device)
        return sampler.sample(
            stds=stds,
            initial_samples=noise * self.diffusion_config["sigma_max"],
            denoiser_fn=self.denoiser,
        )

    def training_step(self, batch, batch_idx):
        target = batch["data_sample"]
        sigma = batch["noise_level"]
        prediction, logvar = self.denoiser(
            batch["noisy_sample"], sigma, return_logvar=True
        )
        cout = (
            self.denoiser.sigma_data
            * sigma
            / (self.denoiser.sigma_data**2 + sigma**2) ** 0.5
        )
        weights = (1 / (cout**2)).reshape(logvar.shape) * torch.exp(-logvar)
        mse_loss = (
            ((prediction - target)**2)
            * weights
            + logvar
        )
        self.log("train/mse", mse_loss.mean(), prog_bar=True)
        return (mse_loss / self.batch_size).sum()

    def validation_step(self, batch, batch_idx):
        B, C, L, H = batch["data_sample"].shape
        noise = torch.randn(
            size=(len(self.validation_sigmas), B, C, L, H), device=self.device
        )
        corrupt_data = batch["data_sample"][None] + torch.stack(
            [n * s for n, s in zip(noise, self.validation_sigmas)]
        )
        predicted_clean = self.denoiser(
            corrupt_data.reshape(B * len(self.validation_sigmas), C, L, H),
            torch.stack(
                [
                    s * torch.ones((B,), device=self.device)
                    for s in self.validation_sigmas
                ]
            ).reshape(B * len(self.validation_sigmas)),
            return_logvar=False,
        ).reshape(len(self.validation_sigmas), B, C, L, H)

        batch_errors = (
            torch.nn.functional.mse_loss(
                predicted_clean,
                batch["data_sample"].repeat((len(self.validation_sigmas), 1, 1, 1, 1)),
                reduction="none",
            )
            .mean(dim=(-3, -2, -1))
        )

        for s, e in zip(self.validation_sigmas, batch_errors):
            self.log(
                f"val/mse/{s}",
                e.mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here

        norms = {
            **{
                f"encoder/grad/{k}": v.item()
                for k, v in L.pytorch.utilities.grad_norm(
                    self.denoiser, norm_type=2
                ).items()
            },
        }
        self.log_dict(norms)

    def configure_optimizers(self):
        if self.optim_config["optimizer"]["type"] == "Adam":
            optimizer = torch.optim.Adam(
                params=self.denoiser.parameters(),
                lr=self.optim_config["optimizer"]["base_learning_rate"],
                betas=self.optim_config["optimizer"]["betas"]
            )
        else:
            raise NotImplementedError("Only Adam implemented")
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=KarrasLRScheduler(
                ref_n_samples=self.optim_config["scheduler"]["ref_n_samples"],
                rampup_n_samples=self.optim_config["scheduler"]["rampup_n_samples"],
                batch_size=self.batch_size
            ),
        )
        return [
            optimizer,
        ], [
            {"scheduler": lr_scheduler, "interval": "step", "frequency": 1},
        ]


class EDM2Diffusion(AbstractDiffusion):
    def __init__(
        self,
        gammas: List[float],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ema_models = torch.nn.ModuleDict(
            {
                f"ema_{str(g).replace('.','-')}": torch.optim.swa_utils.AveragedModel(
                    self.denoiser, avg_fn=KarrasAveragingFunction(gamma=g)
                )
                for g in gammas
            }
        )

    def validation_step(self, batch, batch_idx):
        # This calculates everything for the current model
        B, C, L, H = batch["data_sample"].shape
        noise = torch.randn(
            size=(len(self.validation_sigmas), B, C, L, H), device=self.device
        )
        corrupt_data = batch["data_sample"][None] + torch.stack(
            [n * s for n, s in zip(noise, self.validation_sigmas)]
        )
        for name, dn in {"current": self.denoiser, **self.ema_models}.items():
            predicted_clean = dn(
                corrupt_data.reshape(B * len(self.validation_sigmas), C, L, H),
                torch.stack(
                    [
                        s * torch.ones((B,), device=self.device)
                        for s in self.validation_sigmas
                    ]
                ).reshape(B * len(self.validation_sigmas)),
                return_logvar=False,
            ).reshape(len(self.validation_sigmas), B, C, L, H)

            batch_errors = (
                torch.nn.functional.mse_loss(
                    predicted_clean,
                    batch["data_sample"].repeat(
                        (len(self.validation_sigmas), 1, 1, 1, 1)
                    ),
                    reduction="none",
                )
                .mean(dim=(-3, -2, -1))
            )

            for s, e in zip(self.validation_sigmas, batch_errors):
                self.log(
                    f"val/mse/{s}/{name}",
                    e.mean(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)
        for nm in self.ema_models.keys():
            self.ema_models[nm].update_parameters(self.denoiser)

    def configure_predict_step(self, mode: str, **kwargs):
        if mode == "generate":
            self.predict_fn = self.make_generate_fn(**kwargs)
        elif mode == "denoise": 
            self.predict_fn = self.make_denoise_fn(**kwargs)
        else:
            raise NotImplementedError("Not implemented")

    def make_generate_fn(
        self,
        sampler_type: str,
        num_steps: int,
        clf_free_coeff: float=1,
        denoiser_type: str = "ema_6-94",
        rho: float = 5,
        **kwargs):
        if sampler_type == "EDM":
            sampler = EDM(tqdm_disable=True)
        elif sampler_type == "DDIM":
            sampler = DDIM(tqdm_disable=True, **kwargs)
        else:
            raise NotImplementedError("Available samplers are EDM and DDIM")

        stds = create_karras_sigmas(
        N=num_steps,
        sigma_max=self.diffusion_config["sigma_max"],
        sigma_min=self.diffusion_config["sigma_min"],
        rho=rho,
        )
        if denoiser_type != "current":
            base_denoiser = self.ema_models[denoiser_type].eval()
            base_denoiser.requires_grad_(False)
        else:
            base_denoiser = self.denoiser.eval()
            base_denoiser.requires_grad_(False)

        def sampler_fn(batch):
            initial_noise = batch["initial_noise"]
            return sampler.sample(
                stds=stds.to(initial_noise.device),
                initial_samples=initial_noise,
                denoiser_fn=base_denoiser
            )
        return sampler_fn


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.predict_fn(batch)