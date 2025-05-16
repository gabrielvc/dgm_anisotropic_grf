from click import Tuple
import torch
import lightning as L
from typing import Dict, Any, Tuple
import itertools


def make_image_independent_gaussian(mean, logvar, img_dims=(1, 2, 3)):
    dist = torch.distributions.Normal(
        loc=mean, scale=(0.5 * logvar).exp(), validate_args=False
    )
    dist = torch.distributions.Independent(dist, 3, validate_args=False)
    return dist

class AbstractPrior(object):

    def rsample(shape):
        raise NotImplementedError("You should pass a Prior")


class AbstractVAE(L.LightningModule):
    def __init__(
        self,
        encoder_class: torch.nn.Module,
        decoder_class: torch.nn.Module,
        encoder_params: Dict[str, Any],
        decoder_params: Dict[str, Any],
        optim_config,
        kl_start: int = 1_000,
        kl_max: float = 1e-4,
        kl_coeff: float = 1 / 10_000,
        n_images_to_log: int = 4,
        **kwargs
    ):
        super().__init__()
        self.prior = AbstractPrior()
        self.optim_config = optim_config
        self.encoder = encoder_class(**encoder_params)
        self.decoder = decoder_class(**decoder_params)

        self.kl_start = kl_start
        self.kl_max = kl_max
        self.kl_coeff = kl_coeff
        self.n_images_to_log = n_images_to_log


    def encode(self, x: torch.Tensor) -> torch.distributions.Distribution:
        raise NotImplementedError("Encode not implemented")

    def decode(self, latents: torch.Tensor) -> torch.distributions.Distribution:
        raise NotImplementedError("Decoder not implemented")

    def prior_kl(self, latent_distribution: torch.distributions.Distribution) -> torch.Tensor:
        raise NotImplementedError("Prior kl not implemented")
    
    def r_sample(self, sample_shape: Tuple[int]) -> torch.Tensor:
        raise NotImplementedError("Sampling from model not implemented")

    def get_kl_coeff(self):
        if self.global_step < self.kl_start:
            return 0
        return min((self.global_step - self.kl_start) * self.kl_coeff, self.kl_max)

    def elbo(self, images):
        latent_distribution = self.encode(images)
        latent_samples = latent_distribution.rsample((1,))[0]
        data_distribution = self.decode(latent_samples)
        likelihood_observation = data_distribution.log_prob(images).mean()
        kl_prior = self.prior_kl(latent_distribution=latent_distribution).mean()
        kl_coeff = self.get_kl_coeff()
        elbo = likelihood_observation - kl_coeff * kl_prior
        
        return elbo, likelihood_observation, kl_prior, kl_coeff, latent_distribution, data_distribution


    def training_step(self, batch, batch_idx):
        images = batch["data_sample"].float()
        elbo, lk, kl, kl_coeff = self.elbo(images)[:4]

        self.log("train/lk", lk, prog_bar=True)
        self.log("train/kl", kl, prog_bar=True)
        self.log("train/kl_coeff", kl_coeff, prog_bar=True)
        return -elbo

    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.vae_sch_loss = 0
        self.discriminator_sch_loss = 0
        return super().on_validation_batch_start(batch, batch_idx, dataloader_idx)

    def validation_step(self, batch, batch_idx):
        metrics = {}
        images = batch["data_sample"].float()
        elbo, lk, kl, kl_coeff, latent_distribution, data_distribution = self.elbo(images)
        fake_images = data_distribution.rsample((1,))[0]

        metrics["val/elbo"] = elbo
        metrics["val/lk"] = lk
        metrics["val/kl"] = kl
        metrics["val/kl_coeff"] = kl_coeff
        metrics["val/uRMSE"] = torch.linalg.vector_norm(0.5 * (fake_images - images)).mean()/ (images[0].numel() ** 0.5)


        for k, v in metrics.items():
            self.log(
                k, v,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True
            )
        if batch_idx == 0:
            self.reconstructions_per_class = {i: fake_images[i] for i in range(self.n_images_to_log)}
            self.original_per_class = {i: images[i] for i in range(self.n_images_to_log)}


    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here

        norms = {
            **{
                f"encoder/grad/{k}": v.item()
                for k, v in L.pytorch.utilities.grad_norm(
                    self.encoder, norm_type=2
                ).items()
            },
            **{
                f"decoder/grad/{k}": v.item()
                for k, v in L.pytorch.utilities.grad_norm(
                    self.decoder, norm_type=2
                ).items()
            },
        }

        self.log_dict(norms)

    def on_validation_epoch_end(self):
        for cl in self.original_per_class:
            self.logger.experiment.add_image(
                f"{cl}/rec_image",
                self.reconstructions_per_class[cl],
                self.current_epoch,
            )
            self.logger.experiment.add_image(
                f"{cl}/or_image", self.original_per_class[cl], self.current_epoch
            )

        self.original_per_class.clear()
        self.reconstructions_per_class.clear()
        z = self.prior.sample((8,))[:, 0]
        prior_gen_images = self.decode(z).sample((1,))[0]
        for i, img in enumerate(prior_gen_images):
            img = 0.5 * (img + 1)
            self.logger.experiment.add_image(f"gen_image/{i}", img, self.current_epoch)

    def configure_optimizers(self):
        if self.optim_config["optimizer"]["type"] == "Adam":
            optimizer = torch.optim.Adam(
                itertools.chain(self.encoder.parameters(),self.decoder.parameters()), self.optim_config["optimizer"]["base_learning_rate"]
            )
        else:
            raise NotImplementedError("Only Adam implemented")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=self.optim_config["lr_schedule"]["factor"],
                    mode=self.optim_config["lr_schedule"]["metric_mode"],
                ),
                "monitor": self.optim_config["lr_schedule"]["metric_to_track"],
                "frequency": self.optim_config["lr_schedule"]["frequency"],
            },
        }
    

class GaussGaussVAE(AbstractVAE):
    def __init__(
        self,
        prior_mean,
        prior_logvar,
        scale_max,
        variance_type: str,
        **kwargs
    ):

        super().__init__(**kwargs)
        self.dec_scale_max = scale_max
        self.prior = None
        self.prior_mean = prior_mean
        self.prior_logvar = prior_logvar
        self.variance_type = variance_type
        if self.variance_type == "learned":
            self.variance_param = torch.nn.Parameter(
                self.dec_scale_max * torch.ones([], device=self.device)
            )
        elif self.variance_type == "fixed":
            self.variance_param = self.dec_scale_max * torch.ones(
                [], device=self.device
            )
        else:
            self.variance_param = None

        self.save_hyperparameters(
            ignore=["encoder_class", "decoder_class", "prior_mean", "prior_logvar"]
        )

    def encode(self, x):
        mean, logvar = torch.chunk(self.encoder(x), 2, dim=1)
        return make_image_independent_gaussian(mean=mean, logvar=logvar)

    def decode(self, latents):
        if self.variance_type != "diag":
            mean = self.decoder(latents)
            scale_abs = torch.nn.functional.sigmoid(self.variance_param) * (
                self.dec_scale_max
            )
            scale = scale_abs * torch.ones_like(mean)
            self.log("train/scale", scale_abs.detach().item())
        else:
            mean, logvar = torch.chunk(self.decoder(latents), 2, dim=1)
            scale = torch.nn.functional.sigmoid((0.5 * logvar)) * self.dec_scale_max
        dist = torch.distributions.Normal(loc=mean, scale=scale, validate_args=False)
        dist = torch.distributions.Independent(dist, 3, validate_args=False)
        return dist

    def prior_kl(self, latent_distribution):
        if self.prior is None:
            self.prior = make_image_independent_gaussian(
                self.prior_mean.to(self.device), self.prior_logvar.to(self.device)
            )
        return torch.distributions.kl.kl_divergence(p=self.prior, q=latent_distribution)

    def rsample(self, sample_shape: Tuple[int]) -> torch.Tensor:
        latent_samples = self.prior.rsample(sample_shape).squeeze(len(sample_shape))
        data_dist = self.decode(latent_samples)
        return data_dist.rsample((1,))[0]   