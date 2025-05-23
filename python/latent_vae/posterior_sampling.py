# %%
import torch
import matplotlib.pyplot as plt
from pyro.infer.mcmc import nuts
from pyro.infer.mcmc import MCMC
from latent_vae.utils import get_autoencoder_makers_and_configs
from hydra import initialize, compose
from latent_vae.vaes import GaussGaussVAE
from tqdm import tqdm
import h5py


def ula(state, n_steps, lr, score, eta=0.01):
    for i in tqdm(range(n_steps)):
        state_score = score(state)
        update = state_score / (torch.abs(state_score) * eta + 1)
        state = state + lr * update + (2 * lr) ** 0.5 * torch.randn_like(state)
    return state


def make_all_inverse_problems_potentials(
    ckpt_path,
    preset,
    data_path,
    img_shape=(1, 256, 256),
    idx=0,
    std=0.05,
):

    with initialize(version_base=None, config_path="configs/gauss"):
        # config is relative to a module
        cfg = compose(config_name=preset)

    encoder_class, decoder_class, encoder_cfg, decoder_cfg = (
        get_autoencoder_makers_and_configs(
            cfg["autoencoder"],
            variance_type=cfg["vae"]["variance_type"],
            img_shape=img_shape,
        )
    )
    latent_shape_res = encoder_cfg["img_resolution"] // (
        2 ** (len(encoder_cfg["channel_mult"]) - 1)
    )
    latent_shape = (
        cfg["autoencoder"]["latent_channels"],
        latent_shape_res,
        latent_shape_res,
    )
    vae = GaussGaussVAE.load_from_checkpoint(
        ckpt_path,
        encoder_class=encoder_class,
        decoder_class=decoder_class,
        encoder_params=encoder_cfg,
        decoder_params=decoder_cfg,
        prior_mean=torch.zeros((1,) + latent_shape),
        prior_logvar=torch.zeros((1,) + latent_shape),
        optim_config=cfg["optimizers"],
        **cfg["vae"],
    )
    vae = vae.eval()
    vae.requires_grad_(False)

    with h5py.File(data_path, "r") as f:
        mask = torch.from_numpy(f["mask"][idx][None]).cuda()
        x_orig = torch.from_numpy(f["data"][idx][None]).cuda()

    obs_mean = (x_orig[mask == 1]).mean()
    obs_std = (x_orig[mask == 1]).std()
    x_orig = x_orig - obs_mean
    x_orig = x_orig / obs_std

    obs = x_orig[mask == 1]

    # obs = obs + inverse_problem_cfg.std * torch.randn_like(obs)
    obs = obs.cuda().float()

    def log_pot(x):
        diff = ((x_orig[None] - x) * mask[None]).flatten()
        return -0.5 * (diff**2).sum() / std**2

    log_post_ula = lambda z: (
        log_pot(
            vae.decode(z).base_dist.rsample((1,))[
                0,
            ]
        ) - (z**2).sum() / 2
    )

    return x_orig, latent_shape, log_post_ula, LogPostNuts(x_orig, mask, std, vae), vae


class LogPostNuts(object):
    def __init__(self, x_orig, mask, std, vae):
        self.x_orig = x_orig
        self.mask = mask
        self.vae = vae
        self.std = std

    def __call__(self, z):
        _z = z["state"].float()
        x = self.vae.decode(_z[None]).base_dist.loc[0]
        diff = ((self.x_orig - x) * self.mask).flatten()
        return 0.5 * (diff**2).sum() / self.std**2 + (_z ** 2).sum() / 2
        


if __name__ == "__main__":
    tot_samples = 100
    n_chains = 3
    lr = 1e-4
    n_mcmc = 100
    n_warmup = 20
    n_steps_ula = 1000
    file_with_nuts = PATH
    file_without_nuts = PATH
       
    for idx in [2, 1]:
        x_orig, latent_shape, log_post_ula, log_post_nuts, vae = (
            make_all_inverse_problems_potentials(
                idx=idx,
            )
        )
        plt.imshow(x_orig.cpu()[0])
        plt.savefig("orig.png")
        for rd in range(tot_samples // n_chains):
            samples = ula(
                torch.randn((n_chains,) + latent_shape).cuda(),
                score=torch.func.grad(log_post_ula),
                n_steps=n_steps_ula,
                lr=lr,
            )
            samples_dist = vae.decode(samples).base_dist#.sample((1,))[0].detach().cpu()
            samples_mean = samples_dist.loc.detach().cpu().numpy()
            with h5py.File(file_without_nuts, "a") as f:
                if "data_samples" not in f.keys():
                    print("creating dataset")
                    mean_dset = f.create_dataset(
                                name="data_samples",
                                shape=(3, tot_samples, 1, 256, 256),
                                dtype="f",
                                chunks=(1, 1, 1, 256, 256),
                            )
                else:
                    mean_dset = f["data_samples"]
                if "data_samplesv2" not in f.keys():
                    print("creating dataset")
                    samples_dset = f.create_dataset(
                                        name="data_samplesv2",
                                        shape=(3, tot_samples, 1, 256, 256),
                                        dtype="f",
                                        chunks=(1, 1, 1, 256, 256),
                                    ) 
                else:
                    samples_dset = f["data_samplesv2"]    
                mean_dset[idx, rd*n_chains:(rd+1)*n_chains] = samples_mean
                samples_dset[idx, rd*n_chains:(rd+1)*n_chains] = samples_dist.sample((1,))[0].detach().cpu().numpy()
            samples_dist = None
            samples_mean = None

            if n_mcmc > 0:
                kernel = nuts.NUTS(
                    potential_fn=log_post_nuts,
                    adapt_mass_matrix=True,
                    adapt_step_size=True,
                    full_mass=False,
                    jit_compile=True,
                    step_size=lr,
                    max_tree_depth=6
                )

                mcmc = MCMC(
                    kernel,
                    num_samples=n_mcmc,
                    warmup_steps=n_warmup,
                    num_chains=n_chains,
                    initial_params={"state": samples},
                    mp_context="spawn",
                )
                mcmc.run()
                samples = mcmc.get_samples(group_by_chain=True)["state"][:, -1].float()
            samples_dist = vae.decode(samples).base_dist#.sample((1,))[0].detach().cpu()
            samples_mean = samples_dist.loc.detach().cpu().numpy()
            with h5py.File(file_with_nuts, "a") as f:
                if "data_samples" not in f.keys():
                    mean_dset = f.create_dataset(
                                name="data_samples",
                                shape=(3, tot_samples, 1, 256, 256),
                                dtype="f",
                                chunks=(1, 1, 1, 256, 256),
                            )
                else:
                    mean_dset = f["data_samples"]
                if "data_samplesv2" not in f.keys():
                    samples_dset = f.create_dataset(
                                        name="data_samplesv2",
                                        shape=(3, tot_samples, 1, 256, 256),
                                        dtype="f",
                                        chunks=(1, 1, 1, 256, 256),
                                    ) 
                else:
                    samples_dset = f["data_samplesv2"]    
                mean_dset[idx, rd*n_chains:(rd+1)*n_chains] = samples_mean
                samples_dset[idx, rd*n_chains:(rd+1)*n_chains] = samples_dist.sample((1,))[0].detach().cpu().numpy()
            
            fig, axes = plt.subplots(1, len(samples), squeeze=False)
            for ax, s in zip(axes.flatten(), samples_mean):
                ax.imshow(s[0])
                ax.set_xticks([])
                ax.set_yticks([])
            fig.tight_layout()
            fig.savefig("samples.png")

# %%
