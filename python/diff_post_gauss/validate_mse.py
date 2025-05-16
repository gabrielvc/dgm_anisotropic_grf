import torch
torch.set_float32_matmul_precision('medium')
from tqdm import tqdm
import lightning as L
from hydra import initialize, compose
import math
from diffgauss.utils import load_diffusion_net
from datasources.gauss_spde import AmbientGaussH5Dataset
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import click

#%%

config_path = "configs/"

batch_size = 64

#%%
# ref_data_dir = "data/anisotropic_data/validation"
# train_data_dir = "data/anisotropic_data/train"

ref_data_dir = "data/isotropic_data/validation"
train_data_dir = "data/isotropic_data/train"

datasets = []
for dataset_path in os.listdir(ref_data_dir):
    datasets.append(AmbientGaussH5Dataset(path=os.path.join(ref_data_dir, dataset_path), add_maps=False))
ref_dataset = ConcatDataset(datasets)

ref_dataloader = DataLoader(
    dataset=ref_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,  
    drop_last=False,
)

datasets = []
for dataset_path in os.listdir(train_data_dir):
    datasets.append(AmbientGaussH5Dataset(path=os.path.join(train_data_dir, dataset_path), add_maps=False))
train_dataset = ConcatDataset(datasets)
train_dataloader = DataLoader(
    dataset=ref_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,  
    drop_last=False,
)
#%%
denoisers = {
    "new": {"preset": "isotropic_ambient_diffusion_smaller_more_tail_deeper.yaml",
           "ckpt_path": "data/lightning_models_jean_zay/Diff/isotropic_deeper/epoch=79-step=9280.ckpt"},
    }
for name, infos in denoisers.items():
    with initialize(version_base=None, config_path=config_path):
        # config is relative to a module
        diffusion_cfg = compose(config_name=infos["preset"])


    #with trainer.init_module():
    img_shape, denoiser = load_diffusion_net(
        diffusion_cfg, infos["ckpt_path"], None, is_latent=False
    )
    denoiser = denoiser.ema_models["ema_16-97"]
    denoiser.eval()
    denoiser.requires_grad_(False)
    denoisers[name]["denoiser"] = denoiser
    denoisers[name]["diffusion_cfg"] = diffusion_cfg
#%%
import math
stds = torch.exp(torch.linspace(math.log(diffusion_cfg["diffusion"]["sigma_min"]), math.log(diffusion_cfg["diffusion"]["sigma_max"]), 100))

for name, infos in denoisers.items():
    denoiser = infos["denoiser"]
    denoisers[name]["mses"] = {"train": [], "validation": []}
    for batch_train, batch_validation, _ in zip(train_dataloader, ref_dataloader, range(5)):
        mses = {"train": [], "validation": []}
        for std in tqdm(stds):
            noise = torch.randn((batch_size,) + img_shape)
            noisy_train = batch_train["data_sample"] + std * noise
            noisy_val = batch_validation["data_sample"] + std * noise
            with torch.no_grad():
                clean_train = denoiser(noisy_train.cuda(), std.cuda()).cpu()
                clean_val = denoiser(noisy_val.cuda(), std.cuda()).cpu()
            mse_train = ((clean_train - batch_train["data_sample"])**2).flatten(start_dim=1).sum(dim=-1) / (clean_train[0].numel())
            mse_val = ((clean_val - batch_validation["data_sample"])**2).flatten(start_dim=1).sum(dim=-1) / (clean_train[0].numel())

            mses["train"].append(mse_train)
            mses["validation"].append(mse_val) 
        for k in mses:
            mses[k] = torch.stack(mses[k])
            denoisers[name]["mses"][k].append(mses[k])
#%%
for name, infos in denoisers.items():
    for k in infos["mses"]:
        infos["mses"][k] = torch.stack(infos["mses"][k]).permute(0, 2, 1).flatten(start_dim=0, end_dim=1)
#%%

colors = {"train": "green", "validation": "orange"}
for name, infos in denoisers.items():

    for k in infos["mses"]:
        fig, ax = plt.subplots(1, 1)
        ax.plot(stds**2, stds**(2 * (2/3)), label=r"$y = x^\frac{2}{3}$")

        coeffs = {}
        mses = infos["mses"]
        solution = torch.linalg.lstsq(
        torch.cat((torch.log(stds**2)[:,None], torch.ones_like(torch.log(stds**2)[:, None])), dim=-1).float()[:-23],
        torch.log(mses[k].mean(dim=0))[:, None].float()[:-23]
    ).solution
        coeffs[k] = solution[0, 0].item()
        rec = 2 * torch.log(stds) * solution[0, 0] + solution[1, 0]
        ax.scatter(stds**2, mses[k].mean(dim=0), marker='*',c=colors[k], label=f"{k}", alpha=.5)
        ax.plot(stds**2, rec.exp(), label=f"{k} fit", c=colors[k])

        ax.set_ylabel(r"MSE / d", fontsize=20)
        ax.set_xlabel(r"$\sigma^2$", fontsize=20)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.grid(True)
        fig.subplots_adjust(right=1, top=1)
        fig.show()
#%%
coeffs = {}
for k in mses:
    features = torch.cat((torch.log(stds**2)[:, None], torch.ones_like(torch.log(stds**2)[:, None])), dim=-1).float()
    target = torch.log(mses[k].float())[..., None]
    coeffs[k] = torch.linalg.lstsq(
    features[None],
    target
).solution[:, 0, 0]
fig, axes = plt.subplots(2, 2, figsize=(6, 6), sharex="row", sharey="row")
for ax, k in zip(axes[0].flatten(), mses):
    for mse_row in mses[k]:
        ax.plot(stds**2, mse_row, marker='*', alpha=.3)
    ax.set_ylabel(r"MSE / d")
    ax.set_xlabel(r"$\sigma^2$")

    ax.set_yscale("log")
    ax.set_xscale("log")
axes[1, 0].hist(coeffs["train"], density=True)
axes[1, 1].hist(coeffs["validation"], density=True)
fig.show()

#%%
fig, ax = plt.subplots(1, 1)
cout = (
    1
    * stds
    / (1**2 + stds**2) ** 0.5
)
weights = 1 / (cout ** 2)
for name, infos in denoisers.items():
    coeffs = {}
    mses = infos["mses"]
    losses = mses["train"] * weights[None]
    ax.plot(stds, losses.mean(dim=0))
    ax.fill_between(x=stds, y1=torch.quantile(losses, q=.9, dim=0), y2=torch.quantile(losses, q=.10, dim=0), alpha=.5)
    
    #ax.set_yscale("log")
ax.set_xscale("log")
ax.hist((torch.randn(10000)*1.2 -1.2).exp(), bins=torch.linspace(-5e0, 3, 30).exp(), alpha=.5, density=True, label="EDM")
ax.hist((torch.randn(10000)*1.6 + 1).exp(), bins=torch.linspace(-5e0, 3, 30).exp(), alpha=.5, density=True, label="Ours")
ax.set_xlabel(r"$\sigma$", fontsize=20)
ax.set_ylabel(r"$(\sigma^{-2} + 1)$"+r"$MSE(D_{\theta}, \sigma)$", fontsize=15)
ax.set_xlim(0.005, 80)
ax.set_yticks([0, 1, 2])
ax.axhline(1, color='red')
fig.legend(fontsize=20)
fig.subplots_adjust(right=1, left=.25, top=1)
fig.show()

# %%
