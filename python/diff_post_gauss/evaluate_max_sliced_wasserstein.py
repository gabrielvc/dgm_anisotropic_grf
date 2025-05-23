import torch
from torch.utils.data import DataLoader, ConcatDataset
from datasources.gauss_spde import AmbientGaussH5Dataset
from datasources.h5 import H5Dataset
import ot
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt


class GenH5Dataset(H5Dataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs, data_axis=0, data_name="images")

    def format_element(self, index):
        return {
            "data_sample": self._file["images"][index],
        }

ref_data_dir = "path_TO_REFERENCE_DATA"
train_data_dir = "PATH_TO_TRAIN_DATA"

datasets = []
for dataset_path in os.listdir(ref_data_dir):
    datasets.append(
        AmbientGaussH5Dataset(
            path=os.path.join(ref_data_dir, dataset_path), add_maps=False
        )
    )
ref_dataset = ConcatDataset(datasets)

batch_size = 32
ref_dataloader = DataLoader(
    dataset=ref_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    drop_last=False,
)


datasets = []
for dataset_path in os.listdir(train_data_dir):
    datasets.append(
        AmbientGaussH5Dataset(
            path=os.path.join(train_data_dir, dataset_path), add_maps=False
        )
    )
train_dataset = ConcatDataset(datasets)
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    drop_last=False,
)
# %%
n_slices = 2**16
max_slice_gpu = 2**12
max_sws = {}
n_rep = 20
n_samples = 10_000
f_name = f"max_sliced_per_sampler_{n_slices}_{n_samples}.pt"
if f_name not in os.listdir("data"):
    
    max_sws_train = []
    for i in range(n_rep):

        train_samples, ref_samples = [], []
        for batch_train, batch_ref in tqdm(zip(train_dataloader, ref_dataloader)):
            train_samples.append(batch_train["data_sample"])
            ref_samples.append(batch_ref["data_sample"])
            if len(ref_samples) * batch_size > n_samples:
                break

        train_samples = torch.cat(train_samples)
        ref_samples = torch.cat(ref_samples)

        max_sws_train.append(
            max(
                [
                    ot.sliced.max_sliced_wasserstein_distance(
                        train_samples.flatten(start_dim=1).cuda(),
                        ref_samples.flatten(start_dim=1).cuda(),
                        n_projections=max_slice_gpu,
                        seed=(2**i) * (3**j),
                    )
                    .cpu()
                    .item()
                    for j in range(n_slices // max_slice_gpu)
                ]
            )
        )
        print(max_sws_train)
    max_sws = {("train", 0): max_sws_train}
    torch.save(max_sws,  os.path.join("data", f_name))

max_sws = torch.load(os.path.join("data", f_name))
# %%
models_to_test = {
    (
        "NAME_OF_SAMPLER",
        ADDITIONAL_INFO_1,#FOR EXAMPLE NUMBER OF STEPS
        aDDITIONAL_INFO_2,#FOR EXAMPLE RHO, NONE CAN BE PASSED
    ): "PATH_TO_GENERATED_SAMPLER_FOLDER/data.h5",
}


for name, gen_data_path in models_to_test.items():

    gen_dataset = GenH5Dataset(path=gen_data_path)
    gen_dataloader = DataLoader(
        dataset=gen_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=False,
    )
    if name not in max_sws:
        max_sws[name] = []
        for i in range(n_rep):
            dataset = []
            ref_samples = []
            pbar = tqdm(
                zip(gen_dataloader, ref_dataloader),
                desc="-".join([str(i) for i in name]),
            )
            for batch_gen, batch_ref in pbar:
                gen_samples = batch_gen["data_sample"].float()
                dataset.append(gen_samples[gen_samples.std(dim=(-3, -2, -1)) > 0
])
                ref_samples.append(batch_ref["data_sample"].float())
                if len(dataset) * batch_size > n_samples:
                    break
            dataset = torch.concatenate(dataset)
            ref_samples = torch.concatenate(ref_samples)

            max_sws[name].append(
                max(
                    [
                        ot.sliced.max_sliced_wasserstein_distance(
                            dataset.flatten(start_dim=1).cuda(),
                            ref_samples.float().flatten(start_dim=1).cuda(),
                            n_projections=max_slice_gpu,
                            seed=(2**i) * (3**j),
                        )
                        .cpu()
                        .item()
                        for j in range(n_slices // max_slice_gpu)
                    ]
                )
            )
            print(max_sws[name])
        torch.save(max_sws, os.path.join("data", f_name))


# %%

fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(12, 8))

ax.boxplot(
    max_sws.values(),
    tick_labels=["\n".join([str(i) for i in k]) for k in max_sws.keys()],
)
ax.set_ylabel("Max SW", fontsize=20)
ax.tick_params(axis="x", labelsize=17)
fig.show()
# %%
i = 100
dist_fun = lambda x, y: 1 - (x * y).flatten(start_dim=1).sum(dim=-1) / (
    ((x**2).flatten(start_dim=1).sum(dim=-1) ** 0.5)
    * ((y**2).flatten(start_dim=1).sum(dim=-1) ** 0.5)
)
# dist_fun = lambda x, y: ((x - y)**2).flatten(start_dim=1).sum(dim=-1)
closest = dist_fun(dataset[i][None].cuda(), ref_samples.cuda()).cpu().argmin()
print(closest)
# %%
fig, axes = plt.subplots(1, 2)
axes[0].imshow(dataset[i, 0], vmin=-3, vmax=3)
axes[1].imshow(ref_samples[closest, 0], vmin=-3, vmax=3)
print(ref_samples[closest, 0].std())
fig.show()

# %%
#preparing data
import pandas as pd
df = pd.DataFrame.from_records([{"sampler": k[0], "n_steps": k[1], "rho": k[2] if len(k) > 2 else np.nan, "mean": np.mean(v), "std": np.std(v), "len": len(v)} for k, v in max_sws.items()])
df["clt_gap"] = df["std"] * 1.96 / df["len"]**.5
fig, ax = plt.subplots(1, 1)
ax.axhline(df.loc[df["sampler"]=="train","mean"].iloc[0], label="Train", color="red")
ax.fill_between(x=np.arange(0, 10),y1=df.loc[df["sampler"]=="train","mean"].iloc[0] + df.loc[df["sampler"]=="train", "clt_gap"].iloc[0], y2= df.loc[df["sampler"]=="train","mean"].iloc[0]- df.loc[df["sampler"]=="train", "clt_gap"].iloc[0], alpha=.5, color="red")
for (name,), dt in df.loc[df.n_steps==100].groupby(["sampler"]):
    ax.errorbar(x=dt["rho"], y=dt["mean"], yerr=dt["clt_gap"], label=name.upper(), capsize=10)
ax.set_ylabel(r"Max-SW", fontsize=20)
ax.set_xlabel(r"$\rho$", fontsize=20)
ax.set_xlim([0.5, 7.5])
fig.legend(fontsize=20, loc="lower right")
fig.subplots_adjust(top=1, right=1)

#%%
to_save = df
to_save["value"] = to_save.apply(lambda x: f"{x["mean"]:.3f} + {x["clt_gap"]:.3f}", axis=1)
to_save.sort_values(["sampler", "n_steps"]).to_csv("max_sws_isotropic.csv")
