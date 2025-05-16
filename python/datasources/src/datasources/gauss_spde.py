import math
from typing import (
    Dict,
    Tuple,
    Any,
    Union
)
import os
import numpy as np
from torch.utils.data import DataLoader, random_split, Dataset, ConcatDataset
from lightning import LightningDataModule
from torch import (
    Generator,
    float as torch_float,
    randn,
    rand,
    randn_like,
    from_numpy,
    tensor,
    load
)
from torch.nn.functional import one_hot
from torchvision import transforms
from functools import partial
from datasources.h5 import H5Dataset



def condition_dropout(item, ptg):
    if rand((1,)).item() < ptg:
        item["cond"] = 0
    else:
        item["cond"] = item["cond"] + 1 
    return item

def dict_to_tensor(item):
    return {
        k: from_numpy(el).type(torch_float) if isinstance(el, np.ndarray) else el
        for k, el in item.items()
    }

def add_ve_noise(item, sigma_max, sigma_min, log_mean, log_std, **kwargs):

    noise_level = (
        (randn(size=(1,)) * log_std + log_mean).exp().clip(sigma_min, sigma_max)
    )
    return {
        **item,
        "noise_level": noise_level.item(),
        "noisy_sample": item["data_sample"]
        + randn_like(item["data_sample"]) * noise_level,
    }


def quantize_cond(item, factor):
    item["cond"] = int(math.floor(item["cond"] / factor))
    return item

def one_hot_cond(item, label_dim):
    item["cond"] = one_hot(tensor([item["cond"]]), num_classes=label_dim)[0]
    return item

def remove_mean_std(item, mean, std):
    item["data_sample"] = (item["data_sample"] - mean) / std
    return item


class LatentGaussH5Dataset(H5Dataset):
    """Interface for the SPDE Gaussian dataset"""

    def __init__(
        self,
        return_non_latent_data: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs, data_axis=0, data_name="latent_loc")
        self.return_non_latent_data = return_non_latent_data

    def format_element(self, index) -> Dict:

        loc = from_numpy(self._file["latent_loc"][index]).type(torch_float)
        scale =  from_numpy(self._file["latent_scale"][index]).type(torch_float)
        if not self.return_non_latent_data:
            return {
                    "data_sample":loc + scale * randn_like(loc),
                }
        return {
                    "data_sample": loc + scale * randn_like(loc),
                    "data": self._file["data"][index]
                }

class AmbientGaussH5Dataset(H5Dataset):

    def __init__(
        self,
        add_maps: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs, data_axis=0, data_name="data")
        self.add_maps = add_maps

    def format_element(self, index) -> Dict:
        data_sample = self._file["data"][index]
        if self.add_maps:
            data_sample = np.concatenate((data_sample, self._file["maps"][index]), axis=0)
        return {
                "data_sample": data_sample,
            }


class VEDataset(Dataset):
    def __init__(self, base_dataset, diffusion_cfg, **kwargs):
        super().__init__(**kwargs)
        self.base_dataset = base_dataset
        methods = [dict_to_tensor]
        methods += [
            partial(add_ve_noise, **diffusion_cfg),
        ]

        self.transform = transforms.Compose(methods)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index) -> Dict:
        base_data_instance = self.base_dataset[index]
        return self.transform(base_data_instance)


class GaussVAEDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "path/to/dir",
        batch_size: int = 32,
        val_ptg: float = 0.05,
        n_procs: int = os.cpu_count(),
        add_maps: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.n_procs = n_procs
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_ptg = val_ptg
        self.add_maps = add_maps
        

    def setup(self, stage: str):
        if stage == "fit":
            datasets = []
            for dataset_path in os.listdir(self.data_dir):
                datasets.append(AmbientGaussH5Dataset(path=os.path.join(self.data_dir, dataset_path), add_maps=self.add_maps))

            Gauss_dataset = ConcatDataset(datasets)
            self.train, self.val = random_split(
                Gauss_dataset,
                lengths=[1 - self.val_ptg, self.val_ptg],
                generator=Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.n_procs,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.n_procs,
            shuffle=False,
        )


class GaussVEDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "path/to/dir",
        is_latent: bool = False,
        batch_size: int = 32,
        n_procs: int = os.cpu_count(),
        val_ptg: float = 0.05,
        diffusion_cfg: Dict[str, Any] = {},
        return_data: bool = False,
        add_maps: bool=True,
        **kwargs,
    ):
        super().__init__()
        self.n_procs = n_procs
        self.data_dir = data_dir
        self.is_latent = is_latent
        self.batch_size = batch_size
        self.ve_dataset_params = {
            "diffusion_cfg": diffusion_cfg,
            }
        self.add_maps = add_maps
        self.return_data = return_data
        self.val_ptg = val_ptg

    def setup(self, stage: str):
        if stage == "fit":
            if self.is_latent:
                base_dataset = LatentGaussH5Dataset(path=self.data_dir, return_non_latent_data=self.return_data)
            else:
                datasets = []
                for dataset_path in os.listdir(self.data_dir):
                    datasets.append(AmbientGaussH5Dataset(path=os.path.join(self.data_dir, dataset_path), add_maps=self.add_maps))

                base_dataset = ConcatDataset(datasets)
            final_dataset = VEDataset(base_dataset=base_dataset, **self.ve_dataset_params)
            self.train, self.val = random_split(
                final_dataset,
                lengths=[1 - self.val_ptg, self.val_ptg],
                generator=Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.n_procs,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.n_procs,
            shuffle=False,
        )

