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
from datasources.gauss_spde import VEDataset, add_ve_noise, dict_to_tensor


class FlumyH5Dataset(H5Dataset):

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs, data_axis=0, data_name="data")

    def format_element(self, index) -> Dict:
        data_sample = self._file["data"][index]
        return {
                "data_sample": data_sample,
            }



class FlumyVEDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "path/to/dir",
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
        self.batch_size = batch_size
        self.ve_dataset_params = {
            "diffusion_cfg": diffusion_cfg,
            }
        self.add_maps = add_maps
        self.return_data = return_data
        self.val_ptg = val_ptg

    def setup(self, stage: str):
        if stage == "fit":
            datasets = []
            for dataset_path in os.listdir(self.data_dir):
                datasets.append(FlumyH5Dataset(path=os.path.join(self.data_dir, dataset_path)))

            base_dataset = ConcatDataset(datasets)
            final_dataset = VEDataset(base_dataset=base_dataset, **self.ve_dataset_params)
            self.train, self.val = random_split(
                final_dataset,
                lengths=[1 - self.val_ptg, self.val_ptg],
                generator=Generator().manual_seed(42), # same split between train and val
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

