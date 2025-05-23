import torch
import os

import torch

import os
import h5py
import click
from tqdm import tqdm

@click.command()
@click.option(
    "--path",
    help="Path to generated data repository",
    metavar="STR",
    type=str,
    required=True,
)
@click.option(
    "--field_name",
    help="Field name",
    metavar="STR",
    type=str,
    required=True,
)
def cmdline(
    path: str,
    field_name: str,
    **opts,
):
    raw_data_path = os.path.join(path, "raw_data")
    dsets = {}
    with h5py.File(
        os.path.join(path, "data.h5"), "w"
    ) as f:
        start_index = 0
        end_index = 0
        all_files = os.listdir(raw_data_path)
        for i, batch_file in tqdm(enumerate(all_files)):
            try:
                batch_data = torch.load(os.path.join(raw_data_path, batch_file), map_location=torch.device("cpu"))
                if i == 0:
                    batch_size = batch_data[field_name].shape[0]
                    n_samples = batch_size * len(all_files)
                    for k in batch_data:
                        dsets[k] = f.create_dataset(
                            name=k,
                            shape=(n_samples, *batch_data[k].shape[1:]),
                            maxshape=(n_samples, *batch_data[k].shape[1:]),
                            dtype="f",
                            chunks=(1, *batch_data[k].shape[1:]),
                        )
                batch_size = batch_data[field_name].shape[0]
                end_index += batch_size

                for k, v in batch_data.items():
                    dsets[k][start_index:end_index] = v.cpu().numpy()
                start_index = end_index
            except Exception as e:
                print(e)
                print(f"Couldn't read {batch_file}")

        for k in dsets:
            shape = (start_index,) +  dsets[k].shape[1:]
            print(f"resizing to shape {shape}")
            dsets[k].resize(shape)


if __name__ == "__main__":
    cmdline()
