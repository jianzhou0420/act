import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import re
import numpy as np
import h5py
import rlbench
from data_converter import JianRLBenchDataStructure

import matplotlib.pyplot as plt


class JianRLBenchDataset(Dataset):
    def __init__(self):
        # TODO: config task
        self.data_dir = "/media/jian/data/rlbench_hdf5/train/close_jar"
        self.episodes = sorted([d for d in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, d))], key=self.natural_sort_key)

        pass

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        data_dict = dict()
        read_path = os.path.join(self.data_dir, f"episode_{idx}.h5")

        # Open the HDF5 file in read mode
        with h5py.File(read_path, 'r') as f:
            # List all datasets in the file

            for key in list(f.keys()):
                try:
                    data_dict[key] = torch.tensor(f[key][()])
                except:
                    data_dict[key] = list(f[key][()])  # variation_description is a list of strings

        return data_dict

    def natural_sort_key(self, s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


if __name__ == "__main__":
    # validate
    test = JianRLBenchDataset()
    DataLoader1 = DataLoader(test, batch_size=1, shuffle=True, num_workers=1)

    for i, data_dict in enumerate(DataLoader1):

        plt.imshow(data_dict['overhead_rgb'][0][10].permute(1, 2, 0))
        plt.show()
        pass
