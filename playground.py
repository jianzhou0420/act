import h5py
import numpy as np
from datarelated.data_converter import JianRLBenchDataStructure
import torch
with h5py.File('/media/jian/data/rlbench_hdf5/train/close_jar/episode_0.h5', 'r') as f:
    print("Datasets in the file:", list(f.keys()))
    for key in list(f.keys()):
        data = JianRLBenchDataStructure()
        try:
            data.__dict__[key] = torch.tensor(f[key][()])
        except:
            data.__dict__[key] = f[key][()]
        pass
