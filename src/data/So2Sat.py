# Source: https://github.com/cloudtostreet/Sen1Floods11

import os
import json

import numpy as np
import h5py
import torch
import random

from torch.utils.data import Dataset

def collate_fn(batch):
    """
    Collate function for the dataloader.
    Args:
        batch (list): list of dictionaries with keys "label", "name"  and the other corresponding to the modalities used
    Returns:
        dict: dictionary with keys "label", "name"  and the other corresponding to the modalities used
    """
    keys = list(batch[0].keys())
    output = {}
    if 'name' in keys:
        output['name'] = [x['name'] for x in batch]
        keys.remove('name')
    for key in keys:
        output[key] = torch.stack([x[key] for x in batch])
    return output

class So2Sat(Dataset):
    def __init__(
        self,
        path,
        modalities,
        transform,
        split: str = "train",
        norm_path = None,
    ):
        """Initialize the So2Sat dataset.

        Args:
            path (str): Path to the dataset.
            modalities (list): List of modalities to use (e.g., ["s1", "s2"]).
            transform (callable): A function/transform to apply to the data.
            split (str, optional): Split of the dataset ('train', 'val', 'test'). Defaults to 'train'.
            norm_path (str, optional): Path for normalization data. Defaults to None.
        """
        self.path = path
        self.modalities = modalities
        self.transform = transform
        self.split = split

        mapping = {"train": "training", "val": "validation", "test": "testing"}

        # Load HDF5 file
        h5_file = h5py.File(path + "/" + mapping[split] + ".h5", 'r')
        self.labels = torch.from_numpy(np.array(h5_file["label"])).argmax(dim=1)
        if "s1" in self.modalities:
            self.s1 = torch.from_numpy(np.array(h5_file["sen1"])).permute(0, 3, 2, 1)[:, [2, 0], :, :] #[:, [5, 4], :, :]
            ratio_band = self.s1[:, :1, :, :] / (self.s1[:, 1:, :, :] + 1e-4)
            ratio_band = torch.clamp(ratio_band, max=1e4, min=-1e4)
            self.s1 = torch.cat((self.s1[:, :2, :, :], ratio_band), dim=1).float()
        if "s2" in self.modalities:
            self.s2 = torch.from_numpy(np.array(h5_file["sen2"])).permute(0, 3, 2, 1)
        h5_file.close()
        
        self.collate_fn = collate_fn
        self.norm = None
        if norm_path is not None:
            norm = {}
            for modality in self.modalities:
                file_path = os.path.join(norm_path, "NORM_{}_patch.json".format(modality))
                if not(os.path.exists(file_path)):
                    self.compute_norm_vals(norm_path, modality)
                normvals = json.load(open(file_path))
                norm[modality] = (
                    torch.tensor(normvals['mean']).float(),
                    torch.tensor(normvals['std']).float(),
                )
            self.norm = norm  

    def __len__(self):
        return len(self.labels)
    
    def compute_norm_vals(self, folder, sat):
        print("Computing norm values for {}".format(sat))
        means = []
        stds = []
        for i, b in enumerate(self.labels):
            data = self.__getitem__(i)[sat]
            data = data.permute(1, 0, 2, 3)
            means.append(data.to(torch.float32).mean(dim=(1, 2, 3)).numpy())
            stds.append(data.to(torch.float32).std(dim=(1, 2, 3)).numpy())

        mean = np.stack(means).mean(axis=0).astype(float)
        std = np.stack(stds).mean(axis=0).astype(float)

        norm_vals = dict(mean=list(mean), std=list(std))

        with open(os.path.join(folder, "NORM_{}_patch.json".format(sat)), "w") as file:
            file.write(json.dumps(norm_vals, indent=4))

    def __getitem__(self, index):
        output = {"label": self.labels[index], "name": str(index)}
        if "s1" in self.modalities:
            output["s1"] = self.s1[index].unsqueeze(0).float()
            output["s1_dates"] = torch.tensor([0])
        if "s2" in self.modalities:
            output["s2"] = self.s2[index].unsqueeze(0).float()
            output["s2_dates"] = torch.tensor([0])

        if self.norm is not None:
            for modality in self.modalities:
                if len(output[modality].shape) == 4:
                    output[modality] = (output[modality] - self.norm[modality][0][None, :, 
                                                    None, None]) / self.norm[modality][1][None, :, None, None]
                else:
                    output[modality] = (output[modality] - self.norm[modality][0][:, None, None]) / self.norm[modality][1][:, None, None]

        return self.transform(output)
