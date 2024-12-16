from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
import torch
from datetime import datetime
import json

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
    for key in ["s2", "s2_mask"]:
        if key in keys:
            idx = [x[key] for x in batch]
            max_size_0 = max(tensor.size(0) for tensor in idx)
            stacked_tensor = torch.stack([
                    torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, 0, 0, max_size_0 - tensor.size(0)))
                    for tensor in idx
                ], dim=0)
            output[key] = stacked_tensor
            keys.remove(key)
            key = '_'.join([key, "dates"])
            if key in keys:
                idx = [x[key] for x in batch]
                max_size_0 = max(tensor.size(0) for tensor in idx)
                stacked_tensor = torch.stack([
                        torch.nn.functional.pad(tensor, (0, max_size_0 - tensor.size(0)))
                        for tensor in idx
                    ], dim=0)
                output[key] = stacked_tensor
                keys.remove(key)
    for key in keys:
        output[key] = torch.stack([x[key] for x in batch])
    return output

def day_number_in_year(date_arr):
    day_number = []
    for date_string in date_arr:
        date_object = datetime.strptime(str(date_string), '%Y%m%d')
        day_number.append(date_object.timetuple().tm_yday)
    return torch.tensor(day_number)

class TS2CDataset(Dataset):
    def __init__(
        self,
        path,
        modalities,
        transform,
        split: str = "train",
        norm_path = None,
        temporal_dropout = 0,
        num_classes = 0
        ):
        """
        Initializes the dataset.
        Args:
            path (str): path to the dataset
            modalities (list): list of modalities to use
            transform (torch module): transform to apply to the data
            split (str): split to use (train, val, test)
            classes (list): name of the differrent classes
            partition (float): proportion of the dataset to keep
            mono_strict (bool): if True, puts monodate in same condition as multitemporal
        """
        self.transform = transform
        self.modalities = modalities
        self.path = path
        self.split = split
        self.temporal_dropout = temporal_dropout
        self.num_classes = num_classes

        if split == "train":
            folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
            folders.remove('33UVP')
            folders.remove('2019_33UVP')
            folders.remove('33TWN')
        elif split == "val":
            folders = ['33TWN']
        else:
            folders = ['33UVP']

        all_files = []
        for folder in folders:
            folder_path = os.path.join(path, folder)
            for root, _, files in os.walk(folder_path):
                if files != ["dates.csv"]:
                    for file in files:
                        all_files.append(os.path.join(root, file))

        self.files = all_files

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

    def compute_norm_vals(self, folder, sat):
        means = []
        stds = []
        for i, b in enumerate(self.files):
            data = self.__getitem__(i)
            #mask = data[sat + '_mask'][:, 0].flatten()
            data = data[sat]#[mask]
            data = data.permute(1, 0, 2, 3)
            means.append(data.to(torch.float32).mean(dim=(1, 2, 3)).numpy())
            # if np.any(np.isnan(data.to(torch.float32).mean(dim=(1, 2, 3)).numpy())):
            #     print(i)
            #     print(self.files[i])
            stds.append(data.to(torch.float32).std(dim=(1, 2, 3)).numpy())

        mean = np.stack(means).mean(axis=0).astype(float)
        std = np.stack(stds).mean(axis=0).astype(float)

        norm_vals = dict(mean=list(mean), std=list(std))

        with open(os.path.join(folder, "NORM_{}_patch.json".format(sat)), "w") as file:
            file.write(json.dumps(norm_vals, indent=4))

    def __getitem__(self, i):
        """
        Returns an item from the dataset.
        Args:
            i (int): index of the item
        Returns:
            dict: dictionary with keys "label", "name" and the other corresponding to the modalities used
        """
        file_name = self.files[i]
        dates = day_number_in_year(pd.read_csv(('/').join(file_name.split('/')[:-2] + ["dates.csv"]))["acquisition_date"].values)
        bands = pd.read_csv(file_name)
        # mask = bands['Flag'].values
        # mask = torch.from_numpy(mask == 0).unsqueeze(-1).repeat(1, 10)
        mask = torch.ones((len(dates), 10))
        mask[:, 7] = False

        output = {'label': torch.tensor(int(file_name.split('/')[-2])),
                    's2': torch.from_numpy(bands.values[:, [0, 1, 2, 3, 4, 5, 6, 9, 7, 8]]).unsqueeze(-1).unsqueeze(-1),
                    's2_dates': dates,
                    's2_mask': mask.unsqueeze(-1).unsqueeze(-1),
                    }
        
        if self.split == "train" and self.temporal_dropout > 0:
            N = len(output["s2_dates"])
            random_indices = torch.randperm(N)[:(int(N * (1 - self.temporal_dropout)))]
            output["s2"] = output["s2"][random_indices]
            output["s2_dates"] = output["s2_dates"][random_indices]
            output["s2_mask"] = output["s2_mask"][random_indices]
    
        if self.norm is not None:
            for modality in self.modalities:
                if len(output[modality].shape) == 4:
                    output[modality] = (output[modality] - self.norm[modality][0][None, :, 
                                                    None, None]) / self.norm[modality][1][None, :, None, None]
                else:
                    output[modality] = (output[modality] - self.norm[modality][0][:, None, None]) / self.norm[modality][1][:, None, None]

        return self.transform(output)

    def __len__(self):
        return len(self.files)