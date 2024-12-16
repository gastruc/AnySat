from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
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
    for key in ["s1"]:
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

class BraDD(Dataset):
    def __init__(
        self,
        path,
        modalities,
        transform,
        split: str = "train",
        norm_path = None,
        temporal_dropout = 0,
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
        self.path = os.path.join(path, 'Samples')
        self.split = split
        self.temporal_dropout = temporal_dropout

        df = pd.read_csv(os.path.join(path, 'meta.csv'), index_col=0)
        if split == 'val':
            self.meta = df[df['close_set'] == 'validation'].reset_index(drop=True)
        else:
            self.meta = df[df['close_set'] == split].reset_index(drop=True)

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
        for i, b in enumerate(self.meta['file']):
            data = self.__getitem__(i)[sat]
            data = data.permute(1, 0, 2, 3)
            means.append(data.to(torch.float32).mean(dim=(1, 2, 3)).numpy())
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
        file_name = self.meta.loc[i, 'file']
        sample = torch.load(os.path.join(self.path, file_name))
        min_date = min(sample['image_dates'] + sample['label_dates'])
        ratio_band = sample['image'][:, :1, :, :] / (sample['image'][:, 1:, :, :] + 1e-6)
        ratio_band = torch.clamp(ratio_band, max=1e4, min=-1e4)
        image = torch.cat((sample['image'], ratio_band), dim=1)
        image_days = torch.tensor([(x - min_date).days + 1 for x in sample['image_dates']], dtype=torch.long)
        output = {'label': sample['label'][1].long(),
                    's1': image,
                    's1_dates': image_days,
                    'name': file_name,
                    }
        
        if self.split == "train" and self.temporal_dropout > 0:
            N = len(output["s1_dates"])
            random_indices = torch.randperm(N)[:(int(N * (1 - self.temporal_dropout)))]
            output["s1"] = output["s1"][random_indices]
            output["s1_dates"] = output["s1_dates"][random_indices]
    
        if self.norm is not None:
            for modality in self.modalities:
                if len(output[modality].shape) == 4:
                    output[modality] = (output[modality] - self.norm[modality][0][None, :, 
                                                    None, None]) / self.norm[modality][1][None, :, None, None]
                else:
                    output[modality] = (output[modality] - self.norm[modality][0][:, None, None]) / self.norm[modality][1][:, None, None]

        return self.transform(output)

    def __len__(self):
        return len(self.meta)