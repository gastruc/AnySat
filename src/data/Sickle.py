from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
import torch
import glob
from datetime import datetime
import rasterio
import albumentations as A
import cv2
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
    for key in ["s2", "s1", "l8", "l8_mask"]:
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
        date_object = datetime.strptime(date_string, '%Y%m%d')
        day_number.append(date_object.timetuple().tm_yday) # Get the day of the year
    return torch.tensor(day_number)

class Sickle(Dataset):
    def __init__(
        self,
        path,
        modalities,
        transform,
        split: str = "train",
        norm_path = None,
        temporal_dropout = 0,
        img_size = 32,
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
        self.path = path
        self.transform = transform
        self.modalities = modalities
        self.data_path = path
        self.split = split
        self.temporal_dropout = temporal_dropout
        self.bands = {
            "s2": ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B8', 'B11', 'B12'],
            "l8": ["random","SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "random", "ST_B10", "random"],
            "s1": ["VV", "VH"]
        }

        df = pd.read_csv(os.path.join(path, "sickle_dataset_tabular.csv"))
        self.meta = df[df.SPLIT == split].reset_index(drop=True)
        self.plot_ids = set(self.meta.PLOT_ID)
        self.resize = A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_NEAREST)
        self.img_size= img_size

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
        for i, b in enumerate(self.meta["UNIQUE_ID"]):
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
        uid = int(self.meta.iloc[i]["UNIQUE_ID"])

        path = f"{self.data_path}/masks/10m/{uid}.tif"
        with rasterio.open(path, 'r') as fp:
            mask = fp.read()[1]
            plot_mask = fp.read()[0]
            mask -= 1
            mask[mask > 1] = 1
            mask[mask < -1] = 0
            plot_mask[plot_mask==0]=-1
            plot_mask[plot_mask < -1] = 0
            unmatched_plots = set(np.unique(plot_mask)[1:]) - self.plot_ids
            for unmatched_plot in unmatched_plots:
                mask[plot_mask == unmatched_plot] = -1

        output = {'label': torch.from_numpy(self.resize(image=mask)['image']), 'name': uid}

        for modality in self.modalities:
            path = f"{self.data_path}/images/{modality.upper()}/npy/{uid}/*.npz"
            files = glob.glob(path)
            samples = []
            dates = []
            maskes = []
            for file in files:
                data_file = np.load(file)
                all_channels = []
                mask = []
                for band in self.bands[modality]:
                    if band == "random" or band not in data_file.keys():
                        all_channels.append(torch.zeros((self.img_size, self.img_size)))
                        mask.append(0)
                    elif modality == "l8":
                        all_channels.append(torch.from_numpy(self.resize(image=data_file[band])['image']))
                        mask.append(1)
                    else:
                        all_channels.append(torch.from_numpy(self.resize(image=data_file[band])['image']))
                if modality == "s1":
                    stacked = torch.stack(all_channels)
                    ratio_band = stacked [:1, :, :] / (stacked [1:, :, :] + 1e-6)
                    ratio_band = torch.clamp(ratio_band, max=1e4, min=-1e4)
                    stacked = torch.cat((stacked, ratio_band), dim=0)
                else:
                    stacked = torch.stack(all_channels)
                samples.append(stacked)
                if modality == "s2":
                    if os.path.basename(file)[0] =="T": index_date = os.path.basename(file).split("_")[1][:8]
                    else: index_date = os.path.basename(file).split("_")[0][:8]
                    
                elif modality == "s1":
                    index_date = os.path.basename(file).split("_")[4][:8]
                else:
                    index_date = os.path.basename(file).split("_")[2][:8]
                    maskes.append(torch.tensor(mask))
                dates.append(index_date)

            output[modality] = torch.stack(samples)
            output['_'.join([modality, 'dates'])] = day_number_in_year(dates)
            if modality == "l8":
                output['_'.join([modality, 'mask'])] = torch.stack(maskes).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.img_size, self.img_size)

            if self.split == "train" and self.temporal_dropout > 0:
                N = len(output['_'.join([modality, 'dates'])])
                random_indices = torch.randperm(N)[:(int(N * (1 - self.temporal_dropout)))]
                output[modality] = output[modality][random_indices]
                if modality == "l8":
                    output['_'.join([modality, "mask"])] = output['_'.join([modality, "mask"])][random_indices]
                output['_'.join([modality, "dates"])] = output['_'.join([modality, "dates"])][random_indices]


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