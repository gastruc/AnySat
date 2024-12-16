from torch.utils.data import Dataset
import h5py
import numpy as np
import os
import json
from data.utils import subset_dict_by_filename, filter_labels_by_threshold
from skmultilearn.model_selection import iterative_train_test_split
import torch
import rasterio
from datetime import datetime

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
    for key in ["s2", "s1-asc", "s1-des", "s1"]:
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
    if 'name' in keys:
        output['name'] = [x['name'] for x in batch]
        keys.remove('name')
    for key in keys:
        output[key] = torch.stack([x[key] for x in batch])
    return output

def day_number_in_year(date_arr, place=4):
    day_number = []
    for date_string in date_arr:
        date_object = datetime.strptime(str(date_string).split('_')[place][:8], '%Y%m%d')
        day_number.append(date_object.timetuple().tm_yday) # Get the day of the year
    return torch.tensor(day_number)

def replace_nans_with_mean(batch_of_images):
    image_means = torch.nanmean(batch_of_images, dim=(3, 4), keepdim=True)
    image_means[torch.isnan(image_means)] = 0.
    nan_mask = torch.isnan(batch_of_images)
    batch_of_images[nan_mask] = image_means.expand_as(batch_of_images)[nan_mask]
    return batch_of_images

class TreeSAT(Dataset):
    def __init__(
        self,
        path,
        modalities,
        transform,
        split: str = "train",
        classes: list = [],
        partition: float = 1.,
        mono_strict: bool = False,
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
        self.path = path
        self.transform = transform
        self.partition = partition
        if mono_strict:
            d = {"aerial": "aerial", "s2": "s2-mono", "s1": "s1-mono", "s1-asc": "s1-mono"}
            modalities = [d[modality] for modality in modalities]
        self.modalities = modalities
        self.mono_strict = mono_strict
        self.split = split
        self.temporal_dropout = temporal_dropout
        data_path = path + split + "_filenames.lst"
        with open(data_path, 'r') as file:
            self.data_list = [line.strip() for line in file.readlines()]
        self.load_labels(classes)
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
        for i, b in enumerate(self.data_list):
            data = self.__getitem__(i)[sat]
            if len(data.shape) == 4:
                data = data.permute(1, 0, 2, 3)
                means.append(data.to(torch.float32).mean(dim=(1, 2, 3)).numpy())
                stds.append(data.to(torch.float32).std(dim=(1, 2, 3)).numpy())
            else:
                means.append(data.to(torch.float32).mean(dim=(1, 2)).numpy())
                stds.append(data.to(torch.float32).std(dim=(1, 2)).numpy())

        mean = np.stack(means).mean(axis=0).astype(float)
        std = np.stack(stds).mean(axis=0).astype(float)

        norm_vals = dict(mean=list(mean), std=list(std))

        with open(os.path.join(folder, "NORM_{}_patch.json".format(sat)), "w") as file:
            file.write(json.dumps(norm_vals, indent=4))
            
    def load_labels(self, classes):
        with open(self.path + "labels/TreeSatBA_v9_60m_multi_labels.json") as file:
            jfile = json.load(file)
            subsetted_dict = subset_dict_by_filename(self.data_list, jfile)
            labels = filter_labels_by_threshold(subsetted_dict, 0.07)
            lines = list(labels.keys())

        y = [[0 for i in range(len(classes))] for line in lines]
        for i, line in enumerate(lines):
            for u in labels[line]:
                y[i][classes.index(u)] = 1

        self.data_list, self.labels, _, _ = iterative_train_test_split(np.expand_dims(np.array(lines), axis=1), np.array(y), test_size = 1. - self.partition)
        self.data_list = list(np.concatenate(self.data_list).flat)

    def __getitem__(self, i):
        """
        Returns an item from the dataset.
        Args:
            i (int): index of the item
        Returns:
            dict: dictionary with keys "label", "name" and the other corresponding to the modalities used
        """
        name = self.data_list[i]
        output = {'label': torch.tensor(self.labels[i]), 'name': name}
    
        if 'aerial' in self.modalities:
            with rasterio.open(self.path + "aerial/" + name) as f:
                output["aerial"] = torch.FloatTensor(f.read())

        with h5py.File(self.path + "sentinel/" + '.'.join(name.split('.')[:-1]) + ".h5", 'r') as file:
            if 's1-asc' in self.modalities:
                output["s1-asc_dates"] = day_number_in_year(file["sen-1-asc-products"][:])
            if 's1-des' in self.modalities:
                output["s1-des_dates"] = day_number_in_year(file["sen-1-des-products"][:])
            if 's1' in self.modalities:
                s1_asc_dates = day_number_in_year(file["sen-1-asc-products"][:])
                s1_des_dates = day_number_in_year(file["sen-1-des-products"][:])
            if 's2' in self.modalities:
                output["s2"]= torch.tensor(file["sen-2-data"][:])
                output["s2_dates"] = day_number_in_year(file["sen-2-products"][:], place=2)
                N = len(output["s2_dates"])
                if self.split == "train" and self.temporal_dropout > 0:
                    random_indices = torch.randperm(N)[:(int(N * (1 - self.temporal_dropout)))]
                    output["s2"] = output["s2"][random_indices]
                    output["s2_dates"] = output["s2_dates"][random_indices]

        if 's1-asc' in self.modalities:
            output["s1-asc"] = torch.load(self.path + "s1-asc/" + '.'.join(name.split('.')[:-1]) + ".pth")[:, :2, : ,:]
            N = len(output["s1-asc_dates"])
            if self.split == "train" and self.temporal_dropout > 0:
                random_indices = torch.randperm(N)[:(int(N * (1 - self.temporal_dropout)))]
                output["s1-asc"] = output["s1-asc"][random_indices]
                output["s1-asc_dates"] = output["s1-asc_dates"][random_indices]

        if 's1-des' in self.modalities:
            output["s1-des"] = torch.load(self.path + "s1-des/" + '.'.join(name.split('.')[:-1]) + ".pth")[:, :2, : ,:]
            N = len(output["s1-des_dates"])
            if self.split == "train" and self.temporal_dropout > 0:
                random_indices = torch.randperm(N)[:(int(N * (1 - self.temporal_dropout)))]
                output["s1-des"] = output["s1-des"][random_indices]
                output["s1-des_dates"] = output["s1-des_dates"][random_indices]

        if 's1' in self.modalities:
            s1_asc = torch.load(self.path + "s1-asc/" + '.'.join(name.split('.')[:-1]) + ".pth")[:, :2, : ,:]
            s1_des = torch.load(self.path + "s1-des/" + '.'.join(name.split('.')[:-1]) + ".pth")[:, :2, : ,:]
            output["s1"] = torch.cat([s1_asc, s1_des], dim=0)
            output["s1_dates"] = torch.cat([s1_asc_dates, s1_des_dates], dim=0)
            N = len(output["s1_dates"])
            if self.split == "train" and self.temporal_dropout > 0:
                random_indices = torch.randperm(N)[:(int(N * (1 - self.temporal_dropout)))]
                output["s1"] = output["s1"][random_indices]
                output["s1_dates"] = output["s1_dates"][random_indices]

        if "s1-mono" in self.modalities:
            with rasterio.open(self.path + "s1/60m/" + name) as f:
                numpy_array = f.read()
            numpy_array = numpy_array.astype(np.float32)
            output["s1-mono"] = torch.FloatTensor(numpy_array)
            if self.mono_strict:
                output["s1"] = output["s1-mono"][:2, :, :]

        if "s2-mono" in self.modalities:
            with rasterio.open(self.path + "s2/60m/" + name) as f:
                numpy_array = f.read()
            numpy_array = numpy_array.astype(np.float32)
            output["s2-mono"] = torch.FloatTensor(numpy_array)
            if self.mono_strict:
                output["s2"] = output["s2-mono"][:10, :, :]

        if "s2-4season-median" in self.modalities:
            with h5py.File(self.path + "sentinel/" + '.'.join(name.split('.')[:-1]) + ".h5", 'r') as file:
                output_inter = torch.tensor(file["sen-2-data"][:])
                dates = day_number_in_year(file["sen-2-products"][:], place=2)
            l = []
            for i in range (4):
                mask = ((dates >= 92 * i) & (dates < 92 * (i+1)))
                if sum(mask) > 0:
                    r, _ = torch.median(output_inter[mask], dim = 0)
                    l.append(r)
                else:
                    l.append(torch.zeros((output_inter.shape[1], output_inter.shape[-2], output_inter.shape[-1])))
            output["s2-4season-median"] = torch.cat(l)

        if "s2-median" in self.modalities:
            with h5py.File(self.path + "sentinel/" + '.'.join(name.split('.')[:-1]) + ".h5", 'r') as file:
                output["s2-median"], _ = torch.median(torch.tensor(file["sen-2-data"][:]), dim = 0)

        if "s1-4season-median" in self.modalities:
            with h5py.File(self.path + "sentinel/" + '.'.join(name.split('.')[:-1]) + ".h5", 'r') as file:
                dates = day_number_in_year(file["sen-1-asc-products"][:])
            output_inter = torch.load(self.path + "s1-asc/" + '.'.join(name.split('.')[:-1]) + ".pth")[:, :2, : ,:]
            l = []
            for i in range (4):
                mask = ((dates >= 92 * i) & (dates < 92 * (i+1)))
                if sum(mask) > 0:
                    r, _ = torch.median(output_inter[mask], dim = 0)
                    l.append(r)
                else:
                    l.append(torch.zeros((output_inter.shape[1], output_inter.shape[-2], output_inter.shape[-1])))
            output["s1-4season-median"] = torch.cat(l)

        if "s1-median" in self.modalities:
            output["s1-median"], _ = torch.median(torch.load(self.path + "s1-asc/" + '.'.join(name.split('.')[:-1]) + ".pth")[:, :2, : ,:], dim = 0)

        if "s1-mid" in self.modalities:
            images = torch.load(self.path + "s1-asc/" + '.'.join(name.split('.')[:-1]) + ".pth")[:, :2, : ,:]
            output["s1-mid"] = images[images.shape[0]//2]

        if "s2-mid" in self.modalities:
            with h5py.File(self.path + "sentinel/" + '.'.join(name.split('.')[:-1]) + ".h5", 'r') as file:
                images = torch.tensor(file["sen-2-data"][:])
                output["s2-mid"] = images[images.shape[0]//2]

        if self.norm is not None:
            for modality in self.modalities:
                if len(output[modality].shape) == 4:
                    output[modality] = (output[modality] - self.norm[modality][0][None, :, 
                                                    None, None]) / self.norm[modality][1][None, :, None, None]
                else:
                    output[modality] = (output[modality] - self.norm[modality][0][:, None, None]) / self.norm[modality][1][:, None, None]

        return self.transform(output)

    def __len__(self):
        return len(self.data_list)