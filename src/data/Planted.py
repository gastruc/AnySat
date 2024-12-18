from torch.utils.data import Dataset
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
import torch
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
    if 'name' in keys:
        output['name'] = [x['name'] for x in batch]
        keys.remove('name')
    for key in keys:
        output[key] = torch.stack([x[key] for x in batch])
    return output

def milliseconds_to_datetime(milliseconds):
    """Converts milliseconds since the Unix epoch to a datetime object.

    Args:
        milliseconds (int): The number of milliseconds since the Unix epoch.

    Returns:
        datetime.datetime: A datetime object representing the corresponding date and time.
    """
    return [datetime.fromtimestamp(m / 1000).timetuple().tm_yday for m in milliseconds]

class Planted(Dataset):
    def __init__(
        self,
        path,
        modalities,
        transform,
        split: str = "train",
        classes: list = [],
        partition: float = 1.,
        norm_path = None,
        temporal_dropout = 0,
        density_sampling = True,
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
        self.modalities = modalities
        self.data_path = path + split
        self.split = split
        self.temporal_dropout = temporal_dropout
        self.density_sampling = density_sampling
        with open(os.path.join(self.data_path, "labels2.json"), "r") as f:
            self.data = json.load(f)
        self.ids = list(self.data.keys())
        self.load_labels(classes)

        if self.split == "train" and self.density_sampling:
            label_density = self.labels.sum(axis=0) / len(self.labels)
            self.class_weights = np.sqrt(label_density)
        
            self.class_indices = {
                c: np.where(np.array(list(self.data.values())) == c)[0] 
                for c in range(len(classes))
            }

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
        for i, b in enumerate(self.ids):
            data = self.__getitem__(i)[sat]
            data = data.permute(3, 0, 1, 2)
            means.append(data.to(torch.float32).mean(dim=(1, 2, 3)).numpy())
            stds.append(data.to(torch.float32).std(dim=(1, 2, 3)).numpy())

        mean = np.stack(means).mean(axis=0).astype(float)
        std = np.stack(stds).mean(axis=0).astype(float)

        norm_vals = dict(mean=list(mean), std=list(std))

        with open(os.path.join(folder, "NORM_{}_patch.json".format(sat)), "w") as file:
            file.write(json.dumps(norm_vals, indent=4))
            
    def load_labels(self, classes):
        lis = torch.tensor(list(self.data.values()))

        self.labels = np.array(torch.nn.functional.one_hot(lis, num_classes=len(classes)))
        if self.partition < 1:
            self.ids, _, self.labels, _ = train_test_split(np.array(self.ids), self.labels, stratify = self.labels, test_size = 1. - self.partition)

    def __getitem__(self, i):
        """
        Returns an item from the dataset.
        Args:
            i (int): index of the item
        Returns:
            dict: dictionary with keys "label", "name" and the other corresponding to the modalities used
        """
        if self.split == "train" and self.density_sampling:
            class_idx = np.random.choice(
                len(self.class_weights),
                p=self.class_weights / self.class_weights.sum()
                )
            
            # Sample a random index from the chosen class
            i = np.random.choice(self.class_indices[class_idx])

        id = self.ids[i]
        output = {'label': torch.from_numpy(self.labels[i]).float()}

        for modality in self.modalities:
            output[modality] = torch.tensor(np.load(os.path.join(self.data_path, id, '_'.join([id, modality]) + ".npz")
                                                                            , allow_pickle=True)['arr_0']).permute(0, 3, 1, 2)
            if modality == "alos" or modality == "s1":
                ratio_band = output[modality][:, :1, :, :] / (output[modality][:, 1:2, :, :] + 1e-10)
                ratio_band = torch.clamp(ratio_band, max=1e4, min=-1e4)
                output[modality] = torch.cat((output[modality][:, :2, :, :], ratio_band), dim=1)
            output['_'.join([modality, "mask"])] = torch.tensor(np.load(os.path.join(self.data_path, id, 
                                '_'.join([id, modality, "mask"]) + ".npz"), allow_pickle=True)['arr_0']).permute(0, 3, 1, 2)
            output['_'.join([modality, "dates"])] = torch.tensor(milliseconds_to_datetime(np.load(os.path.join(self.data_path, id, 
                            '_'.join([id, modality, "timestamp"]) + ".npz"), allow_pickle=True)['arr_0']))
            N = len(output['_'.join([modality, "dates"])])
            if self.split == "train" and self.temporal_dropout > 0:
                if modality == "alos":
                    random_indices = torch.randperm(N)[:(int(N * 0.9))]
                else:
                    random_indices = torch.randperm(N)[:(int(N * (1 - self.temporal_dropout)))]
                output[modality] = output[modality][random_indices]
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
        return len(list(self.data.keys()))