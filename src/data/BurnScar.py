import os

import numpy as np
import torch
import tifffile as tiff
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from glob import glob
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
    for key in keys:
        output[key] = torch.stack([x[key] for x in batch])
    return output

class BurnScar(Dataset):
    def __init__(
        self,
        path,
        modalities,
        transform,
        split: str = "train",
        norm_path = None,
    ):
        """Initialize the BurnScar dataset.

        Args:
            path (str): Path to the dataset.
            modalities (list): List of modalities to use.
            transform (callable): A function/transform to apply to the data.
            split (str, optional): Split of the dataset ('train', 'val', 'test'). Defaults to 'train'.
            norm_path (str, optional): Path for normalization data. Defaults to None.
        """
        self.path = path
        self.modalities = modalities
        self.transform = transform
        self.split = split

        self.split_mapping = {
            "train": "training",
            "val": "training",
            "test": "validation",
        }

        all_files = sorted(
            glob(
                os.path.join(
                    self.path, self.split_mapping[self.split], "*merged.tif"
                )
            )
        )
        all_targets = sorted(
            glob(
                os.path.join(
                    self.path, self.split_mapping[self.split], "*mask.tif"
                )
            )
        )

        if self.split != "test":
            split_indices = self.get_train_val_split(all_files)
            if self.split == "train":
                indices = split_indices["train"]
            else:
                indices = split_indices["val"]
            self.image_list = [all_files[i] for i in indices]
            self.target_list = [all_targets[i] for i in indices]
        else:
            self.image_list = all_files
            self.target_list = all_targets
            
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


    @staticmethod
    def get_train_val_split(all_files):
        # Fixed stratified sample to split data into train/val.
        # This keeps 90% of datapoints belonging to an individual event in the training set and puts the remaining 10% in the validation set.
        train_idxs, val_idxs = train_test_split(
            np.arange(len(all_files)),
            test_size=0.1,
            random_state=23,
        )
        return {"train": train_idxs, "val": val_idxs}

    def __len__(self):
        return len(self.image_list)
    
    def compute_norm_vals(self, folder, sat):
        means = []
        stds = []
        for i, b in enumerate(self.image_list):
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
        image = tiff.imread(self.image_list[index])
        image = image.astype(np.float32)  # Convert to float32
        image = torch.from_numpy(image).permute(2, 0, 1)

        target = tiff.imread(self.target_list[index])
        target = target.astype(np.int64)  # Convert to int64 (since it's a mask)
        target = torch.from_numpy(target).long()

        invalid_mask = image == 9999
        image[invalid_mask] = 0

        output = {
            "name": self.image_list[index],
            "hls": image.unsqueeze(0),
            "hls_dates": torch.tensor([0]),
            "label": target,
        }
        
        if self.norm is not None:
            for modality in self.modalities:
                if len(output[modality].shape) == 4:
                    output[modality] = (output[modality] - self.norm[modality][0][None, :, 
                                                    None, None]) / self.norm[modality][1][None, :, None, None]
                else:
                    output[modality] = (output[modality] - self.norm[modality][0][:, None, None]) / self.norm[modality][1][:, None, None]

        return self.transform(output)

    