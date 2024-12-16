# Source: https://github.com/cloudtostreet/Sen1Floods11

import os

import geopandas
import numpy as np
import pandas as pd
import rasterio
import torch
from tqdm import tqdm
import json

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

class Sen1Floods11(Dataset):
    def __init__(
        self,
        path,
        modalities,
        transform,
        split: str = "train",
        partition: float = 1.,
        norm_path = None,
        ignore_index: int = -1,
        num_classes: int = 2,
    ):
        """Initialize the Sen1Floods11 dataset.
        Link: https://github.com/cloudtostreet/Sen1Floods11

        Args:
            path (str): Path to the dataset.
            modalities (list): List of modalities to use.
            transform (callable): A function/transform to apply to the data.
            split (str, optional): Split of the dataset ('train', 'val', 'test'). Defaults to 'train'.
            partition (float, optional): Partition of the dataset to use. Defaults to 1.0.
            norm_path (str, optional): Path for normalization data. Defaults to None.
            ignore_index (int, optional): Index to ignore for metrics and loss. Defaults to -1.
        """
        self.path = path
        self.modalities = modalities
        self.transform = transform
        self.split = split
        self.num_classes = num_classes

        self.ignore_index = ignore_index

        self.split_mapping = {"train": "train", "val": "valid", "test": "test"}

        split_file = os.path.join(
            self.path,
            "v1.1",
            f"splits/flood_handlabeled/flood_{self.split_mapping[split]}_data.csv",
        )
        metadata_file = os.path.join(
            self.path, "v1.1", "Sen1Floods11_Metadata.geojson"
        )
        data_root = os.path.join(
            self.path, "v1.1", "data/flood_events/HandLabeled/"
        )

        self.metadata = geopandas.read_file(metadata_file)

        with open(split_file) as f:
            file_list = f.readlines()

        file_list = [f.rstrip().split(",") for f in file_list]

        self.s1_image_list = [
            os.path.join(data_root, "S1Hand", f[0]) for f in file_list
        ]
        self.s2_image_list = [
            os.path.join(data_root, "S2Hand", f[0].replace("S1Hand", "S2Hand"))
            for f in file_list
        ]
        self.target_list = [
            os.path.join(data_root, "LabelHand", f[1]) for f in file_list
        ]
        
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
        
        if partition < 1:
            indices, _ = self.balance_seg_indices(strategy="stratified", label_fraction=partition, num_bins=3)
            self.s1_image_list = [self.s1_image_list[i] for i in indices]
            self.s2_image_list = [self.s2_image_list[i] for i in indices]
            self.target_list = [self.target_list[i] for i in indices]   

    def __len__(self):
        return len(self.s1_image_list)
    
    def compute_norm_vals(self, folder, sat):
        means = []
        stds = []
        for i, b in enumerate(self.s1_image_list):
            data = self.__getitem__(i)[sat]
            data = data.permute(1, 0, 2, 3)
            means.append(data.to(torch.float32).mean(dim=(1, 2, 3)).numpy())
            stds.append(data.to(torch.float32).std(dim=(1, 2, 3)).numpy())

        mean = np.stack(means).mean(axis=0).astype(float)
        std = np.stack(stds).mean(axis=0).astype(float)

        norm_vals = dict(mean=list(mean), std=list(std))

        with open(os.path.join(folder, "NORM_{}_patch.json".format(sat)), "w") as file:
            file.write(json.dumps(norm_vals, indent=4))

    def _get_date(self, index):
        file_name = self.s2_image_list[index]
        location = os.path.basename(file_name).split("_")[0]
        if self.metadata[self.metadata["location"] == location].shape[0] != 1:
            s2_date = pd.to_datetime("01-01-1998", dayfirst=True)
            s1_date = pd.to_datetime("01-01-1998", dayfirst=True)
        else:
            s2_date = pd.to_datetime(
                self.metadata[self.metadata["location"] == location]["s2_date"].item()
            )
            s1_date = pd.to_datetime(
                self.metadata[self.metadata["location"] == location]["s1_date"].item()
            )
        return torch.tensor([s2_date.dayofyear]), torch.tensor([s1_date.dayofyear])

    def __getitem__(self, index):
        with rasterio.open(self.s2_image_list[index]) as src:
            s2_image = src.read()

        with rasterio.open(self.s1_image_list[index]) as src:
            s1_image = src.read()
            # Convert the missing values (clouds etc.)
            s1_image = np.nan_to_num(s1_image)

        with rasterio.open(self.target_list[index]) as src:
            target = src.read(1)

        timestamp = self._get_date(index)

        s2_image = torch.from_numpy(s2_image).float()[[1,2,3,4,5,6,7,8,11,12]]
        s1_image = torch.from_numpy(s1_image).float()
        ratio_band = s1_image[:1, :, :] / (s1_image[1:, :, :] + 1e-10)
        ratio_band = torch.clamp(ratio_band, max=1e4, min=-1e4)
        s1_image = torch.cat((s1_image[:2, :, :], ratio_band), dim=0)
        target = torch.from_numpy(target).long()

        output = {
            "s2": s2_image.unsqueeze(0),
            "s1": s1_image.unsqueeze(0),
            "label": target,
            "s2_dates": timestamp[0],
            "s1_dates": timestamp[1],
        }
        
        if self.norm is not None:
            for modality in self.modalities:
                if len(output[modality].shape) == 4:
                    output[modality] = (output[modality] - self.norm[modality][0][None, :, 
                                                    None, None]) / self.norm[modality][1][None, :, None, None]
                else:
                    output[modality] = (output[modality] - self.norm[modality][0][:, None, None]) / self.norm[modality][1][:, None, None]

        return self.transform(output)

    # Calculate image-wise class distributions for segmentation
    def calculate_class_distributions(self):
        num_classes = self.num_classes
        ignore_index = self.ignore_index
        class_distributions = []

        for idx in tqdm(range(self.__len__()), desc="Calculating class distributions per sample"):
            target = self[idx]['label']

            if ignore_index is not None:
                target=target[(target != ignore_index)]

            total_pixels = target.numel()
            if total_pixels == 0:
                class_distributions.append([0] * num_classes)
                continue
            else:
                class_counts = [(target == i).sum().item() for i in range(num_classes)]
                class_ratios = [count / total_pixels for count in class_counts]
                class_distributions.append(class_ratios)

        return np.array(class_distributions)

    # Function to bin class distributions using ceil
    def bin_class_distributions(self, class_distributions, num_bins=3, logger=None):
        bin_edges = np.linspace(0, 1, num_bins + 1)[1]
        binned_distributions = np.ceil(class_distributions / bin_edges).astype(int) - 1
        return binned_distributions

    def balance_seg_indices(
            self, 
            strategy, 
            label_fraction=1.0, 
            num_bins=3, 
            logger=None):
        """
        Balances and selects indices from a segmentation dataset based on the specified strategy.

        Args:
        dataset : GeoFMDataset | GeoFMSubset
            The dataset from which to select indices, typically containing geospatial segmentation data.
        
        strategy : str
            The strategy to use for selecting indices. Options include:
            - "stratified": Proportionally selects indices from each class bin based on the class distribution.
            - "oversampled": Prioritizes and selects indices from bins with lower class representation.
        
        label_fraction : float, optional, default=1.0
            The fraction of labels (indices) to select from each class or bin. Values should be between 0 and 1.
        
        num_bins : int, optional, default=3
            The number of bins to divide the class distributions into, used for stratification or oversampling.
        
        logger : object, optional
            A logger object for tracking progress or logging messages (e.g., `logging.Logger`)

        ------
        
        Returns:
        selected_idx : list of int
            The indices of the selected samples based on the strategy and label fraction.

        other_idx : list of int
            The remaining indices that were not selected.

        """
        # Calculate class distributions with progress tracking
        class_distributions = self.calculate_class_distributions()

        # Bin the class distributions
        binned_distributions = self.bin_class_distributions(class_distributions, num_bins=num_bins, logger=logger)
        combined_bins = np.apply_along_axis(lambda row: ''.join(map(str, row)), axis=1, arr=binned_distributions)

        indices_per_bin = {}
        for idx, bin_id in enumerate(combined_bins):
            if bin_id not in indices_per_bin:
                indices_per_bin[bin_id] = []
            indices_per_bin[bin_id].append(idx)

        if strategy == "stratified":
            # Select a proportion of indices from each bin   
            selected_idx = []
            for bin_id, indices in indices_per_bin.items():
                num_to_select = int(max(1, len(indices) * label_fraction))  # Ensure at least one index is selected
                selected_idx.extend(np.random.choice(indices, num_to_select, replace=False))
        elif strategy == "oversampled":
            # Prioritize the bins with the lowest values
            sorted_indices = np.argsort(combined_bins)
            selected_idx = sorted_indices[:int(len(dataset) * label_fraction)]

        # Determine the remaining indices not selected
        other_idx = list(set(range(self.__len__())) - set(selected_idx))

        return selected_idx, other_idx