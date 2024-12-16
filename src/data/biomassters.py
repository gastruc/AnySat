import numpy as np
import torch
import pandas as pd
import pathlib
import rasterio
import json
import os
from tifffile import imread
from os.path import join as opj
from tqdm import tqdm

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
    for key in ["s1", "s2"]:
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
    # keys = list(batch[0].keys())
    # output = {}
    # for key in keys:
    #      output[key] = torch.stack([x[key] for x in batch])
    return output

class BioMassters(Dataset):
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
        temporal_dropout: float = 0.0,
    ):
        """Initialize the BioMassters dataset.
        Link: https://huggingface.co/datasets/nascetti-a/BioMassters

        Args:
            path (str): Root path of the dataset.
            modalities (list): List of modalities to use (e.g. ['s1', 's2']).
            transform: Transform to apply to the data.
            split (str, optional): Split of the dataset (train, val, test). Defaults to "train".
            partition (float, optional): Fraction of data to use. Defaults to 1.0.
            norm_path (str, optional): Path to normalization values. Defaults to None.
            ignore_index (int, optional): Index to ignore for metrics and loss. Defaults to -1.
            num_classes (int, optional): Number of classes. Defaults to 2.
        """
        self.path = path
        self.modalities = modalities
        self.transform = transform
        self.split = split
        self.img_size = 256
        self.temporal_dropout = temporal_dropout

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        self.data_path = pathlib.Path(self.path).joinpath(f"{split}_Data_list.csv")
        self.id_list = list(pd.read_csv(self.data_path)['chip_id'])
        
        self.split_path = 'train' if split == 'val' else split
        self.dir_features = pathlib.Path(self.path).joinpath(f'{self.split_path}_features')
        self.dir_labels = pathlib.Path(self.path).joinpath( f'{self.split_path}_agbm')
        
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
            indices, _ = self.balance_reg_indices(strategy="stratified", label_fraction=partition, num_bins=3)
            with open('/home/GAstruc/PhD/' + split + '_indices.json', 'w') as file:
                json.dump([self.id_list[i] for i in indices], file)
            self.id_list = [self.id_list[i] for i in indices]
        
    def __len__(self):
        return len(self.id_list)
    
    def compute_norm_vals(self, folder, sat):
        means = []
        stds = []
        for i, b in enumerate(self.id_list):
            if i%1000 == 0:
                print(i)
            data = self.__getitem__(i)[sat]
            data = data.permute(1, 0, 2, 3)
            means.append(data.to(torch.float32).mean(dim=(1, 2, 3)).numpy())
            stds.append(data.to(torch.float32).std(dim=(1, 2, 3)).numpy())

        mean = np.stack(means).mean(axis=0).astype(float)
        std = np.stack(stds).mean(axis=0).astype(float)

        norm_vals = dict(mean=list(mean), std=list(std))

        with open(os.path.join(folder, "NORM_{}_patch.json".format(sat)), "w") as file:
            file.write(json.dumps(norm_vals, indent=4))

    def __getitem__(self, index, data_only=False):
        # output = {}
        # chip_id = self.id_list[index]
        # fname = str(chip_id)+'_agbm.tif'
        # if not data_only:
        #     output['embeddings'] = torch.from_numpy(np.load('/home/GAstruc/PhD/PhD/data/BioMassters/embeddings/dense_x_00a0d9d4.npy'))
        # with rasterio.open(self.dir_labels.joinpath(fname)) as lbl:
        #     target = lbl.read(1)
        # target = np.nan_to_num(target)

        # target = torch.from_numpy(target).float()
        # output['label'] = target
        # return output
        
        chip_id = self.id_list[index]
        fname = str(chip_id)+'_agbm.tif'
        
        imgs_s1, imgs_s2, s1_dates, s2_dates = [], [], [], []
        month_list = list(range(12))
        
        for month in month_list:
            
            s1_fname = '%s_%s_%02d.tif' % (str.split(fname, '_')[0], 'S1', month)
            s2_fname = '%s_%s_%02d.tif' % (str.split(fname, '_')[0], 'S2', month)

            s1_filepath = self.dir_features.joinpath(s1_fname)
            if s1_filepath.exists():
                try:
                    img_s1 = imread(s1_filepath)
                    m = img_s1 == -9999
                    img_s1 = img_s1.astype('float32')
                    img_s1 = np.where(m, 0, img_s1)
                    H, W = img_s1.shape[:2]
                    img_s1 = img_s1.reshape(H, W, 2, 2)
                    img_s1 = np.transpose(img_s1, (2, 3, 0, 1))
                    imgs_s1.append(torch.from_numpy(img_s1).float())
                    s1_dates.append(month * 30 + 15)
                    s1_dates.append(month * 30 + 15)
                except (ValueError, OSError) as e:
                    continue
            
            s2_filepath = self.dir_features.joinpath(s2_fname)
            if s2_filepath.exists():
                try:
                    img_s2 = imread(s2_filepath)[:, :, :-1]
                    img_s2 = img_s2.astype('float32')
                    img_s2 = np.transpose(img_s2, (2, 0, 1))
                    imgs_s2.append(img_s2)
                    s2_dates.append(month * 30 + 15)
                except (ValueError, OSError) as e:
                    # Skip problematic S2 image silently
                    continue

        imgs_s1 = torch.cat(imgs_s1, dim=0)
        ratio_band = imgs_s1[:, :1, :, :] / (imgs_s1[:, 1:, :, :] + 1e-10)
        ratio_band = torch.clamp(ratio_band, max=1e4, min=-1e4)
        imgs_s1 = torch.cat((imgs_s1[:, :2, :, :], ratio_band), dim=1)
        imgs_s2 = np.stack(imgs_s2, axis=0)
        imgs_s2 = torch.from_numpy(imgs_s2).float()
        
        s1_dates = torch.tensor(s1_dates)
        s2_dates = torch.tensor(s2_dates)
        
        with rasterio.open(self.dir_labels.joinpath(fname)) as lbl:
            target = lbl.read(1)
        target = np.nan_to_num(target)

        target = torch.from_numpy(target).float()

        output = {
            'name': chip_id,
            's2': imgs_s2,
            's1' : imgs_s1,
            's2_dates': s2_dates,
            's1_dates': s1_dates,
            'label': target,
        }
        
        if self.split == "train" and self.temporal_dropout > 0:
            N = len(output["s1_dates"])
            random_indices = torch.randperm(N)[:(int(N * (1 - self.temporal_dropout)))]
            output["s1"] = output["s1"][random_indices]
            output["s1_dates"] = output["s1_dates"][random_indices]
            N = len(output["s2_dates"])
            random_indices = torch.randperm(N)[:(int(N * (1 - self.temporal_dropout)))]
            output["s2"] = output["s2"][random_indices]
            output["s2_dates"] = output["s2_dates"][random_indices]
        
        if self.norm is not None:
            for modality in self.modalities:
                if len(output[modality].shape) == 4:
                    output[modality] = (output[modality] - self.norm[modality][0][None, :, 
                                                    None, None]) / self.norm[modality][1][None, :, None, None]
                else:
                    output[modality] = (output[modality] - self.norm[modality][0][:, None, None]) / self.norm[modality][1][:, None, None]

        return self.transform(output)
    
    # Calculate image-wise distributions for regression
    def calculate_regression_distributions(self):
        distributions = []

        for idx in tqdm(range(self.__len__()), desc="Calculating regression distributions per sample"):
            target = self.__getitem__(idx, data_only=True)['label']
            mean_value = target.mean().item()  # Example for patch-wise mean; adjust as needed for other metrics
            distributions.append(mean_value)

        return np.array(distributions)

    # Function to bin regression distributions
    def bin_regression_distributions(self,regression_distributions, num_bins=3):
        # Define the range for binning based on minimum and maximum values in regression distributions
        binned_distributions = np.digitize(
            regression_distributions, 
            np.linspace(regression_distributions.min(), regression_distributions.max(), num_bins + 1)
        ) - 1
        return binned_distributions


    def balance_reg_indices(
            self, 
            strategy, 
            label_fraction=1.0, 
            num_bins=3):

        """
        Balances and selects indices from a regression dataset based on the specified strategy.

        Args:
        dataset : GeoFMDataset | GeoFMSubset
            The dataset from which to select indices, typically containing geospatial regression data.
        
        strategy : str
            The strategy to use for selecting indices. Options include:
            - "stratified": Proportionally selects indices from each bin based on the binned regression distributions.
            - "oversampled": Prioritizes and selects indices from bins with lower representation.
        
        label_fraction : float, optional, default=1.0
            The fraction of indices to select from each bin. Values should be between 0 and 1.
        
        num_bins : int, optional, default=3
            The number of bins to divide the regression distributions into, used for stratification or oversampling.
        ------
        
        Returns:
        selected_idx : list of int
            The indices of the selected samples based on the strategy and label fraction.

        other_idx : list of int
            The remaining indices that were not selected.

        """

        regression_distributions = self.calculate_regression_distributions()
        binned_distributions = self.bin_regression_distributions(regression_distributions, num_bins=num_bins)

        indices_per_bin = {i: [] for i in range(num_bins)}

        # Populate the indices per bin
        for index, bin_index in enumerate(binned_distributions):
            if bin_index in indices_per_bin:
                indices_per_bin[bin_index].append(index)
        
        if strategy == "stratified":
            # Select fraction of indices from each bin
            selected_idx = []
            for bin_index, indices in indices_per_bin.items():
                num_to_select = int(max(1, len(indices) * label_fraction))  # Ensure at least one index is selected
                selected_idx.extend(np.random.choice(indices, num_to_select, replace=False))
        elif strategy == "oversampled":
            # Prioritize bins with underrepresented values (e.g., high biomass samples)
            sorted_indices = np.argsort(binned_distributions)
            selected_idx = sorted_indices[:int(self.__len__() * label_fraction)]

        other_idx = list(set(range(self.__len__())) - set(selected_idx))

        return selected_idx, other_idx
