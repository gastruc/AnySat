from torch.utils.data import Dataset
import hydra

class MultiDataset(Dataset):
    def __init__(
        self,
        datasets,
        scales
        ):
        """
        Initializes the dataset.
        Args:
            datasets (dict): dictionary with keys name of the dataset and values the corresponding dataset modules
            scales (dict): dictionary with keys name of the dataset and values the list of possible scales to sample
        """
        self.datasets_modules = datasets
        self.datasets = list(datasets.keys())
        self.collate_fn = {dataset: self.datasets_modules[dataset].collate_fn for dataset in self.datasets}
        self.scales = scales

    def __len__(self):
        return sum([self.datasets_modules[dataset].__len__() for dataset in self.datasets])
