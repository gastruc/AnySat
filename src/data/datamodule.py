import lightning as L
from torch.utils.data import DataLoader, Sampler, DistributedSampler
import random
import time
from typing import List, Tuple, Any, Iterator, Dict, Union, Optional
import torch.distributed as dist

class DataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        global_batch_size,
        num_workers,
        num_nodes=1,
        num_devices=1,
        stop_iteration_train=None,
        stop_iteration_val=None,
    ):
        super().__init__()
        self._builders = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }
        self.num_workers = num_workers
        self.stop_iteration_train = stop_iteration_train
        self.stop_iteration_val = stop_iteration_val
        self.batch_size = global_batch_size // (num_nodes * num_devices)
        print(f"Each GPU will receive {self.batch_size} images")
        self.save_hyperparameters(logger=False)

    @property
    def num_classes(self):
        if hasattr(self, "train_dataset"):
            return self.train_dataset.num_classes
        else:
            return self._builders["train"]().num_classes

    def setup(self, stage=None):
        """Setup the datamodule.
        Args:
            stage (str): stage of the datamodule
                Is be one of "fit" or "test" or None
        """
        print("Stage", stage)
        start_time = time.time()
        if stage == "fit" or stage is None:
            self.train_dataset = self._builders["train"]()
            self.val_dataset = self._builders["val"]()
            print(f"Train dataset size: {len(self.train_dataset)}")
            print(f"Val dataset size: {len(self.val_dataset)}")
        else:
            self.test_dataset = self._builders["test"]()
            print(f"Test dataset size: {len(self.test_dataset)}")
        end_time = time.time()
        print(f"Setup took {(end_time - start_time):.2f} seconds")

    def train_dataloader(self):
        return DataLoaderStop(
            dataset=self.train_dataset,
            stop_iteration=self.stop_iteration_train,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoaderStop(
            dataset=self.val_dataset,
            stop_iteration=self.stop_iteration_val,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset.collate_fn,
        )

    def test_dataloader(self):
        return DataLoaderStop(
            dataset=self.test_dataset,
            stop_iteration=None,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=self.test_dataset.collate_fn,
        )
    
class DataModuleMulti(L.LightningDataModule):
    def __init__(
        self,
        train_dataset: Any,
        val_dataset: Any,
        test_dataset: Any,
        global_batch_size: Union[int, Dict[str, int]],
        num_workers: int,
        num_nodes: int = 1,
        num_devices: int = 1,
        stop_iteration_train: Optional[int] = None,
        stop_iteration_val: Optional[int] = None,
        weights_datasets: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self._builders = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }
        self.num_workers = num_workers
        self.stop_iteration_train = stop_iteration_train
        self.stop_iteration_val = stop_iteration_val
        self.batch_size = global_batch_size
        self.save_hyperparameters(logger=False)
        self.weights_datasets = weights_datasets

    @property
    def num_classes(self):
        if hasattr(self, "train_dataset"):
            return self.train_dataset.num_classes
        else:
            return self._builders["train"]().num_classes

    def setup(self, stage=None):
        """Setup the datamodule.
        Args:
            stage (str): stage of the datamodule
                Is be one of "fit" or "test" or None
        """
        print("Stage", stage)
        start_time = time.time()
        if stage == "fit" or stage is None:
            self.train_dataset = self._builders["train"]()
            self.val_dataset = self._builders["val"]()
            print(f"Train dataset size: {len(self.train_dataset)}")
            print(f"Val dataset size: {len(self.val_dataset)}")
        else:
            self.test_dataset = self._builders["test"]()
            print(f"Test dataset size: {len(self.test_dataset)}")
        end_time = time.time()
        print(f"Setup took {(end_time - start_time):.2f} seconds")

    def on_train_epoch_start(self) -> None:
        """Lightning hook that is called when a training epoch begins."""
        self.current_epoch = self.trainer.current_epoch

    def train_dataloader(self):
        dls = {}
        for dataset in self.train_dataset.datasets:
            if dist.is_initialized():
                sampler = DistributedSampler(
                    self.train_dataset.datasets_modules[dataset],
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=True
                )
                dls[dataset] = DataLoader(
                    self.train_dataset.datasets_modules[dataset],
                    batch_size=self.batch_size[dataset],
                    pin_memory=True,
                    drop_last=True,
                    num_workers=self.num_workers,
                    collate_fn=self.train_dataset.collate_fn[dataset],
                    sampler=sampler
                )
            else:
                dls[dataset] = DataLoader(
                    self.train_dataset.datasets_modules[dataset],
                    batch_size=self.batch_size[dataset],
                    pin_memory=True,
                    drop_last=True,
                    num_workers=self.num_workers,
                    collate_fn=self.train_dataset.collate_fn[dataset],
                    shuffle=True
                )
        return MultiBatchDataLoader(dataloaders=dls, cycle=False, scales=self.train_dataset.scales, 
            stop_iteration=self.stop_iteration_train, weights_datasets=self.weights_datasets)

    def val_dataloader(self):
        dls = {}
        for dataset in self.val_dataset.datasets:
            if dist.is_initialized():
                sampler = DistributedSampler(
                    self.val_dataset.datasets_modules[dataset],
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=False
                )
                dls[dataset] = DataLoader(
                    self.val_dataset.datasets_modules[dataset],
                    batch_size=self.batch_size[dataset],
                    pin_memory=True,
                    drop_last=True,
                    num_workers=self.num_workers,
                    collate_fn=self.val_dataset.collate_fn[dataset],
                    sampler=sampler
                )
            else:
                dls[dataset] = DataLoader(
                    self.val_dataset.datasets_modules[dataset],
                    batch_size=self.batch_size[dataset],
                    pin_memory=True,
                    drop_last=True,
                    num_workers=self.num_workers,
                    collate_fn=self.val_dataset.collate_fn[dataset],
                    shuffle=False
                )
        return MultiBatchDataLoader(dataloaders=dls, cycle=False, scales=self.val_dataset.scales, 
            stop_iteration=self.stop_iteration_val, weights_datasets=self.weights_datasets)

    def test_dataloader(self):
        dls = {}
        for dataset in self.test_dataset.datasets:
            if dist.is_initialized():
                sampler = DistributedSampler(
                    self.test_dataset.datasets_modules[dataset],
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=False
                )
                dls[dataset] = DataLoader(
                    self.test_dataset.datasets_modules[dataset],
                    batch_size=self.batch_size[dataset],
                    pin_memory=True,
                    drop_last=False,
                    num_workers=self.num_workers,
                    collate_fn=self.test_dataset.collate_fn[dataset],
                    sampler=sampler
                )
            else:
                dls[dataset] = DataLoader(
                    self.test_dataset.datasets_modules[dataset],
                    batch_size=self.batch_size[dataset],
                    pin_memory=True,
                    drop_last=False,
                    num_workers=self.num_workers,
                    collate_fn=self.test_dataset.collate_fn[dataset],
                    shuffle=False
                )
        return MultiBatchDataLoader(dataloaders=dls, cycle=False, scales=self.test_dataset.scales, 
            stop_iteration=None, weights_datasets=self.weights_datasets)

class MultiBatchDataLoader(DataLoader):
    def __init__(self, dataloaders: Dict[str, DataLoader], scales: Dict[str, List[int]], cycle: bool = True, 
        stop_iteration = None, weights_datasets = None, **kwargs):
        if not dataloaders:
            raise ValueError("dataloaders dictionary cannot be empty")
        
        if not all(isinstance(dl, DataLoader) for dl in dataloaders.values()):
            raise TypeError("All values in dataloaders must be DataLoader instances")
            
        if set(scales.keys()) != set(dataloaders.keys()):
            raise ValueError("scales and dataloaders must have matching keys")
        
        self.dataloaders = dataloaders
        self.scales = scales
        self.cycle = cycle
        self.stop_iteration = stop_iteration
        self.dataset_names = list(dataloaders.keys())
        self.weights_datasets = weights_datasets if weights_datasets is not None else {name: 1 for name in self.dataset_names}

        # Cache length calculations
        self._max_length = max(len(dataloader) for dataloader in self.dataloaders.values())
        self._sum_length = sum(len(dataloader) for dataloader in self.dataloaders.values())
        
        if self.cycle:
            len_data = self._max_length * len(self.dataloaders)
        else:
            len_data = self._sum_length
        self.length = len_data if self.stop_iteration is None else min(len_data, self.stop_iteration)

        self.iterators = {}
        self.exhausted_loaders = set()
        self.epoch = 0  # Add epoch counter
        
        super().__init__(dataset=range(self.length), batch_size=1, shuffle=False)

        # Set up distributed training parameters
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        if dist.is_initialized():
            self.base_seed = random.randint(0, 2**32-1)
            dist.broadcast_object_list([self.base_seed], src=0)
        else:
            self.base_seed = random.randint(0, 2**32-1)

    def __iter__(self):
        self.epoch += 1
        
        # Set epoch for distributed samplers
        if dist.is_initialized():
            for dataloader in self.dataloaders.values():
                if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
                    dataloader.sampler.set_epoch(self.epoch)
        
        # Reset iterators and state
        self.iterators = {name: iter(dataloader) for name, dataloader in self.dataloaders.items()}
        self.exhausted_loaders = set()
            
        return self

    def __next__(self):
        if not self.cycle and len(self.exhausted_loaders) == len(self.dataloaders):
            raise StopIteration

        # Simple weighted random sampling
        available_datasets = [name for name in self.dataset_names if name not in self.exhausted_loaders]
        if not available_datasets:
            if self.cycle:
                self.exhausted_loaders.clear()
                self.iterators = {name: iter(dataloader) for name, dataloader in self.dataloaders.items()}
                available_datasets = self.dataset_names
            else:
                raise StopIteration
                
        dataset_name = random.choice([dataset for dataset in available_datasets for _ in range (self.weights_datasets[dataset])])

        try:
            batch = next(self.iterators[dataset_name])
            return self.add_metadata(batch, dataset_name)
        except StopIteration:
            self.exhausted_loaders.add(dataset_name)
            return self.__next__()

    def __len__(self):
        return self.length  # Use cached length

    def add_metadata(self, batch, dataset_name):
        scale = random.choice(self.scales[dataset_name])
        if isinstance(batch, dict):
            batch["dataset"] = dataset_name
            batch["scale"] = scale
        elif isinstance(batch, (list, tuple)):
            batch = (*batch, dataset_name, scale)
        return batch

    def __del__(self):
        # Clean up iterators
        for iterator in self.iterators.values():
            del iterator
        self.iterators.clear()

class DataLoaderStop(DataLoader):
    """DataLoader that stops after stop_iteration batches."""
    def __init__(self, stop_iteration=None, **kwargs):
        self.stop_iteration = stop_iteration
        super().__init__(**kwargs)

    def __len__(self):
        len_data = super().__len__()
        if self.stop_iteration is not None and len_data > self.stop_iteration:
            return self.stop_iteration
        return super().__len__()

