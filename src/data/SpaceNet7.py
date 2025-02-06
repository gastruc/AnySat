import os
import json

import numpy as np
import torch
import rasterio

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

SN7_TRAIN = [
    'L15-0331E-1257N_1327_3160_13',
    'L15-0358E-1220N_1433_3310_13',
    'L15-0457E-1135N_1831_3648_13',
    'L15-0487E-1246N_1950_3207_13',
    'L15-0577E-1243N_2309_3217_13',
    'L15-0586E-1127N_2345_3680_13',
    'L15-0595E-1278N_2383_3079_13',
    'L15-0632E-0892N_2528_4620_13',
    'L15-0683E-1006N_2732_4164_13',
    'L15-0924E-1108N_3699_3757_13',
    'L15-1015E-1062N_4061_3941_13',
    'L15-1138E-1216N_4553_3325_13',
    'L15-1203E-1203N_4815_3378_13',
    'L15-1204E-1202N_4816_3380_13',
    'L15-1209E-1113N_4838_3737_13',
    'L15-1210E-1025N_4840_4088_13',
    'L15-1276E-1107N_5105_3761_13',
    'L15-1298E-1322N_5193_2903_13',
    'L15-1389E-1284N_5557_3054_13',
    'L15-1438E-1134N_5753_3655_13',
    'L15-1439E-1134N_5759_3655_13',
    'L15-1481E-1119N_5927_3715_13',
    'L15-1538E-1163N_6154_3539_13',
    'L15-1615E-1206N_6460_3366_13',
    'L15-1669E-1153N_6678_3579_13',
    'L15-1669E-1160N_6679_3549_13',
    'L15-1672E-1207N_6691_3363_13',
    'L15-1703E-1219N_6813_3313_13',
    'L15-1709E-1112N_6838_3742_13',
    'L15-1716E-1211N_6864_3345_13',
]
SN7_VAL = [
    'L15-0357E-1223N_1429_3296_13',
    'L15-0361E-1300N_1446_2989_13',
    'L15-0368E-1245N_1474_3210_13',
    'L15-0566E-1185N_2265_3451_13',
    'L15-0614E-0946N_2459_4406_13',
    'L15-0760E-0887N_3041_4643_13',
    'L15-1014E-1375N_4056_2688_13',
    'L15-1049E-1370N_4196_2710_13',
    'L15-1185E-0935N_4742_4450_13',
    'L15-1289E-1169N_5156_3514_13',
    'L15-1296E-1198N_5184_3399_13',
    'L15-1615E-1205N_6460_3370_13',
    'L15-1617E-1207N_6468_3360_13',
    'L15-1669E-1160N_6678_3548_13',
    'L15-1748E-1247N_6993_3202_13',
]
SN7_TEST = [
    'L15-0387E-1276N_1549_3087_13',
    'L15-0434E-1218N_1736_3318_13',
    'L15-0506E-1204N_2027_3374_13',
    'L15-0544E-1228N_2176_3279_13',
    'L15-0977E-1187N_3911_3441_13',
    'L15-1025E-1366N_4102_2726_13',
    'L15-1172E-1306N_4688_2967_13',
    'L15-1200E-0847N_4802_4803_13',
    'L15-1204E-1204N_4819_3372_13',
    'L15-1335E-1166N_5342_3524_13',
    'L15-1479E-1101N_5916_3785_13',
    'L15-1690E-1211N_6763_3346_13',
    'L15-1691E-1211N_6764_3347_13',
    'L15-1848E-0793N_7394_5018_13',
]

class SpaceNet7(Dataset):
    def __init__(
        self,
        path,
        modalities,
        transform,
        split: str = "train",
        norm_path = None,
        img_size = 256,
    ):
        """Initialize the SpaceNet7 dataset.

        Args:
            path (str): Path to the dataset.
            modalities (list): List of modalities to use (e.g., ["planet"]).
            transform (callable): A function/transform to apply to the data.
            split (str, optional): Split of the dataset ('train', 'val', 'test'). Defaults to 'train'.
            norm_path (str, optional): Path for normalization data. Defaults to None.
            img_size (int, optional): Size of the output images. Defaults to 256.
        """
        self.path = path
        self.modalities = modalities
        self.transform = transform
        self.sn7_img_size = 1024  # size of the SpaceNet 7 images
        self.img_size = img_size
        
        metadata_file = self.path + '/metadata_train.json'
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)

        self.split = split
        self.items = []

        if split == 'train':
            self.aoi_ids = list(SN7_TRAIN)
        elif split == 'val':
            self.aoi_ids = list(SN7_VAL)
        elif split == 'test':
            self.aoi_ids = list(SN7_TEST)
        else:
            raise Exception('Unkown split')

        # adding timestamps (only if label exists and not masked) for each AOI
        for aoi_id in self.aoi_ids:
            timestamps = list(self.metadata[aoi_id])
            for timestamp in timestamps:
                if not timestamp['mask'] and timestamp['label']:
                    item = {
                        'aoi_id': timestamp['aoi_id'],
                        'year': timestamp['year'],
                        'month': timestamp['month'],
                    }
                    self.items.append(dict(item))
                    # # tiling the timestamps
                    # for i in range(0, self.sn7_img_size, self.img_size):
                    #     for j in range(0, self.sn7_img_size, self.img_size):
                    #         item['i'] = i
                    #         item['j'] = j
                    #         self.items.append(dict(item))
        
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

    def __len__(self):
        return len(self.items)
    
    def compute_norm_vals(self, folder, sat):
        print("Computing norm values for {}".format(sat))
        means = []
        stds = []
        for i, b in enumerate(self.items):
            data = self.__getitem__(i)[sat]
            data = data.permute(1, 0, 2, 3)
            means.append(data.to(torch.float32).mean(dim=(1, 2, 3)).numpy())
            stds.append(data.to(torch.float32).std(dim=(1, 2, 3)).numpy())

        mean = np.stack(means).mean(axis=0).astype(float)
        std = np.stack(stds).mean(axis=0).astype(float)

        norm_vals = dict(mean=list(mean), std=list(std))

        with open(os.path.join(folder, "NORM_{}_patch.json".format(sat)), "w") as file:
            file.write(json.dumps(norm_vals, indent=4))
            
    def load_planet_mosaic(self, aoi_id: str, year: int, month: int) -> np.ndarray:
        folder = self.path + '/train/' + aoi_id + '/images_masked'
        file = folder + f'/global_monthly_{year}_{month:02d}_mosaic_{aoi_id}.tif'
        with rasterio.open(str(file), mode='r') as src:
            img = src.read(out_shape=(self.sn7_img_size, self.sn7_img_size), resampling=rasterio.enums.Resampling.nearest)
        # 4th band (last oen) is alpha band
        img = img[:-1]
        return img.astype(np.float32)

    def load_building_label(self, aoi_id: str, year: int, month: int) -> np.ndarray:
        folder = self.path + '/train/' + aoi_id + '/labels_raster'
        file = folder + f'/global_monthly_{year}_{month:02d}_mosaic_{aoi_id}_Buildings.tif'
        with rasterio.open(str(file), mode='r') as src:
            label = src.read(out_shape=(self.sn7_img_size, self.sn7_img_size), resampling=rasterio.enums.Resampling.nearest)
        label = (label > 0).squeeze()
        return label.astype(np.int64)

    def __getitem__(self, index):
        item = self.items[index]
        aoi_id, year, month = item['aoi_id'], int(item['year']), int(item['month'])
        
        image = self.load_planet_mosaic(aoi_id, year, month)
        target = self.load_building_label(aoi_id, year, month)

        # cut to tile
        #i, j = item['i'], item['j']
        # image = image[:, i:i + self.img_size, j:j + self.img_size]
        # target = target[i:i + self.img_size, j:j + self.img_size]

        image = torch.from_numpy(image)

        output = {"label": torch.from_numpy(target), "name": str(index)}
        output["planet"] = image#.unsqueeze(0)
        #output["planet_dates"] = torch.tensor([month * 30 + 15])

        if self.norm is not None:
            for modality in self.modalities:
                if len(output[modality].shape) == 4:
                    output[modality] = (output[modality] - self.norm[modality][0][None, :, 
                                                    None, None]) / self.norm[modality][1][None, :, None, None]
                else:
                    output[modality] = (output[modality] - self.norm[modality][0][:, None, None]) / self.norm[modality][1][:, None, None]

        return self.transform(output)
