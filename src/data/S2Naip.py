from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import json
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torch
import rasterio
from datetime import datetime
import glob
import cv2

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
    for key in ["s2", "l8", "s1"]:
        if key in keys:
            idx = [x[key] for x in batch]
            max_size_0 = max(tensor.size(0) for tensor in idx)
            stacked_tensor = torch.stack([
                    torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, 0, 0, max_size_0 - tensor.size(0)))
                    for tensor in idx
                ], dim=0)
            output[key] = stacked_tensor.float()
            keys.remove(key)
            key = '_'.join([key, "dates"])
            idx = [x[key] for x in batch]
            max_size_0 = max(tensor.size(0) for tensor in idx)
            stacked_tensor = torch.stack([
                    torch.nn.functional.pad(tensor, (0, max_size_0 - tensor.size(0)))
                    for tensor in idx
                ], dim=0)
            output[key] = stacked_tensor.long()
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

def map_category_to_meta(category):
    mapping = {
        # Buildings and Structures
        'building': 1,
        'flagpole': 1,
        'lighthouse': 1,
        'obelisk': 1,
        'observatory': 1,
        
        # Transportation Infrastructure
        'aerialway_pylon': 2,
        'airport': 2,
        'gas_station': 2,
        'helipad': 2,
        'parking': 2,
        'road': 2,
        'runway': 2,
        'taxiway': 2,
        
        # Industrial and Energy Infrastructure (chimneys go here)
        'chimney': 3,
        'petroleum_well': 3,
        'power_plant': 3,
        'power_substation': 3,
        'power_tower': 3,
        'satellite_dish': 3,
        'silo': 3,
        'storage_tank': 3,
        'wind_turbine': 3,
        'works': 3,
        
        # Water Features
        'river': 4,
        'fountain': 4,
        
        # Other
        'leisure': 5,
    }
    return mapping.get(category, 5)

def process_geojson_file(path):
    image = np.zeros((512, 512), dtype=np.uint8)
    
    with open(path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            for feature in data.get('features', []):
                properties = feature.get('properties', {})
                category = properties.get('category')
                if not category:
                    continue  # Skip if no category
                
                # Map category to meta category value
                meta_value = map_category_to_meta(category)
                
                geometry = feature.get('geometry', {})
                geom_type = geometry.get('type')
                coordinates = geometry.get('coordinates', [])
                
                if geom_type == 'LineString':
                    # Draw lines on the image
                    points = np.array(coordinates, dtype=np.int32)
                    # Ensure points are within image bounds
                    points = np.clip(points, 0, 511)
                    # Draw the line with thickness 1
                    cv2.polylines(image, [points], isClosed=False, color=meta_value, thickness=2)
                
                elif geom_type == 'MultiLineString':
                    # Draw multiple lines
                    for line in coordinates:
                        points = np.array(line, dtype=np.int32)
                        points = np.clip(points, 0, 511)
                        cv2.polylines(image, [points], isClosed=False, color=meta_value, thickness=2)
                
                elif geom_type == 'Polygon':
                    # Draw filled polygon on the image
                    # Polygons may have multiple rings; coordinates[0] is the exterior ring
                    if len(feature['geometry']['coordinates'][0]) > 0:
                        for polygon in coordinates:
                            points = np.array(polygon, dtype=np.int32)
                            # Ensure points are within image bounds
                            points = np.clip(points, 0, 511)
                            cv2.fillPoly(image, [points], color=meta_value)
                
                elif geom_type == 'MultiPolygon':
                    # Draw multiple polygons
                    if len(feature['geometry']['coordinates'][0]) > 0:
                        for multipolygon in coordinates:
                            for polygon in multipolygon:
                                points = np.array(polygon, dtype=np.int32)
                                points = np.clip(points, 0, 511)
                                cv2.fillPoly(image, [points], color=meta_value)
                
                elif geom_type == 'Point':
                    # Draw a point on the image
                    point = np.array(coordinates, dtype=np.int32)
                    point = np.clip(point, 0, 511)
                    # Draw a small circle to represent the point
                    cv2.circle(image, tuple(point), radius=1, color=meta_value, thickness=-1)
                
                elif geom_type == 'MultiPoint':
                    # Draw multiple points
                    for coord in coordinates:
                        point = np.array(coord, dtype=np.int32)
                        point = np.clip(point, 0, 511)
                        cv2.circle(image, tuple(point), radius=1, color=meta_value, thickness=-1)
                                        
                else:
                    print(f"Unsupported geometry type '{geom_type}' in file {path}")
        
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file {path}: {e}")
        except Exception as e:
            print(f"An error occurred while processing file {path}: {e}")
    
    return torch.from_numpy(image)


class S2NAIP(Dataset):
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
        """
        self.path = path
        self.transform = transform
        self.partition = partition
        self.modalities = modalities
        data_path = path + split + "_dict.json"
        with open(data_path, 'r') as f:
            self.data_dict = json.load(f)
        self.load_labels(classes)
        self.collate_fn = collate_fn
        self.norm = None
        self.split = split
        self.temporal_dropout = temporal_dropout
        self.resize_transform = transforms.Resize((64,64), antialias=True)
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
        for i, b in enumerate(self.data_dict['name']):
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
        if self.partition < 1:
            indices, _, _, _ = train_test_split(np.array(list(range(len(self.data_dict['name'])))), 
                np.zeros(len(self.data_dict['name'])), test_size = 1. - self.partition)
            self.data_dict = {key: [self.data_dict[key][i] for i in indices] for key in self.data_dict.keys()}

    def __getitem__(self, i):
        """
        Returns an item from the dataset.
        Args:
            i (int): index of the item
        Returns:
            dict: dictionary with keys "label", "name" and the other corresponding to the modalities used
        """
        name = self.data_dict['name'][i]
        output = {'label': process_geojson_file(os.path.join(self.path, "openstreetmap", name + ".geojson")), 'name': name}

        if "naip" in self.modalities:
            output["naip"] = torch.from_numpy(np.array(Image.open(os.path.join(self.path, "naip", name + ".png")))).permute(2, 0, 1)
        
        if "s1" in self.modalities:
            with rasterio.open(os.path.join(self.path, "sentinel1", name + ".tif")) as f:
                output["s1"] = torch.FloatTensor(f.read()).view(-1, 2, 64, 64)
            ratio_band = output["s1"][:, :1, :, :] / (output["s1"][:, 1:, :, :] + 1e-10)
            ratio_band = torch.clamp(ratio_band, max=1e4, min=-1e4)
            output["s1"] = torch.cat((output["s1"], ratio_band), dim=1)
            output["s1_dates"] = day_number_in_year(self.data_dict['s1_dates'][i])
            N = len(output["s1_dates"])
            if self.split == "train" and self.temporal_dropout > 0:
                random_indices = torch.randperm(N)[:(int(N * (1 - self.temporal_dropout)))]
                output["s1"] = output["s1"][random_indices]
                output["s1_dates"] = output["s1_dates"][random_indices]

        if "l8" in self.modalities:
            with rasterio.open(os.path.join(self.path, "landsat", name + "_8.tif")) as f:
                l8_bands_8 = torch.from_numpy(np.array(f.read()).astype(np.int16)).unsqueeze(1)
            with rasterio.open(os.path.join(self.path, "landsat", name + "_16.tif")) as f:
                l8_bands_16 = self.resize_transform(torch.from_numpy(np.array(f.read()).astype(np.int16))).view(-1, 10, 64, 64)
            output["l8"] = torch.cat([l8_bands_8, l8_bands_16], dim=1)
            output["l8_dates"] = day_number_in_year(self.data_dict['l8_dates'][i])
            N = len(output["l8_dates"])
            if self.split == "train" and self.temporal_dropout > 0:
                random_indices = torch.randperm(N)[:(int(N * (1 - self.temporal_dropout)))]
                output["l8"] = output["l8"][random_indices]
                output["l8_dates"] = output["l8_dates"][random_indices]

        if "s2" in self.modalities:
            with rasterio.open(os.path.join(self.path, "sentinel2", name + "_8.tif")) as f:
                s2_bands_8 = torch.from_numpy(np.array(f.read()).astype(np.int16)).view(-1, 4, 64, 64)
            with rasterio.open(os.path.join(self.path, "sentinel2", name + "_16.tif")) as f:
                s2_bands_16 = self.resize_transform(torch.from_numpy(np.array(f.read()).astype(np.int16))).view(-1, 6, 64, 64)
            output["s2"] = torch.cat([s2_bands_8, s2_bands_16], dim=1)[:, [0, 1, 2, 4, 5, 6, 3, 7, 8, 9]]
            output["s2_dates"] = day_number_in_year(self.data_dict['s2_dates'][i])
            N = len(output["s2_dates"])
            if self.split == "train" and self.temporal_dropout > 0:
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

    def __len__(self):
        return len(self.data_dict['name'])