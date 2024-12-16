import torch
import random
import torchvision
import torchvision.transforms.v2.functional as TF

class Identity(object):
    def __init__(self):
        pass

    def __call__(self, batch):     
        return batch

class Transform(object):
    def __init__(self, p = 0.0, size = 300, classif = False, center_crop = False):
        self.p = p
        self.size = size
        self.crop = torchvision.transforms.v2.RandomCrop(size=[size, size])
        #if center_crop:
            #self.crop = torchvision.transforms.CenterCrop(size=[size, size])
        self.center_crop = center_crop
        self.semseg = not(classif)

    def __call__(self, batch):
        keys = list(batch.keys())
        keys.remove('label')
        keys.remove('name')

        if self.semseg:
            if self.center_crop:
                batch['aerial-flair'] = batch['aerial-flair'][:, 6:506, 6:506]
                batch['label'] = batch['label'][6:506, 6:506]
            else:
                i, j, h, w = torchvision.transforms.RandomCrop.get_params(
                    batch['aerial-flair'], output_size=(self.size, self.size))
                batch['label'] = TF.crop(batch['label'], i, j, h, w)
                batch['aerial-flair'] = TF.crop(batch['aerial-flair'], i, j, h, w)
        else:
            if 'aerial' in keys:
                batch['aerial'] = self.crop(batch['aerial'])

        if random.random() < self.p:
            for key in keys:
                batch[key] = TF.horizontal_flip(batch[key])
        
        if random.random() < self.p:
            for key in keys:
                batch[key] = TF.vertical_flip(batch[key])

        if random.random() < self.p:
            for key in keys:
                batch[key] = TF.rotate(batch[key], 90)
        
        return batch

class GeoPlexTransform(object):
    def __init__(self, sizes = {}, modalities = {}):
        if 'tsaits' in modalities.keys():
            self.crop_tsaits = torchvision.transforms.v2.RandomCrop(size=[sizes['tsaits'], sizes['tsaits']])
        if 'flair' in modalities.keys():
            self.crop_flair = torchvision.transforms.v2.RandomCrop(size=[sizes['flair'], sizes['flair']])

    def __call__(self, batch):
        keys = list(batch.keys())
 
        if 'aerial' in keys:
            batch['aerial'] = self.crop_tsaits(batch['aerial'])

        if 'aerial-flair' in keys:
            batch['aerial-flair'] = self.crop_flair(batch['aerial-flair'])
        
        return batch
    
class TransformMAE(object):
    def __init__(self, p = 0.5, size = 224, s2_size = 0):
        self.p = p
        self.s2 = False
        self.crop = torchvision.transforms.Resize(size=[size,size], antialias=True)
        if s2_size > 0:
            self.s2 = True
            self.crop2 = torchvision.transforms.Resize(size=[s2_size, s2_size], antialias=True)

    def __call__(self, batch):
        keys = list(batch.keys())
        keys.remove('label')
        keys.remove('name')

        for key in keys:
            if self.s2 and key in ['s2-4season-median', 's2-median']:
                batch[key] = self.crop2(batch[key])
            else:
                batch[key] = self.crop(batch[key]) 

        return batch
    
class TransformDOFA(object):
    def __init__(self, size = 224):
        S1_MEAN = [166.36275909, 88.45542715]
        S1_STD = [64.83126309, 43.07350145]

        S2_MEAN = [114.1099739 , 114.81779093, 126.63977424,  84.33539309,
                97.84789168, 103.94461911, 101.435633  ,  72.32804172,
                56.66528851]
        S2_STD = [77.84352553, 69.96844919, 67.42465279, 64.57022983, 61.72545487,
            61.34187099, 60.29744676, 47.88519516, 42.55886798]
        
        NAIP_MEAN = [123.675, 116.28, 103.53]
        NAIP_STD = [58.395, 57.12, 57.375]

        self.crop = torchvision.transforms.Resize(size=[size, size], antialias=True)
        self.norm_s2 = torchvision.transforms.Normalize(mean=S2_MEAN, std=S2_STD)
        self.norm_s1 = torchvision.transforms.Normalize(mean=S1_MEAN, std=S1_STD)
        self.norm_naip = torchvision.transforms.Normalize(mean=NAIP_MEAN, std=NAIP_STD)

    def __call__(self, batch):
        keys = list(batch.keys())

        if 's1-mono' in keys:
            batch['s1-mono'] = self.norm_s1(self.crop(batch['s1-mono'][[1, 0], :, :]))

        if 's2-mono' in keys:
            batch['s2-mono'] = self.norm_s2(self.crop(batch['s2-mono'][[2, 1, 0, 4, 5, 6, 3, 8, 9], :, :]))

        if 's1-median' in keys:
            batch['s1-median'] = self.norm_s1(self.crop(batch['s1-median'][[1, 0], :, :]))

        if 's1-mid' in keys:
            batch['s1-mid'] = self.norm_s1(self.crop(batch['s1-mid'][[1, 0], :, :]))

        if 's2-median' in keys:
            batch['s2-median'] = self.norm_s2(self.crop(batch['s2-median'][[2, 1, 0, 3, 4, 5, 6, 8, 9], :, :]))

        if 's2-mid' in keys:
            batch['s2-mid'] = self.norm_s2(self.crop(batch['s2-mid'][[2, 1, 0, 3, 4, 5, 6, 8, 9], :, :]))

        if 'aerial' in keys:
            batch['aerial'] = self.norm_naip(self.crop(batch['aerial'][1:, :, :]))

        return batch

class TransformDOFAFLAIR(object):
    def __init__(self, size = 224):
        S2_MEAN = [114.1099739 , 114.81779093, 126.63977424,  84.33539309,
                97.84789168, 103.94461911, 101.435633  ,  72.32804172,
                56.66528851]
        S2_STD = [77.84352553, 69.96844919, 67.42465279, 64.57022983, 61.72545487,
            61.34187099, 60.29744676, 47.88519516, 42.55886798]
        
        NAIP_MEAN = [123.675, 116.28, 103.53]
        NAIP_STD = [58.395, 57.12, 57.375]

        self.crop = torchvision.transforms.Resize(size=[size, size], antialias=True)
        self.norm_s2 = torchvision.transforms.Normalize(mean=S2_MEAN, std=S2_STD)
        self.norm_naip = torchvision.transforms.Normalize(mean=NAIP_MEAN, std=NAIP_STD)

    def __call__(self, batch):
        keys = list(batch.keys())

        if 's2-median' in keys:
            batch['s2-median'] = self.norm_s2(self.crop(batch['s2-median'][[2, 1, 0, 3, 4, 5, 6, 8, 9], :, :]))

        if 's2-mid' in keys:
            batch['s2-mid'] = self.norm_s2(self.crop(batch['s2-mid'][[2, 1, 0, 3, 4, 5, 6, 8, 9], :, :]))

        if 'aerial' in keys:
            batch['aerial'] = self.norm_naip(self.crop(batch['aerial'][[2, 1, 0], :, :]))

        return batch
    
class TransformDOFAPASTIS(object):
    def __init__(self, size = 224):
        S1_MEAN = [166.36275909, 88.45542715]
        S1_STD = [64.83126309, 43.07350145]

        S2_MEAN = [114.1099739 , 114.81779093, 126.63977424,  84.33539309,
                97.84789168, 103.94461911, 101.435633  ,  72.32804172,
                56.66528851]
        S2_STD = [77.84352553, 69.96844919, 67.42465279, 64.57022983, 61.72545487,
            61.34187099, 60.29744676, 47.88519516, 42.55886798]
        
        NAIP_MEAN = [123.675, 116.28, 103.53]
        NAIP_STD = [58.395, 57.12, 57.375]

        self.crop = torchvision.transforms.Resize(size=[size, size], antialias=True)
        self.norm_s2 = torchvision.transforms.Normalize(mean=S2_MEAN, std=S2_STD)
        self.norm_s1 = torchvision.transforms.Normalize(mean=S1_MEAN, std=S1_STD)
        self.norm_naip = torchvision.transforms.Normalize(mean=NAIP_MEAN, std=NAIP_STD)

    def __call__(self, batch):
        keys = list(batch.keys())

        if 's1-median' in keys:
            batch['s1-median'] = self.norm_s1(self.crop(batch['s1-median'][[1, 0], :, :]))

        if 's1-mid' in keys:
            batch['s1-mid'] = self.norm_s1(self.crop(batch['s1-mid'][[1, 0], :, :]))

        if 's2-median' in keys:
            batch['s2-median'] = self.norm_s2(self.crop(batch['s2-median'][[2, 1, 0, 3, 4, 5, 6, 8, 9], :, :]))

        if 's2-mid' in keys:
            batch['s2-mid'] = self.norm_s2(self.crop(batch['s2-mid'][[2, 1, 0, 3, 4, 5, 6, 8, 9], :, :]))

        if 'aerial' in keys:
            batch['aerial'] = self.norm_naip(self.crop(batch['aerial'][[2, 1, 0], :, :]))

        return batch
    