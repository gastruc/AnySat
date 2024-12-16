import math

import torch
from multiprocessing import Value


class MaskCollator(object):

    def __init__(
        self,
        input_size=(6, 6),
        patch_size=1,
        enc_mask_scale=(0.2, 0.8),
        pred_mask_scale=(0.85, 1.0),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        min_keep=6,
        allow_overlap=False
    ):
        super(MaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep  # minimum number of patches to keep
        self.allow_overlap = allow_overlap  # whether to allow overlap b/w enc and pred masks
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        # -- Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1

        return (h, w)

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        h, w = b_size

        def constrain_mask(mask, tries=0):
            """ Helper to restrict given mask to a set of acceptable regions """
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]
        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner
            top = torch.randint(0, self.height - h + 1, (1,))
            left = torch.randint(0, self.width - w + 1, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    if tries > 20:
                        raise ValueError('Could not find a valid mask')
        mask = mask.squeeze()
        if mask.dim() == 0:
            mask = mask.unsqueeze(0)
        # --
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w] = 0
        # --
        return mask, mask_complement

    def __call__(self, batch):
        '''
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        '''
        B = len(batch["label"])

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio)
        e_size = self._sample_block_size(
            generator=g,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1., 1.))
        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for _ in range(B):
            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = masks_C
            if self.allow_overlap:
                acceptable_regions= None

            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)
        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        device = batch['label'].device
        for i, tensor in enumerate(collated_masks_pred):
            collated_masks_pred[i] = tensor.to(device)

        for i, tensor in enumerate(collated_masks_enc):
            collated_masks_enc[i] = tensor.to(device)
        return collated_masks_enc, collated_masks_pred

class MaskCollatorNaive(object):

    def __init__(
        self,
        input_size=(6, 6),
        patch_size=1,
        enc_mask_scale=0.8,
        pred_mask_scale=0.2,
    ):
        super(MaskCollatorNaive, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self._itr_counter = Value('i', -1)  # shared counter across workers

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def __call__(self, batch):
        B = len(batch["label"])
        total_patches = self.height * self.width
        
        # Calculate number of patches to keep for each mask
        n_enc = int(total_patches * (1 - self.enc_mask_scale))  # keep (1-enc_mask_scale)
        n_pred = int(total_patches * self.pred_mask_scale)      # keep pred_mask_scale
        
        # Generate random indices for each batch
        device = batch['label'].device
        collated_masks_enc, collated_masks_pred = [], []
        
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        
        for _ in range(B):
            # For encoder mask (keeping 1-enc_mask_scale of patches)
            enc_indices = torch.randperm(total_patches, generator=g)[:n_enc]
            # For predictor mask (keeping pred_mask_scale of patches)
            pred_indices = torch.randperm(total_patches, generator=g)[n_enc:]
            
            # Add batch dimension to match MaskCollator output format
            collated_masks_enc.append([enc_indices])
            collated_masks_pred.append([pred_indices])
        
        # Convert to tensor and move to device
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        for i, tensor in enumerate(collated_masks_pred):
            collated_masks_pred[i] = tensor.to(device)

        for i, tensor in enumerate(collated_masks_enc):
            collated_masks_enc[i] = tensor.to(device)
        return collated_masks_enc, collated_masks_pred
