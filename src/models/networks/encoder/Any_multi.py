from typing import Callable, Optional
from functools import partial

import torch
import torch.nn as nn
from timm.layers import trunc_normal_, PatchDropout
import itertools

from models.networks.encoder.utils.utils_ViT import Block, CrossBlockMulti
from models.networks.encoder.utils.pos_embed import get_2d_sincos_pos_embed_with_scale

class AnyModule(nn.Module):
    """
    Initializes AnySat encoding module.
    Args:
        spatial_encoder (nn.Module): Neural network module for spatial encoding
        projectors (dict): Dict of all possible projectors
        modalities (dict): Dict of modalities to use
        num_patches (dict): Dict of number of patches by observation for each modality
        embed_dim (int): Embed dimension of transformer blocks. Default: 768
        depth (int): Depth of transformer blocks. Default: 12
        num_heads (int): Number of heads of transformer blocks. Default: 12
        mlp_ratio (float): MLP ratio of transformer blocks. Default: 4.
        qkv_bias (bool): Whether to use bias in QKV projection. Default: True
        qk_scale: Scale factor for QK attention. Default: None
        class_token (bool): If True, add a class token. Default: True
        pre_norm (bool): Whether to apply normalization before transformer blocks. Default: False
        drop_rate (float): Dropout rate. Default: 0.
        patch_drop_rate (float): Patch dropout rate. Default: 0.
        drop_path_rate (float): Drop path rate for transformer blocks. Default: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        norm_layer (Optional[Callable]): Normalization layer. Default: None
        scales (dict): Dict of scales for each dataset
        keep_subpatch (bool): Whether to keep subpatch information. Default: False
        modality_keep (str): Which modality to keep subpatches for. Default: ""
        flash_attn (bool): Whether to use flash attention. Default: True
        release (bool): Whether to initialize hte model as the feature extractor. Default: False
    """
    def __init__(self,
                 spatial_encoder: nn.Module,
                 projectors: dict = {},
                 modalities: dict = {},
                 num_patches: dict = {},
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale = None,
                 class_token: bool = True,
                 pre_norm: bool = False,
                 drop_rate: float = 0.,
                 patch_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 norm_layer: Optional[Callable] = None,
                 scales: dict = {},
                 keep_subpatch: bool = False,
                 modality_keep: str = "",
                 flash_attn: bool = True,
                 release: bool = False,
                 ):
        
        super(AnyModule, self).__init__()
        self.modalities = modalities

        self.num_prefix_tokens = 1 if class_token else 0
        self.embed_dim = embed_dim
        self.keep_subpatch = keep_subpatch
        self.modality_keep = modality_keep

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        if not release:
            self.datasets = list(modalities.keys())
            self.pos_embed = {}
            for dataset in self.datasets:
                for scale in scales[dataset]:
                    num_p = num_patches[dataset] // (scale * scale)
                    self.pos_embed['_'.join([dataset, str(scale)])] = get_2d_sincos_pos_embed_with_scale(
                                                                        embed_dim, 
                                                                        int(num_p ** .5), 
                                                                        scale, 
                                                                        cls_token=class_token
                                                                    )
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()

        modalities_list = sorted(list(set(list(itertools.chain.from_iterable(modalities.values())))))
        for modality in modalities_list:
            if modality.split('-')[-1] == 'mono':
                m = '-'.join(modality.split('-')[:-1])
            else:
                m = modality
            setattr(self, '_'.join(['projector', modality]), projectors[m])

        self.spatial_encoder = spatial_encoder 

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth + 1)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, attn_drop=attn_drop_rate, 
                drop_path=dpr[i], norm_layer=norm_layer, flash_attn=flash_attn) for i in range(depth)] + [CrossBlockMulti(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, modalities=modalities,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[-1], norm_layer=norm_layer, num_patches=num_patches,
                scales=scales, release=release)
                ])
        trunc_normal_(self.cls_token, std=.02)

    def forward_proj(self, x):
        """
        Forward function until masking used during pretraining
        """
        tokens = []
        masks = {}
        out = {}
        pos_embed = self.pos_embed['_'.join([x['dataset'], str(x['scale'])])].to(x['label'].device)
        _, N, _ = pos_embed.shape
        for modality in self.modalities[x['dataset']]:
            if modality == "aerial" or modality == "spot" or modality == "aerial-flair" or modality == "naip":
                token = getattr(self, '_'.join(['projector', modality]))(x[modality], x['scale'])
            else:
                if '_'.join([modality, "mask"]) in list(x.keys()):
                    token = getattr(self, '_'.join(['projector', modality]))(x[modality], 
                        x['_'.join([modality, "dates"])], x['scale'], x['_'.join([modality, "mask"])])
                    if modality != "modis":
                        out['_'.join(['masks', modality])] = get_mask(x['_'.join([modality, "mask"])], modality)
                else:
                    token = getattr(self, '_'.join(['projector', modality]))(x[modality], x['_'.join([modality, "dates"])], x['scale'])
            token = self.spatial_encoder(token, modality, x['dataset'], x['scale'])
            if modality == "modis":
                tokens.insert(0, token.unsqueeze(1))
            else:
                token = token.view(-1, N - 1, self.embed_dim)
                out['_'.join(['tokens', modality])] = token
                tokens.append(token + pos_embed[:, 1:, :])

        tokens = torch.cat(tokens, dim=1)
        return tokens, out
    
    def forward_transformer(self, x, mask, dataset, scale):
        """
        Forward function after masking used during pretraining
        """
        pos_embed = self.pos_embed['_'.join([dataset, str(scale)])].to(x.device)
        if self.cls_token is not None:
            cls_tokens = (self.cls_token + pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
            tokens = torch.cat((cls_tokens, x), dim=1)
        tokens = self.norm_pre(tokens)
        for blk in self.blocks[:-1]:
            tokens = blk(tokens)
        tokens = self.blocks[-1](tokens, mask, dataset=dataset, scale=scale)
        return tokens

    def forward(self, x):
        """
        Complete forward function during training
        """
        tokens = []
        out = {}
        pos_embed = self.pos_embed['_'.join([x['dataset'], str(x['scale'])])].to(x['label'].device)
        _, N, _ = pos_embed.shape
        for modality in self.modalities[x['dataset']]:
            if modality == "aerial" or modality == "spot" or modality == "aerial-flair" or modality == "naip":
                token = getattr(self, '_'.join(['projector', modality]))(x[modality], x['scale'])
            else:
                if '_'.join([modality, "mask"]) in list(x.keys()):
                    token = getattr(self, '_'.join(['projector', modality]))(x[modality], 
                        x['_'.join([modality, "dates"])], x['scale'], x['_'.join([modality, "mask"])])
                else:
                    token = getattr(self, '_'.join(['projector', modality]))(x[modality], x['_'.join([modality, "dates"])], x['scale'])

            if self.keep_subpatch and modality == self.modality_keep:
                token, subs = self.spatial_encoder(token, modality, x['dataset'], x['scale'], keep_subpatch=True)
                out['_'.join(['subpatches'])] = subs.view(-1, N - 1, subs.shape[1], subs.shape[2])
            else:
                token = self.spatial_encoder(token, modality, x['dataset'], x['scale'])
            if modality == "modis":
                tokens.insert(0, token.unsqueeze(1))
            else:
                token = token.view(-1, N - 1, self.embed_dim)
                tokens.append(token + pos_embed[:, 1:, :])

        tokens = torch.cat(tokens, dim=1)
        if self.cls_token is not None:
            cls_tokens = (self.cls_token + pos_embed[:, :1, :]).expand(token.shape[0], -1, -1)
            tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = self.patch_drop(tokens)
        tokens = self.norm_pre(tokens)
        for blk in self.blocks[:-1]:
            tokens = blk(tokens)
        tokens = self.blocks[-1](tokens, dataset=x['dataset'], scale=x['scale'])
        if self.keep_subpatch:
            return tokens, out
        return tokens
    
    def forward_release(self, x, scale, keep_subpatch=False, modality_keep=''):
        tokens = []
        out = {}
        modalities = [mod for mod in x.keys() if not (mod.endswith('_dates') or mod.endswith('_mask'))]
        batch_size = x[modalities[0]].shape[0]
        device = x[modalities[0]].device
        n_modalities = len(modalities)
        modis = ('modis' in modalities)
        pos_embed = None
        for modality in modalities:
            if modality == "aerial" or modality == "spot" or modality == "aerial-flair" or modality == "naip":
                token = getattr(self, '_'.join(['projector', modality]))(x[modality], scale)
            else:
                if '_'.join([modality, "mask"]) in list(x.keys()):
                    token = getattr(self, '_'.join(['projector', modality]))(x[modality], 
                        x['_'.join([modality, "dates"])], scale, x['_'.join([modality, "mask"])])
                else:
                    token = getattr(self, '_'.join(['projector', modality]))(x[modality], x['_'.join([modality, "dates"])], scale)
            
            if pos_embed is None and modality != "modis":
                _, N, C = token.shape
                num_patches = int(N**(1/2))
                pos_embed = get_2d_sincos_pos_embed_with_scale(C, 
                                                       num_patches, 
                                                       scale, 
                                                       cls_token=True).to(device)
            if keep_subpatch and modality == modality_keep:
                token, subs = self.spatial_encoder.forward_release(token, modality, scale, keep_subpatch=True)
                out['_'.join(['subpatches'])] = subs.view(-1, N, subs.shape[1], subs.shape[2])
            else:
                token = self.spatial_encoder.forward_release(token, modality, scale)
            if modality == "modis":
                tokens.insert(0, token.unsqueeze(1))
            else:
                token = token.view(-1, N, self.embed_dim)
                tokens.append(token + pos_embed[:, 1:, :])

        tokens = torch.cat(tokens, dim=1)
        if self.cls_token is not None:
            cls_tokens = (self.cls_token + pos_embed[:, :1, :]).expand(token.shape[0], -1, -1)
            tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = self.patch_drop(tokens)
        tokens = self.norm_pre(tokens)
        for blk in self.blocks[:-1]:
            tokens = blk(tokens)
        tokens = self.blocks[-1].forward_release(tokens, n_modalities=n_modalities, modis=modis, scale=scale)
        if self.keep_subpatch:
            return tokens, out
            tokens = tokens.unsqueeze(2).permute(0, 2, 4, 1, 3)
            tokens = tokens.view(B, 1, C, num_patches, num_patches, scale, scale)
        
        #B, N, D = tokens.shape
        print(tokens.shape)
        tokens = tokens.view(batch_size, -1, num_patches + 1, C)
        print(tokens.shape)
        #.view(batch_size, N, 1, self.final_dim, scale * scale).permute(0, 2, 3, 1, 4)
        tokens = tokens.view(batch_size, 1, self.final_dim, N, scale, scale)
        tokens = tokens.view(batch_size, 1, self.final_dim, num_patches, num_patches, scale, scale).permute(0, 1, 2, 3, 5, 4, 6)
        #tokens = tokens.reshape(B, 1, self.final_dim, size, size).flatten(0, 1)
        return tokens

