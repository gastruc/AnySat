"""
PyTorch Hub configuration for AnySat model.
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add src directory to Python path and create models symbolic link
REPO_ROOT = Path(__file__).parent
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "src"))

# Create symbolic link from src/models to models
if not (REPO_ROOT / "models").exists():
    os.symlink(REPO_ROOT / "src/models", REPO_ROOT / "models")

dependencies = ['torch']

class AnySat(nn.Module):
    """
    AnySat: Earth Observation Model for Any Resolutions, Scales, and Modalities
    
    Args:
        model_size (str): Model size - 'tiny', 'small', or 'base'
        flash_attn (bool): Whether to use flash attention
        **kwargs: Additional arguments to override config
    """
    
    def __init__(self, model_size='base', flash_attn=True, **kwargs):
        super().__init__()
        self.config = get_default_config(model_size)
        
        self.config['flash_attn'] = flash_attn
        
        # Override any additional parameters
        device = None
        for k, v in kwargs.items():
            if k == "device":
                device = v
            else:
                # Update nested dictionary
                keys = k.split('.')
                current = self.config
                for key in keys[:-1]:
                    current = current.setdefault(key, {})
                current[keys[-1]] = v
        
        from src.models.networks.encoder.utils.ltae import PatchLTAEMulti
        from src.models.networks.encoder.utils.patch_embeddings import PatchMLPMulti
        projectors = {}
        for modality in self.config['modalities']['all']:
            if 'T' in self.config['projectors'][modality].keys():
                projectors[modality] = PatchLTAEMulti(**self.config['projectors'][modality])
            else:
                projectors[modality] = PatchMLPMulti(**self.config['projectors'][modality])
                
        del self.config['projectors']
            
        from src.models.networks.encoder.Transformer import TransformerMulti
        self.spatial_encoder = TransformerMulti(**self.config['spatial_encoder'])
        del self.config['spatial_encoder']
        from src.models.networks.encoder.Any_multi import AnyModule  # Import your actual model class
        self.model = AnyModule(projectors=projectors, spatial_encoder=self.spatial_encoder, **self.config)
        
        if device is not None:
            self.model = self.model.to(device)
    
    @classmethod
    def from_pretrained(cls, model_size='base', **kwargs):
        """
        Create a pretrained AnySat model
        
        Args:
            model_size (str): Model size - 'tiny', 'small', or 'base'
            **kwargs: Additional arguments passed to the constructor
        """
        model = cls(model_size=model_size, **kwargs)
        
        checkpoint_urls = {
            'base': 'https://huggingface.co/g-astruc/AnySat/resolve/main/models/AnySat.pth',
            # 'small': 'https://huggingface.co/gastruc/anysat/resolve/main/anysat_small_geoplex.pth', COMING SOON
            # 'tiny': 'https://huggingface.co/gastruc/anysat/resolve/main/anysat_tiny_geoplex.pth' COMING SOON
        }
        
        checkpoint_url = checkpoint_urls[model_size]
        state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, progress=True)['state_dict']
        
        model.model.load_state_dict(state_dict)
        return model
    
    def forward(self, x, patch_size, **kwargs):
        return self.model.forward_release(x, patch_size // 10, **kwargs)

# Hub entry points
def anysat(pretrained=False, **kwargs):
    """PyTorch Hub entry point"""
    if pretrained:
        return AnySat.from_pretrained(**kwargs)
    return AnySat(**kwargs)

def anysat_tiny(pretrained=False, **kwargs):
    return anysat(pretrained=pretrained, model_size='tiny', **kwargs)

def anysat_small(pretrained=False, **kwargs):
    return anysat(pretrained=pretrained, model_size='small', **kwargs)

def anysat_base(pretrained=False, **kwargs):
    return anysat(pretrained=pretrained, model_size='base', **kwargs)

def get_default_config(model_size='base'):
    """Get default configuration based on model size"""
    dim = 768 if model_size == 'base' else (512 if model_size == 'small' else 256)
    depth = 6 if model_size == 'base' else (4 if model_size == 'small' else 2)
    heads = 12 if model_size == 'base' else (8 if model_size == 'small' else 4)
    base_config = {
        'modalities': {
            'all': ['aerial', 'aerial-flair', 'spot', 'naip', 's2', 's1-asc', 's1', 'alos', 'l7', 'l8', 'modis']
            },
        'projectors': {
            'aerial': {
                'patch_size': 10,
                'in_chans': 4,
                'embed_dim': dim,
                'bias': False,
                'mlp': [dim, dim*2, dim]
                },
            'aerial-flair': {
                'patch_size': 10,
                'in_chans': 5,
                'embed_dim': dim,
                'bias': False,
                'mlp': [dim, dim*2, dim]
                },
            'spot': {
                'patch_size': 10,
                'in_chans': 3,
                'embed_dim': dim,
                'bias': False,
                'resolution': 1.0,
                'mlp': [dim, dim*2, dim]
                }, 
            'naip': {
                'patch_size': 8,
                'in_chans': 4,
                'embed_dim': dim,
                'bias': False,
                'resolution': 1.25,
                'mlp': [dim, dim*2, dim]
                },
            's2': {
                'in_channels': 10,
                'n_head': 16,
                'd_k': 8,
                'mlp': [dim],
                'mlp_in': [dim//8, dim//2, dim, dim*2, dim],
                'dropout': 0.0,
                'T': 367,
                'in_norm': True,
                'return_att': False,
                'positional_encoding': True,
                },
            's1-asc': {
                'in_channels': 2,
                'n_head': 16,
                'd_k': 8,
                'mlp': [dim],
                'mlp_in': [dim//8, dim//2, dim, dim*2, dim],
                'dropout': 0.2,
                'T': 367,
                'in_norm': False,
                'return_att': False,
                'positional_encoding': True,
                },
            's1': {
                'in_channels': 3,
                'n_head': 16,
                'd_k': 8,
                'mlp': [dim],
                'mlp_in': [dim//8, dim//2, dim, dim*2, dim],
                'dropout': 0.2,
                'T': 367,
                'in_norm': False,
                'return_att': False,
                'positional_encoding': True,
                },
            'alos': {
                'in_channels': 3,
                'n_head': 16,
                'd_k': 8,
                'mlp': [dim],
                'mlp_in': [dim//8, dim//2, dim, dim*2, dim],
                'dropout': 0.2,
                'T': 367,
                'in_norm': False,
                'return_att': False,
                'positional_encoding': True,
                },
            'l7': {
                'in_channels': 6,
                'n_head': 16,
                'd_k': 8,
                'mlp': [dim],
                'mlp_in': [dim//8, dim//2, dim, dim*2, dim],
                'dropout': 0.2,
                'T': 367,
                'in_norm': False,
                'return_att': False,
                'positional_encoding': True,
                },
            'l8': {
                'in_channels': 11,
                'n_head': 16,
                'd_k': 8,
                'mlp': [dim],
                'mlp_in': [dim//8, dim//2, dim, dim*2, dim],
                'dropout': 0.2,
                'T': 366,
                'in_norm': False,
                'return_att': False,
                'positional_encoding': True,
                },
            'modis': {
                'in_channels': 7,
                'n_head': 16,
                'd_k': 8,
                'mlp': [dim],
                'mlp_in': [dim//8, dim//2, dim, dim*2, dim],
                'dropout': 0.2,
                'T': 367,
                'in_norm': False,
                'return_att': False,
                'positional_encoding': True,
                'reduce_scale': 12
                },
            },
        'spatial_encoder': {
            'embed_dim': dim,
            'depth': depth,
            'num_heads': heads,
            'mlp_ratio': 4.0,
            'attn_drop_rate': 0.0,
            'drop_path_rate': 0.0,
            'modalities': {
                'all': ['aerial', 'aerial-flair', 'spot', 'naip', 's2', 's1-asc', 's1', 'alos', 'l7', 'l8', 'modis']
                },
            'scales': {},
            'input_res': {
                'aerial': 2,
                'aerial-flair': 2,
                'spot': 10,
                'naip': 10,
                's2': 10,
                's1-asc': 10,
                's1-des': 10,
                's1': 10,
                'l8': 10,
                'l7': 30,
                'alos': 30,
                'modis': 250
                }
            },
        'num_patches': {},
        'embed_dim': dim,
        'depth': depth,
        'num_heads': heads,
        'mlp_ratio': 4.0,
        'class_token': True,
        'pre_norm': False,
        'drop_rate': 0.0,
        'patch_drop_rate': 0.0,
        'drop_path_rate': 0.0,
        'attn_drop_rate': 0.0,
        'scales': {},
        'flash_attn': True,
        'release': True,
    }

    return base_config