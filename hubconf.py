"""
PyTorch Hub configuration for AnySat model.
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from omegaconf import OmegaConf
import warnings
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

# Add src directory to Python path and create models symbolic link
REPO_ROOT = Path(__file__).parent
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "src"))

# Create symbolic link from src/models to models
if not (REPO_ROOT / "models").exists():
    os.symlink(REPO_ROOT / "src/models", REPO_ROOT / "models")

dependencies = ['torch', 'torchvision', 'omegaconf', 'timm', 'easydict', 'omegaconf']

# Load base configurations
CONFIG_PATH = REPO_ROOT / "configs"

class AnySat(nn.Module):
    """
    AnySat: Earth Observation Model for Any Resolutions, Scales, and Modalities
    
    Args:
        model_size (str): Model size - 'tiny', 'small', or 'base'
        modalities (list): List of modalities to use. If None, uses all available
        **kwargs: Additional arguments to override config
    """
    
    def __init__(self, model_size='base', flash_attn=True, **kwargs):
        super().__init__()
        self.config = self._load_config(model_size)
        
        self.config.model.flash_attn = flash_attn
        
        # Override any additional parameters
        device = None
        for k, v in kwargs.items():
            if k == "device":
                device = v
            else:
                OmegaConf.update(self.config, k, v)
        
        # Use Hydra's instantiate instead of manual instantiation
        with warnings.catch_warnings():
            # Ignore all warnings during model initialization
            warnings.filterwarnings('ignore')
            self.model = instantiate(self.config.model)
            if device is not None:
                self.model = self.model.to(device)
    
    @staticmethod
    def _load_config(model_size):
        """Load model configuration using Hydra"""
        # Clear any existing Hydra configuration
        GlobalHydra.instance().clear()
        
        # Get relative config path from MODELS_CONFIG
        config_path = str(Path("configs") / Path("model").parent)
        config_name = "model/AnySat"
        if model_size != 'base':
            config_name += f"_{model_size}"
        
        # Initialize Hydra with relative config path
        with initialize(version_base=None, config_path=config_path):
            # Safely register eval resolver if not already registered
            if not OmegaConf.has_resolver("eval"):
                OmegaConf.register_new_resolver("eval", eval)
            # Load the configuration
            cfg = compose(config_name=config_name)
            return cfg
    
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
            'base': 'https://huggingface.co/gastruc/anysat/resolve/main/models/AnySat.pth',
            # 'small': 'https://huggingface.co/gastruc/anysat/resolve/main/anysat_small_geoplex.pth', COMING SOON
            # 'tiny': 'https://huggingface.co/gastruc/anysat/resolve/main/anysat_tiny_geoplex.pth' COMING SOON
        }
        
        checkpoint_url = checkpoint_urls[model_size]
        state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, progress=True)
        
        #state_dict = torch.load(REPO_ROOT / ".models/AnySat_Base_clean.pth", map_location='cpu')["state_dict"]
        
        model.model.load_state_dict(state_dict)
        return model
    
    def forward(self, x, scale, **kwargs):
        return self.model.forward_release(x, scale // 10, **kwargs)

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