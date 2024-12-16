import torch
from torch import nn

class Late_fusion(nn.Module):
    """
    Initialize Late fusion module. For all modalities, initialize a specific encoder and then perform Late fusion.
    """
    def __init__(
        self, 
        encoder, 
        mlp,
        modalities: list = [],
        ):
        super().__init__()
        for i in range(len(modalities)):
            setattr(self, '_'.join(['encoder', modalities[i]]), encoder[modalities[i]])
        self.modalities = modalities
        self.mlp = mlp.instance

    def forward(self, x):
        """
        Forward pass of the network.
        Concatenates different encodings before feeding the mlp
        """
        tokens = []
        for modality in self.modalities:
            token = getattr(self, '_'.join(['encoder', modality]))(x)
            tokens.append(token)
            
        tokens = torch.cat(tokens, dim=1)
        out = self.mlp(tokens)
        return out