import torch
from torch import nn
from hydra.utils import instantiate

class MonoModal(nn.Module):
    """
    Initialize encoder and mlp
    Args:
        encoder (nn.Module): that encodes data
        omni (bool): If True, takes class token as encoding of data
        mlp (nn.Module): that returns prediction
    """
    def __init__(
        self, 
        encoder, 
        mlp,
        modalities: list = [],
        omni: bool = False):
        super().__init__()
        self.encoder = encoder
        self.omni = omni
        self.mlp = mlp.instance

    def forward(self, x):
        """
        Forward pass of the network
        """
        out = self.encoder(x)
        if self.omni:
            out = out[:, 0]
        out = self.mlp(out)
        return out

class MonoModalMulitScale(nn.Module):
    """
    Initialize encoder and mlp
    Args:
        encoder (nn.Module): that encodes data
        omni (bool): If True, takes class token as encoding of data
        mlp (nn.Module): that returns prediction
    """
    def __init__(
        self, 
        encoder, 
        mlp,
        modalities: dict = {},
        scales: dict = {},
        omni: bool = False):
        super().__init__()
        self.encoder = encoder
        self.scales = scales
        self.omni = omni
        self.mlp = mlp.instance

    def forward(self, x):
        """
        Forward pass of the network
        """
        outs = []
        for scale in self.scales[x['dataset']]:
            x['scale'] = scale
            out = self.encoder(x)
            if self.omni:
                out = out[:, 0]
            out = self.mlp(out)
            outs.append(out)
        outs = torch.stack(outs, dim=0)
        outs = outs.mean(dim=0)
        return outs

class MonoModalFLAIR(nn.Module):
    """
    Initialize encoder and mlp
    Args:
        encoder (nn.Module): that encodes data
        omni (bool): If True, takes class token as encoding of data
        mlp (nn.Module): that returns prediction
    """
    def __init__(
        self, 
        encoder,
        mlp,
        modalities: dict = {},
        scales: dict = {},
        omni: bool = False):
        super().__init__()
        self.encoder = encoder
        self.scales = scales
        self.omni = omni
        self.mlp = mlp.instance

    def forward(self, x):
        """
        Forward pass of the network
        """
        outs = []
        img = x['aerial-flair']
        if img.shape[2] != 500:
            return self.forward_test(x)
        out = self.encoder(x)
        if self.omni:
            out = out[:, 0]
        if 'dataset' in x.keys():
            out = self.mlp(out, x['scale'])
        else:
            out = self.mlp(out)#, x['aerial-flair'])
        return out

    def forward_test(self, x):
        img = x['aerial-flair']
        logits = torch.zeros((img.shape[0], self.mlp.final_dim, img.shape[2], img.shape[3])).to(img.device)
        for i in range (2):
            for j in range (2):
                x['aerial-flair'] = img[:, :, 12*i:(500 + 12*i), 12*j:(500 + 12*j)]
                if 'dataset' in x.keys():
                    outs = []
                    for scale in self.scales[x['dataset']]:
                        x['scale'] = scale
                        out = self.encoder(x)
                        if self.omni:
                            out = out[:, 0]
                        out = self.mlp(out, scale)
                        outs.append(out)
                    outs = torch.stack(outs, dim=0)
                    outs = outs.mean(dim=0)
                else:
                    out = self.encoder(x)
                    if self.omni:
                        out = out[:, 0]
                    outs = self.mlp(out)#, x['aerial-flair'])
                logits[:, :, 12*i:(500 + 12*i), 12*j:(500 + 12*j)] += outs
        logits[:, :, 12:500, 12:500] = logits[:, :, 12:500, 12:500] / 4
        logits[:, :, 0:12, 12:500] = logits[:, :, 0:12, 12:500] / 2
        logits[:, :, 500:, 12:500] = logits[:, :, 500:, 12:500] / 2
        logits[:, :, 12:500, 0:12] = logits[:, :, 12:500, 0:12] / 2
        logits[:, :, 12:500, 500:] = logits[:, :, 12:500, 500:] / 2
        return logits