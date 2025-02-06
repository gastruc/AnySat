import torch
import torch.nn as nn

class Fine(nn.Module):
    """
    Initialize Fine Tuning of OmniSat after pretraining
    Args:
        encoder (torch.nn.Module): initialized model
        path (str): path of checkpoint of model to load
        output_size (int): size of output returned by encoder
        inter_dim (list): list of hidden dims of mlp after encoder
        p_drop (float): dropout parameter of mlp after encoder
        name (str): name of the weights from checkpoint to use
        freeze (bool); if True, freeze encoder to perform linear probing
        n_class (int): output_size of mlp
        pooling_method (str): type of pooling of tokens after transformer
        modalities (list): list of modalities to use
        last_block (bool): if True freeze all encoder except last block of transformer
        proj_only (bool): if True, load only weights from projectors
    """
    def __init__(self, 
                 encoder: torch.nn.Module,
                 path: str = '',
                 output_size: int = 256,
                 inter_dim: list = [],
                 p_drop: float = 0.3,
                 name: str = 'encoder',
                 freeze: bool = True,
                 freeze_new: bool = False,
                 n_class: int = 15,
                 pooling_method: str = 'token',
                 modalities: list = [],
                 last_block: bool = False,
                 proj_only: bool = False,
                 patch_size: int = 1,
                 proj_size: int = 1,
                ):
        super().__init__()

        self.size = output_size
        self.freeze = freeze
        self.freeze_new = freeze_new
        self.global_pool = pooling_method
        self.patch_size = patch_size
        self.keep_subpatch = encoder.keep_subpatch
        if self.keep_subpatch:
            self.size = self.size * 2

        for i in range(len(modalities)):
            if modalities[i].split('-')[-1] == 'mono':
                modalities[i] = '-'.join(modalities[i].split('-')[:-1])

        target = (name == "target_encoder")
        
        u = torch.load(path)
        d = {}
        for key in u["state_dict"].keys():
            if name in key:
                if target:
                    if 'projector' in key:
                        if any([(modality + ".") in key for modality in modalities]):
                            d['.'.join(key.split('.')[1:])] = u["state_dict"][key]
                    else:
                        if not(proj_only) and not('predictor.' in key):
                            d['.'.join(key.split('.')[1:])] = u["state_dict"][key]
                else:
                    if not('target_encoder' in key):
                        if 'projector' in key:
                            if any([modality in key for modality in modalities]):
                                d['.'.join(key.split('.')[2:])] = u["state_dict"][key]
                        else:
                            if not(proj_only) and not('predictor' in key):
                                d['.'.join(key.split('.')[2:])] = u["state_dict"][key]

        if not(proj_only) and not(freeze_new) and proj_size == 1:
            encoder.load_state_dict(d)
        else:
            encoder.load_state_dict(d, strict=False)

        del u
        del d
                 
        self.model = encoder

        if last_block:
            model_parameters = self.model.named_parameters()
            for name, param in model_parameters:
                if len(name.split(".")) > 1:
                    if name.split(".")[1] == "5":
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    param.requires_grad = False

        if freeze_new:
            model_parameters = self.model.named_parameters()
            for name, param in model_parameters:
                if 'projector' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False
                
        self.n_class = n_class
        self.proj_size = proj_size
        # set n_class to 0 if we want headless model
        if not(self.global_pool) and not(self.keep_subpatch):
            n_class = n_class * patch_size * patch_size
        if n_class:
            layers = [nn.LayerNorm(self.size)]
            if len(inter_dim) > 0:
                layers.append(nn.Linear(self.size, inter_dim[0]))
                #layers.append(nn.BatchNorm1d(inter_dim[0]))
                layers.append(nn.Dropout(p = p_drop))
                layers.append(nn.ReLU())
                for i in range(len(inter_dim) - 1):
                    layers.append(nn.Linear(inter_dim[i], inter_dim[i + 1]))
                    #layers.append(nn.BatchNorm1d(inter_dim[i + 1]))
                    layers.append(nn.Dropout(p = p_drop))
                    layers.append(nn.ReLU())
                layers.append(nn.Linear(inter_dim[-1], n_class * proj_size * proj_size))
            else:
                layers.append(nn.Linear(self.size, n_class * proj_size * proj_size))
            self.head = nn.Sequential(*layers)
        
    def forward(self,x):
        """
        Forward pass of the network. Perform pooling of tokens after transformer 
        according to global_pool argument.
        """
        x = self.model(x)
        if self.global_pool:
            if self.global_pool == 'avg':
                x = x[:, 1:].mean(dim=1)
            elif self.global_pool == 'max':
                x ,_ = torch.max(x[:, 1:],1)
            else:
                x = x[:, 0]
            if self.n_class:
                x = self.head(x)   
            return x
        if self.n_class:
            if self.keep_subpatch:
                if self.proj_size > 1:
                    f, out = x
                    x = f[:, 1:].unsqueeze(2).repeat(1, 1, out['subpatches'].shape[2], 1)
                    dense_x = torch.cat([x, out['subpatches']], dim = 3)
                    x = self.head(dense_x)
                    B, N, _, D = x.shape
                    num_patches = int(N**(1/2))
                    size = num_patches * self.patch_size * self.proj_size

                    x = x.unsqueeze(2).permute(0, 2, 4, 1, 3)
                    x = x.view(B, 1, D, N, self.patch_size, self.patch_size)
                    x = x.view(B, 1, self.proj_size, self.proj_size, self.n_class, N, self.patch_size, self.patch_size).permute(0, 1, 4, 5, 6, 2, 7, 3)
                    x = x.reshape(B, 1, self.n_class, N, self.proj_size * self.patch_size, self.patch_size * self.proj_size)
                    x = x.view(B, 1, self.n_class, num_patches, num_patches, self.patch_size * self.proj_size, self.patch_size * self.proj_size)
                    x = x.permute(0, 1, 2, 3, 5, 4, 6).reshape(B, 1, self.n_class, size, size).flatten(0, 1)
                    return x
                
                f, out = x
                x = f[:, 1:].unsqueeze(2).repeat(1, 1, out['subpatches'].shape[2], 1)
                dense_x = torch.cat([x, out['subpatches']], dim = 3)
                x = self.head(dense_x)
                B, N, _, D = x.shape
                num_patches = int(N**(1/2))
                size = num_patches * self.patch_size

                x = x.unsqueeze(2).permute(0, 2, 4, 1, 3)
                x = x.view(B, 1, self.n_class, N, self.patch_size, self.patch_size)
                x = x.view(B, 1, self.n_class, num_patches, num_patches, self.patch_size, self.patch_size).permute(0, 1, 2, 3, 5, 4, 6)
                x = x.reshape(B, 1, self.n_class, size, size).flatten(0, 1)
                return x
            x = x[:, 1:]
            x = self.head(x)
            B, N, D = x.shape
            num_patches = int(N**(1/2))
            size = num_patches * self.patch_size

            x = x.view(B, N, 1, D).view(B, N, 1, self.n_class, self.patch_size * self.patch_size).permute(0, 2, 3, 1, 4)
            x = x.view(B, 1, self.n_class, N, self.patch_size, self.patch_size)
            x = x.view(B, 1, self.n_class, num_patches, num_patches, self.patch_size, self.patch_size).permute(0, 1, 2, 3, 5, 4, 6)
            x = x.reshape(B, 1, self.n_class, size, size).flatten(0, 1)
            return x
        return x[:, 1:]


class Fine_multi(nn.Module):
    """
    Initialize Fine Tuning of OmniSat after pretraining
    Args:
        encoder (torch.nn.Module): initialized model
        path (str): path of checkpoint of model to load
        output_size (int): size of output returned by encoder
        inter_dim (list): list of hidden dims of mlp after encoder
        p_drop (float): dropout parameter of mlp after encoder
        name (str): name of the weights from checkpoint to use
        freeze (bool); if True, freeze encoder to perform linear probing
        n_class (int): output_size of mlp
        pooling_method (str): type of pooling of tokens after transformer
        modalities (list): list of modalities to use
        last_block (bool): if True freeze all encoder except last block of transformer
        proj_only (bool): if True, load only weights from projectors
    """
    def __init__(self, 
                 encoder: torch.nn.Module,
                 path: str = '',
                 output_size: int = 256,
                 inter_dim: list = [],
                 p_drop: float = 0.3,
                 name: str = 'encoder',
                 freeze: bool = True,
                 n_class: int = 15,
                 pooling_method: str = 'token',
                 modalities: dict = {},
                 last_block: bool = False,
                 proj_only: bool = False,
                 scales: dict = {},
                ):
        super().__init__()

        self.size = output_size
        self.freeze = freeze
        self.global_pool = pooling_method
        self.scales = scales
        self.keep_subpatch = encoder.keep_subpatch
        if self.keep_subpatch:
            self.size = self.size * 2

        target = (name == "target_encoder")

        modalities = [m for name in modalities.keys() for m in modalities[name]]
        
        u = torch.load(path)
        d = {}
        for key in u["state_dict"].keys():
            if name in key:
                if target:
                    if 'projector' in key:
                        if any([modality in key for modality in modalities]):
                            d['.'.join(key.split('.')[1:])] = u["state_dict"][key]
                    else:
                        if not(proj_only) and not('predictor.' in key):
                            d['.'.join(key.split('.')[1:])] = u["state_dict"][key]
                else:
                    if not('target_encoder' in key):
                        if 'projector' in key:
                            if any([modality in key for modality in modalities]):
                                d['.'.join(key.split('.')[2:])] = u["state_dict"][key]
                        else:
                            if not(proj_only) and not('predictor' in key):
                                d['.'.join(key.split('.')[2:])] = u["state_dict"][key]

        if not(proj_only):
            encoder.load_state_dict(d)
        else:
            encoder.load_state_dict(d, strict=False)

        del u
        del d
                 
        self.model = encoder

        if last_block:
            model_parameters = self.model.named_parameters()
            for name, param in model_parameters:
                if len(name.split(".")) > 1:
                    if name.split(".")[1] == "5":
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    param.requires_grad = False

        if self.freeze:
            for param in self.model.parameters():
                    param.requires_grad = False
                
        self.n_class = n_class
        # set n_class to 0 if we want headless model
        if not(self.global_pool) and not(self.keep_subpatch):
            n_class = n_class * patch_size * patch_size
        if n_class:
            if len(inter_dim) > 0:
                layers = [nn.Linear(self.size, inter_dim[0])]
                #layers.append(nn.BatchNorm1d(inter_dim[0]))
                layers.append(nn.Dropout(p = p_drop))
                layers.append(nn.ReLU())
                for i in range(len(inter_dim) - 1):
                    layers.append(nn.Linear(inter_dim[i], inter_dim[i + 1]))
                    #layers.append(nn.BatchNorm1d(inter_dim[i + 1]))
                    layers.append(nn.Dropout(p = p_drop))
                    layers.append(nn.ReLU())
                layers.append(nn.Linear(inter_dim[-1], n_class))
            else:
                layers = [nn.Linear(self.size, n_class)]
            self.head = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass of the network. Perform pooling of tokens after transformer 
        according to global_pool argument.
        """
        outs = []
        for scale in self.scales[x['dataset']]:
            x['scale'] = scale
            out = self.model(x)
            if self.global_pool:
                if self.global_pool == 'avg':
                    out = out[:, 1:].mean(dim=1)
                elif self.global_pool == 'max':
                    out,_ = torch.max(out[:, 1:],1)
                else:
                    out = out[:, 0]
                if self.n_class:
                    out = self.head(out)   
            elif self.n_class:
                if self.keep_subpatch:
                    f, mid = out
                    f = f[:, 1:].unsqueeze(2).repeat(1, 1, mid['subpatches'].shape[2], 1)
                    dense_x = torch.cat([f, mid['subpatches']], dim = 3)
                    out = self.head(dense_x)
                    B, N, _, D = out.shape
                    num_patches = int(N**(1/2))
                    size = num_patches * scale

                    out = out.unsqueeze(2).permute(0, 2, 4, 1, 3)
                    out = out.view(B, 1, self.n_class, N, scale, scale)
                    out = out.view(B, 1, self.n_class, num_patches, num_patches, scale, scale).permute(0, 1, 2, 3, 5, 4, 6)
                    out = out.reshape(B, 1, self.n_class, size, size).flatten(0, 1)
                else:
                    out = out[:, 1:]
                    out = self.head(out)
                    B, N, D = out.shape
                    num_patches = int(N**(1/2))
                    size = num_patches * scale

                    out = out.view(B, N, 1, D).view(B, N, 1, self.n_class, scale * scale).permute(0, 2, 3, 1, 4)
                    out = out.view(B, 1, self.n_class, N, scale, scale)
                    out = out.view(B, 1, self.n_class, num_patches, num_patches, scale, scale).permute(0, 1, 2, 3, 5, 4, 6)
                    out = out.reshape(B, 1, self.n_class, size, size).flatten(0, 1)
            else:
                out = out[:, 1:]
            outs.append(out)

        outs = torch.stack(outs, dim=0)
        outs = outs.mean(dim=0)
        return outs

if __name__ == "__main__":
    _ = Fine()