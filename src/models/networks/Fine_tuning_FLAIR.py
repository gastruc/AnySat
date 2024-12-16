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
                 n_class: int = 15,
                 pooling_method: str = 'token',
                 modalities: list = [],
                 last_block: bool = False,
                 proj_only: bool = False,
                 patch_size: int = 1,
                 proj_size: int = 1,
                 resolution: float = 0.2,
                 norm=nn.InstanceNorm1d,
                 activation=nn.ReLU,
                 scales: dict = {}
                ):
        super().__init__()

        self.size = output_size
        self.freeze = freeze
        self.scales = scales
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
                
        self.final_dim = n_class

        self.proj_size = proj_size
        if resolution < 1:
            self.patch_size = int(patch_size / resolution)
        else:
            self.patch_size = patch_size

        if not(self.global_pool):
            n_class = n_class * proj_size * proj_size

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
                layers.append(nn.Linear(inter_dim[-1], n_class))
            else:
                layers.append(nn.Linear(self.size, n_class))
            self.head = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass of the network. Perform pooling of tokens after transformer 
        according to global_pool argument.
        """
        keys = x.keys()
        img = x['aerial-flair']
        if img.shape[2] != 500:
            return self.forward_test(x)

        if 'scale' in keys:
            scale = x['scale']
        x = self.model(x)
        if 'dataset' in keys:
            features, out = x
            patch_size = int(scale / 0.2)
            features = features[:, 1:].unsqueeze(2).repeat(1, 1, out['subpatches'].shape[2], 1)
            dense_x = torch.cat([features, out['subpatches']], dim = 3)
            out = self.head(dense_x)
            B, N, _, D = out.shape
            num_patches = int(N**(1/2))
            size = num_patches * patch_size * self.proj_size
            out = out.unsqueeze(2).permute(0, 2, 4, 1, 3)
            out = out.view(B, 1, D, N, patch_size, patch_size)
            out = out.view(B, 1, self.proj_size, self.proj_size, self.final_dim, N, patch_size, patch_size).permute(0, 1, 4, 5, 6, 2, 7, 3)
            out = out.reshape(B, 1, self.final_dim, N, self.proj_size * patch_size, patch_size * self.proj_size)
            out = out.view(B, 1, self.final_dim, num_patches, num_patches, patch_size * self.proj_size, patch_size * self.proj_size)
            out = out.permute(0, 1, 2, 3, 5, 4, 6).reshape(B, 1, self.final_dim, size, size).flatten(0, 1)
            return out
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
        if self.final_dim:
            if self.keep_subpatch:
                features, out = x
                features = features[:, 1:].unsqueeze(2).repeat(1, 1, out['subpatches'].shape[2], 1)
                dense_x = torch.cat([features, out['subpatches']], dim = 3)
                x = self.head(dense_x)
                B, N, _, D = x.shape
                num_patches = int(N**(1/2))
                size = num_patches * self.patch_size * self.proj_size

                x = x.unsqueeze(2).permute(0, 2, 4, 1, 3)
                x = x.view(B, 1, D, N, self.patch_size, self.patch_size)
                x = x.view(B, 1, self.proj_size, self.proj_size, self.final_dim, N, self.patch_size, self.patch_size).permute(0, 1, 4, 5, 6, 2, 7, 3)
                x = x.reshape(B, 1, self.final_dim, N, self.proj_size * self.patch_size, self.patch_size * self.proj_size)
                x = x.view(B, 1, self.final_dim, num_patches, num_patches, self.patch_size * self.proj_size, self.patch_size * self.proj_size)
                x = x.permute(0, 1, 2, 3, 5, 4, 6).reshape(B, 1, self.final_dim, size, size).flatten(0, 1)
                return x
        return x

    def forward_test(self, x):
        img = x['aerial-flair']
        logits = torch.zeros((img.shape[0], self.final_dim, img.shape[2], img.shape[3])).to(img.device)
        for i in range (2):
            for j in range (2):
                x['aerial-flair'] = img[:, :, 12*i:(500 + 12*i), 12*j:(500 + 12*j)]
                if 'dataset' in x.keys():
                    outs = []
                    for scale in self.scales[x['dataset']]:
                        x['scale'] = scale
                        features, out = self.model(x)
                        patch_size = int(scale / 0.2)
                        features = features[:, 1:].unsqueeze(2).repeat(1, 1, out['subpatches'].shape[2], 1)
                        dense_x = torch.cat([features, out['subpatches']], dim = 3)
                        out = self.head(dense_x)
                        B, N, _, D = out.shape
                        num_patches = int(N**(1/2))
                        size = num_patches * patch_size * self.proj_size
                        out = out.unsqueeze(2).permute(0, 2, 4, 1, 3)
                        out = out.view(B, 1, D, N, patch_size, patch_size)
                        out = out.view(B, 1, self.proj_size, self.proj_size, self.final_dim, N, patch_size, patch_size).permute(0, 1, 4, 5, 6, 2, 7, 3)
                        out = out.reshape(B, 1, self.final_dim, N, self.proj_size * patch_size, patch_size * self.proj_size)
                        out = out.view(B, 1, self.final_dim, num_patches, num_patches, patch_size * self.proj_size, patch_size * self.proj_size)
                        out = out.permute(0, 1, 2, 3, 5, 4, 6).reshape(B, 1, self.final_dim, size, size).flatten(0, 1)
                        outs.append(out)
                    outs = torch.stack(outs, dim=0)
                    out = outs.mean(dim=0)
                else:
                    features, out = self.model(x)
                    features = features[:, 1:].unsqueeze(2).repeat(1, 1, out['subpatches'].shape[2], 1)
                    dense_x = torch.cat([features, out['subpatches']], dim = 3)
                    out = self.head(dense_x)
                    B, N, _, D = out.shape
                    num_patches = int(N**(1/2))
                    size = num_patches * self.patch_size * self.proj_size

                    out = out.unsqueeze(2).permute(0, 2, 4, 1, 3)
                    out = out.view(B, 1, D, N, self.patch_size, self.patch_size)
                    out = out.view(B, 1, self.proj_size, self.proj_size, self.final_dim, N, self.patch_size, self.patch_size).permute(0, 1, 4, 5, 6, 2, 7, 3)
                    out = out.reshape(B, 1, self.final_dim, N, self.proj_size * self.patch_size, self.patch_size * self.proj_size)
                    out = out.view(B, 1, self.final_dim, num_patches, num_patches, self.patch_size * self.proj_size, self.patch_size * self.proj_size)
                    out = out.permute(0, 1, 2, 3, 5, 4, 6).reshape(B, 1, self.final_dim, size, size).flatten(0, 1)
                logits[:, :, 12*i:(500 + 12*i), 12*j:(500 + 12*j)] += out

        x['aerial-flair'] = img[:, :, 6:506, 6:506]
        if 'dataset' in x.keys():
            outs = []
            for scale in self.scales[x['dataset']]:
                x['scale'] = scale
                features, out = self.model(x)
                patch_size = int(scale / 0.2)
                features = features[:, 1:].unsqueeze(2).repeat(1, 1, out['subpatches'].shape[2], 1)
                dense_x = torch.cat([features, out['subpatches']], dim = 3)
                out = self.head(dense_x)
                B, N, _, D = out.shape
                num_patches = int(N**(1/2))
                size = num_patches * patch_size * self.proj_size
                out = out.unsqueeze(2).permute(0, 2, 4, 1, 3)
                out = out.view(B, 1, D, N, patch_size, patch_size)
                out = out.view(B, 1, self.proj_size, self.proj_size, self.final_dim, N, patch_size, patch_size).permute(0, 1, 4, 5, 6, 2, 7, 3)
                out = out.reshape(B, 1, self.final_dim, N, self.proj_size * patch_size, patch_size * self.proj_size)
                out = out.view(B, 1, self.final_dim, num_patches, num_patches, patch_size * self.proj_size, patch_size * self.proj_size)
                out = out.permute(0, 1, 2, 3, 5, 4, 6).reshape(B, 1, self.final_dim, size, size).flatten(0, 1)
                outs.append(out)
            outs = torch.stack(outs, dim=0)
            out = outs.mean(dim=0)
        else:
            features, out = self.model(x)
            features = features[:, 1:].unsqueeze(2).repeat(1, 1, out['subpatches'].shape[2], 1)
            dense_x = torch.cat([features, out['subpatches']], dim = 3)
            out = self.head(dense_x)
            B, N, _, D = out.shape
            num_patches = int(N**(1/2))
            size = num_patches * self.patch_size * self.proj_size

            out = out.unsqueeze(2).permute(0, 2, 4, 1, 3)
            out = out.view(B, 1, D, N, self.patch_size, self.patch_size)
            out = out.view(B, 1, self.proj_size, self.proj_size, self.final_dim, N, self.patch_size, self.patch_size).permute(0, 1, 4, 5, 6, 2, 7, 3)
            out = out.reshape(B, 1, self.final_dim, N, self.proj_size * self.patch_size, self.patch_size * self.proj_size)
            out = out.view(B, 1, self.final_dim, num_patches, num_patches, self.patch_size * self.proj_size, self.patch_size * self.proj_size)
            out = out.permute(0, 1, 2, 3, 5, 4, 6).reshape(B, 1, self.final_dim, size, size).flatten(0, 1)
        logits[:, :, 6:506, 6:506] += out        
        logits[:, :, 12:500, 12:500] = logits[:, :, 12:500, 12:500] / 5
        logits[:, :, 0:6, 12:500] = logits[:, :, 0:6, 12:500] / 2
        logits[:, :, 506:, 12:500] = logits[:, :, 506:, 12:500] / 2
        logits[:, :, 12:500, 0:6] = logits[:, :, 12:500, 0:6] / 2
        logits[:, :, 12:500, 506:] = logits[:, :, 12:500, 506:] / 2
        logits[:, :, 6:12, 12:500] = logits[:, :, 6:12, 12:500] / 3
        logits[:, :, 500:506, 12:500] = logits[:, :, 500:506, 12:500] / 3
        logits[:, :, 12:500, 6:12] = logits[:, :, 12:500, 6:12] / 3
        logits[:, :, 12:500, 500:506] = logits[:, :, 12:500, 500:506] / 3
        return logits


if __name__ == "__main__":
    _ = Fine()