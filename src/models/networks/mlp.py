import torch
from torch import nn

import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(
        self,
        initial_dim=512,
        hidden_dim=[128, 32, 2],
        final_dim=2,
        norm=nn.InstanceNorm1d,
        activation=nn.ReLU,
    ):
        """
        Initializes an MLP Classification Head
        Args:
            initial_dim (int): dimension of input layer
            hidden_dim (list): list of hidden dimensions for the MLP
            final_dim (int): dimension of output layer
            norm (nn.Module): normalization layer
            activation (nn.Module): activation layer
        """
        super().__init__()
        dim = [initial_dim] + hidden_dim + [final_dim]
        args = self.init_layers(dim, norm, activation)
        self.mlp = nn.Sequential(*args)

    def init_layers(self, dim, norm, activation):
        """Initializes the MLP layers."""
        args = [nn.LayerNorm(dim[0])]
        for i in range(len(dim) - 1):
            args.append(nn.Linear(dim[i], dim[i + 1]))
            if i < len(dim) - 2:
                args.append(norm(dim[i + 1]))
                #args.append(norm(4, dim[i + 1]))
                args.append(activation())
        return args

    def forward(self, x):
        """
        Predicts output
        Args:
            x: torch.Tensor with features
        """
        return self.mlp(x)
    
class Identity(nn.Module):
    def __init__(
        self,
    ):
        """
        Initializes a module that computes Identity
        """
        super().__init__()

    def forward(self, x):
        """
        Computes Identity
        """
        return x

class MLPSemSeg(nn.Module):
    def __init__(
        self,
        initial_dim=512,
        hidden_dim=[128, 32, 2],
        final_dim=2,
        norm=nn.InstanceNorm1d,
        activation=nn.ReLU,
        patch_size: int = 1,
    ):
        """
        Initializes an MLP Classification Head
        Args:
            initial_dim (int): dimension of input layer
            hidden_dim (list): list of hidden dimensions for the MLP
            final_dim (int): dimension of output layer
            norm (nn.Module): normalization layer
            activation (nn.Module): activation layer
        """
        super().__init__()
        dim = [initial_dim] + hidden_dim + [final_dim * patch_size * patch_size]
        args = self.init_layers(dim, norm, activation)
        self.mlp = nn.Sequential(*args)
        self.patch_size = patch_size
        self.final_dim = final_dim

    def init_layers(self, dim, norm, activation):
        """Initializes the MLP layers."""
        args = [nn.LayerNorm(dim[0])]
        for i in range(len(dim) - 1):
            args.append(nn.Linear(dim[i], dim[i + 1]))
            if i < len(dim) - 2:
                #args.append(norm(dim[i + 1]))
                args.append(activation())
        return args

    def forward(self, x):
        """
        Predicts output
        Args:
            x: torch.Tensor with features
        """
        x = x[:, 1:]
        x = self.mlp(x)
        B, N, D = x.shape
        num_patches = int(N**(1/2))
        size = num_patches * self.patch_size

        x = x.view(B, N, 1, D).view(B, N, 1, self.final_dim, self.patch_size * self.patch_size).permute(0, 2, 3, 1, 4)
        x = x.view(B, 1, self.final_dim, N, self.patch_size, self.patch_size)
        x = x.view(B, 1, self.final_dim, num_patches, num_patches, self.patch_size, self.patch_size).permute(0, 1, 2, 3, 5, 4, 6)
        x = x.reshape(B, 1, self.final_dim, size, size).flatten(0, 1)
        return x
    
class MLPDenseSemSeg(nn.Module):
    def __init__(
        self,
        initial_dim=512,
        hidden_dim=[128, 32, 2],
        final_dim=2,
        norm=nn.InstanceNorm1d,
        activation=nn.ReLU,
        patch_size: int = 1,
    ):
        """
        Initializes an MLP Classification Head
        Args:
            initial_dim (int): dimension of input layer
            hidden_dim (list): list of hidden dimensions for the MLP
            final_dim (int): dimension of output layer
            norm (nn.Module): normalization layer
            activation (nn.Module): activation layer
        """
        super().__init__()
        dim = [initial_dim * 2] + hidden_dim + [final_dim]
        args = self.init_layers(dim, norm, activation)
        self.mlp = nn.Sequential(*args)
        self.patch_size = patch_size
        self.final_dim = final_dim

    def init_layers(self, dim, norm, activation):
        """Initializes the MLP layers."""
        args = [nn.LayerNorm(dim[0])]
        for i in range(len(dim) - 1):
            args.append(nn.Linear(dim[i], dim[i + 1]))
            if i < len(dim) - 2:
                #args.append(norm(dim[i + 1]))
                args.append(activation())
        return args

    def forward(self, features):
        """
        Predicts output
        Args:
            x: torch.Tensor with features
        """
        x, out = features
        x = x[:, 1:].unsqueeze(2).repeat(1, 1, out['subpatches'].shape[2], 1)
        dense_x = torch.cat([x, out['subpatches']], dim = 3)
        x = self.mlp(dense_x)
        B, N, _, D = x.shape
        num_patches = int(N**(1/2))
        size = num_patches * self.patch_size

        x = x.unsqueeze(2).permute(0, 2, 4, 1, 3)
        x = x.view(B, 1, self.final_dim, N, self.patch_size, self.patch_size)
        x = x.view(B, 1, self.final_dim, num_patches, num_patches, self.patch_size, self.patch_size).permute(0, 1, 2, 3, 5, 4, 6)
        x = x.reshape(B, 1, self.final_dim, size, size).flatten(0, 1)
        return x
    
class MLPDenseSemSegFLAIR(nn.Module):
    def __init__(
        self,
        initial_dim=512,
        hidden_dim=[128, 32, 2],
        final_dim=2,
        norm=nn.InstanceNorm1d,
        activation=nn.ReLU,
        patch_size: int = 1,
        resolution: float = 1.0,
        proj_size: int = 1,
    ):
        """
        Initializes an MLP Classification Head
        Args:
            initial_dim (int): dimension of input layer
            hidden_dim (list): list of hidden dimensions for the MLP
            final_dim (int): dimension of output layer
            norm (nn.Module): normalization layer
            activation (nn.Module): activation layer
        """
        super().__init__()
        self.proj_size = proj_size
        dim = [initial_dim * 2] + hidden_dim + [final_dim * proj_size * proj_size]
        args = self.init_layers(dim, norm, activation)
        self.mlp = nn.Sequential(*args)
        if resolution < 1:
            self.patch_size = int(patch_size / resolution)
        else:
            self.patch_size = patch_size
        self.final_dim = final_dim

    def init_layers(self, dim, norm, activation):
        """Initializes the MLP layers."""
        args = [nn.LayerNorm(dim[0])]
        for i in range(len(dim) - 1):
            args.append(nn.Linear(dim[i], dim[i + 1]))
            if i < len(dim) - 2:
                args.append(activation())
        return args

    def forward(self, features):
        """
        Predicts output
        Args:
            x: torch.Tensor with features
        """
        x, out = features
        x = x[:, 1:].unsqueeze(2).repeat(1, 1, out['subpatches'].shape[2], 1)
        dense_x = torch.cat([x, out['subpatches']], dim = 3)
        x = self.mlp(dense_x)
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

class MLPDenseSemSegFLAIRmulti(nn.Module):
    def __init__(
        self,
        initial_dim=512,
        hidden_dim=[128, 32, 2],
        final_dim=2,
        norm=nn.InstanceNorm1d,
        activation=nn.ReLU,
        resolution: float = 1.0,
        proj_size: int = 1,
    ):
        """
        Initializes an MLP Classification Head
        Args:
            initial_dim (int): dimension of input layer
            hidden_dim (list): list of hidden dimensions for the MLP
            final_dim (int): dimension of output layer
            norm (nn.Module): normalization layer
            activation (nn.Module): activation layer
        """
        super().__init__()
        self.proj_size = proj_size
        dim = [initial_dim * 2] + hidden_dim + [final_dim * proj_size * proj_size]
        args = self.init_layers(dim, norm, activation)
        self.mlp = nn.Sequential(*args)
        if resolution < 10:
            self.resolution = resolution
        self.final_dim = final_dim

    def init_layers(self, dim, norm, activation):
        """Initializes the MLP layers."""
        args = [nn.LayerNorm(dim[0])]
        for i in range(len(dim) - 1):
            args.append(nn.Linear(dim[i], dim[i + 1]))
            if i < len(dim) - 2:
                args.append(activation())
        return args

    def forward(self, features, patch_size):
        """
        Predicts output
        Args:
            x: torch.Tensor with features
        """
        if self.resolution:
            patch_size = int(patch_size / self.resolution)
        x, out = features
        x = x[:, 1:].unsqueeze(2).repeat(1, 1, out['subpatches'].shape[2], 1)
        dense_x = torch.cat([x, out['subpatches']], dim = 3)
        x = self.mlp(dense_x)
        B, N, _, D = x.shape
        num_patches = int(N**(1/2))
        size = num_patches * patch_size * self.proj_size

        x = x.unsqueeze(2).permute(0, 2, 4, 1, 3)
        x = x.view(B, 1, D, N, patch_size, patch_size)
        x = x.view(B, 1, self.proj_size, self.proj_size, self.final_dim, N, patch_size, patch_size).permute(0, 1, 4, 5, 6, 2, 7, 3)
        x = x.reshape(B, 1, self.final_dim, N, self.proj_size * patch_size, patch_size * self.proj_size)
        x = x.view(B, 1, self.final_dim, num_patches, num_patches, patch_size * self.proj_size, patch_size * self.proj_size)
        x = x.permute(0, 1, 2, 3, 5, 4, 6).reshape(B, 1, self.final_dim, size, size).flatten(0, 1)
        return x

class MLPDenseSemSegDeconv(nn.Module):
    def __init__(
        self,
        initial_dim=512,
        hidden_dim=[128, 32, 2],
        final_dim=2,
        norm=nn.InstanceNorm1d,
        activation=nn.ReLU,
        patch_size: int = 1,
    ):
        """
        Initializes an MLP Classification Head
        Args:
            initial_dim (int): dimension of input layer
            hidden_dim (list): list of hidden dimensions for the MLP
            final_dim (int): dimension of output layer
            norm (nn.Module): normalization layer
            activation (nn.Module): activation layer
        """
        super().__init__()
        self.final_dim = final_dim
        self.deconv = nn.ModuleList([nn.Conv2d(initial_dim * 2, initial_dim // 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(initial_dim // 2, final_dim, kernel_size=1, stride=1, padding=0, bias=True)
            ])
        self.patch_size = patch_size

    def forward(self, features):
        """
        Predicts output
        Args:
            x: torch.Tensor with features
        """
        x, out = features
        x = x[:, 1:].unsqueeze(2).repeat(1, 1, out['subpatches'].shape[2], 1)
        x = torch.cat([x, out['subpatches']], dim = 3)
        B, N, _, D = x.shape
        num_patches = int(N**(1/2))
        size = num_patches * self.patch_size

        x = x.unsqueeze(2).permute(0, 2, 4, 1, 3)
        x = x.view(B, 1, D, N, self.patch_size, self.patch_size)
        x = x.view(B, 1, D, num_patches, num_patches, self.patch_size, self.patch_size).permute(0, 1, 2, 3, 5, 4, 6)
        x = x.reshape(B, 1, D, size, size).flatten(0, 1)
        for i in range (len(self.deconv)):
            x = self.deconv[i](x)
        return x

class BilinearUpsampleLayer(nn.Module):
    def __init__(self, output_size):
        super(BilinearUpsampleLayer, self).__init__()
        self.output_size = output_size  # Desired output size as (height, width)

    def forward(self, x):
        return F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=True)

class MLPDenseSemSegDeconvFLAIR(nn.Module):
    def __init__(
        self,
        initial_dim=512,
        hidden_dim=[128, 32, 2],
        final_dim=2,
        norm=nn.InstanceNorm1d,
        activation=nn.ReLU,
        patch_size: int = 1,
        resolution: float = 1.0,
        proj_size: int = 1,
    ):
        """
        Initializes an MLP Classification Head
        Args:
            initial_dim (int): dimension of input layer
            hidden_dim (list): list of hidden dimensions for the MLP
            final_dim (int): dimension of output layer
            norm (nn.Module): normalization layer
            activation (nn.Module): activation layer
        """
        super().__init__()
        self.final_dim = final_dim
        self.deconv = nn.ModuleList([nn.Conv2d(initial_dim * 2, initial_dim, kernel_size=1, stride=1, padding=0, bias=True),
            BilinearUpsampleLayer(output_size=(125, 125)),
            nn.Conv2d(initial_dim, initial_dim // 2, kernel_size=3, stride=1, padding=1, bias=True),
            BilinearUpsampleLayer(output_size=(500, 500)),
            nn.Conv2d(initial_dim // 2, initial_dim // 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(initial_dim // 4, final_dim, kernel_size=1, stride=1, padding=0, bias=True),
            ])
        if resolution < 1:
            self.patch_size = int(patch_size / resolution)
        else:
            self.patch_size = patch_size

    def forward(self, features):
        """
        Predicts output
        Args:
            x: torch.Tensor with features
        """
        x, out = features
        x = x[:, 1:].unsqueeze(2).repeat(1, 1, out['subpatches'].shape[2], 1)
        x = torch.cat([x, out['subpatches']], dim = 3)
        B, N, _, D = x.shape
        num_patches = int(N**(1/2))
        size = num_patches * self.patch_size

        x = x.unsqueeze(2).permute(0, 2, 4, 1, 3)
        x = x.view(B, 1, D, N, self.patch_size, self.patch_size)
        x = x.view(B, 1, D, num_patches, num_patches, self.patch_size, self.patch_size).permute(0, 1, 2, 3, 5, 4, 6)
        x = x.reshape(B, 1, D, size, size).flatten(0, 1)
        for i in range (len(self.deconv)):
            x = self.deconv[i](x)
        return x

class DecoderDeconvNoIndicesAerial(nn.Module):
    """
    Decoder for aerial data with deconvolutions without index bypass
    """
    def __init__(self,
                 in_channels: int = 10,
                 embed_dim: int = 128
                 ):
        super(DecoderDeconvNoIndicesAerial, self).__init__()
        self.decode = nn.ModuleList([nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=4, stride=1, padding=0, bias=False),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=0, bias=True),
            nn.ConvTranspose2d(embed_dim, in_channels, kernel_size=3, stride=2, padding=1, bias=True),
            ])

    def forward(self, x):
        x = x[:, 1:]
        i = x.shape[0] * x.shape[1]
        sizes = [torch.Size([i, 4, 50, 50])]
        shape = x.shape
        x = x.unsqueeze(-1).unsqueeze(-1).flatten(0,1)
        for i in range (len(self.decode) - 1):
            x = self.decode[i](x)
        x = self.decode[-1](x, output_size=sizes[-1])
        x = x.view(shape[0], shape[1], x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)
        return x.flatten(3, 4)

class MLPDenseSemSegFLAIR2(nn.Module):
    def __init__(
        self,
        initial_dim=512,
        hidden_dim=[128, 32, 2],
        final_dim=2,
        norm=nn.InstanceNorm1d,
        activation=nn.ReLU,
        patch_size: int = 1,
        resolution: float = 1.0,
        proj_size: int = 1,
    ):
        """
        Initializes an MLP Classification Head
        Args:
            initial_dim (int): dimension of input layer
            hidden_dim (list): list of hidden dimensions for the MLP
            final_dim (int): dimension of output layer
            norm (nn.Module): normalization layer
            activation (nn.Module): activation layer
        """
        super().__init__()
        self.proj_size = proj_size
        dim = [initial_dim * 2] + hidden_dim + [final_dim * proj_size * proj_size]
        args = self.init_layers(dim, norm, activation)
        self.mlp = nn.Sequential(*args)
        if resolution < 1:
            self.patch_size = int(patch_size / resolution)
        else:
            self.patch_size = patch_size
        self.final_dim = final_dim
        self.deconv = nn.ModuleList([nn.Conv2d(final_dim, final_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(final_dim, final_dim, kernel_size=3, stride=1, padding=1, bias=True)
            ])

    def init_layers(self, dim, norm, activation):
        """Initializes the MLP layers."""
        args = [nn.LayerNorm(dim[0])]
        for i in range(len(dim) - 1):
            args.append(nn.Linear(dim[i], dim[i + 1]))
            if i < len(dim) - 2:
                args.append(activation())
        return args

    def forward(self, features):
        """
        Predicts output
        Args:
            x: torch.Tensor with features
        """
        x, out = features
        x = x[:, 1:].unsqueeze(2).repeat(1, 1, out['subpatches'].shape[2], 1)
        dense_x = torch.cat([x, out['subpatches']], dim = 3)
        x = self.mlp(dense_x)
        B, N, _, D = x.shape
        num_patches = int(N**(1/2))
        size = num_patches * self.patch_size * self.proj_size

        x = x.unsqueeze(2).permute(0, 2, 4, 1, 3)
        x = x.view(B, 1, D, N, self.patch_size, self.patch_size)
        x = x.view(B, 1, self.proj_size, self.proj_size, self.final_dim, N, self.patch_size, self.patch_size).permute(0, 1, 4, 5, 6, 2, 7, 3)
        x = x.reshape(B, 1, self.final_dim, N, self.proj_size * self.patch_size, self.patch_size * self.proj_size)
        x = x.view(B, 1, self.final_dim, num_patches, num_patches, self.patch_size * self.proj_size, self.patch_size * self.proj_size)
        x = x.permute(0, 1, 2, 3, 5, 4, 6).reshape(B, 1, self.final_dim, size, size).flatten(0, 1)
        for i in range (len(self.deconv)):
            x = self.deconv[i](x)
        return x
    
class MLPDenseSemSegFLAIR3(nn.Module):
    def __init__(
        self,
        initial_dim=512,
        hidden_dim=[128, 32, 2],
        final_dim=2,
        norm=nn.InstanceNorm1d,
        activation=nn.ReLU,
        patch_size: int = 1,
        resolution: float = 1.0,
        proj_size: int = 1,
    ):
        """
        Initializes an MLP Classification Head
        Args:
            initial_dim (int): dimension of input layer
            hidden_dim (list): list of hidden dimensions for the MLP
            final_dim (int): dimension of output layer
            norm (nn.Module): normalization layer
            activation (nn.Module): activation layer
        """
        super().__init__()
        self.proj_size = 1
        dim = [initial_dim * 2] + hidden_dim + [initial_dim // 2]
        args = self.init_layers(dim, norm, activation)
        self.mlp = nn.Sequential(*args)
        if resolution < 1:
            self.patch_size = int(patch_size / resolution)
        else:
            self.patch_size = patch_size
        self.inter_dim = initial_dim // 2
        self.final_dim = final_dim
        self.upsampling = BilinearUpsampleLayer(output_size=(500, 500))
        self.deconv = nn.ModuleList([nn.Conv2d(initial_dim // 2 + 5, initial_dim // 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(initial_dim // 4, final_dim, kernel_size=3, stride=1, padding=1, bias=True)
            ])

    def init_layers(self, dim, norm, activation):
        """Initializes the MLP layers."""
        args = [nn.LayerNorm(dim[0])]
        for i in range(len(dim) - 1):
            args.append(nn.Linear(dim[i], dim[i + 1]))
            if i < len(dim) - 2:
                args.append(activation())
        return args

    def forward(self, features, x_modality):
        """
        Predicts output
        Args:
            x: torch.Tensor with features
        """
        x, out = features
        x = x[:, 1:].unsqueeze(2).repeat(1, 1, out['subpatches'].shape[2], 1)
        dense_x = torch.cat([x, out['subpatches']], dim = 3)
        x = self.mlp(dense_x)
        B, N, _, D = x.shape
        num_patches = int(N**(1/2))
        size = num_patches * self.patch_size * self.proj_size

        x = x.unsqueeze(2).permute(0, 2, 4, 1, 3)
        x = x.view(B, 1, D, N, self.patch_size, self.patch_size)
        x = x.view(B, 1, self.proj_size, self.proj_size, self.inter_dim, N, self.patch_size, self.patch_size).permute(0, 1, 4, 5, 6, 2, 7, 3)
        x = x.reshape(B, 1, self.inter_dim, N, self.proj_size * self.patch_size, self.patch_size * self.proj_size)
        x = x.view(B, 1, self.inter_dim, num_patches, num_patches, self.patch_size * self.proj_size, self.patch_size * self.proj_size)
        x = x.permute(0, 1, 2, 3, 5, 4, 6).reshape(B, 1, self.inter_dim, size, size).flatten(0, 1)
        x = self.upsampling(x)
        x = torch.cat([x, x_modality], dim=1)
        for i in range (len(self.deconv)):
            x = self.deconv[i](x)
        return x
