import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from models.networks.encoder.Pse import PixelSetEncoder
from models.networks.encoder.Tae import TemporalAttentionEncoder

modality_to_channels = {'s2': 10, 's1': 3, 'alos': 3, 'l7': 6}

class PseTae(nn.Module):
    """
    Pixel-Set encoder + Temporal Attention Encoder sequence classifier
    """

    def __init__(self, modalities, mlp1=[10, 32, 64], pooling='mean_std', mlp2=[132, 128], with_extra=True,
                 extra_size=4,
                 n_head=4, d_k=32, d_model=None, mlp3=[512, 128, 128], dropout=0.2, T=1000, len_max_seq=24,
                 output_dim=128, positions=None):
        super(PseTae, self).__init__()
        self.modality = modalities[0]
        input_dim = modality_to_channels[self.modality]
        mlp1.insert(0, input_dim)
        mlp2.insert(0, mlp1[-1] * len(pooling.split('_')) + extra_size)
        mlp3.insert(0, mlp2[-1] * n_head)
        mlp3.append(output_dim)
        self.spatial_encoder = PixelSetEncoder(input_dim, mlp1=mlp1, pooling=pooling, mlp2=mlp2, with_extra=with_extra,
                                               extra_size=extra_size)
        self.temporal_encoder = TemporalAttentionEncoder(in_channels=mlp2[-1], n_head=n_head, d_k=d_k, d_model=d_model,
                                                         n_neurons=mlp3, dropout=dropout,
                                                         T=T, len_max_seq=len_max_seq, positions=positions)

    def forward(self, x):
        """
         Args:
            input(tuple): (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
            Pixel-Set : Batch_size x Sequence length x Channel x Number of pixels
            Pixel-Mask : Batch_size x Sequence length x Number of pixels
            Extra-features : Batch_size x Sequence length x Number of features
        """
        input = x[self.modality]
        input_shape = input.shape
        input = input.view(input_shape[0], input_shape[1], input_shape[2],
                                           input_shape[3] * input_shape[4])
        """
        seq_length = max(tensor.shape[0] for tensor in input)
        input_shape = input[0].shape
        concatenated_tensor = [torch.zeros((seq_length, input_shape[1], 
                                           input_shape[2] * input_shape[3]), dtype=torch.float32) for i in range (len(input))]
        mask = torch.zeros((len(concatenated_tensor), seq_length, input_shape[2] * input_shape[3]))
        for i, tensor in enumerate(input):
            concatenated_tensor[i][:tensor.shape[0], :, :] += tensor.view(tensor.shape[0], input_shape[1], 
                                           input_shape[2] * input_shape[3])
            mask[i, :tensor.shape[0], :] += torch.ones(tensor.shape[0], input_shape[2] * input_shape[3])
        concatenated_tensor = torch.stack(concatenated_tensor)
        out = self.spatial_encoder((concatenated_tensor, mask))
        """
        mask = torch.zeros((input_shape[0], input_shape[1],
                                           input_shape[3] * input_shape[4]))
        for i in range (len(input)):
            if torch.max(input[i]) > 0:
                mask[i] += torch.ones((input_shape[1], input_shape[3] * input_shape[4]))

        out = self.spatial_encoder((input, mask.to(input.device)))
        out = self.temporal_encoder(out, x['_'.join([self.modality, "dates"])])
        return out