import torch
from torch import nn
import torch.nn.functional as F

class CrossEntropyWeighted(nn.Module):
    def __init__(self, num_classes):
        super(CrossEntropyWeighted, self).__init__()
        self.weights = torch.ones(num_classes).float()
        self.weights[-1] = 0

    def forward(self, x, y):
        """
        Args:
            x: torch.Tensor BxN that contains the logits
            y: dict that contains "label": torch.Tensor BxN
        Returns:
            torch.Tensor: CrossEntropy loss between x and y: torch.Tensor([B]) while having a 0 weight 
        """
        self.weights = self.weights.to(x.device)
        return {"cross_entropy_loss": nn.functional.cross_entropy(x.flatten(2, 3), y["label"].flatten(1, 2).long(), weight=self.weights)}

class CrossEntropyIgnore(nn.Module):
    def __init__(self):
        super(CrossEntropyIgnore, self).__init__()

    def forward(self, x, y):
        """
        Args:
            x: torch.Tensor BxN that contains the logits
            y: dict that contains "label": torch.Tensor BxN
        Returns:
            torch.Tensor: CrossEntropy loss between x and y: torch.Tensor([B]) while ignoring -1 index
        """
        if len(y["label"].shape) > 1:
            x = x.flatten(2, 3)
            label = y["label"].flatten(1, 2)
        else:
            label = y["label"]
        return {"cross_entropy_loss": nn.functional.cross_entropy(x, label.long(), ignore_index=-1)}

class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, x, y):
        """
        Args:
            x: torch.Tensor BxN that contains the logits
            y: dict that contains "label": torch.Tensor BxN
        Returns:
            torch.Tensor: CrossEntropy loss between x and y: torch.Tensor([B])
        """
        return {"cross_entropy_loss": nn.functional.cross_entropy(x, y["label"])}
    
class BCEWithLogs(nn.Module):
    def __init__(self):
        super(BCEWithLogs, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, x, y):
        """
        Args:
            x: torch.Tensor BxN that contains the logits
            y: dict that contains "label": torch.Tensor BxN
        Returns:
            torch.Tensor: BCE loss between x and y: torch.Tensor([B])
        """
        return {"bce_loss": self.loss(x.float(), y["label"].float())}
    
class MILNCE(nn.Module):
    """Multiple Instance Learning Noise Contrastive Estimation (MIL-NCE) loss.
    
    This loss function implements a contrastive learning approach that handles multiple modalities
    and patches within each modality. It computes similarities between different modality features
    while handling potential masks for valid/invalid patches.

    Args:
        modalities (list): List of modality names to process
        tau (float, optional): Temperature parameter for scaling the logits. Defaults to 0.1.
            Lower values make the model more confident about its predictions.
    """

    def __init__(self, modalities, tau=0.1):
        super(MILNCE, self).__init__()
        self.tau = tau

    def cosine_similarity(self, a, b, normalize=True):
        if normalize:
            w1 = a.norm(p=2, dim=1, keepdim=True)
            w2 = b.norm(p=2, dim=1, keepdim=True)
            sim_matrix = torch.mm(a, b.t()) / (w1 * w2.t()).clamp(min=1e-8)
        else:
            sim_matrix = torch.mm(a, b.t())
        return sim_matrix

    def forward(self, input, y):
        x = input
        modalities = [item.split('_')[1] for item in list(x.keys()) if item.startswith('tokens')]

        features = [x[f'tokens_{modality}'] for modality in modalities]
        n_patches = features[0].shape[1] 
        n_tokens = n_patches * features[0].shape[0]
        features = torch.cat(features, dim=0).flatten(0, 1)
        
        # Compute similarity matrix
        logits = self.cosine_similarity(features, features, normalize=True)
        
        # Set diagonal blocks to -inf efficiently
        diag_mask = torch.block_diag(*[torch.ones(n_patches, n_patches) for _ in range(len(logits)//n_patches)])
        logits.masked_fill_(diag_mask.bool().to(logits.device), float('-inf'))

        # Handle masks if present
        masks = [item.split('_')[1] for item in list(x.keys()) if item.startswith('masks')]
        if masks:
            # Combine all masks efficiently
            mask = torch.cat([x[f'masks_{modality}'] for modality in modalities], 
                           dim=1).flatten(0, 1).float()
            
            # Create mask matrix in one operation
            mask_matrix = mask.unsqueeze(-1) @ mask.unsqueeze(0)
            
            # Apply mask
            logits.masked_fill_(~mask_matrix.bool(), float('-inf'))
            valid_entries = mask_matrix.bool()
            
            # Compute loss only on valid entries
            loss = torch.logsumexp(logits[valid_entries].view(-1, valid_entries.sum(1).max()) / self.tau, dim=1).sum()
        else:
            loss = torch.logsumexp(logits / self.tau, dim=1).sum()

        # Compute positive examples efficiently
        idx = torch.tensor([[i + j * n_tokens for j in range(len(modalities)) if j != k] 
                          for k in range(len(modalities)) for i in range(n_tokens)],
                         device=logits.device)
        pos_logits = torch.gather(logits, 1, idx)
        
        if masks:
            valid_pos = pos_logits > float('-inf')
            pos_logits = pos_logits[valid_pos.any(dim=1)]

        loss += -torch.logsumexp(pos_logits / self.tau, dim=1).sum()
        
        return {
            "contrastive_loss": loss / len(features),
            "logits": logits
        }

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, x, y):
        """
        Args:
            x: torch.Tensor BxN
            y: torch.Tensor BxN
        Returns:
            torch.Tensor: MSE loss between x and y: torch.Tensor([B, N])
        """
        return {"mse_loss": F.smooth_l1_loss(x['predicted_tokens'], y['target'])}

class MSESemSegLoss(nn.Module):
    def __init__(self):
        super(MSESemSegLoss, self).__init__()
        self.softplus = nn.Softplus()

    def forward(self, x, y):
        """
        Args:
            x: torch.Tensor Bx1xHxW
            y: dict that contains "label": torch.Tensor BxHxW
        Returns:
            torch.Tensor: MSE loss between x and y: torch.Tensor([B])
        """
        return {"mse_loss": F.mse_loss(self.softplus(x.flatten(2,3)), y['label'].flatten(1,2))}

LOSSES = {
    "crossentropyweighted": CrossEntropyWeighted,
    "crossentropyignore": CrossEntropyIgnore,
    "crossentropy": CrossEntropy,
    "bce": BCEWithLogs,
    "mil-nce": MILNCE,
    "mse": MSELoss,
    "mse-semseg": MSESemSegLoss,
}
AVERAGE = {False: lambda x: x, True: lambda x: x.mean(dim=-1)}


class Losses(nn.Module):
    """The Losses meta-object that can take a mix of losses."""

    def __init__(self, mix={}, modalities=[], patch_size=50, num_classes=0):
        """Initializes the Losses object.
        Args:
            mix (dict): dictionary with keys "loss_name" and values weight
        """
        super(Losses, self).__init__()
        assert len(mix)
        self.init_losses(mix, modalities, patch_size, num_classes)

    def init_losses(self, mix, modalities, patch_size, num_classes):
        """Initializes the losses.
        Args:
            mix (dict): dictionary with keys "loss_name" and values weight
        """
        self.loss = {}
        for m, v in mix.items():
            m = m.lower()
            try:
                if m in ["mil-nce", "mse_patch"]:
                    self.loss[m] = (LOSSES[m](modalities), v)
                elif m in ["crossentropyweighted"]:
                    self.loss[m] = (LOSSES[m](num_classes), v)
                else:
                    self.loss[m] = (LOSSES[m](), v)
            except KeyError:
                raise KeyError(f"Loss {m} not found in {LOSSES.keys()}")

    def forward(self, x, y, average=True):
        """Computes the losses.
        Args:
            x: dict that contains "gps": torch.Tensor Bx2 or "label": torch.Tensor BxN
            y: dict that contains "gps": torch.Tensor Bx2 or "label": torch.Tensor BxN
            average (bool): whether to average the losses or not
        Returns:
            dict: dictionary with losses
        """
        output = {"loss": 0}
        for loss_name, (loss, weight) in self.loss.items():
            loss_output = loss(x, y)
            for k, v in loss_output.items():
                if k.endswith("_loss"):
                    v = AVERAGE[average](v)
                    output["loss"] += weight * v
                output[k] = v
        return output