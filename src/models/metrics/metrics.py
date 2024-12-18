from torchmetrics import Metric
from torchmetrics import F1Score, Accuracy, JaccardIndex
from torchmetrics.segmentation import MeanIoU
import torch
import os
import torch.nn.functional as F
import torch.nn as nn

class MetricsAccuracy(Metric):
    """
    Computes the Ovearall Accuracy
    Args:
        modalities (list): list of modalities used
        num_classes (int): number of classes
        save_results (bool): if True saves prediction in a csv file 
        get_classes (bool): if True returns the classwise F1 Score
    """

    def __init__(
        self,
        modalities: list = [],
        num_classes: int = 15,
        save_results: bool = False,
        get_classes: bool = False,
        multilabel: bool = False
    ):
        super().__init__()
        self.get_classes = get_classes
        task = "multilabel" if multilabel else "multiclass"
        self.acc = Accuracy(task=task, num_classes=num_classes)
        self.save_results = save_results
        self.multiclass = not(multilabel)
        if save_results:
            self.results = {}

    def update(self, pred, gt):
        self.acc(pred, gt['label'])
        if self.save_results:
            for i, name in enumerate(gt['name']):
                self.results[name] = list(pred.cpu()[i].numpy())

    def compute(self):
        return {'OA': self.acc.compute()}

class MetricsMonoModal(Metric):
    """
    Computes the micro, macro and weighted F1 Score for multi label classification
    Args:
        modalities (list): list of modalities used
        num_classes (int): number of classes
        save_results (bool): if True saves prediction in a csv file 
        get_classes (bool): if True returns the classwise F1 Score
    """

    def __init__(
        self,
        modalities: list = [],
        num_classes: int = 15,
        save_results: bool = False,
        get_classes: bool = False,
        multilabel: bool = True
    ):
        super().__init__()
        self.get_classes = get_classes
        task = "multilabel" if multilabel else "multiclass"
        self.f1 = F1Score(task=task, average = "none", num_labels=num_classes, num_classes=num_classes)
        self.f1_micro = F1Score(task=task, average = "micro", num_labels=num_classes, num_classes=num_classes)
        self.f1_weighted = F1Score(task=task, average = "weighted", num_labels=num_classes, num_classes=num_classes)
        self.save_results = save_results
        self.multiclass = not(multilabel)
        if save_results:
            self.results = {}

    def update(self, pred, gt):
        if self.multiclass:
            gt['label'] = gt['label'].argmax(dim=1)
        self.f1(pred, gt['label'])
        self.f1_micro(pred, gt['label'])
        self.f1_weighted(pred, gt['label'])
        if self.save_results:
            for i, name in enumerate(gt['name']):
                self.results[name] = list(pred.cpu()[i].numpy())

    def compute(self):
        if self.get_classes:
            f1 = self.f1.compute()
            out = {'F1_Score_macro': sum(f1)/len(f1), 'F1_Score_micro': self.f1_micro.compute(), 'F1_Score_weighted': self.f1_weighted.compute()}
            for i in range(len(f1)):
                out['_'.join(['F1_classe', str(i)])] = f1[i]
            return out
        f1 = self.f1.compute()
        out = {'F1_Score_macro': sum(f1)/len(f1), 'F1_Score_micro': self.f1_micro.compute(), 'F1_Score_weighted': self.f1_weighted.compute()}
        if self.save_results:
            out['results'] = self.results
            return out
        return out

class NoMetrics(Metric):
    """
    Computes no metrics or saves a batch of reconstruction to visualise them
    Args:
        save_reconstructs (bool): if True saves a batch of reconstructions
        modalities (list): list of modalities used
        save_dir (str): where to save reconstructions
    """

    def __init__(
        self,
        save_reconstructs: bool = False,
        modalities: list = [],
        save_dir: str = '',
    ):
        super().__init__()
        self.save_dir = save_dir
        self.save_recons = save_reconstructs
        self.modalities = modalities
        if self.save_recons:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.saves = {}
            for modality in self.modalities:
                self.saves[modality] = []
                self.saves['_'.join(['gt', modality])] = []

    def update(self, pred, gt):
        if self.save_recons:
            recons, _ = pred
            for modality in self.modalities:
                if modality == 'aerial':
                    preds = recons['_'.join(['reconstruct', modality])]
                    target = gt[modality][:, :, :300, :300]
                else:
                    preds, mask = recons['_'.join(['reconstruct', modality])]
                    target = gt[modality][mask[:, 0], mask[:, 1]]
                indice = torch.randint(0, len(preds), (1,)).item()
                self.saves[modality].append(preds[indice])
                self.saves['_'.join(['gt', modality])].append(target[indice])

    def compute(self):
        if self.save_recons:
            for key in self.saves.keys():
                for i, tensor in enumerate(self.saves[key]):
                    torch.save(tensor.cpu(), self.save_dir + key + str(i) + ".pt")
        return {}
    
class MetricsContrastif(Metric):
    """
    Computes metrics for contrastive. Given embeddings for all tokens, we compute the cosine similarity matrix.
    The metric computed is the accuracy of the M -1 minimum distances of each line (except diagonal of course) 
    being the same token across other modalities with M the number of modalities.
    Args:
        modalities (list): list of modalities used
    """

    def __init__(
        self,
        modalities: list = [],
    ):
        super().__init__()
        self.modalities = modalities
        self.n_k = len(self.modalities)

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

        for i in range(len(modalities)):
            self.add_state(modalities[i], default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits):
        size = len(logits) // self.n_k
        labels = torch.arange(size).unsqueeze(1)
        labels = torch.cat([labels + i * len(labels) for i in range(self.n_k)], dim=1)
        labels = torch.cat([labels for _ in range(self.n_k)]).to(logits.device)
        for i in range(self.n_k):
            _, top_indices = torch.topk(logits[i * size:(i + 1) * size], k=self.n_k, dim=1, largest=True)
            self.__dict__[self.modalities[i]] += (torch.sum(torch.tensor([top_indices[i, j] in labels[i] 
                                                for i in range(top_indices.size(0)) for j in range(self.n_k)])) - len(top_indices)) / (self.n_k - 1)
        self.count += len(logits)

    def compute(self):
        dict = {}
        for i in range(len(self.modalities)):
            dict['_'.join(['acc', self.modalities[i]])] = self.__dict__[self.modalities[i]] / self.count
        return dict

class MetricsContrastifMulti(Metric):
    """
    Computes metrics for contrastive. Given embeddings for all tokens, we compute the cosine similarity matrix.
    The metric computed is the accuracy of the M -1 minimum distances of each line (except diagonal of course) 
    being the same token across other modalities with M the number of modalities.
    Args:
        modalities (list): list of modalities used
    """

    def __init__(
        self,
        modalities: dict = {},
    ):
        super().__init__()
        self.modalities = modalities

        for dataset in self.modalities.keys():
            self.add_state(dataset + "_count", default=torch.tensor(0), dist_reduce_fx="sum")
            for i in range(len(modalities[dataset])):
                self.add_state(dataset + "_" + modalities[dataset][i], default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, dataset):
        modalities = self.modalities[dataset]
        n_modalities = len(modalities)
        size = len(logits) // n_modalities
        labels = torch.arange(size).unsqueeze(1)
        labels = torch.cat([labels + i * len(labels) for i in range(n_modalities)], dim=1)
        labels = torch.cat([labels for _ in range(n_modalities)]).to(logits.device)
        for i in range(n_modalities):
            _, top_indices = torch.topk(logits[i * size:(i + 1) * size], k=n_modalities, dim=1, largest=True)
            self.__dict__[dataset + "_" + modalities[i]] += (torch.sum(torch.tensor([top_indices[i, j] in labels[i] 
                                                for i in range(top_indices.size(0)) for j in range(n_modalities)])) - 
                                                len(top_indices)) / (n_modalities - 1)
        self.__dict__[dataset + "_count"] += len(logits)

    def compute(self):
        dict = {}
        for dataset in self.modalities.keys():
            for i in range(len(self.modalities[dataset])):
                dict['_'.join(['acc', dataset, self.modalities[dataset][i]])] = self.__dict__[dataset + "_" + 
                                self.modalities[dataset][i]] / self.__dict__[dataset + "_count"]
        return dict
    
class MetricsSemSeg(Metric):
    """
    Computes mIoU for semantic segmentation
    Args:
        modalities (list): list of modalities used
        num_classes (int): number of classes
        save_results (bool): if True saves prediction in a csv file 
        get_classes (bool): if True returns the classwise F1 Score
    """

    def __init__(
        self,
        modalities: list = [],
        num_classes: int = 15,
        save_results: bool = False,
        get_classes: bool = False
    ):
        super().__init__()
        self.modality = modalities[0]
        self.num_classes = num_classes
        self.get_classes = get_classes
        self.miou = MeanIoU(num_classes=num_classes, per_class=True)
        self.save_results = save_results
        if save_results:
            self.results = {}

    def update(self, pred, gt):
        label = gt['label'].flatten(0, 1).long()
        self.miou(torch.nn.functional.one_hot(pred.flatten(2, 3).permute(0, 2 ,1).flatten(0, 1).argmax(dim=1), num_classes=self.num_classes), 
                  torch.nn.functional.one_hot(label, num_classes=self.num_classes))
        if self.save_results:
            for i, name in enumerate(gt['name']):
                self.results[name] = list(pred.cpu()[i].numpy())

    def compute(self):
        if self.get_classes:
            miou = self.miou.compute()
            out = {'mIoU': sum(miou)/len(miou)}
            for i in range(len(miou[:-1])):
                out['_'.join(['IoU', str(i)])] = miou[i]
            return out
        miou = self.miou.compute()
        out = {
            'IoU': miou[1].item(),  # IoU of the foreground class
        }
        if self.save_results:
            out['results'] = self.results
            return out
        return out

class MetricsSemSegJ(Metric):
    """
    Computes the IoU for binary segmentation
    Args:
        modalities (list): list of modalities used
        num_classes (int): number of classes
        save_results (bool): if True saves prediction in a csv file 
        get_classes (bool): if True returns the classwise F1 Score
    """

    def __init__(
        self,
        modalities: list = [],
        num_classes: int = 15,
        save_results: bool = False,
        get_classes: bool = False,
        save_dir: str = "",
    ):
        super().__init__()
        self.modality = modalities[0]
        self.num_classes = num_classes
        self.get_classes = get_classes
        self.miou = JaccardIndex(task="multiclass", num_classes=2, ignore_index=-1)
        self.save_results = save_results
        if save_results:
            self.save_dir = save_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

    def update(self, pred, gt):
        self.miou(pred.flatten(2, 3).permute(0, 2 ,1).flatten(0, 1).argmax(dim=1), 
                  gt['label'].flatten(1, 2).flatten(0, 1).long())
        if self.save_results:
            for i, name in enumerate(gt['name']):
                np.save(self.save_dir + str(name) + '.npy', pred.cpu()[i].numpy())
                np.save(self.save_dir + str(name) + '_gt.npy', gt['label'].cpu()[i].numpy())

    def compute(self):
        if self.get_classes:
            miou = self.miou.compute()
            out = {'mIoU': sum(miou[:-1])/len(miou[:-1])}
            for i in range(len(miou[:-1])):
                out['_'.join(['IoU', str(i)])] = miou[i]
            return out
        miou = self.miou.compute()
        out = {
            'IoU': miou,
        }
        return out

import numpy as np
import torch


class Metric(object):
    """Base class for all metrics.
    From: https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py
    """

    def reset(self):
        pass

    def add(self):
        pass

    def value(self):
        pass


class ConfusionMatrix(Metric):
    """Constructs a confusion matrix for a multi-class classification problems.

    Does not support multi-label, multi-class problems.

    Keyword arguments:
    - num_classes (int): number of classes in the classification problem.
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.

    Modified from: https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
    """

    def __init__(self, num_classes, normalized=False, device='cpu', lazy=True):
        super().__init__()
        if device == 'cpu':
            self.conf = np.ndarray((num_classes, num_classes), dtype=np.int64)
        else:
            self.conf = torch.zeros((num_classes, num_classes)).cuda()
        self.normalized = normalized
        self.num_classes = num_classes
        self.device = device
        self.reset()
        self.lazy = lazy

    def reset(self):
        if self.device == 'cpu':
            self.conf.fill(0)
        else:
            self.conf = torch.zeros(self.conf.shape).cuda()

    def add(self, predicted, target):
        """Computes the confusion matrix

        The shape of the confusion matrix is K x K, where K is the number
        of classes.

        Keyword arguments:
        - predicted (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        predicted scores obtained from the model for N examples and K classes,
        or an N-tensor/array of integer values between 0 and K-1.
        - target (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        ground-truth classes for N examples and K classes, or an N-tensor/array
        of integer values between 0 and K-1.

        """

        # If target and/or predicted are tensors, convert them to numpy arrays
        if self.device == 'cpu':
            if torch.is_tensor(predicted):
                predicted = predicted.cpu().numpy()
            if torch.is_tensor(target):
                target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if len(predicted.shape) != 1:
            assert predicted.shape[1] == self.num_classes, \
                'number of predictions does not match size of confusion matrix'
            predicted = predicted.argmax(1)
        else:
            if not self.lazy:
                assert (predicted.max() < self.num_classes) and (predicted.min() >= 0), \
                    'predicted values are not between 0 and k-1'

        if len(target.shape) != 1:
            if not self.lazy:
                assert target.shape[1] == self.num_classes, \
                    'Onehot target does not match size of confusion matrix'
                assert (target >= 0).all() and (target <= 1).all(), \
                    'in one-hot encoding, target values should be 0 or 1'
                assert (target.sum(1) == 1).all(), \
                    'multi-label setting is not supported'
            target = target.argmax(1)
        else:
            if not self.lazy:
                assert (target.max() < self.num_classes) and (target.min() >= 0), \
                    'target values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.num_classes * target

        if self.device == 'cpu':
            bincount_2d = np.bincount(
                x.astype(np.int64), minlength=self.num_classes ** 2)
            assert bincount_2d.size == self.num_classes ** 2
            conf = bincount_2d.reshape((self.num_classes, self.num_classes))
        else:
            bincount_2d = torch.bincount(
                x, minlength=self.num_classes ** 2)

            conf = bincount_2d.view((self.num_classes, self.num_classes))
        self.conf += conf

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf


class IoU(Metric):
    """Computes the intersection over union (IoU) per class and corresponding
    mean (mIoU).

    Intersection over union (IoU) is a common evaluation metric for semantic
    segmentation. The predictions are first accumulated in a confusion matrix
    and the IoU is computed from it as follows:

        IoU = true_positive / (true_positive + false_positive + false_negative).

    Keyword arguments:
    - num_classes (int): number of classes in the classification problem
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.
    - ignore_index (int or iterable, optional): Index of the classes to ignore
    when computing the IoU. Can be an int, or any iterable of ints.
    """

    def __init__(self, num_classes, normalized=False, ignore_index=None, cm_device='cpu', lazy=True, visu=False, save_dir=''):
        super().__init__()
        self.conf_metric = ConfusionMatrix(num_classes, normalized, device=cm_device, lazy=lazy)
        self.lazy = lazy
        self.visu = visu
        if self.visu:
            self.save_dir = save_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def reset(self):
        self.conf_metric.reset()

    def update(self, predicted, tg):
        """Adds the predicted and target pair to the IoU metric.

        Keyword arguments:
        - predicted (Tensor): Can be a (N, K, H, W) tensor of
        predicted scores obtained from the model for N examples and K classes,
        or (N, H, W) tensor of integer values between 0 and K-1.
        - target (Tensor): Can be a (N, K, H, W) tensor of
        target scores for N examples and K classes, or (N, H, W) tensor of
        integer values between 0 and K-1.

        """
        target = tg['label']
        # Dimensions check
        assert predicted.size(0) == target.size(0), \
            'number of targets and predicted outputs do not match'
        assert predicted.dim() == 3 or predicted.dim() == 4, \
            "predictions must be of dimension (N, H, W) or (N, K, H, W)"
        assert target.dim() == 3 or target.dim() == 4, \
            "targets must be of dimension (N, H, W) or (N, K, H, W)"

        # If the tensor is in categorical format convert it to integer format
        if predicted.dim() == 4:
            _, predicted = predicted.max(1)
        if target.dim() == 4:
            _, target = target.max(1)

        self.conf_metric.add(predicted.view(-1), target.view(-1))
        if self.visu:
            for i, name in enumerate(tg['name']):
                np.save(self.save_dir + str(name) + '.npy', predicted.cpu()[i].numpy())
                np.save(self.save_dir + str(name) + '_gt.npy', target.cpu()[i].numpy())

    def value(self):
        """Computes the IoU and mean IoU.

        The mean computation ignores NaN elements of the IoU array.

        Returns:
            Tuple: (IoU, mIoU). The first output is the per class IoU,
            for K classes it's numpy.ndarray with K elements. The second output,
            is the mean IoU.
        """
        conf_matrix = self.conf_metric.value()
        if self.ignore_index is not None:
            conf_matrix[:, self.ignore_index] = 0
            conf_matrix[self.ignore_index, :] = 0
        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)

        return iou, np.nanmean(iou)

    def compute(self):
        conf_matrix = self.conf_metric.value()
        if torch.is_tensor(conf_matrix):
            conf_matrix = conf_matrix.cpu().numpy()
        if self.ignore_index is not None:
            conf_matrix[:, self.ignore_index] = 0
            conf_matrix[self.ignore_index, :] = 0
        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)
        miou = float(np.nanmean(iou))
        acc = float(np.diag(conf_matrix).sum() / conf_matrix.sum() * 100)
        out =  {'mIoU': miou, 'acc': acc}
        return out

class MetricsBinarySemSeg(Metric):
    """
    Computes IoU Score for binary segmentation tasks
    Args:
        modalities (list): list of modalities used
        save_results (bool): if True saves prediction in a csv file 
        threshold (float): threshold for binary prediction (default: 0.5)
    """

    def __init__(
        self,
        modalities: list = [],
        save_results: bool = False,
        threshold: float = 0.5
    ):
        super().__init__()
        self.modality = modalities[0]
        self.threshold = threshold
        self.miou = MeanIoU(num_classes=2, per_class=True)  # Binary: 2 classes (0 and 1)
        self.save_results = save_results
        if save_results:
            self.results = {}

    def update(self, pred, gt):
        # Convert predictions to binary using threshold
        pred_binary = (pred.sigmoid() > self.threshold).float()
        
        # Convert to one-hot encoding
        pred_one_hot = torch.nn.functional.one_hot(
            pred_binary.flatten(2, 3).permute(0, 2, 1).flatten(0, 1).long(), 
            num_classes=2
        )
        gt_one_hot = torch.nn.functional.one_hot(
            gt['label'].flatten(1, 2).flatten(0, 1).long(), 
            num_classes=2
        )
        
        self.miou(pred_one_hot, gt_one_hot)

        if self.save_results:
            for i, name in enumerate(gt['name']):
                self.results[name] = pred_binary.cpu()[i].numpy()

    def compute(self):
        miou = self.miou.compute()
        # For binary segmentation, we typically care about IoU of class 1 (foreground)
        out = {
            'mIoU': miou[1].item(),  # IoU of the foreground class
            'IoU_background': miou[0].item(),
            'IoU_foreground': miou[1].item()
        }
        
        if self.save_results:
            out['results'] = self.results
        
        return out


class MetricsReg(Metric):
    """
    Computes the Root Mean Square Error (RMSE) for regression tasks by applying a softplus activation 
    to the predictions before computing the MSE loss.

    Args:
        modalities (list): List of modalities used in the model. Currently not used in the implementation
                          but kept for consistency with other metric classes.

    Attributes:
        softplus (nn.Softplus): Softplus activation function applied to predictions
        mse (torch.Tensor): Accumulated mean squared error
        total_samples (torch.Tensor): Counter for total number of samples processed
    """

    def __init__(
        self,
        modalities: list = [],
    ):
        super().__init__()
        self.softplus = nn.Softplus()
        self.mse = torch.tensor(0.0)
        self.total_samples = torch.tensor(0)

    def update(self, pred, gt):
        self.mse += F.mse_loss(self.softplus(pred), gt['label']).cpu()
        self.total_samples += 1
        
    def compute(self):
        rmse = torch.sqrt(self.mse / self.total_samples)
        out = {'RMSE': rmse.item()}            
        return out
    
class SegPangaea(Metric):
    """
    SegPangaea is a class for evaluating segmentation models using a confusion matrix approach.

    Attributes:
        num_classes (int): Number of classes in the segmentation task
        ignore_index (int): Index value to ignore when computing metrics
        confusion_matrix (torch.Tensor): Matrix of shape (num_classes, num_classes) to store predictions

    Methods:
        update(pred, gt):
            Updates the confusion matrix with new predictions and ground truth.
            Args:
                pred (torch.Tensor): Model predictions
                gt (dict): Dictionary containing ground truth labels under 'label' key
                
        compute():
            Computes various metrics from the accumulated confusion matrix.
            Returns:
                dict: Dictionary containing the following metrics:
                    - mIoU: Mean Intersection over Union across all classes
                    - mF1: Mean F1 score across all classes  
                    - mAcc: Mean pixel accuracy
    """

    def __init__(self, num_classes, ignore_index):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = torch.zeros(num_classes, num_classes)

    def update(self, pred, gt):
        label = gt['label'].flatten(1, 2)
        pred = torch.argmax(pred, dim=1).flatten(1, 2)
        valid_mask = label != self.ignore_index
        pred, target = pred[valid_mask], label[valid_mask]
        count = torch.bincount(
            (pred * self.num_classes + target), minlength=self.num_classes ** 2
        )
        self.confusion_matrix = self.confusion_matrix.to(pred.device)
        self.confusion_matrix += count.view(self.num_classes, self.num_classes)

    def compute(self):
        # Calculate IoU for each class
        intersection = torch.diag(self.confusion_matrix)
        union = self.confusion_matrix.sum(dim=1) + self.confusion_matrix.sum(dim=0) - intersection
        iou = (intersection / (union + 1e-6))

        # Calculate precision and recall for each class
        precision = intersection / (self.confusion_matrix.sum(dim=0) + 1e-6)
        recall = intersection / (self.confusion_matrix.sum(dim=1) + 1e-6)

        # Calculate F1-score for each class
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        # Calculate mean IoU, mean F1-score, and mean Accuracy
        miou = iou.mean().item()
        mf1 = f1.mean().item()
        macc = (intersection.sum() / (self.confusion_matrix.sum() + 1e-6)).item()

        # Convert metrics to CPU and to Python scalars
        iou = iou.cpu()
        f1 = f1.cpu()
        precision = precision.cpu()
        recall = recall.cpu()

        # Prepare the metrics dictionary
        metrics = {
            "mIoU": miou,
            "mF1": mf1,
            "mAcc": macc,
        }

        return metrics


