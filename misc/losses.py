import torch
import torch.nn.functional as F
import numpy as np

def linear_rampup(current, rampup_length=6400):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __init__(self):
        super().__init__()

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, lambda_u, rampup_length):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, lambda_u * linear_rampup(epoch, rampup_length)


def dice_loss(reference, logits):
    """Computes the dice loss between as 1 - dsc( reference, sigmoid(logits) ). Tensor shapes are ``[batch, 1, rows,
    cols]``. The dice score is computed along the batch and is averaged at the end.

    Parameters
    ----------
    reference : torch.Tensor
        The reference data, e.g. a ground truth binary image. Shape is `[batch, 1, rows, cols]`.
    logits : torch.Tensor
        Logits predicted for a particular model, with the same shape as input.

    Returns
    -------
    torch.Tensor
        A single valued tensor with the average dice loss across the elements of the batch.
    """
    eps = 1.
    probabilities = torch.sigmoid(logits)
    ab = torch.sum(reference * probabilities, dim=(1, 2, 3))
    a = torch.sum(reference, dim=(1, 2, 3))
    b = torch.sum(probabilities, dim=(1, 2, 3))
    dsc = (2 * ab + eps) / (a + b + eps)
    dsc = torch.mean(dsc)
    return 1 - dsc


class SupervisedLoss(object):
    def __call__(self, outputs_x, targets_x):
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        return Lx

