import numpy as np
import torch
from torch.autograd import Variable
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import morphology
from statistics import mean


def make_one_hot(labels, C=28):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)

    target = Variable(target)

    return target


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch, opt):
    lr = opt.max_lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = linear_rampup(epoch, opt.lr_rampup) * (opt.max_lr - opt.initial_lr) + opt.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if opt.lr_rampdown_epochs:
        assert opt.lr_rampdown_epochs >= opt.epochs
        lr *= cosine_rampdown(epoch, opt.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_current_consistency_weight(epoch, rampup_length):
    return sigmoid_rampup(epoch, rampup_length)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def distance_perclass(vol_output, ground_truth, num_classes):
    sz = vol_output.shape[0]
    hsd_perclass = torch.zeros(num_classes)
    msd_perclass = torch.zeros(num_classes)
    for i in range(num_classes):
        GT = (ground_truth == i).astype(np.int)
        Pred = (vol_output == i).astype(np.int)
        hsd1 = []
        hsd2 = []
        for j in range(sz):
            hsd1.append(directed_hausdorff(GT[j], Pred[j])[0])
            hsd2.append(directed_hausdorff(Pred[j], GT[j])[0])
        hs1 = mean(hsd1)
        hs2 = mean(hsd2)
        hsd = max(hs1, hs2)
        hsd_perclass[i] = hsd
        msd = (hs1 + hs2) / 2
        msd_perclass[i] = msd
    return hsd_perclass, msd_perclass


def dice_score_perclass(vol_output, ground_truth, num_classes):
    dice_perclass = torch.zeros(num_classes)
    for i in range(num_classes):
        GT = (ground_truth == i).float()
        Pred = (vol_output == i).float()
        inter = torch.sum(torch.mul(GT, Pred))
        union = torch.sum(GT) + torch.sum(Pred) + 0.0001
        dice_perclass[i] = (2 * torch.div(inter, union))
    return dice_perclass


def dice_score_per_epoch(output, correct_labels, num_classes):
    ds = dice_score_perclass(output, correct_labels, num_classes)
    ds_mean = torch.mean(ds)
    return ds_mean.item()


def dice_score_perclass_COVID(vol_output, ground_truth, num_classes):
    dice_perclass = []
    for i in range(num_classes):
        GT = (ground_truth == i).float()
        Pred = (vol_output == i).float()
        if torch.sum(GT) != 0:
            inter = torch.sum(torch.mul(GT, Pred))
            union = torch.sum(GT) + torch.sum(Pred)
            dice_perclass.append((2 * torch.div(inter, union)).item())
    return dice_perclass


def dice_score_per_epoch_COVID(output, correct_labels, num_classes):
    ds = dice_score_perclass_COVID(output, correct_labels, num_classes)
    ds_mean = mean(ds)
    return ds_mean, ds


def surfd(input1, input2, sampling=1, connectivity=1):
    input_1 = np.atleast_1d(input1.astype(np.int))
    input_2 = np.atleast_1d(input2.astype(np.int))

    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    S = input_1 - morphology.binary_erosion(input_1, conn)
    Sprime = input_2 - morphology.binary_erosion(input_2, conn)

    dta = morphology.distance_transform_edt(~S, sampling)
    dtb = morphology.distance_transform_edt(~Sprime, sampling)

    sds = np.concatenate([np.ravel(dta[Sprime != 0]), np.ravel(dtb[S != 0])])

    return sds





