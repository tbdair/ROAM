import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data.dataMALC import get_dataset
from models.MixUpUnet import UNet
from misc.Utilities import make_one_hot, dice_score_per_epoch
from misc.losses import SemiLoss
import time
import copy
import os
from random import randrange
import argparse


parser = argparse.ArgumentParser(description='ROAM ')

parser.add_argument('--data_path', default='', type=str, metavar='PATH',
                    help='path to the dataset(validation or testing)')

parser.add_argument('--checkpoint_dir', default='', type=str, metavar='PATH',
                    help='path to the trained model')

parser.add_argument('--model_name', default='roam_malc_semi', type=str, metavar='PATH',
                    help='model to be saved')

parser.add_argument('--epochs', type=int, default=80)

parser.add_argument('--iter', type=int, default=375, help='number of iteration')

parser.add_argument('--learning_rate', type=int, default=0.002)

parser.add_argument('--batch_size', type=int, default=4)

parser.add_argument('--n_classes', type=int, default=28)

parser.add_argument('--weight_decay', type=int, default=0.002)

parser.add_argument('--img_channel', type=int, default=1)

parser.add_argument('--img_width', type=int, default=256)

parser.add_argument('--img_height', type=int, default=256)

parser.add_argument('--alpha', type=int, default=0.75)

parser.add_argument('--T', type=int, default=0.5,  help='Sharpening parameter')

parser.add_argument('--lambda_u', type=int, default=75, help='unlabeled coefficient')

parser.add_argument('--layers', default='MixIMix1MixL', type=str, metavar='PATH',
                    help='layers to be mixed; MixIMix1MixL, MixI, Mix2')

args = parser.parse_args()


def train(train_loader, unl_train_loader, model, optimizer, epoch, train_criterion):
    model.train()
    count = 0
    tr_loss = 0
    tr_lx = 0
    tr_lu = 0
    tr_w = 0
    trainloader_l_iter = enumerate(train_loader)
    trainloader_u_iter = enumerate(unl_train_loader)

    print(">>Train<<")
    for batch_idx in range(0, args.iter):

        # Check if the label loader has a batch available
        try:
            _, sample_batched_l = next(trainloader_l_iter)
        except:
            # Curr loader doesn't have data, then reload data
            del trainloader_l_iter
            trainloader_l_iter = enumerate(train_loader)
            _, sample_batched_l = next(trainloader_l_iter)

        # Check if the unlabel loader has a batch available
        try:
            _, sample_batched_u = next(trainloader_u_iter)
        except:
            # Curr loader doesn't have data, then reload data
            del trainloader_u_iter
            trainloader_u_iter = enumerate(unl_train_loader)
            _, sample_batched_u = next(trainloader_u_iter)

        # Supervised Samples
        input_x = sample_batched_l[0].type(torch.cuda.FloatTensor)
        targets_x = sample_batched_l[1].type(torch.cuda.LongTensor)
        targets_x = targets_x[:, np.newaxis, :, :]
        targets_x = make_one_hot(targets_x, args.n_classes)
        batch_size = input_x.size(0)

        # Un Supervised Samples
        u_input1 = sample_batched_u[0].type(torch.cuda.FloatTensor)
        u_input1 = u_input1[:batch_size]

        # Label Guessing
        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u1 = model(u_input1)
            p = torch.softmax(outputs_u1, dim=1)
            # print(outputs_u1.shape)
            # sharpening guessed labels of unlabel samples
            pt = p ** (1 / args.T)
            targets_u1 = pt / pt.sum(dim=1, keepdim=True)
            targets_u1 = targets_u1.detach()

        # Start of Code for normal mixed mixup
        all_inputs = torch.cat([input_x, u_input1], dim=0)
        all_targets = torch.cat([targets_x, targets_u1], dim=0)

        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)
        idx = torch.randperm(all_inputs.size(0))

        target_a, target_b = all_targets, all_targets[idx]
        mixed_target = l * target_a + (1 - l) * target_b

        if args.layers == 'MixI':
            mixed_logits = model(all_inputs, "MixI", l, idx)
        elif args.layers == 'Mix2':
            mixed_logits = model(all_inputs, "Mix2", l, idx)
        elif args.layers == 'MixIMix1MixL':
            loc = randrange(2)
            if loc == 0:
                mixed_logits = model(all_inputs, "MixI", l, idx)
            elif loc == 1:
                mixed_logits = model(all_inputs, "Mix1", l, idx)
            elif loc == 2:
                mixed_logits = model(all_inputs, "MixL", l, idx)
        else:
            mixed_logits = model(all_inputs, "MixI", l, idx)

        logits_x = mixed_logits[:batch_size]
        logits_u = mixed_logits[batch_size:]

        Lx, Lu, w = train_criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:],
                                    epoch, args.lambda_u, args.epochs)

        loss = Lx + w * Lu

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_weight(model)
        tr_loss += loss.item()
        tr_lx += Lx.item()
        tr_lu += Lu.item()
        tr_w += w
        count += 1

    tr_loss = tr_loss / float(count)
    tr_lx = tr_lx / float(count)
    tr_lu = tr_lu / float(count)
    tr_w = tr_w / float(count)
    print("Total loss: {:.4f}".format(tr_loss))
    print("S. loss:    {:.4f}".format(tr_lx))
    print("U. loss:    {:.4f}".format(tr_lu))
    print("Lamda:      {:.4f}".format(tr_w))


def validate(model, valid_loader, criterion, model_name):
    model.eval()
    out_list = []
    y_list = []
    loss_arr = []
    for batch_idx, sample_batched in enumerate(valid_loader):
        X = sample_batched[0].type(torch.cuda.FloatTensor)
        y = sample_batched[1].type(torch.cuda.LongTensor)

        out = model(X)
        loss = criterion(out, y)
        loss_arr.append(loss.item())
        _, batch_output = torch.max(out, dim=1)
        out_list.append(batch_output.cpu())

        y_list.append(y.cpu())

        del X, y, batch_output, out, loss

    out_arr, y_arr = torch.cat(out_list), torch.cat(y_list)

    dice = dice_score_per_epoch(out_arr, y_arr, args.n_classes)

    print("{} : loss: {:.4f}".format(model_name, np.mean(loss_arr)))
    print("{} : dice: {:.4f}".format(model_name, dice))
    return dice


def update_weight(model):
    wd = 0.02 * args.weight_decay
    for param in model.parameters():
        param.data.mul_(1 - wd)


def run_train(model, train_loader, unl_train_loader, valid_loader, optimizer, train_criterion, validation_criterion):

    model_best_dice = 0
    since = time.time()
    for epoch in range(0, args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))

        train(train_loader, unl_train_loader, model, optimizer, epoch, train_criterion)

        # Saving model checkpoint
        if (epoch + 1) % 5 == 0:
            model_wts = copy.deepcopy(model.state_dict())
            model_name = args.model_name + str(epoch) + '.pt'
            torch.save(model_wts, os.path.join(args.checkpoint_dir, model_name))
            del model_wts, model_name

        validate(model, train_loader, validation_criterion, "Model")

        print(">>Validation<<")

        dice = validate(model, valid_loader, validation_criterion, "Model")

        if dice >= model_best_dice:
            model_best_dice = dice
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(args.checkpoint_dir, args.model_name+'.pt'))
            del best_model_wts
        del dice

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    print('Best Epoch Dices Were model = {}'.format(model_best_dice))
    print('---------------------')


def run():

    trainDS = get_dataset(args.data_path, 'train')
    valDS = get_dataset(args.data_path, 'val')
    unlabel_DS = get_dataset(args.data_path, 'unlabel')

    print("Train size: %i" % len(trainDS))
    print("Unlabeled Train size: %i" % len(unlabel_DS))
    print("Validation size: %i" % len(valDS))
    print("LR: {}".format(args.learning_rate))
    print("WD: {}".format(args.weight_decay))
    print("Layers: {}".format(args.layers))

    print('**********************************************************************')

    train_loader = torch.utils.data.DataLoader(trainDS, batch_size=args.batch_size, shuffle='True',
                                               num_workers=4, pin_memory=True)

    unl_train_loader = torch.utils.data.DataLoader(unlabel_DS, batch_size=args.batch_size, shuffle='True',
                                                   num_workers=4, pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(valDS, batch_size=args.batch_size, shuffle='False',
                                               num_workers=4, pin_memory=True)

    model = UNet(input_channels=args.img_channel, n_classes=args.n_classes)

    model = nn.DataParallel(model)

    model = model.cuda()

    train_criterion = SemiLoss()

    validation_criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    run_train(model, train_loader, unl_train_loader, valid_loader, optimizer, train_criterion, validation_criterion)


if __name__ == '__main__':

    run()
























