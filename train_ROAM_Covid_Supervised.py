import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data.dataCOVID import get_dataset_cv
from models.MixUpUnet import UNet
from misc.Utilities import make_one_hot, dice_score_per_epoch_COVID
from misc.losses import SupervisedLoss
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

parser.add_argument('--model_name', default='roam_covid_sup', type=str, metavar='PATH',
                    help='model to be saved')

parser.add_argument('--epochs', type=int, default=100)

parser.add_argument('--iter', type=int, default=100, help='number of iteration')

parser.add_argument('--learning_rate', type=int, default=0.02)

parser.add_argument('--batch_size', type=int, default=8)

parser.add_argument('--n_classes', type=int, default=5)

parser.add_argument('--weight_decay', type=int, default=0.002)

parser.add_argument('--img_channel', type=int, default=1)

parser.add_argument('--img_width', type=int, default=256)

parser.add_argument('--img_height', type=int, default=256)

parser.add_argument('--alpha', type=int, default=1)

parser.add_argument('--layers', default='Mix2', type=str, metavar='PATH',
                    help='layers to be mixed; MixIMix1MixL, MixI, Mix2')

args = parser.parse_args()


def train(train_loader, unl_train_loader, model, optimizer, train_criterion):

    model.train()
    count = 0
    tr_loss = 0
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
        input_x = input_x[:, np.newaxis, :, :]
        targets_x = targets_x[:, np.newaxis, :, :]
        targets_x = make_one_hot(targets_x, args.n_classes)

        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)
        idx = torch.randperm(input_x.size(0))

        target_a, target_b = targets_x, targets_x[idx]
        mixed_target = l * target_a + (1 - l) * target_b

        if args.layers == 'MixI':
            mixed_logits = model(input_x, "MixI", l, idx)
        elif args.layers == 'Mix2':
            mixed_logits = model(input_x, "Mix2", l, idx)
        elif args.layers == 'MixIMix1MixL':
            loc = randrange(2)
            if loc == 0:
                mixed_logits = model(input_x, "MixI", l, idx)
            elif loc == 1:
                mixed_logits = model(input_x, "Mix1", l, idx)
            elif loc == 2:
                mixed_logits = model(input_x, "MixL", l, idx)
        else:
            mixed_logits = model(input_x, "MixI", l, idx)

        loss = train_criterion(mixed_logits, mixed_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_weight(model)
        tr_loss += loss.item()
        count += 1

    tr_loss = tr_loss / float(count)
    print("Total loss: {:.4f}".format(tr_loss))


def validate(model, valid_loader, criterion, model_name):
    model.eval()
    out_list = []
    y_list = []
    loss_arr = []
    for batch_idx, sample_batched in enumerate(valid_loader):
        X = sample_batched[0].type(torch.cuda.FloatTensor)
        y = sample_batched[1].type(torch.cuda.LongTensor)
        X = X[:, np.newaxis, :, :]

        out = model(X)
        loss = criterion(out, y)
        loss_arr.append(loss.item())
        _, batch_output = torch.max(out, dim=1)
        out_list.append(batch_output.cpu())

        y_list.append(y.cpu())

        del X, y, batch_output, out, loss

    out_arr, y_arr = torch.cat(out_list), torch.cat(y_list)

    dice, _ = dice_score_per_epoch_COVID(out_arr, y_arr, args.n_classes)

    print("{} : loss: {:.4f}".format(model_name, np.mean(loss_arr)))
    print("{} : dice: {:.4f}".format(model_name, dice))
    return dice


def update_weight(model):
    wd = 0.02 * args.weight_decay
    for param in model.parameters():
        param.data.mul_(1 - wd)


def run_train(model, train_loader, valid_loader, optimizer, train_criterion, validation_criterion):

    model_best_dice = 0
    since = time.time()
    for epoch in range(0, args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))

        train(train_loader, train_loader, model, optimizer, train_criterion)

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

    trainDS, valDS = get_dataset_cv(1, args.data_path)

    print("Train size: %i" % len(trainDS))
    print("Validation size: %i" % len(valDS))
    print("LR: {}".format(args.learning_rate))
    print("WD: {}".format(args.weight_decay))
    print("Layers: {}".format(args.layers))

    print('**********************************************************************')

    train_loader = torch.utils.data.DataLoader(trainDS, batch_size=args.batch_size, shuffle='True',
                                               num_workers=4, pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(valDS, batch_size=args.batch_size, shuffle='False',
                                               num_workers=4, pin_memory=True)

    model = UNet(input_channels=args.img_channel, n_classes=args.n_classes)

    model = nn.DataParallel(model)

    model = model.cuda()

    train_criterion = SupervisedLoss()

    validation_criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    run_train(model, train_loader, valid_loader, optimizer, train_criterion, validation_criterion)


if __name__ == '__main__':

    run()


















