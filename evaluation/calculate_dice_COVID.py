import torch
from models.UNet import UNet
from misc.Utilities import dice_score_per_epoch_COVID
import os
import numpy as np
from data.dataCOVID import get_dataset_test
import argparse


parser = argparse.ArgumentParser(description='ROAM ')

parser.add_argument('--data_path', default='', type=str, metavar='PATH',
                    help='path to the dataset(validation or testing)')

parser.add_argument('--checkpoint_dir', default='', type=str, metavar='PATH',
                    help='path to the trained model')

parser.add_argument('--model_name', default='', type=str, metavar='PATH',
                    help='model to loaded')

args = parser.parse_args()


def run():

    model = UNet(input_channels=1, n_classes=5)
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, args.model_name)))
    model = model.cuda()
    model.eval()

    dataVal, labelVal = get_dataset_test(1, args.data_path)

    sz = dataVal.shape[0]

    for i in range(sz):

        inpt = dataVal[i, :, :]
        gt = labelVal[i, :, :]
        gt = torch.from_numpy(gt[np.newaxis, :, :]).type(torch.cuda.LongTensor)
        inpt = torch.from_numpy(inpt[np.newaxis, np.newaxis, :, :]).type(torch.cuda.FloatTensor)

        out = model(inpt)
        _, batch_output = torch.max(out, dim=1)
        dice, dices = dice_score_per_epoch_COVID(batch_output, gt, 5)
        print("********Image: ({})*******".format(i))
        print("Dice = {}".format(dice))

        for cls in range(len(dices)):
            print("Class ({}): Dice = {}".format(cls, dices[cls]))


if __name__ == '__main__':

    run()

