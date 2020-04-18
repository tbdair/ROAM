import torch
import torch.nn as nn
from data.dataMALC import get_dataset
from models.UNet import UNet
from misc.Utilities import distance_perclass
import os
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='ROAM ')

parser.add_argument('--data_path', default='', type=str, metavar='PATH',
                    help='path to the dataset(validation or testing)')

parser.add_argument('--checkpoint_dir', default='', type=str, metavar='PATH',
                    help='path to the trained model')

parser.add_argument('--model_name', default='', type=str, metavar='PATH',
                    help='model to loaded')

parser.add_argument('--batch_size', type=int, default=8)

args = parser.parse_args()


def run():

    image_datasets = get_dataset(args.data_path, 'test')

    model = UNet(input_channels=1, n_classes=28)

    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, args.model_name)))
    model = nn.DataParallel(model)
    model = model.cuda()

    test_loader = torch.utils.data.DataLoader(image_datasets, batch_size=args.batch_size, shuffle='False',
                                              num_workers=4, pin_memory=True)
    model.eval()
    out_list = []
    y_list = []

    for batch_idx, sample_batched in enumerate(test_loader):
        X = sample_batched[0].type(torch.cuda.FloatTensor)
        y = sample_batched[1].type(torch.cuda.LongTensor)
        y_list.append(y.cpu())

        out = model(X)
        _, batch_output = torch.max(out, dim=1)
        out_list.append(batch_output.cpu())
        del batch_output, out

    y_arr = np.concatenate(y_list)
    out_arr = np.concatenate(out_list)

    hsd, msd = distance_perclass(out_arr, y_arr, 28)

    hsd_mean = torch.mean(hsd)
    msd_mean = torch.mean(msd)
    print("HSD = {}".format(hsd_mean.item()))
    print("MSD = {}".format(msd_mean.item()))


if __name__ == '__main__':

    run()




