import matplotlib.pyplot as plt
import torch
from models.UNet import UNet
import os
import numpy as np
from scipy import ndimage
import torch.nn as nn
from data.dataMALC import get_dataset
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


def overlay_pred_img(num_class, prd_batch, gt_batch, inpt_batch):

    sz = inpt_batch.shape[0]

    for slic in range(sz):
        ipt = inpt_batch[slic, :, :, :]
        gt = gt_batch[slic, :, :]
        prd = prd_batch[slic, :, :]

        ipt = ipt * 255
        ipt = np.squeeze(ipt)

        ipt = ndimage.rotate(ipt, -90)
        gt = ndimage.rotate(gt, -90)
        prd = ndimage.rotate(prd, -90)

        plt.imshow(ipt, cmap='gray')
        plt.imshow(gt, cmap='CMRmap', alpha=0.5, vmin=0, vmax=num_class - 1)
        plt.savefig('GT_'+str(slic)+'.png')

        plt.imshow(ipt, cmap='gray')
        plt.imshow(prd, cmap='CMRmap', alpha=0.5, vmin=0, vmax=num_class - 1)
        plt.savefig('Ours_'+str(slic)+'.png')


def run():
    image_datasets = get_dataset(args.data_path, 'test')

    model = UNet(input_channels=1, n_classes=28)

    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, args.model_name)))
    # model = nn.DataParallel(model)
    model = model.cuda()

    test_loader = torch.utils.data.DataLoader(image_datasets, batch_size=args.batch_size, shuffle='False',
                                              num_workers=4, pin_memory=True)
    model.eval()

    for batch_idx, sample_batched in enumerate(test_loader):
        inp = sample_batched[0].type(torch.cuda.FloatTensor)
        gt = sample_batched[1].type(torch.cuda.LongTensor)
        prd = model.predict(inp)

        overlay_pred_img(28, prd, gt.cpu().numpy(), inp.cpu().numpy())


if __name__ == '__main__':

    run()

