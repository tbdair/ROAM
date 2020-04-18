import torch
from models.UNet import UNet
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from data.dataCOVID import get_dataset_test
import argparse


parser = argparse.ArgumentParser(description='ROAM ')

parser.add_argument('--data_path', default='', type=str, metavar='PATH',
                    help='path to the data vol')

parser.add_argument('--checkpoint_dir', default='', type=str, metavar='PATH',
                    help='path to the trained model')

parser.add_argument('--model_name', default='', type=str, metavar='PATH',
                    help='model to loaded')

args = parser.parse_args()


def overlay_pred_img(name, num_class, prd, inpt):
  inpt = inpt * 255
  inpt = np.squeeze(inpt)
  plt.imshow(inpt, cmap='gray')
  plt.imshow(prd, cmap='CMRmap', alpha=0.25, vmin=0, vmax=num_class - 1)
  plt.savefig(name+'.png')


def run():

    model = UNet(input_channels=1, n_classes=5)
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, args.model_name)))
    model = model.cuda()
    model.eval()

    data, segm = get_dataset_test(1, args.data_path)

    sz = data.shape[0]
    seg_nifti = np.zeros((256, 256, sz))

    for i in range(sz):

        inpt = data[i, :, :]
        inpt = inpt[np.newaxis, np.newaxis, :, :]
        gt = segm[i, :, :]
        prd = model.predict(inpt)
        seg_nifti[:, :, i] = prd
        name = "Prd_" + str(i)
        overlay_pred_img(name, 5, prd, inpt)

        name = "GT_" + str(i)
        overlay_pred_img(name, 5, gt, inpt)

    seg_nifti = np.zeros((256, 256, sz))

    savefile = nib.Nifti1Image(seg_nifti, None)
    nib.save(savefile, "Predictions.nii")


if __name__ == '__main__':

    run()

