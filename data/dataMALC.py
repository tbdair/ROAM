import os
import numpy as np
import torch.utils.data as data
import h5py


class ImdbData(data.Dataset):
    def __init__(self, X, y):
        self.X = X if len(X.shape) == 4 else X[:, np.newaxis, :, :]
        self.y = y

    def __getitem__(self, index):
        img = self.X[index]
        label = self.y[index]
        return img, label

    def __len__(self):
        return len(self.X)


def get_dataset(path, phase):

    if phase =='train' or phase =='unlabel':
        data_train = h5py.File(os.path.join(path+'/'+phase, "Data_train.h5"), 'r')
        label_train = h5py.File(os.path.join(path+'/'+phase, "Label_train.h5"), 'r')
        data = data_train['data'][()]
        label = label_train['label'][()]
    elif phase =='val':
        data_train = h5py.File(os.path.join(path+'/'+phase, "Data_Test.h5"), 'r')
        label_train = h5py.File(os.path.join(path+'/'+phase, "Label_Test.h5"), 'r')
        data = data_train['data'][()]
        label = label_train['label'][()]
    elif phase == 'test':
        data_train = h5py.File(os.path.join(path, "Data_Test.h5"), 'r')
        label_train = h5py.File(os.path.join(path, "Label_Test.h5"), 'r')
        data = data_train['data'][()]
        label = label_train['label'][()]
    return ImdbData(data, label)

