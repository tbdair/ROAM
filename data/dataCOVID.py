import numpy as np
import torch.utils.data as data
import nibabel as nib


class ImdbData(data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        img = self.X[index]
        label = self.y[index]
        return img, label

    def __len__(self):
        return len(self.X)


class ImdbDataU(data.Dataset):
    def __init__(self, X):
        self.X = X

    def __getitem__(self, index):
        img = self.X[index]
        return img

    def __len__(self):
        return len(self.X)


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_dataset_test(fold, path):

    idx = [63, 14, 90, 5, 95, 34, 2, 74, 49, 48, 84, 45, 62, 59, 57, 20, 89, 71, 96, 99, 7, 68, 52, 25, 55, 39, 92, 11, 42, 65, 73, 88, 98, 70, 12, 58, 51, 30, 50, 66, 17, 18, 15, 24, 26, 8, 67, 72, 69, 80, 75, 46, 29, 10, 22, 78, 9, 82, 13, 1, 94, 61, 41, 3, 87, 93, 36, 81, 38, 54, 4, 97, 32, 27, 40, 43, 91, 76, 6, 23, 37, 86, 53, 0, 79, 21, 64, 16, 56, 60, 35, 19, 85, 33, 44, 77, 31, 47, 83, 28]

    data = np.load(path+'COVID.npy')
    segm = np.load(path +'COVID_Seg.npy')

    n = 20
    chnks = list(divide_chunks(idx, n))

    dataVal = data[chnks[fold]]
    labelVal = segm[chnks[fold]]

    return dataVal, labelVal


def get_dataset_cv(fold, path):

    idx = [63, 14, 90, 5, 95, 34, 2, 74, 49, 48, 84, 45, 62, 59, 57, 20, 89, 71, 96, 99, 7, 68, 52, 25, 55, 39, 92, 11, 42, 65, 73, 88, 98, 70, 12, 58, 51, 30, 50, 66, 17, 18, 15, 24, 26, 8, 67, 72, 69, 80, 75, 46, 29, 10, 22, 78, 9, 82, 13, 1, 94, 61, 41, 3, 87, 93, 36, 81, 38, 54, 4, 97, 32, 27, 40, 43, 91, 76, 6, 23, 37, 86, 53, 0, 79, 21, 64, 16, 56, 60, 35, 19, 85, 33, 44, 77, 31, 47, 83, 28]

    data = np.load(path + 'COVID.npy')
    segm = np.load(path + 'COVID_Seg.npy')

    n = 20
    chnks = list(divide_chunks(idx, n))

    dataVal = data[chnks[fold]]
    labelVal = segm[chnks[fold]]

    del chnks[fold]
    trnIdx = chnks[0]+chnks[1]+chnks[2]+chnks[3]
    dataTrain = data[trnIdx]
    labelTrain = segm[trnIdx]

    return ImdbData(dataTrain, labelTrain), ImdbData(dataVal, labelVal)


def get_dataset_unl(path):

    coviddata = nib.load(path+'unlCovid.nii').get_fdata()
    noncoviddata = nib.load(path+'unlNonCovid.nii').get_fdata()

    idx = [115, 127, 180,  44, 182, 200,   0,  83, 129, 134, 151, 147, 155,  71,
        196, 189, 201, 170, 144, 108,  47, 135,  72, 142, 141, 133,  46,   2,
        130, 193, 140,  90,  68, 175,   1,  74, 169,  23, 157, 194,  55, 100,
        124, 104,  97, 174, 199, 139,  58, 103,  89,  37, 131,  81, 120, 173,
          6,  43,  27, 122,  52, 184,  53,   8, 102,  86,  26, 137,  51,  73,
         70,  17,  76, 152, 118,  16,  48,  75, 113,  45,  19, 161, 105, 177,
         96,  21, 101, 191,  25,  35, 190, 156, 186, 176,  14,  50,  49, 188,
        116,  11, 111, 112,  87,  56,  36, 172,  61, 146,  67, 162, 136, 167,
        164, 128,  60, 195,  85, 154, 123,  77, 149, 187,  29, 185, 192,   7,
        138,  33, 168,  18, 109,  63,  82,  40,  62, 160,  91,  80,  93, 110,
         66, 121,  38,  64, 150, 158,  10,  69,  92, 163, 183, 165,  15, 126,
          9,  41, 178, 114,  34, 198, 106,  79,   4,  20, 179,  32,  57,  59,
        197,  94,  54, 171,  22, 145,  78, 117,  88, 119,  95,  65,  31, 125,
        143, 159,   3, 153, 181,  28,  98, 148,  99,   5,  39, 132,  84,  13,
        107,  30, 166,  12,  42,  24]

    data = np.concatenate((coviddata, noncoviddata), axis=2)
    data = data[:, :, idx]

    return ImdbDataU(data.transpose((2, 0, 1)))





