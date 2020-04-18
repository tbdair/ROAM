"""
MixUp UNet
The main UNet model implementation
"""

import torch
import torch.nn as nn
import numpy as np


# Create Convolution Block
def create2DConvBlock(in_channels,out_channels,kernel_size=3,stride=1,padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def max_pooling():
    return nn.MaxPool2d(2)


def create2DUpConvBlock(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    def __init__(self, input_channels, n_classes, n_filters=64, kernel_size=3, stride=1, padding=1):
        super().__init__()

        # Encoding Path
        self.conv1 = create2DConvBlock(input_channels, n_filters, kernel_size, stride, padding)
        self.conv2 = create2DConvBlock(n_filters, n_filters*2, kernel_size, stride, padding)
        self.conv3 = create2DConvBlock(n_filters*2, n_filters*4, kernel_size, stride, padding)
        self.conv4 = create2DConvBlock(n_filters*4, n_filters*8, kernel_size, stride, padding)
        self.conv5 = create2DConvBlock(n_filters*8, n_filters*16, kernel_size, stride, padding)
        self.down_pooling = max_pooling()

        # Decoding Path
        self.upconv6 = create2DUpConvBlock(n_filters*16, n_filters*8)
        self.conv6 = create2DConvBlock(n_filters*16, n_filters*8)
        self.upconv7 = create2DUpConvBlock(n_filters*8, n_filters*4)
        self.conv7 = create2DConvBlock(n_filters*8, n_filters*4)
        self.upconv8 = create2DUpConvBlock(n_filters*4, n_filters*2)
        self.conv8 = create2DConvBlock(n_filters*4, n_filters*2)
        self.upconv9 = create2DUpConvBlock(n_filters*2, n_filters)
        self.conv9 = create2DConvBlock(n_filters*2, n_filters)
        self.conv10 = nn.Conv2d(n_filters, n_classes, 1)

        # Weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x_input, mode="Normal", l=1, idx=0):
        if mode == "Normal":
            # Encoding input
            x1_conv1 = self.conv1(x_input)
            pool1 = self.down_pooling(x1_conv1)
            x2_conv2 = self.conv2(pool1)
            pool2 = self.down_pooling(x2_conv2)
            x3_conv3 = self.conv3(pool2)
            pool3 = self.down_pooling(x3_conv3)
            x4_conv4 = self.conv4(pool3)
            pool4 = self.down_pooling(x4_conv4)
            x5_conv5 = self.conv5(pool4)
            # Decoding
            pool6 = self.upconv6(x5_conv5)
            x6_conv6 = torch.cat([pool6, x4_conv4], dim=1)
            x6_conv6 = self.conv6(x6_conv6)
            pool7 = self.upconv7(x6_conv6)
            x7_conv7 = torch.cat([pool7, x3_conv3], dim=1)
            x7_conv7 = self.conv7(x7_conv7)
            pool8 = self.upconv8(x7_conv7)
            x8_conv8 = torch.cat([pool8, x2_conv2], dim=1)
            x8_conv8 = self.conv8(x8_conv8)
            pool9 = self.upconv9(x8_conv8)
            x9_conv9 = torch.cat([pool9, x1_conv1], dim=1)
            x9_conv9 = self.conv9(x9_conv9)
            output = self.conv10(x9_conv9)
            return output

        if mode == "MixI":
            # Encoding input
            x_a, x_b = x_input, x_input[idx]
            x_input = l * x_a + (1 - l) * x_b

            x1_conv1 = self.conv1(x_input)
            pool1 = self.down_pooling(x1_conv1)
            x2_conv2 = self.conv2(pool1)
            pool2 = self.down_pooling(x2_conv2)
            x3_conv3 = self.conv3(pool2)
            pool3 = self.down_pooling(x3_conv3)
            x4_conv4 = self.conv4(pool3)
            pool4 = self.down_pooling(x4_conv4)
            x5_conv5 = self.conv5(pool4)
            # Decoding
            pool6 = self.upconv6(x5_conv5)
            x6_conv6 = torch.cat([pool6, x4_conv4], dim=1)
            x6_conv6 = self.conv6(x6_conv6)
            pool7 = self.upconv7(x6_conv6)
            x7_conv7 = torch.cat([pool7, x3_conv3], dim=1)
            x7_conv7 = self.conv7(x7_conv7)
            pool8 = self.upconv8(x7_conv7)
            x8_conv8 = torch.cat([pool8, x2_conv2], dim=1)
            x8_conv8 = self.conv8(x8_conv8)
            pool9 = self.upconv9(x8_conv8)
            x9_conv9 = torch.cat([pool9, x1_conv1], dim=1)
            x9_conv9 = self.conv9(x9_conv9)
            output = self.conv10(x9_conv9)
            return output

        if mode == "Mix1":
            # Encoding input
            x1_conv1 = self.conv1(x_input)
            x_a, x_b = x1_conv1, x1_conv1[idx]
            x1_conv1 = l * x_a + (1 - l) * x_b

            pool1 = self.down_pooling(x1_conv1)
            x2_conv2 = self.conv2(pool1)
            pool2 = self.down_pooling(x2_conv2)
            x3_conv3 = self.conv3(pool2)
            pool3 = self.down_pooling(x3_conv3)
            x4_conv4 = self.conv4(pool3)
            pool4 = self.down_pooling(x4_conv4)
            x5_conv5 = self.conv5(pool4)
            # Decoding
            pool6 = self.upconv6(x5_conv5)
            x6_conv6 = torch.cat([pool6, x4_conv4], dim=1)
            x6_conv6 = self.conv6(x6_conv6)
            pool7 = self.upconv7(x6_conv6)
            x7_conv7 = torch.cat([pool7, x3_conv3], dim=1)
            x7_conv7 = self.conv7(x7_conv7)
            pool8 = self.upconv8(x7_conv7)
            x8_conv8 = torch.cat([pool8, x2_conv2], dim=1)
            x8_conv8 = self.conv8(x8_conv8)
            pool9 = self.upconv9(x8_conv8)
            x9_conv9 = torch.cat([pool9, x1_conv1], dim=1)
            x9_conv9 = self.conv9(x9_conv9)
            output = self.conv10(x9_conv9)
            return output

        if mode == "Mix2":
            # Encoding input
            x1_conv1 = self.conv1(x_input)
            pool1 = self.down_pooling(x1_conv1)

            x2_conv2 = self.conv2(pool1)
            x_a, x_b = x2_conv2, x2_conv2[idx]
            x2_conv2 = l * x_a + (1 - l) * x_b

            pool2 = self.down_pooling(x2_conv2)
            x3_conv3 = self.conv3(pool2)
            pool3 = self.down_pooling(x3_conv3)
            x4_conv4 = self.conv4(pool3)
            pool4 = self.down_pooling(x4_conv4)
            x5_conv5 = self.conv5(pool4)
            # Decoding
            pool6 = self.upconv6(x5_conv5)
            x6_conv6 = torch.cat([pool6, x4_conv4], dim=1)
            x6_conv6 = self.conv6(x6_conv6)
            pool7 = self.upconv7(x6_conv6)
            x7_conv7 = torch.cat([pool7, x3_conv3], dim=1)
            x7_conv7 = self.conv7(x7_conv7)
            pool8 = self.upconv8(x7_conv7)
            x8_conv8 = torch.cat([pool8, x2_conv2], dim=1)
            x8_conv8 = self.conv8(x8_conv8)
            pool9 = self.upconv9(x8_conv8)
            x9_conv9 = torch.cat([pool9, x1_conv1], dim=1)
            x9_conv9 = self.conv9(x9_conv9)
            output = self.conv10(x9_conv9)
            return output

        if mode == "Mix3":
            # Encoding input
            x1_conv1 = self.conv1(x_input)
            pool1 = self.down_pooling(x1_conv1)
            x2_conv2 = self.conv2(pool1)
            pool2 = self.down_pooling(x2_conv2)

            x3_conv3 = self.conv3(pool2)
            x_a, x_b = x3_conv3, x3_conv3[idx]
            x3_conv3 = l * x_a + (1 - l) * x_b

            pool3 = self.down_pooling(x3_conv3)
            x4_conv4 = self.conv4(pool3)
            pool4 = self.down_pooling(x4_conv4)
            x5_conv5 = self.conv5(pool4)
            # Decoding
            pool6 = self.upconv6(x5_conv5)
            x6_conv6 = torch.cat([pool6, x4_conv4], dim=1)
            x6_conv6 = self.conv6(x6_conv6)
            pool7 = self.upconv7(x6_conv6)
            x7_conv7 = torch.cat([pool7, x3_conv3], dim=1)
            x7_conv7 = self.conv7(x7_conv7)
            pool8 = self.upconv8(x7_conv7)
            x8_conv8 = torch.cat([pool8, x2_conv2], dim=1)
            x8_conv8 = self.conv8(x8_conv8)
            pool9 = self.upconv9(x8_conv8)
            x9_conv9 = torch.cat([pool9, x1_conv1], dim=1)
            x9_conv9 = self.conv9(x9_conv9)
            output = self.conv10(x9_conv9)
            return output

        if mode == "Mix4":
            # Encoding input
            x1_conv1 = self.conv1(x_input)
            pool1 = self.down_pooling(x1_conv1)
            x2_conv2 = self.conv2(pool1)
            pool2 = self.down_pooling(x2_conv2)
            x3_conv3 = self.conv3(pool2)
            pool3 = self.down_pooling(x3_conv3)

            x4_conv4 = self.conv4(pool3)
            x_a, x_b = x4_conv4, x4_conv4[idx]
            x4_conv4 = l * x_a + (1 - l) * x_b

            pool4 = self.down_pooling(x4_conv4)
            x5_conv5 = self.conv5(pool4)
            # Decoding
            pool6 = self.upconv6(x5_conv5)
            x6_conv6 = torch.cat([pool6, x4_conv4], dim=1)
            x6_conv6 = self.conv6(x6_conv6)
            pool7 = self.upconv7(x6_conv6)
            x7_conv7 = torch.cat([pool7, x3_conv3], dim=1)
            x7_conv7 = self.conv7(x7_conv7)
            pool8 = self.upconv8(x7_conv7)
            x8_conv8 = torch.cat([pool8, x2_conv2], dim=1)
            x8_conv8 = self.conv8(x8_conv8)
            pool9 = self.upconv9(x8_conv8)
            x9_conv9 = torch.cat([pool9, x1_conv1], dim=1)
            x9_conv9 = self.conv9(x9_conv9)
            output = self.conv10(x9_conv9)
            return output

        if mode == "Mix5":
            # Encoding input
            x1_conv1 = self.conv1(x_input)
            pool1 = self.down_pooling(x1_conv1)
            x2_conv2 = self.conv2(pool1)
            pool2 = self.down_pooling(x2_conv2)
            x3_conv3 = self.conv3(pool2)
            pool3 = self.down_pooling(x3_conv3)
            x4_conv4 = self.conv4(pool3)
            pool4 = self.down_pooling(x4_conv4)

            x5_conv5 = self.conv5(pool4)
            x_a, x_b = x5_conv5, x5_conv5[idx]
            x5_conv5 = l * x_a + (1 - l) * x_b
            # Decoding
            pool6 = self.upconv6(x5_conv5)
            x6_conv6 = torch.cat([pool6, x4_conv4], dim=1)
            x6_conv6 = self.conv6(x6_conv6)
            pool7 = self.upconv7(x6_conv6)
            x7_conv7 = torch.cat([pool7, x3_conv3], dim=1)
            x7_conv7 = self.conv7(x7_conv7)
            pool8 = self.upconv8(x7_conv7)
            x8_conv8 = torch.cat([pool8, x2_conv2], dim=1)
            x8_conv8 = self.conv8(x8_conv8)
            pool9 = self.upconv9(x8_conv8)
            x9_conv9 = torch.cat([pool9, x1_conv1], dim=1)
            x9_conv9 = self.conv9(x9_conv9)
            output = self.conv10(x9_conv9)
            return output

        if mode == "MixL":
            # Encoding input
            x1_conv1 = self.conv1(x_input)
            pool1 = self.down_pooling(x1_conv1)
            x2_conv2 = self.conv2(pool1)
            pool2 = self.down_pooling(x2_conv2)
            x3_conv3 = self.conv3(pool2)
            pool3 = self.down_pooling(x3_conv3)
            x4_conv4 = self.conv4(pool3)
            pool4 = self.down_pooling(x4_conv4)
            x5_conv5 = self.conv5(pool4)
            # Decoding
            pool6 = self.upconv6(x5_conv5)
            x6_conv6 = torch.cat([pool6, x4_conv4], dim=1)
            x6_conv6 = self.conv6(x6_conv6)
            pool7 = self.upconv7(x6_conv6)
            x7_conv7 = torch.cat([pool7, x3_conv3], dim=1)
            x7_conv7 = self.conv7(x7_conv7)
            pool8 = self.upconv8(x7_conv7)
            x8_conv8 = torch.cat([pool8, x2_conv2], dim=1)
            x8_conv8 = self.conv8(x8_conv8)
            pool9 = self.upconv9(x8_conv8)
            x9_conv9 = torch.cat([pool9, x1_conv1], dim=1)

            x9_conv9 = self.conv9(x9_conv9)
            x_a, x_b = x9_conv9, x9_conv9[idx]
            x9_conv9 = l * x_a + (1 - l) * x_b

            output = self.conv10(x9_conv9)
            return output

    def forward_embddings(self, x_input):
        # Encoding input
        x1_conv1 = self.conv1(x_input)
        pool1 = self.down_pooling(x1_conv1)
        x2_conv2 = self.conv2(pool1)
        pool2 = self.down_pooling(x2_conv2)
        x3_conv3 = self.conv3(pool2)
        pool3 = self.down_pooling(x3_conv3)
        x4_conv4 = self.conv4(pool3)
        pool4 = self.down_pooling(x4_conv4)
        x5_conv5 = self.conv5(pool4)
        # Decoding
        pool6 = self.upconv6(x5_conv5)
        x6_conv6 = torch.cat([pool6, x4_conv4], dim=1)
        x6_conv6 = self.conv6(x6_conv6)
        pool7 = self.upconv7(x6_conv6)
        x7_conv7 = torch.cat([pool7, x3_conv3], dim=1)
        x7_conv7 = self.conv7(x7_conv7)
        pool8 = self.upconv8(x7_conv7)
        x8_conv8 = torch.cat([pool8, x2_conv2], dim=1)
        x8_conv8 = self.conv8(x8_conv8)
        pool9 = self.upconv9(x8_conv8)
        x9_conv9 = torch.cat([pool9, x1_conv1], dim=1)
        x9_conv9 = self.conv9(x9_conv9)
        output = self.conv10(x9_conv9)
        return output, x9_conv9

    def predict(self, X, device=0, enable_dropout=False):
        """
        Predicts the outout after the model is trained.
        Inputs:
        - X: Volume to be predicted
        """
        self.eval()

        if type(X) is np.ndarray:
            X = torch.tensor(X, requires_grad=False).type(torch.FloatTensor).cuda(device, non_blocking=True)
        elif type(X) is torch.Tensor and not X.is_cuda:
            X = X.type(torch.FloatTensor).cuda(device, non_blocking=True)

        if enable_dropout:
            self.enable_test_dropout()

        with torch.no_grad():
            out = self.forward(X)

        max_val, idx = torch.max(out, 1)
        idx = idx.data.cpu().numpy()
        prediction = np.squeeze(idx)
        del X, out, idx, max_val
        return prediction



