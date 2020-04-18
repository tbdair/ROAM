
import numpy as np
import nibabel as nib
from skimage.transform import resize

def transform_to_hu(image):
    intercept = -1024
    slope = 1
    hu_image = image * slope + intercept
    return hu_image


def win_scale(data, wl, ww, dtype, out_range):
    """
    Scale pixel intensity data using specified window level, width, and intensity range.
    """

    data_new = np.empty(data.shape, dtype=np.double)
    data_new.fill(out_range[1] - 1)

    data_new[data <= (wl - ww / 2.0)] = out_range[0]
    data_new[(data > (wl - ww / 2.0)) & (data <= (wl + ww / 2.0))] = \
        ((data[(data > (wl - ww / 2.0)) & (data <= (wl + ww / 2.0))] - (wl - 0.5)) / (ww - 1.0) + 0.5) * (
                    out_range[1] - out_range[0]) + out_range[0]
    data_new[data > (wl + ww / 2.0)] = out_range[1] - 1

    return data_new.astype(dtype)


def window_image(image, img_min, img_max):
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    return window_image


def get_pixels_hu(image):
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = -1024
    slope = 1

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


data_path = ""

lbeled = nib.load(data_path).get_fdata()
lbeled = np.squeeze(lbeled)
sz = lbeled.shape[0]

hu_image = transform_to_hu(lbeled)
out_range = [-1606.5217391304348, 597.8046699576582]
lung_image = win_scale(lbeled, -504.4, 2204.3, np.double, out_range)


lung_image = (lung_image - np.min(lung_image)) / (np.max(lung_image) - np.min(lung_image))

new_nifti = np.zeros((256, 256, sz))

for i in range(35):
    frame = lung_image[:,:,i]

    resized_img = resize(frame, (256, 256), preserve_range=True)
    new_nifti[:, :, i] = resized_img


savefile = nib.Nifti1Image(new_nifti, None)
nib.save(savefile, "Vol_Proc.nii")



