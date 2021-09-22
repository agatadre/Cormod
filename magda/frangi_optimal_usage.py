from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import math
import argparse
import glob
from pydicom import dcmread
import cv2
from FRANGI.frangiFilrer2D import FrangiFilter2D
from skimage.filters import frangi, hessian


if __name__ == '__main__':
    print("Type DICOM file path:\n")
    file_path = "numer2\\exam.dcm"
    # if file_path[-4:-1] != ".dcm":
    #    file_path = "exam.dcm"

    ds = dcmread(file_path)
    images = ds.pixel_array
    print(f"SHAPE of pixel_array.......:{images.shape}")

    img_test1 = images[0]
    img_test2 = images[15]

    """
    =============
    Frangi filter
    =============
    
    The Frangi and hybrid Hessian filters can be used to detect continuous
    edges, such as vessels, wrinkles, and rivers.
    """

    image = img_test1
    image = image.astype(float)

    fig, ax = plt.subplots(nrows=2, ncols=4, subplot_kw={'adjustable': 'box'})

    ax[0, 0].imshow(image, cmap=plt.cm.gray)
    ax[0, 0].set_title('Original image - 1')
    ax[0, 0].axis('off')

    ax[0, 1].hist(image.ravel(), 256, [0, 256])
    ax[0, 1].set_title('1 - hist')

    ax[0, 2].imshow(frangi(image), cmap=plt.cm.gray)
    ax[0, 2].set_title('Frangi filter result - 1')
    ax[0, 2].axis('off')

    ax[0, 3].hist(frangi(image).ravel(), 256, [0, 256])
    ax[0, 3].set_title('1 frangi - hist')

    image = image.max()-image
    image = image.astype(float)

    ax[1, 0].imshow(image, cmap=plt.cm.gray)
    ax[1, 0].set_title('Original image - 2')
    ax[1, 0].axis('off')

    ax[1, 1].hist(image.ravel(), 256, [0, 256])
    ax[1, 1].set_title('2 - hist')

    ax[1, 2].imshow(frangi(image, black_ridges=False), cmap=plt.cm.gray)
    ax[1, 2].set_title('Frangi filter result - 2 (inverted, black_ridges=False)')
    ax[1, 2].axis('off')

    ax[1, 3].hist(frangi(image).ravel(), 256, [0, 256])
    ax[1, 3].set_title('2 frangi - hist')

    for a in ax:
        for aa in a:
            aa.axis('off')

    plt.show()

