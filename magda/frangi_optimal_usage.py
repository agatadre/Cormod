import matplotlib.pyplot as plt
from pydicom import dcmread
from skimage.filters import frangi, hessian

"""
=============
Frangi filter
=============

The Frangi and hybrid Hessian filters can be used to detect continuous
edges, such as vessels, wrinkles, and rivers.
"""


def display_4_img_hist(images, titles, plotHistograms=False):
    ncols = 4 if plotHistograms else 2

    fig, axs = plt.subplots(nrows=2, ncols=ncols)
    num_of_img = images.__len__()
    axs = axs.ravel()

    for ind in range(num_of_img):
        axs[2 * ind].imshow( images[ind], cmap=plt.cm.gray, aspect='equal' )
        axs[2 * ind].set_title( titles[ind], fontsize=8, pad=0.1 )
        axs[2 * ind].set_axis_off()

    if plotHistograms:
        axs[1].hist(images[0].ravel(), 256, [0, 256])
        axs[1].set_title('hist')
        axs[3].hist(images[1].ravel(), 256, [0, 256])
        axs[3].set_xscale('log')
        axs[3].set_title('hist log')
        axs[5].hist(images[2].ravel(), 256, [0, 256])
        axs[5].set_title('hist')
        axs[7].hist(images[3].ravel(), 256, [0, 256])
        axs[7].set_xscale('log')
        axs[7].set_title('hist log')

 #   fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.2)


if __name__ == '__main__':
    file_path = "numer2\\exam.dcm"
    ds = dcmread(file_path)

    """ TYPE = float """
    images = ds.pixel_array.astype(float)
    TYPE = images.dtype

    image = images[15]

    image_frangi = (255 * frangi(image)).astype('uint8')    # po prostu ucina końcówki, bez zaokrąglania

    image_inv = (image.max() - image)  # cv2.sub(image.max() - image)

    images_res = [image,
                  image_frangi,
                  image_inv,  # check minus operation
                  (255 * frangi(image_inv, black_ridges=False)).astype('uint8') ]
    titles = ['1. Original Image',
              '1. Frangi filter result',
              '2. Inverted Image',
              '2. Frangi filter result (black_ridges=False)']

    display_4_img_hist(images=images_res, titles=titles, plotHistograms=True)

    plt.show()
