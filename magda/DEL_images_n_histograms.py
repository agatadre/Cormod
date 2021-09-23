import matplotlib.pyplot as plt
import math
from pydicom import dcmread


"""
Poboczne.
Wyświetlanie histogramów serii obrazów - dla Agaty.
"""

def display_multiple_img_hist(images, rows=1, type='gray'):
    cols = 2 * math.ceil(images.shape[0]/rows)
    fig, axs = plt.subplots( nrows=rows, ncols=cols )
    num_of_img = images.shape[0]
    axs = axs.ravel()
    for ind in range(num_of_img):
        axs[2 * ind].imshow( images[ind], type, aspect='equal' )
        axs[2 * ind].set_title( ind, fontsize=8, pad=0.1 )
        axs[2 * ind].set_axis_off()
        axs[2 * ind + 1].hist(images[ind].ravel(), 256, [0, 256])
        axs[2 * ind + 1].set_title(f"{ind} hist", fontsize=8, pad=0.1 )
        axs[2 * ind + 1].set_axis_off()
    for ind in range(num_of_img*2, rows*cols):
        axs[ind].set_axis_off()
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.2)


if __name__ == '__main__':
    print("Type DICOM file path:\n")
    file_path = "numer1\\exam.dcm"
    # if file_path[-4:-1] != ".dcm":
    #    file_path = "exam.dcm"

    ds = dcmread(file_path)
    images = ds.pixel_array.astype(float)
    print(f"SHAPE of pixel_array.......:{images.shape}")

    display_multiple_img_hist(images, 4, 'gray')

    plt.show()

