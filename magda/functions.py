import math
import matplotlib.pyplot as plt
import numpy as np

"""
Funkcje używane w wielu miejscach.
W przyszłości do podziału między klasy (tak?), na razie umieszczam je tutaj.
"""


def print_dicom_info(dic_file):
    print(dic_file['Modality'])
    print(dic_file['StudyDate'])
    print(dic_file['SOPClassUID'])
    print(dic_file['Manufacturer'])
    print(dic_file['ManufacturerModelName'])
    print(dic_file['DeviceSerialNumber'])
    print(dic_file['SoftwareVersions'])
    print(f"Transfer Syntax..........: {dic_file.file_meta.TransferSyntaxUID}")
    print(dic_file['SamplesPerPixel'])  # 1 -> grayscale
    print(dic_file['PhotometricInterpretation'])
    print(dic_file['Rows'])
    print(dic_file['Columns'])
    if dic_file[0x0028, 0x0002].value != 1:
        print(f"Planar Configuration (order of pix val e.g. rgb)...............: {dic_file[0x0028, 0x0006].value}")
    print(dic_file['FrameTime'].__str__() + " ms")  # Nominal time (in msec) per individual frame.
    print(dic_file['NumberOfFrames'])  # Number of frames in a Multi-frame Image. See C.7.6.6.1.1 for further explanation.
    print(dic_file['RecommendedDisplayFrameRate'])  # Recommended rate at which the frames of a Multi-frame image should be displayed in frames/second.
    print(dic_file['CineRate'])
    # print(ds['ActualFrameDuration'])
    # print(ds['ContentTime']) #The time the image pixel data creation started. Required if image is part of a series in which the images are temporally related.


def create_multiple_img_fig(images, rows=1, type='gray', title='', addColorbar=False, img_titles=''):
    if images.shape[0] > 30:
        rows = 6
    elif images.shape[0] > 20:
        rows = 5
    elif images.shape[0] > 15:
        rows = 4
    elif images.shape[0] > 8:
        rows = 3
    elif images.shape[0] > 2:
        rows = 2
    else:
        rows = 1

    cols = math.ceil(images.shape[0]/rows)
    fig, axs = plt.subplots( nrows=rows, ncols=cols )
    num_of_img = images.shape[0] if isinstance(images, np.ndarray) else len(images)
    if isinstance(images, np.ndarray):
        num_of_img = images.shape[0]
    else:
        num_of_img = len(images)
    axs = axs.ravel()

    for ind in range(num_of_img):
        img_ret = axs[ind].imshow( images[ind], type, aspect='equal' )
        if img_titles != '':
            axs[ind].set_title(ind, fontsize=8, pad=0.1)
        else:
            axs[ind].set_title(ind, fontsize=8, pad=0.1 )
        axs[ind].set_axis_off()

    for ind in range(num_of_img, rows*cols):
        axs[ind].set_axis_off()

    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.2)
    if title != '':
        fig.canvas.manager.set_window_title(title)
    if addColorbar:
        fig.colorbar(img_ret)

    return fig


def save_np_as_png(filename, fig=None, img=None):
    if fig == None:
        plt.imsave(f'{filename}.png', img, format='png', cmap=plt.cm.gray)
    else:
        fig.savefig(f'{filename}.png',
                    format='png',
                    bbox_inches='tight',
                    pad_inches=0.5)
