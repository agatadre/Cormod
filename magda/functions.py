import math
import matplotlib.pyplot as plt


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


def display_multiple_img(images, rows=1, type='gray', title='', addColorbar=False):
    cols = math.ceil(images.shape[0]/rows)
    fig, axs = plt.subplots( nrows=rows, ncols=cols )
    num_of_img = images.shape[0]
    axs = axs.ravel()
    for ind in range(num_of_img):
        img_ret = axs[ind].imshow( images[ind], type, aspect='equal' )
        axs[ind].set_title( ind, fontsize=8, pad=0.1 )
        axs[ind].set_axis_off()
    for ind in range(num_of_img, rows*cols):
        axs[ind].set_axis_off()
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.2)
    if title != '':
        fig.canvas.set_window_title(title)
    if addColorbar:
        fig.colorbar(img_ret)


def save_np_as_png(img, filename):
    plt.imsave(f'{filename}.png', img, format='png', cmap=plt.cm.gray)
