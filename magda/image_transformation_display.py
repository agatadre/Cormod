from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import math
import argparse
import glob
from pydicom import dcmread
import cv2


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


def display_top_blackhat_res(img, filterSize=(10,10), title=''):
    fig, ax = plt.subplots(2, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, filterSize)

    tophat_img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    blackhat_img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

    ax[0, 0].imshow(img, 'gray')
    ax[0, 0].set_title('original')
    ax[0, 1].imshow(tophat_img, 'gray')
    ax[0, 1].set_title(f'tophat, kernel size{filterSize[0]}')
    ax[0, 2].imshow(blackhat_img, 'gray')
    ax[0, 2].set_title(f'blackhat, {filterSize[0]}')

    res_img = cv2.subtract(img, blackhat_img)
    im = ax[1, 0].imshow(res_img, 'gray')
    ax[1, 0].set_title('res = img - black')
    #    fig.colorbar(im)

    res_img = cv2.subtract(cv2.add(img, tophat_img), blackhat_img)
    im = ax[1, 1].imshow(res_img, 'gray')
    ax[1, 1].set_title('res = (img + top) - black')

    res_img = cv2.add( img, tophat_img)
    im = ax[1, 2].imshow(res_img, 'gray')
    ax[1, 2].set_title('res = img + top')


    fig.canvas.set_window_title(f'{filterSize[0]} {title}')
    fig.tight_layout()


def display_threshholding_img(img, th_val=127, blockSize=11, C=0, title=''):
    ret, th1 = cv2.threshold(img, th_val, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, blockSize=blockSize, C=C)
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, blockSize=blockSize, C=C)

    titles = ['Original Image', f'Global Thresholding (v = {th_val})',
              f'Adaptive Mean Thresholding: blockSize {blockSize}, C {C}', 'Adaptive Gaussian Thresholding']
    images_th = [img, th1, th2, th3]
    fig, ax = plt.subplots(2, 2)
    ax = ax.ravel()
    for i in range(4):
        ax[i].imshow(images_th[i], 'gray')
        ax[i].set_title(titles[i])
        ax[i].set_axis_off()

    if title != '':
        fig.canvas.set_window_title(f'{title}')
    fig.tight_layout()


def blackhat_transformations(img, kernelSize=(37,37), blurSize=5, th_val=10, title=''):
    b0 = cv2.morphologyEx( img,
                           cv2.MORPH_BLACKHAT,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernelSize) )
    b1 = cv2.medianBlur(b0, blurSize)
    _, b2 = cv2.threshold(b0, th_val, 255, cv2.THRESH_BINARY)
    _, b3 = cv2.threshold(b1, th_val, 255, cv2.THRESH_BINARY)

    titles = [f'0 - blackhat, kernel = {kernelSize[0]}', '1 - medianBlurred blackhat',
              'Adaptive Mean Th', 'Adaptive Mean Th of blurred']
    images_b = [b0, b1, b2, b3]
    fig, ax = plt.subplots(2, 2)
    ax = ax.ravel()
    for i in range(4):
        ax[i].imshow(images_b[i], 'gray')
        ax[i].set_title(titles[i])
        ax[i].set_axis_off()

    if title != '':
        fig.canvas.set_window_title(title)
    else:
        fig.canvas.set_window_title('Blackhat - transformations')
    fig.tight_layout()


if __name__ == '__main__':
    print("Type DICOM file path:\n")
    file_path = "numer2\\exam.dcm"
    # if file_path[-4:-1] != ".dcm":
    #    file_path = "exam.dcm"

    ds = dcmread(file_path)
    print_dicom_info(ds)
    images = ds.pixel_array
    print(f"SHAPE of pixel_array.......:{images.shape}")

# poprzedni kod
    #
    # img_test2 = images[15]
    # cv2.imshow('Test img 2', img_test2)
    #
    # filterSize = (12, 12)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
    # img = img_test2
    #
    # opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # tophat_img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    # blackhat_img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    # cv2.imshow('Opening image', opening)
    # cv2.imshow('Closing image', closing)
    # cv2.imshow('Tophat image', tophat_img)
    # cv2.imshow('Blackhat image', blackhat_img)
    #
    #
    #
    # # filterSize = (40, 40)
    # # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
    # # img = img_test2
    # #
    # # opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # # closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # # tophat_img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    # # blackhat_img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    # # cv2.imshow('Opening image 2', opening)
    # # cv2.imshow('Closing image 2', closing)
    # # cv2.imshow('Tophat image 2', tophat_img)
    # # cv2.imshow('Blackhat image 2', blackhat_img)
    #
    # plt.hist(blackhat_img.ravel(), 256, [0, 256])
    # plt.show()
    #
    # # apply binary thresholding
    # bin_img = cv2.threshold(blackhat_img, 8, 255, cv2.THRESH_BINARY)[1]
    #
    # cv2.imshow('Binary image', bin_img)
#poprzedni kod - koniec

# nowy sposób filtracji

    img = images[15]
    display_top_blackhat_res(img, (37, 37), title='original - top/blackhat op.')

    # img_blurred = cv2.medianBlur(img, 5)
    # display_top_blackhat_res(img_blurred, (37, 37), title='blur 5 - top/blackhat op.')

    # threshholding
    display_threshholding_img(img, th_val=110, blockSize=11, C=2, title='original - thresholding')
    tophat_img   = cv2.morphologyEx( img,
                                     cv2.MORPH_TOPHAT,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 37)))
    blackhat_img = cv2.morphologyEx( img,
                                     cv2.MORPH_BLACKHAT,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 37)) )

    res_img = cv2.subtract(img, blackhat_img)
    res_img = cv2.GaussianBlur(res_img, (5, 5), 0)
    display_threshholding_img(res_img, th_val=100, blockSize=131, C=2, title='-blac & gausBlur - thresholding')

    res_img = cv2.subtract(cv2.add(img, tophat_img), blackhat_img)
    res_img = cv2.medianBlur(res_img, 5)
    display_threshholding_img(res_img, th_val=100, blockSize=17, C=2, title='-black +top & medianBlur - thresholding')


    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Frame 'Relative Time' (n) = Frame Delay + Frame Time * (n-1)



#
# # =========================================================================
# # ======= preprocessing === TOP-HAT OPERATOR (MORPHOLOGICAL) ==============
#
# # Defining the kernel to be used in Top-Hat TODO - jaki filtr jest najlepszy??
# filterSize = (200, 200)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
#
# # Applying the Top-Hat operation
# images_tophat = np.empty((0, images.shape[1], images.shape[2]), dtype=images.dtype)
# for ind in range(images.shape[0]):
#     img = images[ind]
#     tophat_img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
#     tophat_img = np.reshape(tophat_img, (1, 512, 512))
#     images_tophat = np.append(images_tophat, tophat_img, axis=0)
#
# display_multiple_img(images_tophat, 5, math.ceil(images_tophat.shape[0]/5))
#
# # Computing horizontal line integrals (sum over x)
# hl_integrals = np.sum(images_tophat, axis=1)
#
# """ The vertical motion between two successive frames is estimated by identifying the shift along the
# vertical axis that minimizes the sum of squared differences between the corresponding @Hn vectors (@horizontal line integrals).
# """
#
# h1 = 0
# h2 = 0
#
# sq_diff = []
#
# for ind in range(hl_integrals.shape[0]):
#     h2 = ind
#     sum = np.sum((hl_integrals[h1] - hl_integrals[h2]) ** 2)
#     sq_diff = np.append(sq_diff, sum)
#
# print(sq_diff)
#
# plt.plot(range(hl_integrals.shape[0] - 1), sq_diff[1:])
# plt.title("Surrogate ECG signal")
# plt.show()
