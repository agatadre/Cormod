import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pydicom import dcmread
from skimage.filters import frangi, hessian
from functions import display_multiple_img
import cv2
from skimage import img_as_ubyte


""" 
Computing horizontal line integrals (sum over x).
The vertical motion between two successive frames is estimated by identifying the shift along the
vertical axis that minimizes the sum of squared differences between the corresponding vectors (horizontal line integrals).
"""
def surrogate_ECG_horline_integrals(images, max_shift=-1):
    type = images.dtype
    hor_integrals = np.sum(images, axis=2)
    num_of_img = hor_integrals.shape[0]

    height = hor_integrals.shape[1]
    max_shift = height - 1 if max_shift is -1 else max_shift

    best_shifts = np.empty([0])

    for ind in range(num_of_img - 1):
        vec_1 = hor_integrals[ind]
        vec_2 = np.concatenate((np.zeros(max_shift, dtype=type),
                                hor_integrals[ind + 1],
                                np.zeros(max_shift, dtype=type)))
        sum_results = np.empty([0])

        """ pierwszy obraz jest stały, a drugi "przesuwamy" względem niego """
        for sh in range(-max_shift, max_shift+1):
            window = vec_2[max_shift + sh: height + max_shift + sh]
            s = np.sum((vec_1 - window) ** 2)
            sum_results = np.append(sum_results, s)

#        print(f"IMG {ind} - Array of sum for every shift applied")
#        print(sum_results)

        best_sh = np.argmin(sum_results) - max_shift
        best_shifts = np.append(best_shifts, best_sh)
    return best_shifts


def plot_ECG_signal(heights, title=''):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.scatter(range(len(images)), images)
    ax.plot(range(len(heights)), heights, marker='o', label='normal vals')
    ax.plot(range(len(heights)), abs(heights), marker='o', color='red', label='abs')
    ax.plot(range(len(heights)), np.zeros([heights.shape[0]]))
    ax.set_title(title)
    ax.legend(frameon=False)


def test_algorithm():
    print("============= test =============")
    test1 = np.array([
        [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        ])

    display_multiple_img(test1, rows=11)

    test_sh = surrogate_ECG_horline_integrals(test1, max_shift=3)
    test_sh = abs(test_sh)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(len(test_sh)), test_sh,  marker='o')
    plt.show()

    print("============= end test =============")


def show_results_from_dicom():
    print("Type DICOM file path:\n")
    file_path = "numer2\\exam.dcm"
    ds = dcmread(file_path)
    images = ds.pixel_array
    TYPE = images.dtype

    print("============== filtering - frangi ==================")
    images_f = images.astype('float')

    images_frangi = np.empty((0, images.shape[1], images.shape[2]), dtype=TYPE)

    for ind in range(images.shape[0]):
        img = frangi_scaled(images_f[ind])
#        img = frangi(images_f[ind])
        img = np.reshape(img, (1, 512, 512))
        images_frangi = np.append(images_frangi, img, axis=0)

    maximal = np.max(images_frangi)
    print(f'=============== MAX = {maximal}')
    # display_multiple_img(images_frangi, 4, 'gray', addColorbar=True)

    thresh = 20
    bin_imgs = cv2.threshold(images_frangi, thresh, 255, cv2.THRESH_BINARY)[1]
    display_multiple_img(bin_imgs, 4, 'gray', title=f'binary{thresh}')
    #
    # bin_imgs = cv2.threshold(images_frangi, 10.0, 255.0, cv2.THRESH_BINARY_INV)[1]
    # display_multiple_img(bin_imgs, 4, 'gray', title='binary inv 10')

    #    plt.show()

    print("============== algorithm ==================")

    m_sh = 200
    # best_shifts_1 = surrogate_ECG_horline_integrals(images_frangi, max_shift=m_sh)
    # best_shifts_2 = surrogate_ECG_horline_integrals(images, max_shift=m_sh)
    best_shifts_3 = surrogate_ECG_horline_integrals(bin_imgs, max_shift=m_sh)

    """ 
        Plotting best shifts between each two adjacent images (0:1, 1:2 ... N-1:N)
    """
    print("============== Surrogate ECG ==================")

    # plot_ECG_signal(best_shifts_1, f"Surrogate ECG signal - frangi, max_sh = {m_sh}")
    # plot_ECG_signal(best_shifts_2, f"Surrogate ECG signal - original images, max_sh = {m_sh}")
    plot_ECG_signal(best_shifts_3, f"Surrogate ECG signal - thresh frangi, max_sh = {m_sh}")

    plt.show()


def clearImageEdges(img):
     imgM = np.copy(img)
     imgM[0:512, 0:30] = 0
     imgM[0:43, 0:512] = 0
     imgM[0:512, 475:512] = 0
     imgM[470:512, 0:512] = 0
     return imgM


''' Maska: thresholding, rozmycie + wykresy '''
def my_mask(img, th_val=127):
    ret, th_img = cv2.threshold(img, th_val, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    dil_img = cv2.dilate(th_img, kernel, iterations=3)

    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax.imshow(img, 'gray')
    ax = fig.add_subplot(132)
    ax.imshow(th_img, 'gray')
    ax = fig.add_subplot(133)
    ax.imshow(dil_img, 'gray')

    return dil_img


def maskImage(img, threshold=100):
     imgM = np.copy(img)
     kernel = np.ones((5, 5), np.uint8)
     imgM = cv2.dilate(img, kernel, iterations=3)
     # imgM = cv2.bilateralFilter(imgM, 100, 20, 600, borderType=cv2.BORDER_CONSTANT)
     for i in range(imgM.shape[0]):
         for j in range(imgM.shape[1]):
             if imgM[i][j] < threshold:
                 imgM[i][j] = 0
             else:
                 imgM[i][j] = 1

     return imgM


def frangi_scaled(img):
    return (255 * frangi(img)).astype('uint8')


def disp_top_blackhat_frangi_and_thresh_resuts(img, thresh=15):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 37))
    tophat_img = cv2.morphologyEx(img_2d, cv2.MORPH_TOPHAT, kernel)
    blackhat_img = cv2.morphologyEx(img_2d, cv2.MORPH_BLACKHAT, kernel)

    i_add = cv2.add(img_2d, tophat_img)
    i_sub = cv2.subtract(img_2d, blackhat_img)
    i_add_sub = cv2.subtract(cv2.add(img_2d, tophat_img), blackhat_img)

    fig = plt.figure()
    ax = fig.add_subplot(251)
    ax.imshow(img_2d, 'gray')

    res_img = frangi_scaled(img_2d)
    ret, th_img = cv2.threshold(res_img, thresh, 255, cv2.THRESH_BINARY)
    ax = fig.add_subplot(252)
    ax.imshow(res_img, 'gray')
    ax.set_title('classic frangi')
    ax = fig.add_subplot(257)
    ax.imshow(th_img, 'gray')

    res_img = frangi_scaled(i_sub)
    ret, th_img = cv2.threshold(res_img, thresh, 255, cv2.THRESH_BINARY)
    ax = fig.add_subplot(253)
    ax.imshow(res_img, 'gray')
    ax.set_title('sub blackhat - frangi')
    ax = fig.add_subplot(258)
    ax.imshow(th_img, 'gray')

    res_img = frangi_scaled(i_add)
    ret, th_img = cv2.threshold(res_img, thresh, 255, cv2.THRESH_BINARY)
    ax = fig.add_subplot(254)
    ax.imshow(res_img, 'gray')
    ax.set_title('add tophat - frangi')
    ax = fig.add_subplot(259)
    ax.imshow(th_img, 'gray')

    res_img = frangi_scaled(i_add_sub)
    ret, th_img = cv2.threshold(res_img, thresh, 255, cv2.THRESH_BINARY)
    ax = fig.add_subplot(255)
    ax.imshow(res_img, 'gray')
    ax.set_title('add top-, sub blackhat - frangi')
    ax = fig.add_subplot(2, 5, 10)
    ax.imshow(th_img, 'gray')


if __name__ == '__main__':
    show_results_from_dicom()
    # test_algorithm()

    ds = dcmread("numer2\\exam.dcm")
    ind = 18
    img_2d = ds.pixel_array[ind].astype('float')
    img_2d_scaled = (np.maximum(img_2d, 0) / img_2d.max()) * 255.0

    # ======================== TOP-, BLACKHAT =========================
    # disp_top_blackhat_frangi_and_thresh_resuts(img_2d, 10)
    # plt.show()
    # =====================================================================

    # Frame 'Relative Time' (n) = Frame Delay + Frame Time * (n-1)
