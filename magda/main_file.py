import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pydicom import dcmread
from skimage.filters import frangi, hessian
from functions import display_multiple_img, print_dicom_info
import cv2
import math
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
    ax.plot(range(1, len(heights)+1), heights, marker='o', label='normal vals')
    ax.plot(range(1, len(heights)+1), abs(heights), marker='o', color='red', label='abs')
    ax.plot(range(1, len(heights)+1), np.zeros([heights.shape[0]]))
    ax.set_xticks(range(1, len(heights)+1))
    ax.set_xticklabels(range(1, len(heights)+1))
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
         [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
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
    ax.plot(range(1, len(test_sh)+1), test_sh,  marker='o')
    ax.set_xticks(range(1, len(test_sh)+1))
    ax.set_xticklabels(range(1, len(test_sh)+1))
    plt.show()

    print("============= end test =============")


"""Wyświetlenie przefiltrowanych i ztreshowanych obrazów z numer2/exam.dcm
Wyliczenie sygnału ECG dla wyników po frangi i po thresholdingu"""
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

    m_sh = math.ceil(images.shape[1]*0.1)
    best_shifts_1 = surrogate_ECG_horline_integrals(images_frangi, max_shift=m_sh)
    # best_shifts_2 = surrogate_ECG_horline_integrals(images, max_shift=m_sh)
    best_shifts_3 = surrogate_ECG_horline_integrals(bin_imgs, max_shift=m_sh)

    """ 
        Plotting best shifts between each two adjacent images (0:1, 1:2 ... N-1:N)
    """
    print("============== Surrogate ECG ==================")

    plot_ECG_signal(best_shifts_1, f"Surrogate ECG signal - frangi, max_sh = {m_sh}")
    # plot_ECG_signal(best_shifts_2, f"Surrogate ECG signal - original images, max_sh = {m_sh}")
    plot_ECG_signal(best_shifts_3, f"Surrogate ECG signal - thresh frangi, max_sh = {m_sh}")

    plt.show()


def ED_finder_algorithm(images, max_shift=None, title=None):
    print("============== algorithm ==================")
    if max_shift is None:
        max_shift = math.ceil(images.shape[1] * 0.1)
    best_shifts = surrogate_ECG_horline_integrals(images, max_shift=max_shift)

    """ 
        Plotting best shifts between each two adjacent images (0:1, 1:2 ... N-1:N)
    """
    print("============== Surrogate ECG ==================")

    if title is None:
        title = f"Surrogate ECG signal, max_sh = {max_shift}"
    plot_ECG_signal(best_shifts, title)
    plt.show()


def quick_image_filtering(images, thresh=10, min_size=700):
    TYPE = images.dtype
    display_multiple_img(images, 4, 'gray', addColorbar=True, title=f'original projection')

    print("============== filtering - frangi ==================")
    images_frangi = np.empty((0, images.shape[1], images.shape[2]), dtype=TYPE)

    for ind in range(images.shape[0]):
        img = frangi_scaled(images[ind])
        img = np.reshape(img, (1, 512, 512))
        images_frangi = np.append(images_frangi, img, axis=0)

    maximal = np.max(images_frangi)
    print(f'=============== MAX = {maximal}')

    bin_imgs = cv2.threshold(images_frangi, thresh, 255, cv2.THRESH_BINARY)[1]
    display_multiple_img(bin_imgs, 4, 'gray', title=f'thresholded projection, th={thresh}')

    print("============== removing small objects ==================")
    images_reduced = np.empty((0, images.shape[1], images.shape[2]), dtype=TYPE)

    for img in bin_imgs:
        red = np.reshape(remove_small_objects(img, min_size), (1, 512, 512))
        images_reduced = np.append(images_reduced, red, axis=0)

    display_multiple_img(images_reduced, 4, 'gray', title=f'reduced binary projection, th={thresh} min_s={min_size}')

    # canny1 = cv2.Canny(images[18], 150, 190)
    # canny2 = cv2.Canny(images_frangi[18], 50, 100)
    # canny3 = cv2.Canny(reduced_img, 100, 200)
    #
    # images_res = np.stack((canny1, canny2, canny3))
    # titles = ['canny - Original image',
    #           'canny - Frangi',
    #           'canny - reduced ']
    # display_multiple_img(images_res, 2, 'gray', title=f'binary{thresh}', img_titles=titles)
    plt.show()
    return images_reduced


def remove_small_objects(binary_map, min_size=100):
    connectivity = 4  # You need to choose 4 or 8 for connectivity type

    # find all your connected components (white blobs in your image)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity)

    # taking out the background which is also considered a component
    nlabels = nlabels - 1
    areas = stats[1:, cv2.CC_STAT_AREA]  # get CC_STAT_AREA component as stats[label, COLUMN]

    # your answer image
    result = np.zeros(labels.shape, np.uint8)
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nlabels):
        if areas[i] >= min_size:
            result[labels == i + 1] = 255

    return result


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


""" Obraz najpierw zamieniony jest na float, żeby dostosować do go filtru frangi.
Wynik jest z zakresu (0, 1), więc dokonwywane jest skalowanie do (0, 255)
i zamiana na typ całkowity. 
"""
def frangi_scaled(img):
    img_float = img.astype('float')
    return np.around(255 * frangi(img_float)).astype('uint8')


def disp_top_blackhat_frangi_and_thresh_resuts(img, thresh=15):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 37))
    tophat_img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    blackhat_img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

    i_add = cv2.add(img, tophat_img)
    i_sub = cv2.subtract(img, blackhat_img)
    i_add_sub = cv2.subtract(cv2.add(img, tophat_img), blackhat_img)

    fig = plt.figure()
    ax = fig.add_subplot(251)
    ax.imshow(img, 'gray')

    res_img = frangi_scaled(img)
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
    # test_algorithm()

    print("Type DICOM file path:\n")
    file_path = "numer2\\exam.dcm"
    ds = dcmread(file_path)
    images = ds.pixel_array
    print_dicom_info(ds)

    projection_reduced = quick_image_filtering(images, thresh=15)
    ED_finder_algorithm(projection_reduced)

    # show_results_from_dicom()


    # ds = dcmread("numer2\\exam.dcm")
    # ind = 18
    # img_2d = ds.pixel_array[ind].astype('float')
    #
    # plt.imshow(img_2d, 'gray')
    #
    # plt.show()

    # img_2d_scaled = (np.maximum(img_2d, 0) / img_2d.max()) * 255.0
    #
    # # ======================== TOP-, BLACKHAT =========================
    # disp_top_blackhat_frangi_and_thresh_resuts(img_2d, 10)
    # plt.show()
    # =====================================================================

    # Frame 'Relative Time' (n) = Frame Delay + Frame Time * (n-1)
