import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pydicom import dcmread
from skimage.filters import frangi, hessian
from functions import create_multiple_img_fig, print_dicom_info
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

    best_shifts_neighbours = np.empty([0])

    for ind in range(num_of_img - 1):
        vec_1 = hor_integrals[ind]
        vec_2 = np.concatenate((np.zeros(max_shift, dtype=type),
                                hor_integrals[ind + 1],
                                np.zeros(max_shift, dtype=type)))
        sum_results = np.empty([0])

        """ pierwszy obraz jest stały, a drugi "przesuwamy" względem niego """
        for sh in range(-max_shift, max_shift + 1):
            window = vec_2[max_shift + sh: height + max_shift + sh]
            s = np.sum((vec_1 - window) ** 2)
            sum_results = np.append(sum_results, s)

        #        print(f"IMG {ind} - Array of sum for every shift applied")
        #        print(sum_results)

        best_sh = np.argmin(sum_results) - max_shift
        best_shifts_neighbours = np.append(best_shifts_neighbours, best_sh)

    best_shifts_toward_first = np.empty([0])
    vec_1 = hor_integrals[0]
    for ind in range(num_of_img - 1):
        vec_2 = np.concatenate((np.zeros(max_shift, dtype=type),
                                hor_integrals[ind + 1],
                                np.zeros(max_shift, dtype=type)))
        sum_results = np.empty([0])

        """ pierwszy obraz jest stały, a drugi "przesuwamy" względem niego """
        for sh in range(-max_shift, max_shift + 1):
            window = vec_2[max_shift + sh: height + max_shift + sh]
            s = np.sum((vec_1 - window) ** 2)
            sum_results = np.append(sum_results, s)

        #        print(f"IMG {ind} - Array of sum for every shift applied")
        #        print(sum_results)

        best_sh = np.argmin(sum_results) - max_shift
        best_shifts_toward_first = np.append(best_shifts_toward_first, best_sh)
    return best_shifts_neighbours, best_shifts_toward_first


def plot_ECG_signal(heights, use_average=False, title='', fig_title=''):
    heights_plot = heights
    heights_abs_plot = abs(heights)
    if use_average:
        heights_plot = moving_average(heights_plot, 3)
        heights_abs_plot = moving_average(heights_abs_plot, 3)
    fig = plt.figure()
    if fig_title != '':
        fig.canvas.manager.set_window_title(fig_title)
    ax = fig.add_subplot(111)
    # ax.scatter(range(len(images)), images)
    ax.plot(range(1, len(heights_plot) + 1), heights_plot, marker='o', label='normal vals')
    ax.plot(range(1, len(heights_plot) + 1), heights_abs_plot, marker='o', color='red', label='abs')
    ax.plot(range(1, len(heights_plot) + 1), np.zeros([heights_plot.shape[0]]))

    all_N = len(heights_plot) + 1
    N = math.ceil(all_N / 5)
    xind = []
    xlabels = []
    for i in range(N):
        xlabels.append(str((i * 5) + 1))
        xind.append((i * 5) + 1)
    ax.set_xticks(xind)
    ax.set_xticklabels(xlabels)
    # ax.set_xticks(range(1, len(heights)+1))
    # ax.set_xticklabels(range(1, len(heights)+1))

    ax.set_title(title)
    ax.legend(frameon=False)


def ED_finder_algorithm(images, max_shift=None, title=None):
    print("============== algorithm ==================")
    if max_shift is None:
        max_shift = math.ceil(images.shape[1] * 0.1)
    best_shifts_neighbours, best_shifts_toward_first = surrogate_ECG_horline_integrals(images, max_shift=max_shift)

    """ 
        Plotting best shifts between each two adjacent images (0:1, 1:2 ... N-1:N)
    """
    print("============== Surrogate ECG ==================")

    if title is None:
        title = f"Surrogate ECG signal, max_sh = {max_shift}"
    plot_ECG_signal(best_shifts_neighbours, title)
    plt.show()


def quick_image_filtering(images, thresh=10, min_size=700, showPlots=False):
    TYPE = images.dtype
    if showPlots:
        create_multiple_img_fig(images, 4, 'gray', addColorbar=True, title=f'original projection')

    print("============== filtering - frangi ==================")
    images_frangi = np.empty((0, images.shape[1], images.shape[2]), dtype=TYPE)

    for ind in range(images.shape[0]):
        img = frangi_scaled(images[ind])
        img = np.reshape(img, (1, 512, 512))
        images_frangi = np.append(images_frangi, img, axis=0)

    maximal = np.max(images_frangi)
    print(f'=============== MAX = {maximal}')

    bin_imgs = cv2.threshold(images_frangi, thresh, 255, cv2.THRESH_BINARY)[1]

    if showPlots:
        create_multiple_img_fig(bin_imgs, 4, 'gray', title=f'thresholded projection, th={thresh}')

    print("============== removing small objects ==================")
    images_reduced = np.empty((0, images.shape[1], images.shape[2]), dtype=TYPE)

    for img in bin_imgs:
        red = np.reshape(remove_small_objects(img, min_size), (1, 512, 512))
        images_reduced = np.append(images_reduced, red, axis=0)

    if showPlots:
        create_multiple_img_fig(images_reduced, 4, 'gray',
                                title=f'reduced binary projection, th={thresh} min_s={min_size}')

    # canny1 = cv2.Canny(images[18], 150, 190)
    # canny2 = cv2.Canny(images_frangi[18], 50, 100)
    # canny3 = cv2.Canny(reduced_img, 100, 200)
    #
    # images_res = np.stack((canny1, canny2, canny3))
    # titles = ['canny - Original image',
    #           'canny - Frangi',
    #           'canny - reduced ']
    # create_multiple_img_fig(images_res, 2, 'gray', title=f'binary{thresh}', img_titles=titles)

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


def wykres_na_podstawie_artykulu_hor_line_integrals(images_reduced):
    hor_integrals = np.sum(images_reduced, axis=2).transpose()
    hor_integrals = hor_integrals / 512

    all_N = hor_integrals.shape[1]
    N = math.ceil(all_N / 5)
    xind = []
    xlabels = []
    for i in range(N):
        xlabels.append(str((frame_time * i * 5) / 100))
        xind.append(i * 5)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(hor_integrals, 'gray')
    ax.set_xticks(xind)
    ax.set_xticklabels(xlabels)
    ax.set_aspect(aspect=0.1)
    ax.set_xlabel('Czas (s)')
    ax.set_ylabel('Całka po linii poziomej dla współrzędnej pionowej obrazu')
    plt.show()

def wykresy_ecg_ze_srednia_oraz_sasiadami_i_pierwszym(images_reduced):
    print("============== algorithm ==================")
    max_shift = math.ceil(images_reduced.shape[1] * 0.1)
    bs_neighbours, bs_toward_first = surrogate_ECG_horline_integrals(images_reduced, max_shift=max_shift)

    """ 
        Plotting best shifts between each two adjacent images (0:1, 1:2 ... N-1:N)
    """
    print("============== Surrogate ECG ==================")

    plot_ECG_signal(bs_neighbours, use_average=False,
                    title=f"Surrogate ECG signal (neighbours), max_sh = {max_shift}", fig_title=filepath)
    plot_ECG_signal(bs_neighbours, use_average=True,
                    title=f"Surrogate ECG signal (neighbours) with mov average, max_sh = {max_shift}",
                    fig_title=filepath)

    plot_ECG_signal(bs_toward_first, use_average=False,
                    title=f"Surrogate ECG signal (to first), max_sh = {max_shift}", fig_title=filepath)
    plot_ECG_signal(bs_toward_first, use_average=True,
                    title=f"Surrogate ECG signal (to first) with mov average, max_sh = {max_shift}", fig_title=filepath)


def moving_average(array, num_of_elem):
    return np.convolve(array, np.ones(num_of_elem), 'valid') / num_of_elem


if __name__ == '__main__':

    best_src = ["best_src\\s1exam2.dcm", "best_src\\s3exam14.dcm", "best_src\\s6exam8.dcm", "best_src\\s8exam2.dcm"]
    for filepath in best_src:
        print("\n============== image: " + filepath + " ==================")
        ds = dcmread(filepath)
        images = ds.pixel_array
        print_dicom_info(ds)
        frame_time = ds['FrameTime'].value if 'FrameTime' in ds else 0
        TYPE = images.dtype

        create_multiple_img_fig(images, 4, 'gray', addColorbar=False, title=f'original projection')
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(images[12], 'gray')

        images_red = quick_image_filtering(images, thresh=10, showPlots=False)
        wykresy_ecg_ze_srednia_oraz_sasiadami_i_pierwszym(images_red)

    plt.show()

    # ===========================================
    # create_multiple_img_fig(images, 4, 'gray', addColorbar=True, title=f'original projection')
    #
    # projection_reduced = quick_image_filtering(images, thresh=15)
    # ED_finder_algorithm(projection_reduced)

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
