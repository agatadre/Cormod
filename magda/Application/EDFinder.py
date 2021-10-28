import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pydicom import dcmread
from skimage.filters import frangi, hessian
from functions import create_multiple_img_fig, print_dicom_info, save_np_as_png
import cv2
import math
from skimage import img_as_ubyte


class EDFinder:
    filename = ''

    """
    Computing horizontal line integrals (sum over x).
    The vertical motion between two successive frames is estimated by identifying the shift along the
    vertical axis that minimizes the sum of squared differences between the corresponding vectors (horizontal line integrals).
    """
    @staticmethod
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
            for sh in range(-max_shift, max_shift + 1):
                window = vec_2[max_shift + sh: height + max_shift + sh]
                s = np.sum((vec_1 - window) ** 2)
                sum_results = np.append(sum_results, s)

            #        print(f"IMG {ind} - Array of sum for every shift applied")
            #        print(sum_results)

            best_sh = np.argmin(sum_results) - max_shift
            best_shifts = np.append(best_shifts, best_sh)
        return best_shifts

    """ Obraz najpierw zamieniony jest na float, żeby dostosować do go filtru frangi.
    Wynik jest z zakresu (0, 1), więc dokonwywane jest skalowanie do (0, 255)
    i zamiana na typ całkowity.
    """
    @staticmethod
    def frangi_scaled(img):
        img_float = img.astype('float')
        return np.around(255 * frangi(img_float)).astype('uint8')

    @staticmethod
    def quick_image_filtering(images, thresh=10, min_size=700):
        TYPE = images.dtype


        print("============== filtering - frangi ==================")
        images_frangi = np.empty((0, images.shape[1], images.shape[2]), dtype=TYPE)

        for ind in range(images.shape[0]):
            img = EDFinder.frangi_scaled(images[ind])
            img = np.reshape(img, (1, 512, 512))
            images_frangi = np.append(images_frangi, img, axis=0)

        maximal = np.max(images_frangi)
        print(f'=============== MAX = {maximal}')

        bin_imgs = cv2.threshold(images_frangi, thresh, 255, cv2.THRESH_BINARY)[1]

        print("============== removing small objects ==================")
        images_reduced = np.empty((0, images.shape[1], images.shape[2]), dtype=TYPE)

        for img in bin_imgs:
            red = np.reshape(EDFinder.__remove_small_objects(img, min_size), (1, 512, 512))
            images_reduced = np.append(images_reduced, red, axis=0)

        # canny1 = cv2.Canny(images[18], 150, 190)
        # canny2 = cv2.Canny(images_frangi[18], 50, 100)
        # canny3 = cv2.Canny(reduced_img, 100, 200)
        #
        # images_res = np.stack((canny1, canny2, canny3))
        # titles = ['canny - Original image',
        #           'canny - Frangi',
        #           'canny - reduced ']
        # create_multiple_img_fig(images_res, 2, 'gray', title=f'binary{thresh}', img_titles=titles)

        print("============== displaying results ==================")
        # create_multiple_img_fig(images, 4, 'gray', addColorbar=True, title=f'original projection')
        # create_multiple_img_fig(bin_imgs, 4, 'gray', title=f'thresholded projection, th={thresh}')


        fig = create_multiple_img_fig(images_reduced, 4, 'gray',
                             title=f'reduced binary projection, th={thresh} min_s={min_size}')
        save_np_as_png(EDFinder.filename + '_reduced_images', fig=fig)
        plt.close(fig=fig)
        return images_reduced

    @staticmethod
    def __remove_small_objects(binary_map, min_size=100):
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

    @staticmethod
    def plot_ECG_signal(heights, title=''):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax.scatter(range(len(images)), images)
        ax.plot(range(1, len(heights) + 1), heights, marker='o', label='normal vals')
        ax.plot(range(1, len(heights) + 1), abs(heights), marker='o', color='red', label='abs')
        ax.plot(range(1, len(heights) + 1), np.zeros([heights.shape[0]]))
        ax.set_xticks(range(1, len(heights) + 1))
        ax.set_xticklabels(range(1, len(heights) + 1))
        ax.set_title(title)
        ax.legend(frameon=False)
        return fig

    @staticmethod
    def find_EDPhase(projection, max_shift=None, plot_ECG_signal=True, ECG_signal_title=''):

        projection_reduced = EDFinder.quick_image_filtering(projection, thresh=15)

        print("============== algorithm ==================")
        if max_shift is None:
            max_shift = math.ceil(projection.shape[1] * 0.1)
        best_shifts = EDFinder.surrogate_ECG_horline_integrals(projection, max_shift=max_shift)

        if plot_ECG_signal:
            """ 
                Plotting best shifts between each two adjacent images (0:1, 1:2 ... N-1:N)
            """
            print("============== Surrogate ECG ==================")
            if ECG_signal_title is None:
                ECG_signal_title = f"Surrogate ECG signal, max_sh = {max_shift}"
            fig = EDFinder.plot_ECG_signal(best_shifts, ECG_signal_title)
            save_np_as_png(EDFinder.filename + '_ECG_signal', fig=fig)
            plt.close(fig=fig)

        return best_shifts

