import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pydicom import dcmread
from skimage.filters import frangi, hessian
from image_transformation_display import display_multiple_img


""" 
Computing horizontal line integrals (sum over x).
The vertical motion between two successive frames is estimated by identifying the shift along the
vertical axis that minimizes the sum of squared differences between the corresponding vectors (horizontal line integrals).
"""
def surrogate_ECG_horline_integrals(images):
    hor_integrals = np.sum(images, axis=1)
    num_of_img = hor_integrals.shape[0]

    height = hor_integrals.shape[1]

    best_shifts = np.empty([0])

    for ind in range(num_of_img - 1):
        vec_1 = hor_integrals[ind]
        vec_2 = np.concatenate((np.zeros(height, dtype=TYPE),
                                hor_integrals[ind + 1],
                                np.zeros(height, dtype=TYPE)))
        sum_results = np.empty([0])

        # pierwszy obraz jest stały, a drugi "przesuwamy" względem niego
        for sh in range(-height + 1, height):
            window = vec_2[height + sh: 2 * height + sh]
            s = np.sum((vec_1 - window) ** 2)
            sum_results = np.append(sum_results, s)

        print(f"IMG {ind} - Array of sum for every shift applied")
        print(sum_results)

        best_sh = np.argmin(sum_results) - height + 1
        best_shifts = np.append(best_shifts, best_sh)
    return best_shifts


if __name__ == '__main__':
    print("Type DICOM file path:\n")
    file_path = "numer2\\exam.dcm"
    # if file_path[-4:-1] != ".dcm":
    #    file_path = "exam.dcm"

    ds = dcmread(file_path)
    images = ds.pixel_array.astype('float')
    TYPE = images.dtype

    # ===============
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow( frangi(images[18]), 'gray' )
    ax.set_axis_off()
    frangi_s = frangi(images[18])
    plt.imsave('frangi111.png', frangi_s, format='png' , cmap=plt.cm.gray )
    plt.show()
    # ===============

    print(f"SHAPE of pixel_array.......:{images.shape}")

    display_multiple_img(images, 4, 'gray')

    """
    =============
    Frangi filter
    =============

    The Frangi and hybrid Hessian filters can be used to detect continuous
    edges, such as vessels, wrinkles, and rivers.
    """
    print("============== filtering - frangi ==================")

    images_frangi = np.empty((0, images.shape[1], images.shape[2]), dtype=TYPE)

    for ind in range(images.shape[0]):
        img = frangi(images[ind])
        print(f'{ind}  img = frangi(images[ind]) :')
        print(img)
        img = np.reshape(img, (1, 512, 512))
        print(f'{ind}  img = fnp.reshape(img, (1, 512, 512)) :')
        print(img)
        images_frangi = np.append(images_frangi, img, axis=0)
        print(f'{ind}  images_frangi :')
        print(images_frangi)
        print("================================")

    maximal = np.max(images_frangi)
    print(f'=============== MAX = {maximal}')
    display_multiple_img(images_frangi, 4, 'gray', addColorbar=True)

    # bin_imgs = cv2.threshold(images_frangi, 10.0, 255.0, cv2.THRESH_BINARY)[1]
    # display_multiple_img(bin_imgs, 4, 'gray', title='binary 10')
    #
    # bin_imgs = cv2.threshold(images_frangi, 10.0, 255.0, cv2.THRESH_BINARY_INV)[1]
    # display_multiple_img(bin_imgs, 4, 'gray', title='binary inv 10')

    plt.show()

    print("============== algorithm ==================")

    best_shifts_1 = surrogate_ECG_horline_integrals(images_frangi)
    best_shifts_2 = surrogate_ECG_horline_integrals(images)

    """ 
        Plotting best shifts between each two adjacent images (0:1, 1:2 ... N-1:N)
    """
    print("============== Surrogate ECG ==================")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(len(best_shifts_1)), best_shifts_1)
    ax.set_title("Surrogate ECG signal - frangi")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(len(best_shifts_2)), best_shifts_2)
    ax.set_title("Surrogate ECG signal - original images")

    plt.show()

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
