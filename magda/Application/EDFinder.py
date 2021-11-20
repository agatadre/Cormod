import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
import matplotlib.image as mpimg
import numpy as np
from pydicom import dcmread
from skimage.filters import frangi, hessian
from functions import create_multiple_img_fig, print_dicom_info, save_np_as_png
import cv2
import math
import matplotlib.figure
from skimage import img_as_ubyte
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)


class EDFinder:
    filename = ''
    mainWindow = None
    projection_reduced = None
    bs_neighbours = None
    bs_toward_first = None
    max_shift = 0

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
                squares = np.square((vec_1 - window))
                s = np.sum(squares)
                sum_results = np.append(sum_results, s)

            best_sh = np.argmin(sum_results) - max_shift
            best_shifts_neighbours = np.append(best_shifts_neighbours, best_sh)
            # print(f"IMG {ind} - min(sum_results)" + str(np.min(sum_results)))

        print('toward first')
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
                squares = np.square((vec_1 - window))
                s = np.sum(squares)
                sum_results = np.append(sum_results, s)

            best_sh = np.argmin(sum_results) - max_shift
            best_shifts_toward_first = np.append(best_shifts_toward_first, best_sh)
            # print(f"IMG {ind} - min(sum_results)" + str(np.min(sum_results)))
        return best_shifts_neighbours, best_shifts_toward_first

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
        height = images.shape[1]
        width = images.shape[2]
        images_frangi = np.empty((0, height, width), dtype=TYPE)

        for ind in range(images.shape[0]):
            img = EDFinder.frangi_scaled(images[ind])
            img = np.reshape(img, (1, height, width))
            images_frangi = np.append(images_frangi, img, axis=0)

        maximal = np.max(images_frangi)
        print(f'=============== MAX = {maximal}')

        bin_imgs = cv2.threshold(images_frangi, thresh, 255, cv2.THRESH_BINARY)[1]

        print("============== removing small objects ==================")
        images_reduced = np.empty((0, height, width), dtype=TYPE)

        for img in bin_imgs:
            red = np.reshape(EDFinder.__remove_small_objects(img, min_size), (1, height, width))
            images_reduced = np.append(images_reduced, red, axis=0)

        # print("============== displaying results ==================")
        # fig = create_multiple_img_fig(images_reduced, 4, 'gray',
        #                      title=f'reduced binary projection, th={thresh} min_s={min_size}')
        # save_np_as_png(EDFinder.filename + '_reduced_images', fig=fig)
        # plt.close(fig=fig)
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
    def find_EDPhase(projection):
        thresh = 10
        EDFinder.projection_reduced = EDFinder.quick_image_filtering(projection, thresh=thresh)

        newWindow = ECG_signal_window()

    @staticmethod
    def find_best_shifts(start_frame):
        print("============== algorithm ==================")
        #TODO - max_shift
        EDFinder.max_shift = math.ceil(EDFinder.projection_reduced.shape[1] * 0.05)

        short_proj = EDFinder.projection_reduced[start_frame:]
        max_shift = EDFinder.max_shift
        EDFinder.bs_neighbours, EDFinder.bs_toward_first = EDFinder.surrogate_ECG_horline_integrals(
            short_proj, max_shift=max_shift)


class ECG_signal_window:
    def __init__(self):
        self.newW = tk.Toplevel(EDFinder.mainWindow)  # bezposrednie tworzenie okna
        self.newW.title("ED signal options")
        self.newW.geometry("850x550")

        ttk.Label(self.newW, text='Set first OK frame:').grid(row=0, column=0)
        self.start_frame = tk.IntVar()
        ttk.Entry(self.newW, textvariable=self.start_frame, width=5).grid(row=0, column=1)
        self.btn_start = ttk.Button(self.newW, text='Start', command=self.invoke_ed_finder_alg)
        self.btn_start.grid(row=0, column=2)

        self.button_hli = ttk.Button(self.newW, text='hor line integrals',
                                     command=self.wykres_na_podstawie_artykulu_hor_line_integrals)
        self.button_hli.grid(row=0, column=3)
        self.button_hli.state(['disabled'])

        self.button_ecg_n = ttk.Button(self.newW, text='ECG - neighbours',
                                       command=self.wykres_ecg_neighbours)
        self.button_ecg_n.grid(row=0, column=4)
        self.button_ecg_n.state(['disabled'])

        self.button_ecg_f = ttk.Button(self.newW, text='ECG - to first',
                                       command=self.wykres_ecg_to_first)
        self.button_ecg_f.grid(row=0, column=5)
        self.button_ecg_f.state(['disabled'])

        self.fig = matplotlib.figure.Figure()
        self.fig.set_size_inches(8, 5, forward=True)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, self.newW)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=6)

        # initial view
        self.save_multiple_img_fig()

    def plot_ECG_signal(self, heights, title=''):
        fig = matplotlib.figure.Figure()
        fig.set_size_inches(8, 5, forward=True)
        ax = fig.add_subplot(111)

        # ax.scatter(range(len(images)), images)
        ax.plot(range(1, len(heights) + 1), heights, marker='o', label='normal vals')
        ax.plot(range(1, len(heights) + 1), abs(heights), marker='o', color='red', label='abs')
        ax.plot(range(1, len(heights) + 1), np.zeros([heights.shape[0]]))

        all_N = len(heights) + 1
        N = math.ceil(all_N / 5)
        xind = []
        xlabels = []
        for i in range(N):
            xlabels.append(str((i * 5) + 1 + self.start_frame.get()))
            xind.append((i * 5) + 1)
        ax.set_xticks(xind)
        ax.set_xticklabels(xlabels)

        # ax.set_xticks(range(1, len(heights) + 1))
        # ax.set_xticklabels(range(1, len(heights) + 1))
        ax.set_title(title)
        ax.legend(frameon=False)

        self.fig = fig
        self.ax = ax
        self.canvas.figure = fig
        self.canvas.draw()

    def wykres_na_podstawie_artykulu_hor_line_integrals(self):
        images_reduced = EDFinder.projection_reduced
        frame_time = EDFinder.mainWindow.get_frame_time()

        hor_integrals = np.sum(images_reduced, axis=2).transpose()
        hor_integrals = hor_integrals / 512

        all_N = hor_integrals.shape[1]
        N = math.ceil(all_N / 5)
        xind = []
        xlabels = []
        for i in range(N):
            xlabels.append(str((frame_time * i * 5) / 100))
            xind.append(i * 5)

        fig = matplotlib.figure.Figure()
        fig.set_size_inches(5, 5, forward=True)
        ax = fig.add_subplot(111)
        ax.imshow(hor_integrals, 'gray')
        ax.set_xticks(xind)
        ax.set_xticklabels(xlabels)
        ax.set_aspect(aspect=0.1)
        ax.set_xlabel('Czas (s)')
        ax.set_ylabel('Całka po linii poziomej dla współrzędnej pionowej obrazu')

        self.fig = fig
        self.ax = ax
        self.canvas.figure = fig
        self.canvas.draw()

        save_np_as_png(EDFinder.filename + '_hli', fig=self.fig)

    def wykres_ecg_neighbours(self):
        max_shift = EDFinder.max_shift
        self.plot_ECG_signal(EDFinder.bs_neighbours, title=f"Surrogate ECG signal (neighbours), max_sh = {max_shift}")

        save_np_as_png(EDFinder.filename + '_ECG_signal_neighbours', fig=self.fig)

    def wykres_ecg_to_first(self):
        max_shift = EDFinder.max_shift
        self.plot_ECG_signal(EDFinder.bs_toward_first, title=f"Surrogate ECG signal (to first), max_sh = {max_shift}")

        save_np_as_png(EDFinder.filename + '_ECG_signal_to_first', fig=self.fig)

    def save_multiple_img_fig(self):
        images = EDFinder.projection_reduced

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

        cols = math.ceil(images.shape[0] / rows)

        fig = matplotlib.figure.Figure()
        axs = fig.subplots(nrows=rows, ncols=cols)
        num_of_img = images.shape[0] if isinstance(images, np.ndarray) else len(images)
        axs = axs.ravel()

        for ind in range(num_of_img):
            axs[ind].imshow(images[ind], 'gray', aspect='equal')
            axs[ind].set_title(ind, fontsize=8, pad=0.1)
            axs[ind].set_axis_off()

        for ind in range(num_of_img, rows * cols):
            axs[ind].set_axis_off()

        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.2)

        self.fig = fig
        self.ax = axs
        self.canvas.figure = fig
        self.canvas.draw()

        save_np_as_png(EDFinder.filename + '_filtered_projection', fig=fig)

    def invoke_ed_finder_alg(self):
        start_frame = self.start_frame.get()
        EDFinder.find_best_shifts(start_frame)
        self.button_hli.state(['!disabled'])
        self.button_ecg_n.state(['!disabled'])
        self.button_ecg_f.state(['!disabled'])

