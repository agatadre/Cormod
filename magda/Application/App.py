import tkinter as tk
from tkinter import ttk
from pydicom.errors import InvalidDicomError
from pydicom import dcmread
from Application.FrMenu import FrMenuPanel
from Application.FrProjectionAnimator import FrProjectionAnimator
from Application.FrCropImages import FrCropImages
from functions import create_multiple_img_fig, print_dicom_info
import numpy as np

class App(tk.Tk):
    def __init__(self, center=False):
        super().__init__()
        self.ds_context = {}
        self.filenames = {}
        self.frames_context = {}

        self.window_width = 800
        self.window_height = 600

        if center:
            # get the screen dimension
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()
            # find the center point
            center_x = int(screen_width / 2 - self.window_width / 2)
            center_y = int(screen_height / 2 - self.window_height / 2)
            # set the position of the window to the center of the screen
            self.geometry(f'{self.window_width}x{self.window_height}+{center_x}+{center_y}')
        else:
            self.geometry(f'{self.window_width}x{self.window_height}+50+50')

        self.title('Cormod')
        self.iconbitmap('../assets/logoIcon.ico')  # only for ico format
        # self.iconphoto(False, tk.PhotoImage(file='../assets/logoIcon_16x16.png'))  # for png/jpg format
        self.resizable(True, True)
        # self.attributes('-toolwindow', True)  # windows only (remove the minimize/maximize button)
        # self.attributes('-topmost', 1)

        self.__active_frame = None
        self.__active_frame_name = ''
        self.__active_projection = None

        self.__create_widgets()

    def __create_widgets(self):
        # layout of the root window
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        # create the MenuPanel Frame and locate it
        self.__menu_panel_frame = FrMenuPanel(self)
        self.__menu_panel_frame.grid(column=0, row=0, sticky='nsew')

    """Load dicom file to App"""
    def load_chosen_dicom(self, index, filename):
        try:
            ds = dcmread(filename)
        except InvalidDicomError:
            print('The the file does not appear to be DICOM. Maybe the header is lacking.')

        if ds is not None:
            self.filenames[index] = filename
            self.ds_context[index] = ds
            self.frames_context[index] = ds.pixel_array

    def crop_projection(self, values):
        frames = self.frames_context[self.__active_projection]
        height = values['n']-values['s']
        width = values['e']-values['w']
        cropped = np.empty((0, height, width), dtype=frames.dtype)

        for idx in range(frames.shape[0]):
            img = frames[idx]
            img = img[values['s']:values['n'], values['w']:values['e']]
            img = np.reshape(img, (1, height, width))
            cropped = np.append(cropped, img, axis=0)

        self.frames_context[self.__active_projection] = cropped

    def show_component(self, name):
        if not self.ds_context.__contains__(self.__active_projection):
            return

        self.__active_frame_name = name

        if name == 'crop_images':
            self.__active_frame = FrCropImages(self)
            self.__active_frame.grid(column=0, row=1, sticky='nsew')
            array = self.ds_context[self.__active_projection].pixel_array
            self.__active_frame.load_dicom_image(array[0])

        elif name == 'projection_animator':
            self.__active_frame = FrProjectionAnimator(self)
            self.__active_frame.grid(column=0, row=1, sticky='nsew')

            dicom = self.ds_context[self.__active_projection]
            frame_time = dicom['FrameTime'].value if 'FrameTime' in dicom else 500
            frames = self.frames_context[self.__active_projection]
            self.__active_frame.load_frames(frames, frame_time)

    def change_active(self, which):
        self.__active_projection = which
        self.show_component(self.__active_frame_name)


if __name__ == "__main__":
    app = App(center=True)
    app.mainloop()
