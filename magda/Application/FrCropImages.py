import math
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.figure
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib
from pathlib import Path
from Application.EDFinder import EDFinder


class FrCropImages(ttk.Frame):

    def __init__(self, container):
        super().__init__(container)
        self.mainApp = container
        ttk.Style().configure('PA.TFrame', background='#ffffff')
        self['style'] = 'PA.TFrame'

        self.__frame = None
        self.__cropped_frame = None
        self.__vmin = 0
        self.__vmax = 255

        self.left = 0
        self.down = 0
        self.right = 0
        self.top = 0
        self.__fig = matplotlib.figure.Figure()
        self.__fig.patch.set_facecolor('#9dc2bb')
        self.__ax = self.__fig.add_subplot(111)
        self.__ax.set_aspect('equal', adjustable='box')

        self.__create_widgets()

    def __create_widgets(self):
        self.__btn_panel = FrButtons(self)

        self.__canvas = FigureCanvasTkAgg(self.__fig, self)
        self.__canvas.draw()

        # drawtype is 'box' or 'line' or 'none'
        #TODO - spancords='data'
        self.__rs = RectangleSelector(self.__ax, self.line_select_callback,
                                      drawtype='box', useblit=False, button=[1],  # use only left button
                                      minspanx=5, minspany=5, spancoords='data', interactive=True,
                                      rectprops=dict(facecolor='#ffba49', edgecolor='red', linewidth=2, alpha=0.2, fill=True))

        # setup the grid layout manager
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=7)
        self.rowconfigure(0, weight=1)

        self.__btn_panel.grid(column=0, row=0, sticky='nsew')
        self.__canvas.get_tk_widget().grid(column=1, row=0, sticky='nsew')

    def line_select_callback(self, eclick, erelease):
        self.left, self.down = int(eclick.xdata), int(eclick.ydata)
        self.right, self.top = int(erelease.xdata), int(erelease.ydata)
        print(f'l,d ({self.left}, {self.down}) --> r,t ({self.right}, {self.top})')
        self.left = math.ceil(self.__rs.extents[0])
        self.right = math.ceil(self.__rs.extents[1])
        self.down = math.ceil(self.__rs.extents[2])
        self.top = math.ceil(self.__rs.extents[3])
        print(f'EXT ({self.left}, {self.down}, {self.right}, {self.top})')

    def redraw_image(self, image):
        self.__ax.imshow(image, cmap='gray', vmin=self.__vmin, vmax=self.__vmax)
        self.__canvas.draw()

    def load_dicom_image(self, image):
        self.__frame = image
        self.__vmin = np.amin(image)
        self.__vmax = np.amax(image)
        self.redraw_image(image)
        self.__btn_panel.btn_crop.state(['!disabled'])

    def crop_image(self):
        self.__rs.extents = (0, 0, 0, 0)
        self.__rs.set_active(False)
        for artist in self.__rs.artists:
            artist.set_visible(False)
        self.__rs.update()

        imgM = np.copy(self.__frame)
        imgM = imgM[self.down:self.top, self.left:self.right]
        self.__cropped_frame = imgM
        self.redraw_image(self.__cropped_frame)

        self.__rs.set_active(True)
        self.__btn_panel.btn_reset.state(['!disabled'])
        self.__btn_panel.btn_crop.state(['disabled'])
        self.__btn_panel.btn_save.state(['!disabled'])

    def reset_image(self):
        self.redraw_image(self.__frame)
        self.__btn_panel.btn_crop.state(['!disabled'])
        self.__btn_panel.btn_save.state(['disabled'])

    def save_cropped(self):
        self.__frame = self.__cropped_frame
        values = dict(
            w=int(self.left),
            s=int(self.down),
            e=int(self.right),
            n=int(self.top)
        )
        self.mainApp.crop_projection(values)
        self.__btn_panel.btn_crop.state(['!disabled'])
        self.__btn_panel.btn_save.state(['disabled'])

class FrButtons(ttk.Frame):
    def __init__(self, container):
        super().__init__(container)
        self.container = container
        ttk.Style().configure('Btn.TFrame', background='#ddbea9')
        self['style'] = 'Btn.TFrame'
        ttk.Style().configure('Btn.TLabel', background='#ddbea9')

        self.btn_reset = ttk.Button(self, text='Reset', command=container.reset_image)
        self.btn_crop = ttk.Button(self, text='Crop', command=container.crop_image)
        self.btn_save = ttk.Button(self, text='Save changes', command=container.save_cropped)
        self.btn_reset.state(['disabled'])
        self.btn_crop.state(['disabled'])
        self.btn_save.state(['disabled'])

        self.btn_reset.pack(ipadx=10, ipady=2, pady=5)
        self.btn_crop.pack(ipadx=10, ipady=2, pady=5)
        self.btn_save.pack(ipadx=10, ipady=2, pady=5)
