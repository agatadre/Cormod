import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib


class FrProjectionAnimator(ttk.Frame):

    def __init__(self, container):
        super().__init__(container)
        self.mainApp = container
        ttk.Style().configure('PA.TFrame', background='#bdfdcb')
        self['style'] = 'PA.TFrame'

        self.__frames = None
        self.__frame_time = None
        self.__animation = None

        self.__create_widgets()

    def __create_widgets(self):
        self.__btn_start = ttk.Button(self, text='Start', command=self.__start_animation)
        self.__btn_start.state(['disabled'])

        self.__fig, self.__ax = plt.subplots()
        self.__ax.set_aspect('equal', adjustable='box')
        self.__canvas = FigureCanvasTkAgg(self.__fig, self)
        self.__canvas.draw()

        # setup the grid layout manager
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=7)
        self.rowconfigure(0, weight=1)
        self.__btn_start.grid(column=0, row=0)
        self.__canvas.get_tk_widget().grid(column=1, row=0, sticky='nsew')

    def load_frames(self, dicom):
        self.__frames = dicom.pixel_array
        self.__frame_time = dicom['FrameTime'].value if 'FrameTime' in dicom else 500
        self.__btn_start.state(['!disabled'])

    def frames_len(self):
        return self.__frames.shape[0]

    def __update_animation(self, frame_number):
        img = self.__frames[frame_number]
        im = plt.imshow(img, cmap='gray', animated=True)
        # text = self.__ax.set_title(f'Frame no. {frame_number}', fontsize=20)
        return im,text,

    def __start_animation(self):
        self.__animation = FuncAnimation(fig=self.__fig,
                                         func=self.__update_animation,
                                         frames=self.frames_len(),
                                         blit=True,
                                         interval=self.__frame_time,
                                         repeat=False)


