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
        self.frames = None
        self.animation = None
        self.frame_time = 0
        self.playing = False

        ttk.Style().configure('PA.TFrame', background='#bdfdcb')
        self['style'] = 'PA.TFrame'

        self.__create_widgets()

    def __create_widgets(self):
        # setup the grid layout manager
        # self.columnconfigure(0, weight=1)

        # ttk.Button(self, text='Start', command=self.start_animation).grid(column=0, row=0)

        self.baseFrame = ttk.Frame(self, width=400, height=400, bg="gray")
        self.baseFrame.place(x=150, y=40)
        self.baseLabel = ttk.Label(self.baseFrame, bg="gray")
        self.baseFrame.place(x=150, y=40)

        # canvas = tk.Canvas(self, width=800, height=800)
        # canvas.pack()  # this makes it visible
        #
        # # Loads and create image (put the image in the folder)
        # img = tk.PhotoImage(file="mega.png")
        # image = canvas.create_image(10, 10, anchor=tk.NW, image=img)

    def load_frames(self, dicom):
        self.frames = dicom.pixel_array
        self.frame_time = dicom['FrameTime'].value if 'FrameTime' in dicom else 10

    def frames_len(self):
        return self.frames.size[0]

    def __update_animation(self, frame_number):
        img = self.frames[frame_number]
        img_disp = self.axes.imshow(img, 'gray')
        self.axes.set_title(f'Frame no. {frame_number}', fontsize=20)
        return img_disp
        # ax.set_axis_off()

    def start_animation(self):
        # self.animation = FuncAnimation(fig=self.__figure,
        #                                func=self.__update_animation,
        #                                frames=22,
        #                                interval=self.__frame_time)
        self.playing = True
        if self.play_count == 1:
            # print(ind)
            frame = self.my_frames[ind]
            ind += 1
            self.baseLabel.configure(image=frame)
            if ind > len(self.my_frames) - 1:
                ind = 0
            self.last_index = ind
            self.frameCounterLabel.configure(text="Current Frame: %d" % (self.last_index))
            self.loop = self.master.after(self.fpsValue, self.play_anim, ind, anim, 1)
        else:
            return


    def update_anim(self, i):
        # im_normed = np.random.random((64, 64))
        # self.im = self.axes.imshow(im_normed)
        # self.axes.set_title("Angle: {}*pi/10".format(i), fontsize=20)
        # self.axes.set_axis_off()
        # return self.im,
        self.line.set_ydata(self.A * np.sin(self.x + self.v * i))  # update the data
        return self.line,

    def init(self):
        im_normed = np.random.random((64, 64))
        self.im = self.axes.imshow(im_normed)
        self.axes.set_title("Angle: {}*pi/10", fontsize=20)
        self.axes.set_axis_off()
        return self.im,
