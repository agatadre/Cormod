import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib
matplotlib.use("TkAgg")

class Window(ttk.Frame):

    def __init__(self, master = None):
        ttk.Frame.__init__(self, master)
        self.master = master
        self.init_window()


    def Clear(self):
        print("clear")
        self.textAmplitude.insert(0, "1.0")
        self.textSpeed.insert(0, "1.0")


    def Plot(self):
        self.v = float(self.textSpeed.get())
        self.A = float(self.textAmplitude.get())


    def animate(self,i):
        self.line.set_ydata(self.A*np.sin(self.x+self.v*i))  # update the data
        return self.line,


    def init_window(self):
        self.master.title("Use Of FuncAnimation in tkinter based GUI")
        self.pack(fill='both', expand=1)

        #Create the controls, note use of grid
        self.labelSpeed = ttk.Label(self,text="Speed (km/Hr)",width=12)
        self.labelSpeed.grid(row=0,column=1)
        self.labelAmplitude = ttk.Label(self,text="Amplitude",width=12)
        self.labelAmplitude.grid(row=0,column=2)

        self.textSpeed = ttk.Entry(self,width=12)
        self.textSpeed.grid(row=1,column=1)
        self.textAmplitude = ttk.Entry(self,width=12)
        self.textAmplitude.grid(row=1,column=2)

        self.textAmplitude.insert(0, "1.0")
        self.textSpeed.insert(0, "1.0")
        self.v = 1.0
        self.A = 1.0


        self.buttonPlot = ttk.Button(self,text="Plot",command=self.Plot,width=12)
        self.buttonPlot.grid(row=2,column=1)

        self.buttonClear = ttk.Button(self,text="Clear",command=self.Clear,width=12)
        self.buttonClear.grid(row=2,column=2)


        self.buttonClear.bind(lambda e:self.Clear)



        tk.Label(self,text="SHM Simulation").grid(column=0, row=3)

        self.fig = plt.Figure()

        self.x = 20*np.arange(0, 2*np.pi, 0.01)        # x-array


        self.ax = self.fig.add_subplot(111)
        self.line, = self.ax.plot(self.x, np.sin(self.x))


        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(column=0,row=4)


        self.ani = FuncAnimation(self.fig, self.animate, np.arange(1, 200), interval=25, blit=False)




class FrProjectionAnimator(ttk.Frame):

    def __init__(self, container):
        super().__init__(container)
        self.mainApp = container
        self.frames = None
        self.animation = None
        self.frame_time = 0

        ttk.Style().configure('PA.TFrame', background='#bdfdcb')
        self['style'] = 'PA.TFrame'

        self.__create_widgets()

    def __create_widgets(self):
        # setup the grid layout manager
        self.columnconfigure(0, weight=1)

        # ttk.Button(self, text='Start', command=self.start_animation).grid(column=0, row=0)

        # create a figure
        fig, ax = plt.subplots(figsize=(3,3), dpi=100)
        self.figure = fig
        self.axes = ax

        self.v = 1.0
        self.A = 1.0
        self.x = 20 * np.arange(0, 2 * np.pi, 0.01)  # x-array
        self.line, = self.axes.plot(self.x, np.sin(self.x))

        self.canvas = FigureCanvasTkAgg(fig, master=self)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(column=0, row=1)

        # canvas = tk.Canvas(self, width=800, height=800)
        # canvas.pack()  # this makes it visible
        #
        # # Loads and create image (put the image in the folder)
        # img = tk.PhotoImage(file="mega.png")
        # image = canvas.create_image(10, 10, anchor=tk.NW, image=img)

        for widget in self.winfo_children():
            widget.grid(padx=0, pady=3)

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
        self.line.set_ydata(self.A * np.sin(self.x + self.v))  # update the data
        self.anim = FuncAnimation(self.figure, self.update_anim, frames=np.arange(0, 20), interval=50)

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
