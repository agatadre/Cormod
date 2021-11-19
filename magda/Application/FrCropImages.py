import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.figure
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

        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.__fig = matplotlib.figure.Figure()
        self.__ax = self.__fig.add_subplot(111)
        self.__ax.set_aspect('equal', adjustable='box')

        self.__create_widgets()

    def __create_widgets(self):
        self.__btn_panel = FrButtons(self)

        self.rect = Rectangle((0, 0), 1, 1)
        self.__ax.add_patch(self.rect)

        self.__canvas = FigureCanvasTkAgg(self.__fig, self)
        self.__canvas.draw()
        self.__canvas.mpl_connect('button_press_event', self.on_press)
        self.__canvas.mpl_connect('button_release_event', self.on_release)
        # self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        # self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

        # setup the grid layout manager
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=7)
        self.rowconfigure(0, weight=1)

        self.__btn_panel.grid(column=0, row=0, sticky='nsew')
        self.__canvas.get_tk_widget().grid(column=1, row=0, sticky='nsew')

    def on_press(self, event):
        print('press')
        self.x0 = event.xdata
        self.y0 = event.ydata

    def on_release(self, event):
        print('release')
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        self.__ax.figure.canvas.draw()


class FrButtons(ttk.Frame):
    def __init__(self, container):
        super().__init__(container)
        self.container = container
        ttk.Style().configure('Btn.TFrame', background='#ddbea9')
        self['style'] = 'Btn.TFrame'
        ttk.Style().configure('Btn.TLabel', background='#ddbea9')


