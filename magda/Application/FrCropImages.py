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

        self.left = 0.0
        self.down = 0.0
        self.right = 0.0
        self.top = 0.0
        self.__fig = matplotlib.figure.Figure()
        self.__ax = self.__fig.add_subplot(111)
        self.__ax.set_aspect('equal', adjustable='box')

        self.__create_widgets()

    def __create_widgets(self):
        self.__btn_panel = FrButtons(self)

        self.__canvas = FigureCanvasTkAgg(self.__fig, self)
        self.__canvas.draw()

        # drawtype is 'box' or 'line' or 'none'
        self.__rs = RectangleSelector(self.__ax, self.line_select_callback,
                                      drawtype='box', useblit=False,
                                      button=[1],  # use only left button
                                      minspanx=5, minspany=5,
                                      spancoords='pixels',
                                      interactive=True)

        # setup the grid layout manager
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=7)
        self.rowconfigure(0, weight=1)

        self.__btn_panel.grid(column=0, row=0, sticky='nsew')
        self.__canvas.get_tk_widget().grid(column=1, row=0, sticky='nsew')

    def line_select_callback(self, eclick, erelease):
        self.left, self.down = eclick.xdata, eclick.ydata
        self.right, self.top = erelease.xdata, erelease.ydata
        print("l,d (%3.2f, %3.2f) --> r,t (%3.2f, %3.2f)" % (self.left, self.down, self.right, self.top))
        # print(" The button you used were: %s %s" % (eclick.button, erelease.button))

class FrButtons(ttk.Frame):
    def __init__(self, container):
        super().__init__(container)
        self.container = container
        ttk.Style().configure('Btn.TFrame', background='#ddbea9')
        self['style'] = 'Btn.TFrame'
        ttk.Style().configure('Btn.TLabel', background='#ddbea9')


