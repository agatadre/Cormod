import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.figure
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib
from pathlib import Path
from Application.EDFinder import EDFinder


class FrProjectionAnimator(ttk.Frame):

    def __init__(self, container):
        super().__init__(container)
        self.mainApp = container
        ttk.Style().configure('PA.TFrame', background='#ffffff')
        self['style'] = 'PA.TFrame'

        self.__frames = None
        self.__frame_time = None
        self.__animation = None
        self.__actual_frame_num = None
        self.__frame_disp = None
        self.__anim_running = False

        self.__fig = matplotlib.figure.Figure()
        self.__ax = self.__fig.add_subplot(111)
        self.__ax.set_aspect('equal', adjustable='box')
        self.__time_text = self.__ax.text(0.5, 0.05, "", bbox={'facecolor': 'red',
                                                               'alpha': 0.5, 'pad': 2},
                                          transform=self.__ax.transAxes, ha="center")

        self.__create_widgets()

    def __create_widgets(self):
        self.__btn_panel = FrButtons(self)
        self.__canvas = FigureCanvasTkAgg(self.__fig, self)
        self.__canvas.draw()

        # setup the grid layout manager
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=7)
        self.rowconfigure(0, weight=1)

        self.__btn_panel.grid(column=0, row=0, sticky='nsew')
        self.__canvas.get_tk_widget().grid(column=1, row=0, sticky='nsew')

    def load_frames(self, projection, frame_time):
        self.__frames = projection
        self.__frame_time = frame_time
        self.__actual_frame_num = 0
        self.__frame_disp = self.__ax.imshow(self.__frames[0], cmap='gray', animated=True)
        self.__time_text.set_text(self.__actual_frame_num)
        self.__canvas.draw()

        time_text = 'Frame time: ' + str(self.__frame_time) + ' ms'
        self.__btn_panel.children['frameTimeLabel'].configure(text=time_text)

        self.__btn_panel.btn_start.state(['!disabled'])
        self.__btn_panel.btn_plot_ECG.state(['!disabled'])

    def frames_len(self):
        return self.__frames.shape[0]

    def __update_animation(self, i):
        self.__actual_frame_num = i
        self.__frame_disp.set_array(self.__frames[i])
        self.__time_text.set_text( i )
        return self.__frame_disp, self.__time_text,

    def start_animation(self):
        interval = self.__btn_panel.interval.get() if self.__btn_panel.interval.get() > 0 else self.__frame_time
        self.__actual_frame_num = 0
        self.__animation = FuncAnimation(fig=self.__fig,
                                         func=self.__update_animation,
                                         frames=self.frames_len(),
                                         blit=True,
                                         interval=interval,
                                         repeat=True,
                                         repeat_delay=1000)
        self.__anim_running = True

        self.__btn_panel.btn_pause.state(['!disabled'])
        self.__btn_panel.btn_left.state(['disabled'])
        self.__btn_panel.btn_right.state(['disabled'])
        self.__btn_panel.btn_start.state(['disabled'])

    def pause_animation(self):
        if self.__anim_running:
            self.__animation.pause()
        else:
            self.__animation.resume()
        self.__anim_running = not self.__anim_running

        self.__btn_panel.btn_pause
        self.__btn_panel.btn_left.state(['!disabled'])
        self.__btn_panel.btn_right.state(['!disabled'])
        self.__btn_panel.btn_start.state(['!disabled'])

    def move_to_next_frame(self):
        if str( self.__btn_panel.btn_left['state'] ) == 'normal' and self.__actual_frame_num < self.frames_len() - 1:
            next_f = self.__actual_frame_num + 1

            self.__frame_disp.set_array(self.__frames[next_f])
            self.__time_text.set_text(next_f)
            self.__actual_frame_num = next_f
            self.__canvas.draw()

    def move_to_prev_frame(self):
        if str( self.__btn_panel.btn_left['state'] ) == 'normal' and self.__actual_frame_num > 0:
            prev_f = self.__actual_frame_num - 1

            self.__frame_disp.set_array(self.__frames[prev_f])
            self.__time_text.set_text(prev_f)
            self.__actual_frame_num = prev_f
            self.__canvas.draw()

    def call_plot_ECG_signal(self):
        p = Path(self.mainApp.filename)
        parts = p.parts
        pp = Path('../../results_images_edfinder/').joinpath(parts[-2] + '_' + parts[-1][:-4])  # without extension (.dcm)

        EDFinder.filename = str(pp)
        EDFinder.find_EDPhase(self.__frames)


class FrButtons(ttk.Frame):
    def __init__(self, container):
        super().__init__(container)
        self.container = container
        ttk.Style().configure('Btn.TFrame', background='#ddbea9')
        self['style'] = 'Btn.TFrame'
        ttk.Style().configure('Btn.TLabel', background='#ddbea9')

        self.l_frame_time = ttk.Label(self, name='frameTimeLabel', text='Frame time: --- ms', style='Btn.TLabel')
        interval_label = ttk.Label(self, text='Set interval (ms):', style='Btn.TLabel')
        self.interval = tk.IntVar()
        interval_entry = ttk.Entry(self, textvariable=self.interval, width=5)
        self.btn_start = ttk.Button(self, text='Start', command=container.start_animation)
        self.btn_start.state(['disabled'])
        self.btn_left = ttk.Button(self, text='<', command=container.move_to_prev_frame)
        self.btn_left.state(['disabled'])
        self.btn_right = ttk.Button(self, text='>', command=container.move_to_next_frame)
        self.btn_right.state(['disabled'])
        self.btn_pause = ttk.Button(self, text='Pause', command=container.pause_animation)
        self.btn_pause.state(['disabled'])
        self.btn_plot_ECG = ttk.Button(self, text='Plot ECG', command=container.call_plot_ECG_signal)
        self.btn_plot_ECG.state(['disabled'])

        self.l_frame_time.pack(
            ipadx=10,
            ipady=2,
            pady=5
        )
        interval_label.pack()
        interval_entry.pack()
        self.btn_start.pack(
            ipadx=10,
            ipady=2,
            pady=5
        )
        self.btn_left.pack()
        self.btn_right.pack()
        self.btn_pause.pack(
            ipadx=10,
            ipady=2,
            pady=5
        )
        self.btn_plot_ECG.pack(
            ipadx=10,
            ipady=2,
            pady=5
        )

        # self.columnconfigure(0, weight=1)
        # self.columnconfigure(1, weight=7)
        # self.rowconfigure(0, weight=1)
        # self.rowconfigure(1, weight=1)
        # self.btn_start.grid(column=0, row=0, sticky='n')
        # self.btn_pause.grid(column=0, row=1, sticky='n')
