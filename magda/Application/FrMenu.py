import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from functions import create_multiple_img_fig, print_dicom_info


class FrMenuPanel(ttk.Frame):
    def __init__(self, container):
        super().__init__(container)
        self.__mainApp = container
        self.__filenames = {}
        self.__active = None

        ttk.Style().configure('MP.TFrame', background='#eac871')
        self['style'] = 'MP.TFrame'

        self.__create_widgets()

    """Create all elements and set layout to grid."""
    def __create_widgets(self):
        ttk.Style().configure('Menu.TLabel', background='#eac871')
        ttk.Style().configure('Menu.TRadiobutton', background='#eac871')

        source_frame_A = ttk.Frame(self, name='sourceFrameA', style='MP.TFrame', relief='ridge', padding=5)
        source_frame_A.columnconfigure(0, weight=1)
        ttk.Label(source_frame_A, text="Opened dicom file A:", style='Menu.TLabel')\
            .grid(column=0, row=0, sticky='w')
        ttk.Button(source_frame_A, text="Browse", command=lambda: self.__browse_files('A'))\
            .grid(column=0, row=0, sticky='e')
        ttk.Label(source_frame_A, name='filenameLabelA', text='-', style='Menu.TLabel') \
            .grid(column=0, row=1, sticky='w', columnspan=2)

        source_frame_B = ttk.Frame(self, name='sourceFrameB', style='MP.TFrame', relief='ridge', padding=5)
        source_frame_B.columnconfigure(0, weight=1)
        ttk.Label(source_frame_B, text="Opened dicom file B:", style='Menu.TLabel') \
            .grid(column=0, row=0, sticky='w')
        ttk.Button(source_frame_B, text="Browse", command=lambda: self.__browse_files('B')) \
            .grid(column=0, row=0, sticky='e')
        ttk.Label(source_frame_B, name='filenameLabelB', text='-', style='Menu.TLabel') \
            .grid(column=0, row=1, sticky='w', columnspan=2)

        choose_frame = ttk.Frame(self, name='chooseFrame', style='MP.TFrame', padding=5)
        ttk.Label(choose_frame, text="Set active projection:", style='Menu.TLabel') \
            .pack(side='left')
        self.__active = tk.StringVar()
        ttk.Radiobutton(choose_frame, style='Menu.TRadiobutton', text="A", variable=self.__active, value='A', command=self.__set_active)\
            .pack(side='left')
        ttk.Radiobutton(choose_frame, style='Menu.TRadiobutton', text="B", variable=self.__active, value='B', command=self.__set_active)\
            .pack(side='left')

        button_frame = ttk.Frame(self, name='buttonFrame', style='MP.TFrame', padding=5)
        ttk.Button(button_frame, text='Crop images', command=lambda: self.__mainApp.show_component(name='crop_images'))\
            .grid(column=0, row=0)
        ttk.Button(button_frame, text='Projection animator', command=lambda: self.__mainApp.show_component(name='projection_animator'))\
            .grid(column=1, row=0)

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        source_frame_A.grid(column=0, row=0, sticky='nsew')
        source_frame_B.grid(column=1, row=0, sticky='nsew')
        choose_frame.grid(column=0, row=1, sticky='nsew')
        button_frame.grid(column=0, row=2, sticky='nsew')

    """Let user choose a source dicom file and load it"""
    def __browse_files(self, which):
        filename = fd.askopenfilename(
            title='Open a file',
            initialdir='../../src_images/',
            filetypes=(('DICOM files', '*.dcm'), ('All files', '*.*'))
        )

        if filename is not None and filename != '':
            """Load only if user has chosen any file"""
            self.__filenames[which] = filename
            self.children['sourceFrame'+which].children['filenameLabel'+which].configure(text=filename)
            self.children['sourceFrame' + which].children['active'+which+'Btn'].state(['!disabled'])
            self.__mainApp.load_chosen_dicom(which, filename)

    def __set_active(self):
        which = self.__active.get()
        self.__mainApp.change_active(which)

