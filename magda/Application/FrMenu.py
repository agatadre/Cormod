import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from functions import display_multiple_img, print_dicom_info


class FrMenuPanel(ttk.Frame):
    def __init__(self, container):
        super().__init__(container)
        self.__mainApp = container
        self.__filename = ''

        ttk.Style().configure('MP.TFrame', background='#fcfabc')
        self['style'] = 'MP.TFrame'

        self.__create_widgets()

    """Create all elements and set layout to grid."""
    def __create_widgets(self):
        ttk.Style().configure('MP.TLabel', background='#fcfabc')

        top_frame = ttk.Frame(self, name='topFrame', style='MP.TFrame', relief='ridge', padding=5)
        top_frame.columnconfigure(0, weight=1)
        ttk.Label(top_frame, text="Opened dicom file:", style='MP.TLabel')\
            .grid(column=0, row=0, sticky='w')
        ttk.Label(top_frame, name='filenameLabel', text='-', style='MP.TLabel')\
            .grid(column=0, row=1, sticky='w')
        ttk.Button(top_frame, text="Browse", command=self.__browse_files)\
            .grid(column=0, row=0, sticky='e')

        button_frame = ttk.Frame(self, name='buttonFrame', style='MP.TFrame', padding=5)
        ttk.Button(button_frame, text='Projection animator', command=self.__mainApp.show_component(name='projection_animator'))\
            .grid(column=0, row=0)
        # ttk.Button(button_frame, text='Second func')\
        #     .grid(column=1, row=0)
        # ttk.Button(self, text='Third func').grid(column=0, row=2)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0)
        self.rowconfigure(1)
        top_frame.grid(column=0, row=0, sticky='nsew')
        button_frame.grid(column=0, row=1, sticky='nsew')

    """Let user choose a source dicom file and load it"""
    def __browse_files(self):
        filename = fd.askopenfilename(
            title='Open a file',
            initialdir='../../src_images/',
            filetypes=(('DICOM files', '*.dcm'), ('All files', '*.*'))
        )

        if filename is not None and filename != '':
            """Load only if user has chosen any file"""
            self.__filename = filename
            self.children['topFrame'].children['filenameLabel'].configure(text=self.__filename)
            self.__mainApp.load_chosen_dicom(self.__filename)
