import cv2
import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt


class FullAngiographyResults:
    def __init__(self, inputdir):
        """
        :param inputdir: a directory with source dicom files.
        """
        self.__inputdir       = inputdir
        self.filenames      = None
        self.projections    = np.array([])

    def load_dicoms(self, num_of_files=0):
        self.filenames = [f for f in os.listdir(self.inputdir)]

        for f in self.filenames[:num_of_files]:  # remove "[:10]" to convert all images
            ds  = pydicom.dcmread(self.inputdir + f)
            self.projections = np.append(self.projections, ds.pixel_array)

    def load_dicom(self, filename):
        self.filenames = [filename]
        ds = pydicom.read_file

    def save_projection_to_png(self, projection, filename, outdir='./projection'):
        try:
            os.mkdir(outdir)
        except FileExistsError:
            print('Results will be saved to already existing directory.')
        except FileNotFoundError:
            print('Invalid directory path. Files not saved')
            return None

        for i, img in enumerate(projection):
            cv2.imwrite(os.path.join(outdir, filename) + str(i) + '.png', img, format='png', cmap=plt.cm.gray)

    def print_dicom_info(self, file_index):
        dic_file = pydicom.read_file(self.inputdir + self.filenames[file_index])

        print(dic_file['Modality'])
        print(dic_file['StudyDate'])
        print(dic_file['SOPClassUID'])
        print(dic_file['Manufacturer'])
        print(dic_file['ManufacturerModelName'])
        print(dic_file['DeviceSerialNumber'])
        print(dic_file['SoftwareVersions'])
        print(f"Transfer Syntax..........: {dic_file.file_meta.TransferSyntaxUID}")
        print(dic_file['SamplesPerPixel'])  # 1 -> grayscale
        print(dic_file['PhotometricInterpretation'])
        print(dic_file['Rows'])
        print(dic_file['Columns'])
        if dic_file[0x0028, 0x0002].value != 1:
            print(f"Planar Configuration (order of pix val e.g. rgb)...............: {dic_file[0x0028, 0x0006].value}")
        print(dic_file['FrameTime'].__str__() + " ms")  # Nominal time (in msec) per individual frame.
        print(dic_file[
                  'NumberOfFrames'])  # Number of frames in a Multi-frame Image. See C.7.6.6.1.1 for further explanation.
        print(dic_file[
                  'RecommendedDisplayFrameRate'])  # Recommended rate at which the frames of a Multi-frame image should be displayed in frames/second.
        print(dic_file['CineRate'])
        # print(ds['ActualFrameDuration'])
        # print(ds['ContentTime']) #The time the image pixel data creation started. Required if image is part of a series in which the images are temporally related.

