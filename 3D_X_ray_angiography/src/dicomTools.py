import os
from os.path import isfile, join
import uuid
import png
import cv2
import pydicom
import numpy as np
from itertools import combinations

class Result:
    NO_FILE = "no file"
    NO_FRAMES = "no frames"

class DicomFile:
    def __init__(self, dicom_file_path):
        self.__dicom_file_path = dicom_file_path
        self.__dicom_file = None
        self.__dicom_plan = None
        self.__open_dicom_file()

    def __del__(self):
        self.__close_dicom_file()

    def __open_dicom_file(self):
        # Making sure that the mri file exists
        if not os.path.exists(self.__dicom_file_path):
            raise Exception('File "%s" does not exists' % self.__dicom_file_path)

        # open mri and get data
        self.__dicom_file = open(self.__dicom_file_path, 'rb')
        self.__dicom_plan = pydicom.read_file(self.__dicom_file)

    def __get_exam_name(self):
        #it works only if dicom is placed as: exam/view/dicomFile
        self.__exam_name = self.__dicom_file_path.split('/')[-3]

    def __close_dicom_file(self):
        if self.__dicom_file is not None:
            self.__dicom_file.close()
            self.__dicom_file = None
            self.__dicom_plan = None

    def __grey_rescale(self, image_2d, max_val):
        image_2d_scaled = []
        for row in image_2d:
            row_scaled = []
            for col in row:
                col_scaled = int((float(col) / float(max_val)) * 255.0)
                row_scaled.append(col_scaled)
            image_2d_scaled.append(row_scaled)
        return image_2d_scaled

    def __obtain_pixels(self, mri_img):
        image_2d = []
        max_val = 0
        for row in mri_img:
            pixels = []
            for col in row:
                pixels.append(col)
                max_val = max(col, max_val)
            image_2d.append(pixels)
        return image_2d, max_val

    def __image_to_png(self, mri_img, png_file):
        """ Function to convert from a DICOM image to png
            @param mri_img: A single mri image from dicom data
            @param png_file: An opened file like object to write the png data
        """
        # get image shape
        shape = mri_img.shape

        image_2d, max_val = self.__obtain_pixels(mri_img)
        image_2d_scaled = self.__grey_rescale(image_2d, max_val)

        # Writing the PNG file
        w = png.Writer(shape[1], shape[0], greyscale=True)
        w.write(png_file, image_2d_scaled)

    def get_cv_grayscale_imgs(self):
        '''
        :return: numpy array with opencv gray scale pngs from a previously passed dicom file
        '''
        imgs_dir_name = str(uuid.uuid4())
        pngs = self.dicom_to_pngs(imgs_dir_name, return_pngs=True)
        imgs_listed = os.listdir(imgs_dir_name)
        for img in imgs_listed:
            os.remove(f"{imgs_dir_name}/{img}")
        os.removedirs(imgs_dir_name)
        return np.asarray(pngs, dtype=np.uint8)

    def get_grayscale_imgs(self):
        '''
        :return: numpy array with custom gray scale pngs from a previously passed dicom file
        '''
        images = []
        for i, img in enumerate(self.__dicom_plan.pixel_array):
            images.append([])
            images[i], max_val = self.__obtain_pixels(img)
            images[i] = self.__grey_rescale(images[i], max_val)
        return np.array(images, dtype=np.uint8)

    def load_cv_pngs(self, directory_path):
        '''
        :param directory_path: path to a folder with images to load
        :return: numpy array with gray scaled pngs
        '''
        pngs = []
        arr = os.listdir(directory_path)
        sorted_arr = sorted(arr,key=lambda x: int(os.path.splitext(x)[0]))
        for png in sorted_arr:
            pngs.append(cv2.imread(f"{directory_path}/{png}", cv2.IMREAD_GRAYSCALE))
        return np.asarray(pngs)

    def dicom_to_pngs(self, directory_path, return_pngs=None):
        """ Function to convert an MRI file to a
            PNG image files located inside given directory.
            @param directory_path: Full path to non existing directory
            @param return_pngs: after saving pngs to a directory, return ndarray with gray scaled imgs
        """

        # Making sure the directory does not exist
        if os.path.exists(directory_path):
            raise Exception('Directory "%s" already exists' % directory_path)

        # Creating directory
        try:
            os.mkdir(directory_path)
        except OSError:
            print("Creation of the directory %s failed" % directory_path)

        # iterate through all images in dicom file
        pngs = []
        for i, oneImage in enumerate(self.__dicom_plan.pixel_array):
            one_image_path = directory_path + '/' + str(i) + '.png'
            cv2.imwrite(f"{one_image_path}", oneImage)
            if return_pngs:
                pngs.append(cv2.imread(f"{one_image_path}", cv2.IMREAD_GRAYSCALE))
        if return_pngs:
            return pngs

    def getPath(self):
        print(self.__dicom_file_path)
    def get_sod(self):
        # (0018, 1111) Distance Source to Patient
        return self.__dicom_plan[0x0018, 0x1111].value

    def get_sid(self):
        # (0018, 1110) Distance Source to Detector
        return self.__dicom_plan[0x0018, 0x1110].value

    def get_alpha(self):
        # (0018, 1510) Positioner Primary Angle
        return self.__dicom_plan[0x0018, 0x1510].value

    def get_beta(self):
        # (0018, 1511) Positioner Secondary Angle
        return self.__dicom_plan[0x0018, 0x1511].value

    def get_pixel_spacing(self):
        # (0018, 1164) Image Pixel Spacing
        return self.__dicom_plan[0x0018, 0x1164].value

    def get_number_fps(self):
        #(0018, 0040) Cine Rate
        try:
            return int(self.__dicom_plan[0x0018, 0x0040].value)
        except:
            return Result.NO_FRAMES

    def _return_metadata(self):
        lines = ['Distance Source to Detector (SID) = ' + str(self.get_sid()) + '\n',
                 'Distance Source to Patient (SOD) = ' + str(self.get_sod()) + '\n',
                 'Positioner Primary Angle (alpha) = ' + str(self.get_alpha()) + '\n',
                 'Positioner Secondary Angle (beta) = ' + str(self.get_beta()) + '\n',
                 'Image Pixel Spacing (x,y) = ' + str(self.get_pixel_spacing())]
        return lines

    def save_metadata_to_file(self, text_file_path):
        text_file = open(text_file_path, "w")
        text_file.writelines(self._return_metadata())
        text_file.close()

def get_closest_projections(directory_path, epsilon=2):
    """
    Chooses projections with the lowest angle difference
    :param directory_path: path to directory which contains more than one projection
    :param epsilon: minimal difference in degrees, because we don't want almost the same projections
    :return: tuple (tuple of two dicom files with closest projections, value of angle difference)
    """
    # get all dicom files from all directories in given path
    dicom_paths = []
    projection_dirs = [join(directory_path, d) for d in os.listdir(directory_path)
                       if os.path.isdir(join(directory_path, d))]
    for dir_path in projection_dirs:
        for (_, _, filenames) in os.walk(dir_path):
            for filename in filenames:
                if filename.endswith('.dcm'):
                    dicom_paths.append(join(dir_path, filename))
            break

    # get all dicom pairs and search for a one with minimal angle difference
    min_diff = 1000
    min_pair = (None, None)
    for (dicom_path_a, dicom_path_b) in combinations(dicom_paths, 2):
        dicom_a = DicomFile(dicom_path_a)
        dicom_b = DicomFile(dicom_path_b)
        alpha_a = dicom_a.get_alpha()
        beta_a = dicom_a.get_beta()
        alpha_b = dicom_b.get_alpha()
        beta_b = dicom_b.get_beta()
        delta_alpha = abs(alpha_b - alpha_a)
        delta_beta = abs(beta_b - beta_a)
        diff = delta_beta + delta_alpha
        if min_diff > diff > epsilon:
            min_diff = diff
            min_pair = (dicom_a, dicom_b)
    return min_pair, min_diff
