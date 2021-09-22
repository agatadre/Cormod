import os
import csv

class Result:
    NO_FILE = "no file"
    NO_FRAMES = "no frames"

class PlainTextTools:
    def __init__(self, text_file_path):
        self.__text_file_path = text_file_path
        self.__text_file = None
        self.__text_plan = None
        self.__open_text_file()


    def __del__(self):
        self.__close_json_file()


    def __open_text_file(self):
        # Making sure that the mri file exists
        if not os.path.exists(self.__text_file_path):
            raise Exception('File "%s" does not exists' % self.__text_file_path)

        # open mri and get data
        self.__text_file = open(self.__text_file_path, 'rb')
        self.__text_plan = self.__text_file.readlines()

    def __get_exam_name(self):
        #it works only if dicom is placed as: exam/view/dicomFile
        self.__exam_name = self.__text_file_path.split('/')[-3]

    def __close_json_file(self):
        if self.__text_file is not None:
            self.__text_file.close()
            self.__text_file = None
            self.__text_plan = None

    def __find_line(self, tag1, tag2):
        for line in self.__text_plan:
            words = line.split()
            if str(words[0]).find(tag1) != -1 and str(words[1]).find(tag2) != -1:
                return words

    def __get_value_from_line(self, line):
            return str(line[-1]).strip("b'\"")

    def get_alpha(self):
        # (0018, 1510) Positioner Primary Angle
        line = self.__find_line("0018", "1510")
        return self.__get_value_from_line(line)

    def get_beta(self):
        # (0018, 1511) Positioner Secondary Angle
        line = self.__find_line("0018", "1511")
        return self.__get_value_from_line(line)

    def get_pixel_spacing(self):
        # (0018, 1164) Image Pixel Spacing
        line = self.__find_line("0018", "1164")
        return self.__get_value_from_line(line)

    def get_number_fps(self):
        try:
            line = self.__find_line("0018", "0040")
            return self.__get_value_from_line(line)
        except:
            return Result.NO_FRAMES

    def save_metadata_to_csv(self, csv_path):
        with open(csv_path, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            row = [self.__get_exam_name(), self.get_alpha(), self.get_beta(), self.get_number_fps()]
            csvwriter.writerow(row)
