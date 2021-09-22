import os
import json
import csv

class Result:
    NO_FILE = "no file"
    NO_FRAMES = "no frames"

class JsonTools:
    def __init__(self, json_file_path):
        self.__json_file_path = json_file_path
        self.__json_file = None
        self.__json_plan = None
        self.__open_json_file()


    def __del__(self):
        self.__close_json_file()


    def __open_json_file(self):
        # Making sure that the mri file exists
        if not os.path.exists(self.__json_file_path):
            raise Exception('File "%s" does not exists' % self.__json_file_path)

        # open mri and get data
        self.__json_file = open(self.__json_file_path, 'rb')
        self.__json_plan = json.load(self.__json_file)

    def __get_exam_name(self):
        #it works only if dicom is placed as: exam/view/dicomFile
        self.__exam_name = self.__json_file_path.split('/')[-3]

    def __close_json_file(self):
        if self.__json_file is not None:
            self.__json_file.close()
            self.__json_file = None
            self.__json_plan = None


    def get_sod(self):
        # (0018, 1111) Distance Source to Patient
        return self.__json_plan["(0018,1111)"]["value"]


    def get_sid(self):
        # (0018, 1110) Distance Source to Detector
        return self.__json_plan["(0018,1110)"]["value"]


    def get_alpha(self):
        # (0018, 1510) Positioner Primary Angle
        return self.__json_plan["(0018,0510)"]["value"]


    def get_beta(self):
        # (0018, 1511) Positioner Secondary Angle
        return self.__json_plan["(0018,1511)"]["value"]

    def get_pixel_spacing(self):
        # (0018, 1164) Image Pixel Spacing
        return self.__json_plan["(0018,1164)"]["value"]

    def get_number_fps(self):
        # (0018, 0040) Cine Rate
        try:
            return int(self.__json_plan["(0018,0040)"]["value"])
        except:
            return Result.NO_FRAMES

    def save_metadata_to_csv(self, csv_path):
        with open(csv_path, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            row = [self.__get_exam_name(), self.get_alpha(), self.get_beta(), self.get_number_fps()]
            csvwriter.writerow(row)
