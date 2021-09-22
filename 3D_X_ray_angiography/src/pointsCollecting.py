import cv2
import csv


class PointsCollecting:

    def __init__(self):
        self.fieldnames = ['x_position', 'y_position']
        self.fieldnames_corresponding = [self.fieldnames[0] + '_1', self.fieldnames[1] + '_1',
                                         self.fieldnames[0] + '_2', self.fieldnames[1] + '_2']

    # NO LONGER IN USE
    def save_points_from_image(self, input_image_path, output_points_path):
        # Mouse callback function
        click_list = []
        img = cv2.imread(input_image_path)

        def callback(event, x, y, flags, param):
            if event == 1:
                click_list.append((x, y))
                cv2.circle(img, (x, y), 5, (255, 0, 0), -1)

        cv2.namedWindow('img')
        cv2.setMouseCallback('img', callback)

        # Mainloop - show the image and collect the data
        while True:
            cv2.imshow('img', img)
            # Wait, and allow the user to quit with the 'esc', 'enter' or 'space' key
            k = cv2.waitKey(1)
            # If user presses 'esc' break
            if k == 27 or k == 13 or k == 32:
                break
        cv2.destroyAllWindows()
        self.__save_points(click_list, output_points_path)

    # NO LONGER IN USE
    def __save_points(self, click_list, output_csv_path):
        # Write data to a spreadsheet
        with open(output_csv_path, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
            # writer = csv.writer(csvfile)
            writer.writeheader()
            for position in click_list:
                x, y = position[0], position[1]
                # writer.writerow((x, y))
                writer.writerow({self.fieldnames[0]: x, self.fieldnames[1]: y})

    def save_corresponding_points_from_images(self, image_1_path, image_2_path, output_points_path):
        # Mouse callback function
        click_lists = [[], []]
        img = [cv2.imread(image_1_path), cv2.imread(image_2_path)]

        def callback_1(event, x, y, flags, param):
            if event == 1:
                click_lists[0].append((x, y))
                cv2.circle(img[0], (x, y), 5, (255, 0, 0), -1)

        def callback_2(event, x, y, flags, param):
            if event == 1:
                click_lists[1].append((x, y))
                cv2.circle(img[1], (x, y), 5, (0, 255, 0), -1)

        cv2.namedWindow('img_1')
        cv2.setMouseCallback('img_1', callback_1)
        cv2.namedWindow('img_2')
        cv2.setMouseCallback('img_2', callback_2)

        # Mainloop - show the image and collect the data
        while True:
            cv2.imshow('img_1', img[0])
            cv2.imshow('img_2', img[1])
            # Wait, and allow the user to quit with the 'esc', 'enter' or 'space' key
            k = cv2.waitKey(1)
            # If user presses 'esc' break
            if k == 27 or k == 13 or k == 32:
                break
        cv2.destroyAllWindows()
        self.__save_corresponding_points_to_csv(click_lists, output_points_path)

    def __save_corresponding_points_to_csv(self, click_lists, output_csv_path):
        # Write data to a spreadsheet
        with open(output_csv_path, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames_corresponding)
            writer.writeheader()
            for click_num in range(len(click_lists[0])):
                x_1, y_1 = click_lists[0][click_num][0], click_lists[0][click_num][1]
                x_2, y_2 = click_lists[1][click_num][0], click_lists[1][click_num][1]
                writer.writerow({self.fieldnames_corresponding[0]: x_1, self.fieldnames_corresponding[1]: y_1,
                                 self.fieldnames_corresponding[2]: x_2, self.fieldnames_corresponding[3]: y_2})

    def load_corresponding_points(self, points_csv_path):
        points = [[], []]
        with open(points_csv_path) as csv_file:
            # reader = csv.reader(csv_file)
            reader = csv.DictReader(csv_file, self.fieldnames_corresponding)
            for row in reader:
                try:
                    points[0].append([
                        int(row.get(self.fieldnames_corresponding[0])), int(row.get(self.fieldnames_corresponding[1]))])
                    points[1].append([
                        int(row.get(self.fieldnames_corresponding[2])), int(row.get(self.fieldnames_corresponding[3]))])
                except ValueError:
                    # skip first row which is not int value, but a field names
                    continue
        return points

    # NO LONGER IN USE
    def load_points(self, points_csv_path):
        points = []
        with open(points_csv_path) as csv_file:
            # reader = csv.reader(csv_file)
            reader = csv.DictReader(csv_file, self.fieldnames)
            for row in reader:
                try:
                    points.append([int(row.get(self.fieldnames[0])), int(row.get(self.fieldnames[1]))])
                except ValueError:
                    # skip first row which is not int value, but a field names
                    continue
        return points

    @staticmethod
    def get_image_size(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        return [img.shape[0], img.shape[1]]