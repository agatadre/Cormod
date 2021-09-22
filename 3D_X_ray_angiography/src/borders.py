import cv2
import numpy as np
from scipy.interpolate import interp1d
from deviceConfiguration import DeviceConfiguration


def get_scaled_average_diameter_size(one_segment_points: np.ndarray, mask_path, conf_dev: DeviceConfiguration,
                                     mask_num):
    avg_diameter_length = get_average_diameter_size(one_segment_points, mask_path)
    # should be scaled by ratio dist(F, 3D_point)/dist(F, 2D_point)
    # but we do not use this scale in our reconstructed vessel
    # TODO scale reconstructed figure by dist(F, 3D_point)/dist(F, 2D_point) ratio and rescale diameter then?
    return avg_diameter_length * conf_dev.pixel_spacing[mask_num]


def get_average_diameter_size(one_segment_points: np.ndarray, mask_path):
    b_points = get_2d_border_points(one_segment_points.astype(int), mask_path)
    diameters_lengths = np.sqrt(np.sum(np.square((b_points[:, 1, :] - b_points[:, 0, :])), axis=1))
    # TODO remove outsiders (too big values, diameters of bifurcation points) before calculating average
    avg_diameter_length = np.average(diameters_lengths)
    return avg_diameter_length


def get_2d_border_points(one_segment_points: np.ndarray, mask_path, visualize=False):
    """
    Calculates border points corresponding with give segment centerline points
    :param one_segment_points: array with 2d points describing centerline of a vessel segment
                                points should look like ndarray( [ [x1, y1], [x2, y2], ... ] )
    :param mask_path: path to an image file (with mask layer) that contains given centerline segment
    :param visualize: whether you want to see visual result or not
    :return: 3 dimensional ndarray (0: centerline point index, 1: border point number(0/1), 2: xy(0/1) coordinates)
    """
    mask = cv2.imread(mask_path)

    points_num = one_segment_points.shape[0]
    border_points = np.empty([points_num, 2, 2])
    for point_num in range(points_num - 1):
        p1 = one_segment_points[point_num, :]
        p2 = one_segment_points[point_num + 1, :]
        # create vector p1->p2
        vec = p2 - p1

        # create perpendicular vector
        per_vec = np.array([vec[1], -vec[0]])

        # translate first point by a perpendicular vector
        pp1 = p1 + per_vec

        # calc linear equation for p1->pp2 line
        polynomial, axis = calc_linear_func(p1[0], pp1[0], p1[1], pp1[1])

        test_y0 = polynomial(0)
        test_y1 = polynomial(1)

        # img = cv2.imread(mask_path)
        # image_size = img.shape[0]
        # x_axis = list(range(image_size))
        # y_axis = polynomial(x_axis)

        border_points[point_num, :, :] = get_line_end_points(mask, p1, polynomial, axis)

        # if it is penultimate point, create two vectors instead of one
        if point_num == one_segment_points.shape[0] - 2:
            pp2 = p2 + per_vec
            polynomial, axis = calc_linear_func(p2[0], pp2[0], p2[1], pp2[1])
            border_points[point_num + 1, :, :] = get_line_end_points(mask, p1, polynomial, axis)
    if visualize:
        cv2.imshow('boders_test', mask)
    return border_points


def calc_linear_func(x1, x2, y1, y2):
    """
    Calculates linear equation that goes through given two points
    :param x1: x coordinate of first point
    :param x2: y coordinate of first point
    :param y1: x coordinate of second point
    :param y2: y coordinate of second point
    :return: linear function, whether it is reversed or not
            axis == 0 -> y = ax + b
            axis == 1 -> x = ay + b
    """
    xs = [x1, x2]
    ys = [y1, y2]

    # we cannot calculate function with the same x point given
    if x1 != x2:
        coefficients = np.polyfit(xs, ys, 1)
    # so we just set high coefficients value to calculate reverse function
    else:
        coefficients = [99, 99]

    # if y axis increases slower that make reverse function
    if abs(coefficients[0]) > 1:
        axis = 1
        coefficients = np.polyfit(ys, xs, 1)
        if y1 == y2:
            print("Cannot calculate function with the same x values, this shouldn't happen")
    else:
        axis = 0

    return np.poly1d(coefficients), axis


def get_line_end_points(mask, start_point, func, axis):
    """
    Calculates line start and line end according to color based detection
    Used for getting 2d vessel diameter
    :param mask: image file (with mask layer) that contains proper centerline segment
    :param start_point: points where to start searching (point inside vessel / center of diameter)
    :param func: linear function that is perpendicular to the vessel
    :param axis: whether the function is reversed or not
    :return: array with line start x,y coordinates and end x,y coordinates
    """
    # TODO Rewrite this spaghetti code
    vessel_color = mask[start_point[1], start_point[0]]
    if np.all([0, 0, 0] == vessel_color):
        print(start_point)
        print('WARNING: centerline point (' + str(start_point) + ') doesnt belong to a vessel!!!')
        # TODO change it in the future, this situation should not happen
        return np.array([[0, 0], [0, 0]])

    if axis == 0:
        right_max_x = start_point[0]
        left_max_x = start_point[0]

        # get rightmost pixel
        while True:
            y = int(np.round(func(right_max_x + 1)))
            if np.all(mask[y, right_max_x + 1] == vessel_color):
                right_max_x += 1
            else:
                break
        # get leftmost pixel
        while True:
            y = int(np.round(func(left_max_x - 1)))
            if np.all(mask[y, left_max_x - 1] == vessel_color):
                left_max_x -= 1
            else:
                break

        right_max_y = int(np.round(func(right_max_x)))
        left_max_y = int(np.round(func(left_max_x)))

    else:
        right_max_y = start_point[1]
        left_max_y = start_point[1]

        # get rightmost pixel
        while True:
            x = int(np.round(func(right_max_y + 1)))
            if np.all(mask[right_max_y + 1, x] == vessel_color):
                right_max_y += 1
            else:
                break
        # get leftmost pixel
        while True:
            x = int(np.round(func(left_max_y - 1)))
            if np.all(mask[left_max_y - 1, x] == vessel_color):
                left_max_y -= 1
            else:
                break

        right_max_x = int(np.round(func(right_max_y)))
        left_max_x = int(np.round(func(left_max_y)))

    # for testing purpose
    mask[right_max_y:right_max_y + 5, right_max_x:right_max_x + 5] = [0, 0, 255]
    mask[left_max_y - 5:left_max_y, left_max_x - 5:left_max_x] = [0, 255, 0]
    # cv2.imshow('boders_test', img)

    return np.array([[left_max_x, left_max_y], [right_max_x, right_max_y]])


def show_centerline_points(mask_path, centerline):
    mask = cv2.imread(mask_path)
    for point in centerline:
        #mask[point[1] - 5:point[1], point[0] - 5:point[0]] = [0, 255, 0]
        #mask[point[1], point[0]] = [0, 0, 255]
        mask[point[1] - 2:point[1] + 2, point[0] - 2:point[0] + 2] = [0, 0, 255]
    cv2.imshow('centerline_points', mask)
    cv2.waitKey(0)
