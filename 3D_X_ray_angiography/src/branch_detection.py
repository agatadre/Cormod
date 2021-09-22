""" Computer vision algorithm for branch detection.
"""

import random
import cv2
import thinning
import numpy as np
from kernels import get_branch_kernels, get_support_kernels

def split_into_segments(prediction, bifurs):
    """
        Function which extract centerline from a vessel and split it into
        segments.
        @param prediction: gray OpenCV image of a vessel
        @param bifurs: bifurcation points previously marked on an
                       image of a vessel
        @return points: bifurcation points, in the future probably aligned by
                        an algorithm
        @return segments: segments of a vessel
    """
    # get centerline, and bifurcation points
    thinned, points_map, points = get_skeleton_and_inter_points(prediction, bifurs)
    contours, hierarchy = cv2.findContours(
        thinned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw segments using different grayscale
    bifurs = points
    sum = np.zeros(prediction.shape, dtype=np.uint8)

    # list of segments
    segments = []

    for contour in contours:
        segment = np.zeros(thinned.shape[:2], dtype=np.uint8)

        cv2.drawContours(segment, [contour], -1, 255, -1)
        # add segment to the list
        segments.append(segment)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        segment = cv2.dilate(segment, kernel, iterations=1)

        #sum[segment > 0] = random.sample(list(range(256)), k=1)[0]

    #segments_vis = cv2.cvtColor(prediction, cv2.COLOR_RGB2GRAY)
    # visualization of a segments, not use now
    #segments_vis = prediction
    #segments_vis = cv2.bitwise_or(sum, sum, mask=segments_vis)
    #segments_vis = sum

    prediction -= thinned
    prediction -= points_map

    return points, segments

def get_skeleton_and_inter_points(prediction, bifurs):
    """
        Function which extracts centerline from a vessel.
        @param prediction: gray OpenCV image of a vessel
        @param bifurs: bifurcation points previously marked on an
                       image of a vessel
        @return points: bifurcation points, in the future propably aligned by
                        an algorithm
        @return thinned: centerline of a vessel
        @return points_map: bifurcation points as a gray OpenCV image
        @return points: List of bifurcation points
    """
    pred = prediction.copy()
    thinned = thinning.guo_hall_thinning(pred) / 255
    kernels = get_branch_kernels()

    h, w = thinned.shape[:2]
    points_map = np.zeros((h, w))
    """
    # Algorithm probably for future development
    points = []
    # Find branch points using unique kernels with size 3x3
    for kernel in kernels:
        input_img = thinned.astype(np.float32)
        kernel = kernel.astype(np.float32)
        result = cv2.matchTemplate(image=input_img, templ=kernel,
                                   method=cv2.TM_SQDIFF, mask=kernel)

        result = result.astype(np.uint8)
        points_map[1:h - 1, 1:w - 1] += (result == 0)
        points.extend(np.argwhere(result == 0))
        points_map, points = filter_false_segments(points_map, points, thinned)

    #points = bifurs
    new_points = np.empty([len(bifurs), 2], dtype=np.int64)

    for i, bifur in enumerate(bifurs):
        x, y = list(bifur)
        for point in points:
            y_p, x_p = list(point)
            if x in range(x_p - 5, x_p + 5) and y in range(y_p - 5, y_p + 5):
                print(bifur)
                new_points[i, :] = [x_p, y_p]
                break

    points = new_points
    """
    # Leave bifurcation points unchanged
    points = bifurs
    # Add points on the map
    for point in points:
        x, y = list(point)
        points_map[y, x] = 1

    # Function which level image after reducing it in a Gimp
    for y in range(0, thinned.shape[0]):
        for x in range(0, thinned.shape[1]):
            if thinned[y, x] > 0.9:
                thinned[y, x] = 1

    # Split thinned vascular tree into segments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    points_map = cv2.dilate(points_map, kernel, iterations=1)
    thinned = np.clip(thinned - points_map, 0, 1)

    points_map = points_map.astype(np.uint8) * 255
    thinned = thinned.astype(np.uint8) * 255

    return thinned, points_map, points

def get_segments(prediction):
    """
        Experimental function, not used in a current version of a software.
        It splits vessel into segments with automatically finded bifurcation
        points. It might be used for vessel reduction.
        @param prediction: gray OpenCV image of a vessel
        @return thinned1: centerline of a vessel
        @return segments: segments of a vessel
        @return segments_vis: visualization of a vessel splitted into a segments
    """
    pred = prediction.copy()
    thinned1 = thinning.guo_hall_thinning(pred) / 255
    kernels = get_branch_kernels()

    thinned = thinned1.copy()
    h, w = thinned.shape[:2]
    points_map = np.zeros((h, w))
    sum = np.zeros(prediction.shape, dtype=np.uint8)
    points = []
    # Find branch points using unique kernels with size 3x3
    for kernel in kernels:
        input_img = thinned.astype(np.float32)
        kernel = kernel.astype(np.float32)
        result = cv2.matchTemplate(image=input_img, templ=kernel,
                                   method=cv2.TM_SQDIFF, mask=kernel)

        result = result.astype(np.uint8)
        points_map[1:h - 1, 1:w - 1] += (result == 0)
        points.extend(np.argwhere(result == 0))
        points_map, points = filter_false_segments(points_map, points, thinned)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    points_map = cv2.dilate(points_map, kernel, iterations=1)
    thinned = np.clip(thinned - points_map, 0, 1)

    points_map = points_map.astype(np.uint8) * 255

    thinned = thinned.astype(np.uint8) * 255

    # get list of segments
    contours, hierarchy = cv2.findContours(
        thinned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = []

    for contour in contours:
        segment = np.zeros(thinned.shape[:2], dtype=np.uint8)
        cv2.drawContours(segment, [contour], -1, 255, -1)
        # add segment to the list

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        segment = cv2.dilate(segment, kernel, iterations=1)
        segments.append(segment)
        sum[segment > 0] = random.sample(list(range(256)), k=1)[0]

    #segments_vis = cv2.cvtColor(prediction, cv2.COLOR_RGB2GRAY)
    segments_vis = prediction
    segments_vis = cv2.bitwise_or(sum, sum, mask=segments_vis)

    return thinned1, segments, segments_vis

def filter_false_segments(points_map, points, thinned):
    """
        Function which filter false segments. Not used in current version
        of a software.
        @param points_map: bifurcation points as a gray OpenCV image
        @param points: list of a bifuraction points
        @param thinned: centerline of a vessel
        @return new_points_map: new bifurcation points as a gray OpenCV image
        @return new_points: new list of a bifuraction points
    """
    thinned = thinned.copy().astype(np.uint8) * 255
    initial_contours, _ = cv2.findContours(
         thinned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    initial_contours_hashes = [hash(contour.tostring()) for contour in initial_contours]
    support_kernels = get_support_kernels()

    # Main loop responsible for filtering false branch points
    new_points = []

    for point in points:
        y, x = list(point)
        areas = []
        for kernel in support_kernels + [np.zeros((3, 3), dtype=np.uint8)]:
            kernel = kernel.astype(np.uint8) * 255
            thinned_tmp = thinned.copy()

            # Mask branch point, split input contour into new contours
            branch_point = thinned_tmp[y:y+3, x:x+3].copy()
            thinned_tmp[y:y+3, x:x+3] = np.logical_and(branch_point, kernel).astype(np.uint8) * 255
            if (thinned_tmp[y:y+3, x:x+3] / 255).sum().astype(np.uint8) == 1:
                continue

            tmp_contours, _ = cv2.findContours(
                 thinned_tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            thinned_tmp[y:y + 3, x:x + 3] = np.logical_and(branch_point, kernel).astype(np.uint8) * 128

            # Find new contours after previous operation
            new_contours = [contour for contour in tmp_contours
                            if hash(contour.tostring()) not in initial_contours_hashes]

            # For each new contour measure an area
            for contour in new_contours:
                contour_drawing = np.zeros(thinned_tmp.shape, dtype=np.uint8)
                cv2.drawContours(contour_drawing, [contour], -1, 1, -1)

                areas.append(contour_drawing.sum())

        # Skip contours with area below the threshold
        if all(area > 30 for area in areas):
            new_points.append(point)

    # Draw new map with points
    new_points_map = np.zeros(points_map.shape, dtype=np.float32)
    for point in new_points:
        y, x = list(point)
        new_points_map[y, x] = 1.0

    return new_points_map, new_points

def pointSegments(mask_path):
    """
        Function which let the user points the segments of an image
        which he or she want to remove from a vessel.
        @param mask_path: path to the mask of an image which should be
                          reduced
        @return click_list: list of a points marked on an image
    """
    img = cv2.imread(mask_path)
    click_list = []
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

    return click_list

def reduceVessel(mask_path):
    """
        Experimental function which reduced vessel.
    """
    # Choose segments for removing
    # and get points which belongs to this segments centerlines
    points = pointSegments(mask_path)
    img = cv2.imread(mask_path, 0)
    thinned, segments, segments_vis = get_segments(img)
    cv2.imshow("segments_vis", segments_vis)
    for point in points:
        for segment in segments:
            if segment[point[1], point[0]] > 0:
                img -= segment
                break

    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    cv2.imshow("after", img)
    cv2.waitKey(0)
