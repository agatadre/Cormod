import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import open3d as o3d
from deviceConfiguration import DeviceConfiguration
import borders
import cv2
import splineInterpolation as splint


def remove_given_indexes(points, errors, cp, idx_to_remove):
    """
    Function that removes points of given indexes from three arrays: points, errors and cp
    :param points: three column array of points coordinates (xyz)
    :param errors: vector of errors corresponding with points from points array
    :param cp: two column array of points correspondence
    :param idx_to_remove: list of indexes that we want to remove
    :return: points, errors and cp arrays with wanted elements removed
    """
    megazord = np.concatenate((points, errors.reshape(-1, 1), cp), axis=1)
    reduced_megazord = np.delete(megazord, idx_to_remove, 0)
    new_points, new_errors, new_cp = reduced_megazord[:, :3], (reduced_megazord[:, 3:4]).flatten(), (
        reduced_megazord[:, 4:6]).astype(int)
    return new_points, new_errors, new_cp


def fit_cubic_spline(points):
    """
    Fitting very accurate cubic spline (if it is possible) that goes through every single point of given array
    :param points: three column array of points coordinates (xyz) used for spline fitting
    :return: fitted spline
    """
    if points.shape[0] < 4:
        print('Warning: To few points to draw cubic spline')
        spline = fit_spline(points, points.shape[0])
    else:
        spline = splint.cubicSplineInterpolate(points[:, 0], points[:, 1], points[:, 2])
        spline = (np.array(spline)).T

    # visualization
    # fig = plt.figure()
    # my_ax = fig.add_subplot(111, projection='3d')
    # my_ax.plot(points[:, 0], points[:, 1], points[:, 2], 'ro')
    # my_ax.plot(spline[:, 0], spline[:, 1], spline[:, 2], color="red")
    # plt.show()

    return spline


def fit_spline(points, checkpoints_num):
    """
    Fitting very smooth spline into the point-cloud
    It should be cubic spline, but if there is too few points for cubic spline fitting we use square or linear functions
    :param points: three column array of points coordinates (xyz) used for spline fitting
    :param checkpoints_num: number of points from points array that will be used for spline fitting
    :return: fitted spline
    """
    points_num = points.shape[0]
    if points_num < checkpoints_num:
        print('Too few points for given number of checkpoints')
        return None
    reduced_points = points

    x = reduced_points[:, 0]
    y = reduced_points[:, 1]
    z = reduced_points[:, 2]

    k_param = 3   # cubic spline
    if reduced_points.shape[0] < 4:
        print('Warning: To few points to draw cubic spline')
        k_param = reduced_points.shape[0] - 1
    if k_param < 1:
        print('ERROR: One point or less given for drawing a spline')
        return None

    coords = [x, y, z]
    try:
        tck, u = interpolate.splprep(coords, w=np.ones(len(coords[0])), k=k_param, s=5000)
    except:
        print('ERROR: Exception while drawing a spline occurred')
        print(reduced_points)
        print(checkpoints_num)
        return None

    # this value should depend somehow on the length of vessel (image size? distance between furthest points)
    num_true_pts = 200
    u_fine = np.linspace(0, 1, num_true_pts)

    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
    x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)

    # visualization
    # fig2 = plt.figure(2)
    # ax3d = fig2.add_subplot(111, projection='3d')
    # ax3d.plot(x, y, z, 'g*')
    # ax3d.plot(x_fine, y_fine, z_fine, 'r')
    # ax3d.plot(x_knots, y_knots, z_knots, 'ro')
    # ax3d.plot(points[:, 0], points[:, 1], points[:, 2], 'go')
    #
    # fig2.show()
    # plt.show()

    spline = np.vstack((x_fine, y_fine, z_fine)).T

    return spline


def get_outsiders_indexes(points, spline):
    """
    If any point in the cloud lies at more than 3 times
    average distance from the fitted spline curve at that
    point, consider it as an outlier and return its index
    :param points: three column array of points coordinates (xyz)
    :param spline: fitted spline
    :return: indexes of elements considered as outliers (this elements should be removed from point-cloud)
    """
    pcd = o3d.geometry.PointCloud()
    distances = []

    # calc distance for every point
    for point in points:
        summed_points = np.vstack((spline, point))
        pcd.points = o3d.utility.Vector3dVector(summed_points)
        added_point_index = len(pcd.points) - 1

        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        [_, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[added_point_index], 2)
        closest_point = np.asarray(pcd.points)[idx[1], :]
        distance = np.sum(np.abs(point - closest_point))
        distances.append(distance)

    sum = 0
    for dist in distances:
        sum += dist
    avg_dist = sum / len(distances)

    idx_to_remove = [i for i, d in zip(range(len(distances)), distances) if d > 3 * avg_dist]
    return idx_to_remove


def remove_outsiders(points, spline, errors, cp, current_spline_len):
    """
    Discards points considered as outliers (3 times average distance from the fitted spline) from the point cloud
    :param points: three column array of points coordinates (xyz)
    :param spline: fitted spline
    :param errors: vector of errors corresponding with points from points array
    :param cp: two column array of points correspondence
    :param current_spline_len: length of spline currently fitted into point-cloud
    :return: reduced points, errors and cp arrays
    """
    current_approximate_points = points[:current_spline_len, :]
    idx_to_remove = get_outsiders_indexes(current_approximate_points, spline)
    new_points, new_errors, new_cp = remove_given_indexes(points, errors, cp, idx_to_remove)

    return new_points, new_errors, new_cp


def remove_points_by_spline_interpolation(points, errors, cp):
    """
    Reduced points by fitting very smooth spline into the point cloud and removing points that are further from spline
    than the average (outliers)
    :param points: three column array of points coordinates (xyz)
    :param errors: vector of errors corresponding with points from points array
    :param cp: two column array of points correspondence
    :return: reduced points, fitted spline, reduced errors and cp arrays
    """
    b = 5
    new_points = points
    new_cp = cp
    new_errors = errors
    last_good_spline = None
    end = False
    print('Num of points before spline based point removal: ', str(points.shape[0]))
    while True:
        if b > new_points.shape[0]:
            b = new_points.shape[0]
            end = True
        current_approximate_points = points[:b, :]
        spline = fit_spline(current_approximate_points, b)

        if spline is None:
            print('Error bad spline')
            break

        last_good_spline = spline
        new_points, new_errors, new_cp = remove_outsiders(new_points, spline, new_errors, new_cp, b)
        b += 5
        if end:
            break
    return new_points, last_good_spline, new_errors, new_cp


def plot_spline_and_points(spline=None, points=None):
    """
    Function used mostly for debugging.
    Shows current visual state of spline or points.
    :param spline: spline points to visualize
    :param points: centerline points to visualize
    :return: None
    """
    fig = plt.figure(3)
    ax3d = fig.add_subplot(111, projection='3d')
    if spline is not None:
        ax3d.plot(spline[:, 0], spline[:, 1], spline[:, 2], 'r')
    if points is not None:
        ax3d.plot(points[:, 0], points[:, 1], points[:, 2], 'ro')
    plt.show()


def reduce_many_to_one_point_correspondence(points: np.ndarray, errors: np.ndarray, cp: np.ndarray,
                                            point_max_num_of_correspondences=1):
    """
    Compress the point-cloud by retaining some of the nearest points.
    For each point along the centerline from every projection plane, at most k nearest point
    correspondences (with distance less than ) are included.
    :param points: three column array of points coordinates (xyz)
    :param errors: vector of errors corresponding with points from points array
    :param cp: two column array of points correspondence
    :param point_max_num_of_correspondences: k parameter from point cloud paper
    :return: reduced points, errors and cp arrays
    """
    seen_indexes = set()
    indexes_to_remove = []

    # getting 2d points number
    points_num = np.max(cp) + 1
    # for each projection
    for projection_num in range(2):
        # for each point
        for point_num in range(points_num):
            multiple_same_corr_points = []
            # search all one to many point correspondence for examined point
            for error, correspondence, index in zip(errors, cp, range(errors.shape[0])):
                if correspondence[projection_num] == point_num:
                    multiple_same_corr_points.append([index, error])
            if len(multiple_same_corr_points) <= point_max_num_of_correspondences:
                break
            multiple_same_corr_points = np.array(multiple_same_corr_points)
            smallest_error_indexes = np.argpartition(multiple_same_corr_points[:, 1].T, point_max_num_of_correspondences)
            big_error_indexes = smallest_error_indexes[point_max_num_of_correspondences:]
            for big_error_index in big_error_indexes:
                # delete only duplicates (points that are not needed in neither of projections)
                index_to_delete = int(multiple_same_corr_points[big_error_index, 0])
                if index_to_delete in seen_indexes:
                    indexes_to_remove.append(index_to_delete)
                else:
                    seen_indexes.add(index_to_delete)

    # delete points
    new_points, new_errors, new_cp = remove_given_indexes(points, errors, cp, indexes_to_remove)

    return new_points, new_errors, new_cp


def no_corr_generate_3d_centerline_from_point_cloud(points, errors, cp, centerlines_points, masks_paths):
    """
    Algorithm 1 implementation from point-cloud paper
    Generating 3D Centerline From Point-Cloud by step by step points reduction
    :param points: three column array of points coordinates (xyz)
    :param errors: vector of errors corresponding with points from points array
    :param cp: two column array of points correspondence
    :param centerlines_points: tuple of two points arrays representing same centerline segment from two different views
    :param masks_paths: paths to mask images (used for diameter calculating)
    :return: 3d points representing segment of the vessel; spline points that approximate vessel centerline;
            cp and errors arrays used for further points reduction in algorithm 2
    """

    # For every point along the 2D centerline in each
    # projection plane, find the point of intersection between
    # projection lines with every point along centerline in other
    # planes, or the nearest orthogonal point in case of
    # non-intersecting projection lines;
    # We moved that step outside of the function
    # points, f, o, errors, cp = get_all_possible_points_from_segments(centerlines_points[0], centerlines_points[1],
    #                                                                 conf_device, known_cp)

    # Visualization - all points
    # plot_spline_and_points(None, points)

    # Scaled diameter seems to be too small
    diameters = []
    for centerline_points, mask_path, idx in zip(centerlines_points, masks_paths, range(len(centerlines_points))):
        # avg_vessel_diameter = borders.get_scaled_average_diameter_size(centerline_points, mask_path, conf_device, idx)
        avg_vessel_diameter = borders.get_average_diameter_size(centerline_points, mask_path)
        diameters.append(avg_vessel_diameter)
    avg_max_vessel_diameter = max(diameters) * 2

    # Retain points with orthogonal distance between projection lines
    # less than the average maximum diameter of the vessel
    idx_to_remove = [i for i, e in zip(range(errors.shape[0]), errors) if e > avg_max_vessel_diameter]
    points, errors, cp = remove_given_indexes(points, errors, cp, idx_to_remove)

    # Compress the point-cloud by retaining some of the
    # nearest points (in terms of orthogonal distance between
    # projection lines) for each point along 2D centerline in
    # each of the projection planes and discarding the rest.
    points, errors, cp = reduce_many_to_one_point_correspondence(points, errors, cp)

    # visualization - error based removal
    # print("avg_max_vessel_diameter:")
    # print(avg_max_vessel_diameter)
    # plot_spline_and_points(None, points)

    # additional outlier removal
    points, errors, cp = statistical_outlier_removal(points, errors, cp)

    # visualization - statistic based removal
    # plot_spline_and_points(None, points)

    # spline interpolation
    points, spline, errors, cp = remove_points_by_spline_interpolation(points, errors, cp)

    # visualization - spline based removal
    # plot_spline_and_points(spline, points)
    # print('points num: ' + str(points.shape[0]))

    return points, spline, cp, errors


def statistical_outlier_removal(points, errors, cp):
    """
    Removes points that are further away from their neighbors compared to the average for the point cloud.
    :param points: three column array of points coordinates (xyz)
    :param errors: vector of errors corresponding with points from points array
    :param cp: two column array of points correspondence
    :return: reduced points, errors and cp arrays
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    new_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=8,
                                                  std_ratio=1)
    idx_to_remove = [i for i in range(points.shape[0]) if i not in ind]
    points, errors, cp = remove_given_indexes(points, errors, cp, idx_to_remove)
    return points, errors, cp


def minimize_reprojection_error(conf_device: DeviceConfiguration, centerlines_points,
                                bifurs_2D, masks_paths, threshold=None):
    """
    Algorithm 2 implementation from point-cloud paper
    Generating optimum 3D centerline from point-cloud by minimizing reprojection error
    :param conf_device: DeviceConfiguration object after calibration with all necessary data (angles, sid, sod etc.)
    :param centerlines_points: tuple of two points arrays representing same centerline segment from two different views
    :param bifurs_2D: array of bifurcation points
    :param masks_paths: paths to mask images (used for diameter calculating)
    :param threshold: optional nonstandard threshold value, that is responsible for marking points as correct or not
    :return: 3d points representing segment of the vessel; spline points that approximate vessel centerline
    """
    max_iter = 10
    if threshold is None:
        threshold = conf_device.objective_function_value + 0.7

    corr_points = [i for i in range(len(bifurs_2D[0]))]
    bifurs1 = np.array(bifurs_2D[0], dtype=np.float64)
    bifurs2 = np.array(bifurs_2D[1], dtype=np.float64)

    conf_device.change_points(bifurs1, bifurs2, corr_points)
    bifurs, _, _, bif_errors = conf_device.run_configured_3d_view_generation()

    points, f, o, errors, cp = get_all_possible_points_from_segments(centerlines_points[0], centerlines_points[1],
                                                                     conf_device, None)
    ############################################################################
    # -- 1 -- do
    ############################################################################
    while max_iter:
        ########################################################################
        # -- 2 -- Run Algorithm 1 to generate the 3D centerline of thevessel;
        ########################################################################
        print("iter no " + str(max_iter))
        print("Number of points before Alg1 = " + str(points.shape[0]))
        points, spline, cp, errors = no_corr_generate_3d_centerline_from_point_cloud(
            points, errors, cp, centerlines_points, masks_paths)
        print("Number of points after Alg1 = " + str(points.shape[0]))
        points_num = points.shape[0]
        incorrect_points_num = 0
        correctness = np.empty([points_num])

        # make sure that we have right points set in our device
        conf_device.change_points(centerlines_points[0], centerlines_points[1], None)

        ########################################################################
        # -- 3 -- Back-project the 3D centerline on the originalprojection planes;
        ########################################################################
        # calculate two ray-traced shadows
        ray_trace_shadow_1 = conf_device.get_ray_traced_shadow(points, f[0], o[0])
        ray_trace_shadow_2 = conf_device.get_ray_traced_shadow(points, f[1], o[1])
        # points are in non corresponding position, so we cant broadcast whole matrix
        # but broadcasting would improve performance

        # transform centerline points in 2D to 3D shadows
        shadow_1, shadow_2 = conf_device.get_image_shadows()

        incorrect_points1 = []
        incorrect_points2 = []
        ########################################################################
        # -- 4 -- Compare with 2D centerlines: if the back-projectionof a point
        # perfectly matches with 2D centerline at allprojection planes, assign 1,
        # otherwise assign 0;
        ########################################################################
        for ray_traced_point_1, ray_traced_point_2, corr, point, idx in zip(
                ray_trace_shadow_1, ray_trace_shadow_2, cp, points, range(points_num)):
            error_1 = np.sum(np.abs(ray_traced_point_1 - shadow_1[corr[0]]))
            error_2 = np.sum(np.abs(ray_traced_point_2 - shadow_2[corr[1]]))
            if error_1 > threshold or error_2 > threshold:
                ################################################################
                # -- 5 -- From the sequence of 1’s and 0’s, construct segments
                # of correct and incorrect reconstructions;
                ################################################################
                point1 = centerlines_points[0][corr[0]]
                point2 = centerlines_points[1][corr[1]]

                incorrect_points1.append(point1)
                incorrect_points2.append(point2)

                correctness[idx] = 0
                incorrect_points_num += 1
            else:
                ################################################################
                # -- 5 -- From the sequence of 1’s and 0’s, construct segments
                # of correct and incorrect reconstructions;
                ################################################################
                correctness[idx] = 1

        incorrect_points1 = np.unique(np.array(incorrect_points1), axis=0)
        incorrect_points2 = np.unique(np.array(incorrect_points2), axis=0)

        # End of algorithm
        if incorrect_points_num == 0:
            break
        correct_points_num = points_num - incorrect_points_num

        ########################################################################
        # -- 6 -- For incorrect segments, generate point cloud,
        # similar to Algorithm 1, where, for each point along incorrect segments
        # on 2D centerline, some of the nearest points are retained.
        # No restriction on the orthogonal distance between projection lines;
        ########################################################################

        if correct_points_num == 0:
            incorrect_points, _, _, incorrect_errors, incorrect_cp = get_all_possible_points_from_segments(
                incorrect_points1, incorrect_points2, conf_device, None)

            print("Number of incorrect_points = " + str(incorrect_points.shape[0]))
            corrected_points, corrected_errors, corrected_cp = incorrect_points_correction(
                incorrect_points, incorrect_errors, incorrect_cp)

            print("Number of corrected_points = " + str(corrected_points.shape[0]))
            points = corrected_points
            errors = corrected_errors
            cp = corrected_cp
        else:
            ####################################################################
            # -- 7 -- For correct segments, keep only the point-to-point
            # correspondences;
            ####################################################################
            correct_points = np.empty([correct_points_num, points.shape[1]])
            correct_errors = np.empty([correct_points_num])
            correct_cp = np.empty([correct_points_num, cp.shape[1]], dtype=int)

            corr_idx = 0
            for i in range(points_num):
                if correctness[i] == 1:
                    correct_points[corr_idx] = points[i]
                    correct_errors[corr_idx] = errors[i]
                    correct_cp[corr_idx] = cp[i]
                    corr_idx += 1

            # Generate point cloud without correct points
            incorrect_points, _, _, incorrect_errors, incorrect_cp = get_all_possible_points_from_segments(
                incorrect_points1, incorrect_points2, conf_device, correct_points)

            print("Number of incorrect_points = " + str(incorrect_points.shape[0]))
            corrected_points, corrected_errors, corrected_cp = incorrect_points_correction(
                incorrect_points, incorrect_errors, incorrect_cp)

            print("Number of corrected_points = " + str(corrected_points.shape[0]))
            ####################################################################
            # -- 8 -- Construct the point-cloud set combining both correctly
            # and incorrectly reconstructed segments;
            ####################################################################
            points = np.vstack((correct_points, corrected_points))
            errors = np.concatenate((correct_errors, corrected_errors))
            cp = np.vstack((correct_cp, corrected_cp))

            # sort by cp order because stacking broke it
            points, errors, cp = sort_points(points, errors, cp)

        max_iter -= 1

    points, errors, cp = distance_sort(points, errors, cp)
    if points.shape[0] == 0:
        print('ERROR: No points')
        return None
    points = check_if_bifurs_exist(points, bifurs)
    spline = fit_cubic_spline_with_max_points_num(points)
    print(points)
    print(spline)
    return points, spline


def check_if_bifurs_exist(points, bifurs):
    """
    Check if bifurcation points is already in the segment.
    If point is not there, it adds it in the end of the array or at its beginning
    :param points: three column array of points coordinates (xyz)
    :param bifurs: array that contains bifurcation points for vessels
    :return: points array with missing bifurcation points added
    """

    for bifur in bifurs:
        # check if bifurcation points is already in the segment
        if bifur not in points:
            first_diff = np.sum(np.abs(points[0] - bifur))
            last_diff = np.sum(np.abs(points[-1] - bifur))

            if first_diff > last_diff:
                # it is the end
                print("Last bifur " + str(bifur) + " not belongs to the segment")
                points = np.vstack((points, bifur))
            else:
                # it is start
                print("First bifur " + str(bifur) + " not belongs to the segment")
                points = np.vstack((bifur, points))

    return points


def get_one_to_one_point_correspondence_indexes(cp):
    """
    Get indexes from cp array that points to array elements with one-to-one correspondence.
    One-to-many or many-to-one points correspondence indexes are rejected.
    :param cp: two column array of points correspondence
    :return: indexes of cp array elements that contains one-to-one point correspondence
    """
    centerline_2d_points_num = np.max(cp) + 1
    point_quantity_and_index = np.zeros([2, centerline_2d_points_num, 2], dtype=int)
    for corr, idx in zip(cp, range(len(cp))):
        for plane in range(2):
            point_quantity_and_index[plane][corr[plane]][0] += 1
            point_quantity_and_index[plane][corr[plane]][1] = idx
    indexes = []
    for i in range(centerline_2d_points_num):
        first_plane_point_instances_num = point_quantity_and_index[0][i][0]
        first_plane_point_idx = point_quantity_and_index[0][i][1]
        if first_plane_point_instances_num == 1:
            second_plane_point_num = cp[first_plane_point_idx][1]
            second_plane_point_instances_num = point_quantity_and_index[1][second_plane_point_num][0]
            if second_plane_point_instances_num == 1:
                indexes.append(first_plane_point_idx)

    if len(indexes) == 0:
        return None

    return indexes


def incorrect_points_correction(inc_points, errors, cp):
    """
    For incorrect segments, generate point cloud, similar
    to Algorithm 1, where, for each point along incorrect
    segments on 2D centerline, some of the nearest points
    are retained. No restriction on the orthogonal distance
    between projection lines;
    :param inc_points: incorrect points array
    :param errors: vector of errors corresponding with points from inc_points array
    :param cp: two column array of points correspondence
    :return: corrected points, errors, and cp
    """
    print('correction algorithm start')
    print('starting incorrect points num:' + str(inc_points.shape[0]))
    points, errors, cp = reduce_many_to_one_point_correspondence(inc_points, errors, cp, 1)
    print('incorrect points num after first step:' + str(inc_points.shape[0]))
    if points.shape[0] > 4:
        points, spline, errors, cp = remove_points_by_spline_interpolation(points, errors, cp)
        print('incorrect points num after second step:' + str(inc_points.shape[0]))

    return points, errors, cp


def fit_cubic_spline_with_max_points_num(points):
    """
    Draw spline precisely through all points
    :param points: three column array of points coordinates (xyz)
    :return: array of points that approximate fitted spline
    """
    return fit_cubic_spline(points)
    # return fit_spline(points, points.shape[0])


def showVessel(vessel, points):
    colors_vessel = np.empty([vessel.shape[0], 3], dtype=np.float64)
    colors_vessel[:] = [0, 0, 0]

    colors_points = np.empty([points.shape[0], 3], dtype=np.float64)
    colors_points[:] = [255, 0, 0]

    pcd_list = []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack([vessel]))
    pcd.colors = o3d.utility.Vector3dVector(colors_vessel)
    pcd_list.append(pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack([points]))
    pcd.colors = o3d.utility.Vector3dVector(colors_points)
    pcd_list.append(pcd)

    o3d.visualization.draw_geometries(pcd_list)


def showShadows(device, f, o, pointCloud3D, mask_paths):
    shadow_points1 = device.get_ray_traced_shadow(pointCloud3D, f[0], o[0])
    shadow_points2 = device.get_ray_traced_shadow(pointCloud3D, f[1], o[1])

    shadow_points1 = np.array(shadow_points1, dtype=np.int64)
    shadow_points2 = np.array(shadow_points2, dtype=np.int64)

    mask1 = cv2.imread(mask_paths[0], 0)
    mask2 = cv2.imread(mask_paths[1], 0)
    vessel1 = np.argwhere(mask1)
    vessel2 = np.argwhere(mask2)

    vessel1 = np.array(vessel1, dtype=np.float64)
    vessel2 = np.array(vessel2, dtype=np.float64)
    device.change_points(vessel1, vessel2, None)

    vessel1, vessel2 = device.get_image_shadows()

    showVessel(vessel1, shadow_points1)
    showVessel(vessel2, shadow_points2)


def get_all_possible_points_from_segments(seg_points_1: np.ndarray, seg_points_2: np.ndarray,
                                          configured_device: DeviceConfiguration, known_points=None):
    """
    For every point along the 2D centerline in each
    projection plane, find the point of intersection between
    projection lines with every point along centerline in other
    planes, or the nearest orthogonal point in case of
    non-intersecting projection lines
    If known_points is given, then this points are skipped
    :param seg_points_1: 2d centerline points on first projection plane
    :param seg_points_2: 2d centerline points on second projection plane
    :param configured_device: DeviceConfiguration object after calibration with all necessary data (angles, sid etc.)
    :param known_points: array with known points correspondence
    :return: generated 3d points array; Tuple(xyz coordinate of first ray source; second);
             Tuple(array of xyz points of first detector plane; second); reprojection errors array, cp array
    """
    sizes = (seg_points_1.shape[0], seg_points_2.shape[0])
    points_num = sizes[0] * sizes[1]
    bigger_seg_points_num = max(sizes)
    smaller_seg_points_num = min(sizes)
    all_points = np.empty([points_num, 3])
    all_errors = np.empty([points_num])
    all_cp = np.empty([points_num, 2], dtype=int)
    f, o = None, None

    first_points_num = smaller_seg_points_num
    second_points_num = bigger_seg_points_num
    if sizes[0] < sizes[1]:
        first_points_num = bigger_seg_points_num
        second_points_num = smaller_seg_points_num

    for i in range(first_points_num):
        corr_points = [i for z in range(second_points_num)]
        configured_device.change_points(seg_points_1, seg_points_2, corr_points)
        new_points, f, o, new_errors = configured_device.run_configured_3d_view_generation()
        all_points[i * second_points_num: (i + 1) * second_points_num, :] = new_points
        all_errors[i * second_points_num: (i + 1) * second_points_num] = new_errors
        all_cp[i * second_points_num: (i + 1) * second_points_num, :] = [[z, i] for z in range(second_points_num)]

    if known_points is not None:
        idx_to_remove = set()
        for point in known_points:
            for i in range(first_points_num):
                for j in range(second_points_num):
                    cmp = point == all_points[i * second_points_num + j]
                    if cmp.all():
                        idx_to_remove.add(i * second_points_num + j)

        # removing indexes
        print("---------- known_points ----------")
        print(known_points)
        print("---------- all_points ----------")
        print(all_points)
        print("---------- idx_to_remove ----------")
        print(idx_to_remove)
        all_points, all_errors, all_cp = remove_given_indexes(all_points, all_errors, all_cp, list(idx_to_remove))
        print("---------- all_points after ----------")
        print(all_points)

        # sort by cp order
        all_points, all_errors, all_cp = sort_points(all_points, all_errors, all_cp)

    return all_points, f, o, all_errors, all_cp


def distance_sort(points: np.ndarray, errors: np.ndarray, cp: np.ndarray):
    """
    Sort points, errors, cp by distance
    We start from first point in array points than we find for a closest point and repeat procedure.
    First point is always where we start.
    :param points: three column array of points coordinates (xyz)
    :param errors: vector of errors corresponding with points from points array
    :param cp: two column array of points correspondence
    :return: sorted points, errors, cps by distance
    """
    points_num = points.shape[0]
    if points_num < 1:
        return points, errors, cp

    sorted_points = np.empty([points_num, 3])
    sorted_errors = np.empty([points_num])
    sorted_cp = np.empty([points_num, 2], dtype=int)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    # set first point
    sorted_indexes = [0]
    sorted_points[0] = points[0]
    sorted_errors[0] = errors[0]
    sorted_cp[0] = cp[0]

    idx_num = 1
    current_point = pcd.points[0]
    while idx_num != points_num:
        # getting closest point
        try_num = 1
        while True:
            [k, idx, _] = pcd_tree.search_knn_vector_3d(current_point, 1 + try_num)
            found_point_idx = idx[try_num]
            if found_point_idx not in sorted_indexes:
                sorted_indexes.append(found_point_idx)
                closest_point = np.asarray(pcd.points)[found_point_idx, :]
                sorted_points[idx_num] = closest_point
                sorted_errors[idx_num] = errors[found_point_idx]
                sorted_cp[idx_num] = cp[found_point_idx]
                current_point = closest_point
                break
            try_num += 1
        idx_num += 1

    return sorted_points, sorted_errors, sorted_cp


def sort_points(points: np.ndarray, errors: np.ndarray, cp: np.ndarray):
    """
    Sort points, errors, cp according to ascending cp order
    :param points: three column array of points coordinates (xyz)
    :param errors: vector of errors corresponding with points from points array
    :param cp: two column array of points correspondence
    :return: sorted points, errors, cps
    """
    indexes = get_sorted_cp_indexes(cp)
    sorted_cp = cp.take(indexes, axis=0)
    sorted_points = points.take(indexes, axis=0)
    sorted_errors = errors.take(indexes)
    return sorted_points, sorted_errors, sorted_cp


def get_sorted_cp_indexes(cp):
    """
    Sort points correspondence (cp array) first by 1 column then by 0 column
    :param cp: two column array of points correspondence
    :return: column sorted cp array
    """
    # sort first by 1 column then by 0 column
    sorted_indexes = np.lexsort((cp[:, 0], cp[:, 1]))
    return sorted_indexes
