import numpy as np
import math as m
from scipy.spatial.transform.rotation import Rotation

def point_p(F1, F2, P1, P2):
    """ Function which solves linear eqations
        L1 and L2 to calculate point P
        @param F1: 3D projection of X-ray source from base view
        @param F2: 3D projection of X-ray source from second view
        @param P1: 3D projection of a 2D point from base view
        @param P2: 3D projection of a 2D point from second view
        @return P: Tuple(3D position of 2D point, error of reproduction)
    """
    # vertexes
    eta = np.subtract(P1, F1)
    tau = np.subtract(P2, F2)

    # From linear equations L1, L2:
    # s*eta*eta' - t*eta*tau' = eta*(F2 - F1)'
    # s*eta*tau' - t*tau*tau' = tau*(F2- F1)'
    #
    # we must calculate parameters s and t to gain P

    # determinant
    eta_eta = float(np.matmul(eta, eta.transpose()))
    tau_tau = float(-np.matmul(tau, tau.transpose()))
    eta_tau1 = float(np.matmul(eta, tau.transpose()))
    eta_tau2 = float(-np.matmul(eta, tau.transpose()))

    F2_F1 = np.subtract(F2, F1)
    eta_F = float(np.matmul(eta, F2_F1.transpose()))
    tau_F = float(np.matmul(tau, F2_F1.transpose()))

    deter = (eta_eta * tau_tau) - (eta_tau1 * eta_tau2)

    # parameters s and t
    s = ((eta_F * tau_tau) - (eta_tau2 * tau_F)) / deter
    t = ((eta_eta * tau_F) - (eta_F * eta_tau1)) / deter

    # calculating points L1s nad L2t
    L1s = F1 + (s * eta)
    L2t = F2 + (t * tau)

    # Point P
    P = (L1s + L2t) / 2

    # calc distance between projection lines for further correction
    distance = np.sum(np.abs(L2t-L1s))

    return P, distance


def calc_m(alpha=0, beta=0):
    """
    Calculates rotational matrix M with given angles
    :param alpha: alpha (LAO[+180] - RAO[-180]) angle in radians
    :param beta: beta (CRA[+90] - CAU[-90]) angle in radians
    :return: rotational matrix M
    """
    rotational_matrix = np.array([[m.cos(beta), m.sin(alpha) * m.sin(beta), m.cos(alpha) * m.sin(beta), 0],
                                 [0, m.cos(alpha), m.sin(alpha), 0],
                                 [-m.sin(beta), -m.sin(alpha) * m.cos(beta), m.cos(alpha) * m.cos(beta), 0],
                                 [0, 0, 0, 1]])
    return rotational_matrix


def rotate_points(points_array, alpha, beta):
    """
    Rotates given points by alpha and beta angle
    :param points_array: array that contains xyz1 coordinates of a points to rotate
    :param alpha: alpha (LAO[+180] - RAO[-180]) angle in radians
    :param beta: beta (CRA[+90] - CAU[-90]) angle in radians
    :return: array that contains xyz1 coordinates of rotated points
    """
    for point_num in range(points_array.shape[0]):
        points_array[point_num, :] = np.matmul(calc_m(alpha, beta), points_array[point_num, :].T)
    return points_array


def calc_real_points(shadow_1, shadow_2, source_1, source_2, corresponding_points_array):
    """
    Calculates points of a real object from given shadows
    :param shadow_1: array that contains xyz1 coordinates of a first shadow
    :param shadow_2: array that contains xyz1 coordinates of a second shadow
    :param source_1: xyz1 coordinate of first ray source
    :param source_2: xyz1 coordinate of second ray source
    :param  corresponding_points_array: array that contains indexes of corresponding points
            in shadow_1 and shadow_2 arrays
    :return: tuple(array that contains xyz1 coordinates of real object points, array of errors of points reproduction)
    """
    num_of_points_to_create = len(corresponding_points_array)
    if num_of_points_to_create > shadow_1.shape[0]:
        print("corresponding points doesn't match with given shadows")
        return None
    if num_of_points_to_create != shadow_1.shape[0]:
        print('warning, non equal points number')

    real_points = []
    errors = []
    for i in range(num_of_points_to_create):
        p, error = point_p(source_1, source_2,
                           shadow_1[i, :],
                           shadow_2[corresponding_points_array[i], :])
        real_points.append(p)
        errors.append(error)

    # list to np.array
    real_points_array = np.array(real_points, dtype=np.float32)
    return real_points_array, errors


def get_xyz_rotation(x_angle, y_angle, z_angle):
    """
    Creates 3-axis rotation by an x, y, z angle
    :param x_angle: value of rotation by x angle given in radians
    :param y_angle: value of rotation by y angle given in radians
    :param z_angle: value of rotation by z angle given in radians
    :return: rotation by an x, y, z angle
    """
    # X axis rotation
    rotation_axis = np.array([1, 0, 0])
    rotation_vector = x_angle * rotation_axis
    rotation_x = Rotation.from_rotvec(rotation_vector)

    # Y axis rotation
    rotation_axis = np.array([0, 1, 0])
    rotation_vector = y_angle * rotation_axis
    rotation_y = Rotation.from_rotvec(rotation_vector)

    # Z axis rotation
    rotation_axis = np.array([0, 0, 1])
    rotation_vector = z_angle * rotation_axis
    rotation_z = Rotation.from_rotvec(rotation_vector)

    return rotation_x * rotation_y * rotation_z


def correct_rotational_misalignment(shadow_1, shadow_2, params):
    """
    Applies theta rotations on shadows to correct rotational misalignment caused by device nonideal mechanical response
    :param shadow_1: array that contains xyz1 coordinates of a first shadow
    :param shadow_2: array that contains xyz1 coordinates of a second shadow
    :param params: dictionary of optimized parameters acquired by device calibration
    :return: corrected shadows arrays
    """
    if params is not None:
        theta_rotation = get_xyz_rotation(params['theta_x'], params['theta_y'], params['theta_z'])
        shadow_1 = theta_rotation.apply(shadow_1[:, :-1])
        shadow_2 = theta_rotation.apply(shadow_2[:, :-1])

        # add lost dimension with ones
        shadow_1 = np.hstack((shadow_1, np.ones((shadow_1.shape[0], 1), dtype=shadow_1.dtype)))
        shadow_2 = np.hstack((shadow_2, np.ones((shadow_2.shape[0], 1), dtype=shadow_2.dtype)))
    return shadow_1, shadow_2


def generate_3d_view(img_points_1, img_points_2, alpha, beta, sid_1, sod_1, sid_2, sod_2, corresponding_points,
                     image_sizes=None, view_id='0', visualize=False, optimized_params=None, pixel_spacing=None):
    """
    Function that creates 3D view from 2 given views of the same object seen from different perspective (angles)
    :param img_points_1: 2D array of xy first shadow (image) points
    :param img_points_2: 2D array of xy second shadow (image) points
    :param alpha: alpha (LAO[+180] - RAO[-180]) angle in radians between views
    :param beta: beta (CRA[+90] - CAU[-90]) angle in radians between views
    :param sid_1: Distance Source to Detector (plane) of the first view
    :param sod_1: Distance Source to Patient (object) of the first view
    :param sid_2: Distance Source to Detector (plane) of the second view
    :param sod_2: Distance Source to Patient (object) of the second view
    :param corresponding_points: array that contains indexes of corresponding points in shadow_1 and shadow_2 arrays
    :param image_sizes: array that contains image width and height
    :param view_id: id (string) appended to name of generated point cloud file
    :param visualize: whether you want to see visualized result or not
    :param optimized_params: dictionary of 11 parameters form device calibration to correct device misalignments
    :param pixel_spacing: array that contains image pixel spacing x and y (in mm),
            so that we can reconstruct real dimensions
    :return: Tuple(array of xyz points of real 3D object; Tuple(xyz coordinate of first ray source; second);
             Tuple(array of xyz points of first detector plane; second); array of errors of points reproduction)
    """

    # Copy all points so we don't change original object
    shadow_points_1 = np.array(img_points_1)
    shadow_points_2 = np.array(img_points_2)

    # translate images to the center (correct x,y coordinates) if image size is given
    if image_sizes is not None:
        shadow_points_1 -= image_sizes[0]/2
        shadow_points_2 -= image_sizes[1]/2

    # convert pixels to real dimensions (mm)
    if pixel_spacing is not None:
        shadow_points_1 *= pixel_spacing
        shadow_points_2 *= pixel_spacing

    # set angles to more accurate if possible
    if optimized_params is not None:
        alpha += optimized_params['alpha']
        beta += optimized_params['beta']

    # sid = 8
    # sod = 4
    f_1 = np.array([0, 0, -sod_1, 1])
    f_z_2 = np.array([0, 0, -sod_2, 1])
    f_2 = np.matmul(calc_m(alpha, beta), f_z_2.T)

    # convert points to 3d
    # by adding one dimension (Z) with zeros
    shadow_points_1 = np.hstack((shadow_points_1, np.zeros((shadow_points_1.shape[0], 1), dtype=shadow_points_1.dtype)))
    shadow_points_2 = np.hstack((shadow_points_2, np.zeros((shadow_points_2.shape[0], 1), dtype=shadow_points_2.dtype)))

    # convert to cartesian
    # by adding one dimension with ones
    shadow_points_1 = np.hstack((shadow_points_1, np.ones((shadow_points_1.shape[0], 1), dtype=shadow_points_1.dtype)))
    shadow_points_2 = np.hstack((shadow_points_2, np.ones((shadow_points_2.shape[0], 1), dtype=shadow_points_2.dtype)))

    # first view rotation
    # we don't need it since first view is the reference
    # a1 = 0  # -m.pi/2
    # b1 = 0  # -m.pi/2
    # rotate_points(shadow_points_1, a1, b1)

    # image detector translation misalignment correction
    if optimized_params is not None:
        delta_o = np.array(
            [optimized_params['delta_o_x'], optimized_params['delta_o_y'], optimized_params['delta_o_z'], 0])
        shadow_points_1 += delta_o
        shadow_points_2 += delta_o

    # second view rotation
    rotate_points(shadow_points_2, alpha, beta)

    # image detector rotational misalignment correction
    shadow_points_1, shadow_points_2 = correct_rotational_misalignment(shadow_points_1, shadow_points_2, optimized_params)

    # calculate translations of the projections
    o_1 = np.array([0, 0, sid_1 - sod_1, 1])
    o_z_2 = np.array([0, 0, sid_2 - sod_2, 1])
    o_2 = np.matmul(calc_m(alpha, beta), o_z_2.T)

    # translate views (shadows) to their real position => add O vector
    shadow_points_1[:, :-1] += o_1[:-1]
    shadow_points_2[:, :-1] += o_2[:-1]

    # add relative isocenter movement to secondary view
    if optimized_params is not None:
        delta_i = np.array(
            [optimized_params['delta_i_x'], optimized_params['delta_i_y'], optimized_params['delta_i_z'], 0])
        shadow_points_2 += delta_i

    # calculate real point position from shadows
    real_object_points, errors = calc_real_points(shadow_points_1, shadow_points_2, f_1, f_2, corresponding_points)

    # END OF CALCULATIONS

    # VISUALIZATION PART
    if visualize:
        visual.visualize_all(
            shadow_points_1, shadow_points_2, real_object_points, f_1, f_2, view_id, corresponding_points)

    return real_object_points[:, :-1], (f_1[:-1], f_2[:-1]), (o_1[:-1], o_2[:-1]), errors


def get_3d_shadows(img_points_1, img_points_2, alpha, beta, sid_1, sod_1, sid_2, sod_2, image_sizes=None,
                   pixel_spacing=None):
    """
    Function translate and rotate given 2d shadows from images to their correct 3d positions
    :param img_points_1: 2D array of xy first shadow (image) points
    :param img_points_2: 2D array of xy second shadow (image) points
    :param alpha: alpha (LAO[+180] - RAO[-180]) angle in radians between views
    :param beta: beta (CRA[+90] - CAU[-90]) angle in radians between views
    :param sid_1: Distance Source to Detector (plane) of the first view
    :param sod_1: Distance Source to Patient (object) of the first view
    :param sid_2: Distance Source to Detector (plane) of the second view
    :param sod_2: Distance Source to Patient (object) of the second view
    :param image_sizes: array that contains image width and height
    :param pixel_spacing: array that contains image pixel spacing x and y (in mm),
            so that we can reconstruct real dimensions
    :return: array of xyz coordinates of new correct 3d position of shadows
    """

    # Copy all points so we don't change original object
    shadow_points_1 = np.array(img_points_1)
    shadow_points_2 = np.array(img_points_2)

    # translate image to the center (correct x,y coordinates) if image size is given
    if image_sizes is not None:
        shadow_points_1 -= image_sizes[0]/2
        shadow_points_2 -= image_sizes[1]/2

    # convert pixels to real dimensions (mm)
    if pixel_spacing is not None:
        shadow_points_1 *= pixel_spacing
        shadow_points_2 *= pixel_spacing

    # convert points to 3d
    # by adding one dimension (Z) with zeros
    shadow_points_1 = np.hstack((shadow_points_1, np.zeros((shadow_points_1.shape[0], 1), dtype=shadow_points_1.dtype)))
    shadow_points_2 = np.hstack((shadow_points_2, np.zeros((shadow_points_2.shape[0], 1), dtype=shadow_points_2.dtype)))

    # convert to cartesian
    # by adding one dimension with ones
    shadow_points_1 = np.hstack((shadow_points_1, np.ones((shadow_points_1.shape[0], 1), dtype=shadow_points_1.dtype)))
    shadow_points_2 = np.hstack((shadow_points_2, np.ones((shadow_points_2.shape[0], 1), dtype=shadow_points_2.dtype)))

    # second view rotation
    rotate_points(shadow_points_2, alpha, beta)

    # calculate translations of the projections
    o_1 = np.array([0, 0, sid_1 - sod_1, 1])
    o_z_2 = np.array([0, 0, sid_2 - sod_2, 1])
    o_2 = np.matmul(calc_m(alpha, beta), o_z_2.T)

    # translate views (shadows) to their real position => add O vector
    shadow_points_1[:, :-1] += o_1[:-1]
    shadow_points_2[:, :-1] += o_2[:-1]
    return shadow_points_1[:, :-1], shadow_points_2[:, :-1]
