from geneticAlg import geneticalgorithm as ga
import numpy as np
from enum import Enum
import imageFormation

class Variable(Enum):
    ALPHA = 0
    BETA = 1
    THETA_X = 2
    THETA_Y = 3
    THETA_Z = 4
    O_X = 5
    O_Y = 6
    O_Z = 7
    I_X = 8
    I_Y = 9
    I_Z = 10

class DeviceConfiguration:
    def __init__(self, img_1_points, img_2_points, corresponding_points, sid_1, sod_1, sid_2, sod_2, alpha_2, beta_2,
                 image_sizes=None, pixel_spacing=None, optimized_params=None):
        self.beta_2 = beta_2
        self.alpha_2 = alpha_2
        self.sod_1 = sod_1
        self.sid_1 = sid_1
        self.sod_2 = sod_2
        self.sid_2 = sid_2
        self.img_1_points = img_1_points
        self.img_2_points = img_2_points
        self.corresponding_points = corresponding_points
        self.image_sizes = image_sizes
        self.optimized_params = optimized_params
        self.pixel_spacing = pixel_spacing
        self.objective_function_value = 0
        print('Objective function without correction:\n' + str(self.__min_function(
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))))

    def change_points(self, new_img_1_points, new_img_2_points, new_corresponding_points):
        """
        Change set of points to a different one.
        Use only with the same images, but a different set of points.
        For example: to change bifurcation points set to centerline points
        :param new_img_1_points: 2D array of xy first shadow (image) points
        :param new_img_2_points: 2D array of xy second shadow (image) points
        :param new_corresponding_points: array that contains indexes of corresponding points
                in shadow_1 and shadow_2 arrays
        :return:
        """
        self.img_1_points = new_img_1_points
        self.img_2_points = new_img_2_points
        self.corresponding_points = new_corresponding_points

    @staticmethod
    def line_plane_collision(plane_normal, plane_point, ray_direction, ray_point, epsilon=1e-6):
        """
        Finds the intersection of a line with a plane
        :param plane_normal: normal vector to the plane
        :param plane_point: any point that belongs to the plane
        :param ray_direction: vector that determines ray direction
        :param ray_point: point that belongs to the line
        :param epsilon: max error
        :return: intersection point(array with xyz coordinates)
        """
        ndotu = plane_normal.dot(ray_direction)
        if abs(ndotu) < epsilon:
            raise RuntimeError("no intersection or line is within plane")

        w = ray_point - plane_point
        si = -plane_normal.dot(w) / ndotu
        psi = w + si * ray_direction + plane_point
        return psi

    def get_ray_traced_shadow(self, real_image_points, f, o, relocate=False, points_correspondence=None):
        """
        Calculates shadow of real image seen from F point on O plane using ray-tracing
        :param real_image_points: array of xyz coordinates of the object
        :param f: xyz coordinates of point of view / X-ray source
        :param o: xyz coordinates of plane / Detector plane location
        :param relocate: whether you need to change point order or not
        :return: array of xyz coordinates of the ray-traced shadow
        """
        num_of_points = real_image_points.shape[0]
        shadow_points = np.empty([num_of_points, 3], dtype=np.float32)
        if points_correspondence is None:
            points_correspondence = self.corresponding_points

        # if corresponding points are not given, put coordinates as they are
        for point_num in range(num_of_points):
            index = points_correspondence[point_num] if relocate else point_num
            shadow_points[index, :] = np.array(
            [self.line_plane_collision(f - o, o, real_image_points[point_num] - f, real_image_points[point_num])])

        return shadow_points

    @staticmethod
    def get_var_dict(xa):
        """
        Returns dictionary with described fields
        :param xa: array of 11 parameters
        :return: parameters dictionary
        """
        optimized_variables = {'alpha': xa[Variable.ALPHA.value],
                               'beta': xa[Variable.BETA.value],
                               'theta_x': xa[Variable.THETA_X.value],
                               'theta_y': xa[Variable.THETA_Y.value],
                               'theta_z': xa[Variable.THETA_Z.value],
                               'delta_o_x': xa[Variable.O_X.value],
                               'delta_o_y': xa[Variable.O_Y.value],
                               'delta_o_z': xa[Variable.O_Z.value],
                               'delta_i_x': xa[Variable.I_X.value],
                               'delta_i_y': xa[Variable.I_Y.value],
                               'delta_i_z': xa[Variable.I_Z.value]}
        return optimized_variables

    def __min_function(self, xa):
        """
        Fitness function
        Calculates value for given individual parameters
        :param xa: array of individual parameters
        :return:    Mean square error value between image representation
                    and ray-tracing projection
        """

        # get real image, so that we can ray-trace back
        real_points, f, o, _ = imageFormation.generate_3d_view(self.img_1_points, self.img_2_points, self.alpha_2,
                                                               self.beta_2, self.sid_1, self.sod_1, self.sid_2, self.sod_2,
                                                               self.corresponding_points, self.image_sizes, '0', False,
                                                               self.get_var_dict(xa), self.pixel_spacing)

        # calculate two ray-traced shadows
        ray_trace_shadow_1 = self.get_ray_traced_shadow(real_points, f[0], o[0])
        ray_trace_shadow_2 = self.get_ray_traced_shadow(real_points, f[1], o[1], True)

        # transform shadows to their real position in 3d
        shadow_1, shadow_2 = self.get_image_shadows()

        # calculate MSE
        value = np.sum(np.square(shadow_1 - ray_trace_shadow_1))
        value += np.sum(np.square(shadow_2 - ray_trace_shadow_2))
        value /= real_points.shape[0]
        return value

    def get_image_shadows(self):
        return imageFormation.get_3d_shadows(self.img_1_points, self.img_2_points, self.alpha_2, self.beta_2,
                                             self.sid_1, self.sod_1, self.sid_2, self.sod_2,
                                             self.image_sizes, self.pixel_spacing)

    def get_calibration_params(self):
        """
        Finds best device calibration params using genetic algorithm
        :return: dictionary with best parameters found and objective function
        """

        # max degrees error for alpha and beta expressed in radians
        max_delta_rad = np.deg2rad(2)
        max_theta_rad = np.deg2rad(2)
        max_translation = 2
        max_isocenter_movement = 2

        # set bounds for minimized parameters
        variables_bounds = np.array([[-max_delta_rad, max_delta_rad],                       # delta_alpha
                                     [-max_delta_rad, max_delta_rad],                       # delta_beta
                                     [-max_theta_rad, max_theta_rad],                       # delta_theta_x (rotation misalignment)
                                     [-max_theta_rad, max_theta_rad],                       # delta_theta_y (rotation misalignment)
                                     [-max_theta_rad, max_theta_rad],                       # delta_theta_z (rotation misalignment)
                                     [-max_translation, max_translation],                   # delta_O_x (translation misalignment)
                                     [-max_translation, max_translation],                   # delta_O_y (translation misalignment)
                                     [-max_translation, max_translation],                   # delta_O_z (translation misalignment)
                                     [-max_isocenter_movement, max_isocenter_movement],     # delta_i_x (2nd view isocenter movement)
                                     [-max_isocenter_movement, max_isocenter_movement],     # delta_i_x (2nd view isocenter movement)
                                     [-max_isocenter_movement, max_isocenter_movement]])    # delta_i_x (2nd view isocenter movement)

        # algorithm parameters modifications, mainly for faster testing
        algorithm_param = {'max_num_iteration': 60,
                           'population_size': 20,
                           'mutation_probability': 0.1,
                           'elit_ratio': 0.01,
                           'crossover_probability': 0.5,
                           'parents_portion': 0.3,
                           'crossover_type': 'uniform',
                           'max_iteration_without_improv': None}

        # model creation
        model = ga(function=self.__min_function,
                   dimension=variables_bounds.shape[0],
                   variable_type='real',
                   variable_boundaries=variables_bounds,
                   algorithm_parameters=algorithm_param)

        model.run(False)
        self.optimized_params = self.get_var_dict(model.output_dict.get('variable'))
        self.objective_function_value = model.best_function
        return model.output_dict

    def run_configured_3d_view_generation(self, view_id='0', visualize=False, no_optimization=False):
        """
        Runs function that creates 3D view with optimized parameters acquired during device calibration
        :param view_id: id (string) appended to name of generated point cloud file
        :param visualize: whether you want to see visualized result or not
        :param no_optimization: True if you do not want to use optimized parameters
        :return: Tuple(array of xyz points of real 3D object; Tuple(xyz coordinate of first ray source; second);
                 Tuple(array of xyz points of first detector plane; second))
        """
        opt = self.optimized_params
        if no_optimization:
            opt = None
        return imageFormation.generate_3d_view(self.img_1_points, self.img_2_points, self.alpha_2, self.beta_2,
                                               self.sid_1, self.sod_1, self.sid_2, self.sod_2,
                                               self.corresponding_points, self.image_sizes, view_id, visualize, opt,
                                               self.pixel_spacing)
