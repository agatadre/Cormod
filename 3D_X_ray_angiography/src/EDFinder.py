import operator
import dicomTools as dct
import numpy as np
import torch
import segmentation_models_pytorch as smp
import data_process_utils as prep
import corr_funcs_jit as corr
from scipy import optimize, ndimage, signal
import math
'''
Most of the functionality is based on: "Extraction of the Best Frames in Coronary Angiograms for Diagnosis and Analysis" M.Dehkordi
and "Nonrigid Image Registration in Digital Subtraction Angiography Using Multilevel B-Spline" M. Nejati
and their references
'''
class EDFinder:
    def __init__(self, dcm_path, cp_params=None, imgs=None, masks=None) -> None:
        '''
        :param dcm_path: path to a dicom file
        :param cp_params: dictionary with experimentally fitted variables if None default set is used
            {"cutoff": minimum histogram's population percentage to use as an extrema in normalization,
             "pop_fractione": fraction of used pixels after strong edge filtering,
             "d_min": minimum distance between control points
             "max_cp": maximum number of control points}
        :param imgs: numpy array with n gray scale images, if None then gray scale images are generated from a dicom file
        :param masks: numpy array with n binary mask, if None then masks will be generated based on gray scale images
        '''
        super().__init__()
        self._vessels_masks = [] if masks is None else masks
        self._gray_frames_cv = [] if imgs is None else imgs
        self._vessel_model = None
        if cp_params is None:
            self.params = {"cutoff": 0.0, "pop_fraction": 0.0001, "d_min": 20, "max_cp": np.inf}
        else:
            self.params = cp_params
        self.dcm_tool = dct.DicomFile(dcm_path)

    def get_end_diastole_frame(self, w=26, dx_max=30, dy_max=30, frame_low=None, frame_up=None, all_extemas=False, optimization=False):
        '''
        :param w: template matching window size - optimal value is in (19, 30)
        :param dx_max: maximum local displacement searched in x-axis
        :param dy_max: maximum local displacement searched in y-axis
        :param frame_low: first and low boundary frame from a view used to be checked as the ED frame if None: auto boundary set
        :param frame_up: last and up boundary frame from a view used to be checked as the ED frame if None: auto boundary set
        :param all_extemas: if False: returns the frame with maximum mean distance else function returns array of all potential ED frames
        :param optimization: if False: use computational search else: use Powells' optimization algorith - less accurate but faster for greater (w, dx, dy) paramaters
        :return: single frame or array of the most potential frames to be ED and their images in grayscale
        '''
        self.w_size = w
        self.dx_max = dx_max
        self.dy_max = dy_max
        self.optimization = optimization
        if self._gray_frames_cv == []:
            self._gray_frames_cv = self.dcm_tool.get_cv_grayscale_imgs()
        best_frame_idx = self._get_best_frame_idx()
        if frame_low is None or frame_up is None:
            first_f_idx, last_f_idx = self._boundry_contrast_frames_idx(best_frame_idx)
        else:
            first_f_idx, last_f_idx = frame_low, frame_up
        strong_edges = self._obtain_strong_edges(best_frame_idx)
        ctrl_pts = self._obtain_ctrl_pts(strong_edges, best_frame_idx)
        state_features = self._calc_state_features(ctrl_pts, best_frame_idx, first_f_idx, last_f_idx)
        best_idx = self._return_best_frames_idx(state_features, all_extemas)
        shifted_idx = np.add(best_idx, first_f_idx)
        return shifted_idx, self._gray_frames_cv[shifted_idx]

    def _get_best_frame_idx(self):
        '''
        Create binary masks from images, based on number of pixels return the most contrast filled frame
        :return: index with the most contrast filled frame
        '''
        if (len(self._gray_frames_cv) == 0):
            raise("No frames to operate on")
        if self._vessels_masks == []:
            self._vessels_masks = self.create_vessel_masks(self._gray_frames_cv)
        vessel_sizes = np.sum(self._vessels_masks, axis=(1,2))
        return np.argmax(vessel_sizes)

    def create_vessel_masks(self, gray_scale_frames):
        '''
        :param gray_scale_frames: array shape:(n, 512, 512) with pngs generated using opencv's gray scale
        :return: ndarray with binary masks shape(n, 512, 512)
        '''
        self._load_vessel_model()
        masks = [self._get_vessel_tree(view) for view in gray_scale_frames]
        return np.asarray(masks)

    def _load_vessel_model(self):
        backbone = "resnet34"
        self._vessel_model = smp.Unet(backbone, classes=1)
        first_layer = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                                      padding=(3, 3), bias=False)
        self._vessel_model.encoder.conv1 = first_layer
        weights = torch.load("../res/kernel_ckpt/epoch=56-avg_val_dice_thresh=0.81.ckpt")["state_dict"]

        self._vessel_model.load_state_dict({k.replace("model.", ""): v for k, v in weights.items()})
        self._vessel_model.eval()

    def _get_vessel_tree(self, view):
        img = torch.tensor(self._preprocess_input(view[np.newaxis, np.newaxis, :, :])).float()
        with torch.no_grad():
            output = self._vessel_model(img)
            output = torch.sigmoid(output).round().cpu().numpy()
            output = np.squeeze(output)
            return np.asarray(output * 255, dtype=np.uint8)

    def _preprocess_input(self, x):
        xp = x / 255.
        xp -= 0.5
        xp *= 2.
        return xp

    def _boundry_contrast_frames_idx(self, best_frame_idx):
        '''
        Based on number of frames per second rate we estimate average heart beat.
        With average heart beat, number of all frames and with best frame, we estimate boundary within we search
        for end-diastole frame
        :param best_frame_idx: index of the most contrast filled frame
        :return: first and last indexes as boundary to search
        '''
        fps = self.dcm_tool.get_number_fps()
        all_frames_no = len(self._gray_frames_cv)
        common_sec_per_beat_l = 1.0
        min_full_heart_cycle = common_sec_per_beat_l * fps * 1.2
        upper_bound_mult = (all_frames_no - (best_frame_idx+1))/all_frames_no
        lower_bound_mult =  (best_frame_idx+1)/all_frames_no
        return int(best_frame_idx - min_full_heart_cycle*lower_bound_mult),\
               int(best_frame_idx + min_full_heart_cycle*upper_bound_mult)

    def _obtain_strong_edges(self, best_frame_idx):
        '''
        X-Y axis Gaussian filter helps to obtain strong edges
        Normalized array will cut pixels under threshold, calculated with predefined population fraction
        :param best_frame_idx: index of best frame in array of all frames
        :return: ndarray (512,512), values [0,1]
        '''
        gradient = ndimage.gaussian_filter(self._gray_frames_cv[best_frame_idx], [1, 1], [1, 1])
        norm_gradient = prep.normalize(gradient, cutoff=self.params["cutoff"])
        norm_hist, norm_bins = np.histogram(norm_gradient, bins=256)
        threshold_i = self._find_threshold_index(norm_gradient, self.params["pop_percentage"])
        norm_gradient[norm_gradient < norm_bins[threshold_i]] = 0
        return norm_gradient

    def _find_threshold_index(self, frame, pop_fraction):
        hist, bins = np.histogram(frame, bins=256)
        hist, bins = hist[::-1], bins[::-1]
        all_pixels = frame.shape[0] ** 2
        sum_pop = 0
        for i, val in enumerate(bins):
            sum_pop += hist[i]
            if sum_pop >= int(all_pixels * pop_fraction):
                return hist.shape[0] - i - 1

    def _obtain_ctrl_pts(self, strong_edges, frame_idx):
        '''
        To maximalize reliability of control points kernel mask is used to merge edges with vessel tree
        Since the strongest edges, in 2D array, are the most probable to be a vessel, so we sort them by value.
        :param strong_edges: ndarray (512,512) values [0,1]
        :param frame_idx: index of the most contrast filled frame
        :return: ndarray of tuples with position (x,y) of a control point - shape (n, 2)
        '''
        vessel_tree = self._vessels_masks[frame_idx]
        listed_gradient = self._convert_to_tuple_flat_list(strong_edges)
        listed_gradient.sort(key=operator.itemgetter(0))
        ctrl_pts = self._get_control_points(listed_gradient, d_min=self.params["d_min"], max_cp=self.params["max_cp"])
        return(self._merge_points(vessel_tree, ctrl_pts))

    def _convert_to_tuple_flat_list(self, arr_2d):
        list_1d = []
        for i in range(arr_2d.shape[0]):
            for j in range(arr_2d.shape[1]):
                if arr_2d[i][j] != 0:
                    list_1d.append((arr_2d[i][j], (i, j)))
        return list_1d

    def _get_control_points(self, sorted_list, d_min=1, max_cp=np.Inf):
        '''
        :param sorted_list: sorted pixels by value; shape:(n,2); (n, (pixel_value, (x,y))
        :param d_min: minimum distance from a point to other so it may be used
        :param max_cp: maximum number of points, if exceeded return previously collected points
        :return: ndarray of tuples (x,y): shape (n,2)
        '''
        cp = []
        inserted_cp = {}
        inserted_cp[str(sorted_list[0])] = 1
        cp.append((sorted_list[0][1][0], sorted_list[0][1][1]))
        for p in sorted_list:
            x, y = p[1][0], p[1][1]
            if len(inserted_cp) >= max_cp:
                return cp
            if p not in inserted_cp and self._moved_from_border(p[1]) and self._distant_enough(p[1], cp, d_min):
                inserted_cp[str(p)] = 1
                cp.append((x, y))
        return np.asarray(cp)

    def _moved_from_border(self, p):
        max_border = self._gray_frames_cv[0].shape[0]
        return p[0]>self.w_size and p[0]<max_border-self.w_size and p[1]>self.w_size and p[1]<max_border-self.w_size

    def _distant_enough(self, p, cp, d_min):
        if d_min == 1:
            return True
        for p2 in cp:
            x1, y1 = p[0], p[1]
            x2, y2 = p2[0], p2[1]
            if x1 == p2[0] and y1 == p2[1]:
                continue
            if math.sqrt((x1-x2)**2 + (y1-y2)**2) < d_min:
                return False
        return True

    def _merge_points(self, mask_img, edges, radius=1):
        '''
        Combine generate vessel tree kernel with Gaussian filter strong edges to increase reliability of control points
        If in 'radius' range any vessel tree's pixel is present next to a strong edge then use this strong edge
        :param mask_img: binary image (512,512) of a vessel tree
        :param edges: (x,y) points
        :param radius: size of mask which sums all pixels within its window
        :return: ndarray shape:(n,2) with (x,y) of points
        '''
        matched_points = []
        for row, col in edges:
            mask = prep.get_correct_slice(mask_img, col, row, radius, radius)
            if np.sum(mask) > 0:
                matched_points.append((col, row))
        return np.asarray(matched_points)

    def _calc_state_features(self, ctrl_pts, best_f_idx, first_f_idx, last_f_idx):
        '''
        :param ctrl_pts: ndarray with position (x,y) of control points - shape (n, 2)
        :param best_f_idx: index of the most contrast filled frame
        :param first_f_idx: first frame to be calculated
        :param last_f_idx: last frame to be calculated
        :return: ndarray shape: (n) with calculated mean distance between control points
        '''
        state_features = []
        state_features.append(self._calc_mean_distance(ctrl_pts))
        basic_ctrl_pts = ctrl_pts
        prev_frame_idx = best_f_idx
        for i in range(best_f_idx, last_f_idx):
            curr_frame_idx = i + 1
            corr_pts = self._find_corresponding_points(curr_frame_idx, prev_frame_idx, basic_ctrl_pts)
            state_features.append(self._calc_mean_distance(corr_pts))
            basic_ctrl_pts = corr_pts
            prev_frame_idx = curr_frame_idx
        state_features.reverse()
        basic_ctrl_pts = ctrl_pts
        prev_frame_idx = best_f_idx
        for i in range(best_f_idx, first_f_idx, -1):
            curr_frame_idx = i - 1
            corr_pts = self._find_corresponding_points(curr_frame_idx, prev_frame_idx, basic_ctrl_pts)
            state_features.append(self._calc_mean_distance(corr_pts))
            basic_ctrl_pts = corr_pts
            prev_frame_idx = curr_frame_idx
        state_features.reverse()
        return np.asarray(state_features)

    def _find_corresponding_points(self, curr_frame_idx, prev_frame_idx, ctrl_pts):
        opt_d = []
        for i in range(len(ctrl_pts)):
            opt_d_i = self._find_corresponding_point(curr_frame_idx, prev_frame_idx, ctrl_pts[i])
            opt_d.append(opt_d_i)
        return np.asarray(opt_d)

    def _find_corresponding_point(self, curr_frame_idx, prev_frame_idx, ctrl_p):
        '''
        There are 2 possible way to a calculate correspondence:
        - optimized, less reliable way based on an optimization algorithm
        - computational - more time consuming but more reliable to find better correspondence
        Both methods are based on template matching
        :param curr_frame_idx: index of a frame having correspondence calculated
        :param prev_frame_idx: index of a frame with a calculated control point
        :param ctrl_p: control point of a previous frame
        :return: ndarray with (x,y) position of a corresponding point
        '''
        curr_frame = self._gray_frames_cv[curr_frame_idx]
        prev_frame = self._gray_frames_cv[prev_frame_idx]
        if self.optimization:
            return self._optimized_correspondence(curr_frame, prev_frame, ctrl_p)
        else:
            return corr._global_correspondence(curr_frame, prev_frame, ctrl_p, self.w_size, self.dx_max, self.dy_max)

    def _calc_mean_distance(self, control_points):
        sum = 0
        for i in range(len(control_points)):
            for j in range(i, len(control_points)):
                x1, y1 = control_points[i][0], control_points[i][1]
                x2, y2 = control_points[j][0], control_points[j][1]
                sum += math.sqrt((x1-x2)**2 + (y1-y2)**2)
        N = len(control_points)
        return sum * 2 / (N * (N - 1))


    def _optimized_correspondence(self, curr_frame, prev_frame, ctrl_p):
        cp_x, cp_y = ctrl_p[0], ctrl_p[1]
        x_l, y_l, x_up, y_up = self._get_neighbourhood_borders(curr_frame, cp_x, cp_y)
        init_pts = [(cp_x+5, cp_y+5), (cp_x+5, cp_y-5), (cp_x-5, cp_y+5), (cp_x-5, cp_y-5), (cp_x, cp_y)]
        opt_d = []
        results = []
        for init_pt in init_pts:
            if init_pt[0] < x_l or init_pt[1] < y_l or init_pt[0] >= x_up or init_pt[1] >= y_up:
                init_pt = cp_x, cp_y
            res = optimize.minimize(self._entropy_min, x0=np.asarray([init_pt[0], init_pt[1]]),
                                    args=(prev_frame, curr_frame, ctrl_p),
                                    method='Powell', bounds=((x_l, x_up), (y_l, y_up)))
            results.append(res.fun)
            opt_d.append(res.x)
        min_ind = np.argmin(results)
        return opt_d[min_ind]

    def _get_neighbourhood_borders(self, img, x, y):
        shape_x = img.shape[0]
        shape_y = img.shape[1]

        x_l = np.maximum(0, x - self.dx_max)
        x_up = np.minimum(shape_x, x + self.dx_max)
        y_l = np.maximum(0, y - self.dy_max)
        y_up = np.minimum(shape_y, y + self.dy_max)
        return (x_l, y_l, x_up, y_up)

    def _entropy_min(self, d, contrast_fr, mask, cp):
        dx, dy = int(d[0]), int(d[1])
        x, y = cp[0], cp[1]
        W = prep.get_correct_slice(contrast_fr, x, y, self.w_size, self.w_size)
        W_m = prep.get_correct_slice(mask, dx, dy, self.w_size, self.w_size)
        W_diff = prep.get_correct_sub(W, W_m)
        return self._calc_entropy(W_diff)

    def _calc_entropy(self, W_diff):
        hist, bin = np.histogram(W_diff, bins=511, range=(-255, 256))
        non_zero_hist = hist[hist != 0]
        norm_hist = non_zero_hist/np.sum(non_zero_hist)
        sim = -np.sum(norm_hist* np.log(norm_hist))
        return sim

    def _return_best_frames_idx(self, state_features, all_extemas):
        extrema_idx = None
        if all_extemas:
            extrema_idx = signal.argrelextrema(state_features, np.greater)
        if all_extemas is False or extrema_idx is None:
            extrema_idx = np.argmax(state_features)
        return extrema_idx
