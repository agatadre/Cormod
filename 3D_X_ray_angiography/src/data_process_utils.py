import numpy as np
from numba import njit

def __get_extremum(view, cutoff, maximum=True):
    hist, bins = np.histogram(view, bins=256)
    met_population = 0
    extrema = np.max(view) if maximum else np.min(view)
    if cutoff == 0:
        return extrema
    if maximum:
        hist, bins = hist[::-1], bins[::-1]
    for i, val in enumerate(bins):
        met_population += hist[i]
        if met_population >= cutoff:
            return val
    return extrema

def __adjust_outliers(view, min, max):
    view[view <= min] = min
    view[view >= max] = max
    return view

def normalize(view, up_l=1, down_l=0, cutoff=0.0):
    upper_limit = up_l
    down_limit = down_l

    pixels_no = view.shape[0]**2
    cutoff = int(pixels_no * cutoff)

    min_val = __get_extremum(view, cutoff, maximum=False)
    max_val = __get_extremum(view, cutoff, maximum=True)
    view = __adjust_outliers(view, min_val, max_val)

    return np.subtract(view, min_val)*((upper_limit-down_limit)/(max_val-min_val)) + down_limit

@njit
def get_correct_sub(W, W_m):
    W_s = W.shape
    W_m_s = W_m.shape
    x_bord = int(np.minimum(W_s[1], W_m_s[1]))
    y_bord = int(np.minimum(W_s[0], W_m_s[0]))
    return np.subtract(W[:y_bord, :x_bord], W_m[:y_bord, :x_bord])

@njit
def get_correct_slice(arr2d, x, y, shift_x, shift_y):
    shape_x = arr2d.shape[0]
    shape_y = arr2d.shape[1]

    abs_shift_x = np.abs(shift_x)
    abs_shift_y = np.abs(shift_y)
    slice_x_l = int(np.maximum(0, x-abs_shift_x))
    slice_x_up = int(np.minimum(shape_x, x+abs_shift_x))
    slice_y_l = int(np.maximum(0, y-abs_shift_y))
    slice_y_up = int(np.minimum(shape_y, y+abs_shift_y))
    return arr2d[slice_y_l:slice_y_up, slice_x_l:slice_x_up]
