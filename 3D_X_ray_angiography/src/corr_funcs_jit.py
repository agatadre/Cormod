import numpy as np
import data_process_utils as prep
from numba import njit

@njit
def _global_correspondence(mask, contrast_fr, ctrl_p, w, dx_max, dy_max):
    min_e = np.inf
    opt_d = (0, 0)
    x = ctrl_p[0]
    y = ctrl_p[1]

    W = prep.get_correct_slice(contrast_fr, x, y, w, w)
    dx = [i for i in range(-dx_max, dx_max)]
    dy = [i for i in range(-dy_max, dy_max)]
    for r in dx:
        for s in dy:
            if x+r-w < 0 or x+r+w > mask.shape[1] or y+s-w < 0 or y+s+w > mask.shape[0]:
                continue
            else:
                W_m = prep.get_correct_slice(mask, x + r, y + s, w, w)
                W_diff = prep.get_correct_sub(W, W_m)
                sim = calc_entropy(W_diff)
                min_e = np.minimum(sim, min_e)
                if min_e == sim:
                    opt_d = (r, s)
    return np.asarray(opt_d, dtype=np.int32)+ctrl_p.astype(np.int32)


@njit
def calc_entropy(W_diff):
    hist, bin = np.histogram(W_diff, bins=511, range=(-255, 256))
    non_zero_hist = hist[hist != 0]
    norm_hist = non_zero_hist/np.sum(non_zero_hist)
    sim = -np.sum(norm_hist* np.log(norm_hist))
    return sim
