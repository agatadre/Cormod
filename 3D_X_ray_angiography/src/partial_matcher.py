from decimal import Decimal
import numpy as np
import deviceConfiguration as dc
from osb import findpathDAG

def find_matching_views(prim_views, sec_views, corresponding_points,
                        alpha_diff, beta_diff, sid1, sod1, sid2, sod2, XA):
    """
    Function to find the most similar frames based on the cardiac phase
    :param prim_views: n 2D bifurcation points over m frames shape:(m,n,2)
    :param sec_views: n 2D bifurcation points over p frames  shape:(p,n,2)
    :param corresponding_points: array containing indexes of corresponding points between views
    :param alpha_diff: alpha (LAO[+180] - RAO[-180]) angle in radians between views
    :param beta_diff: beta (CRA[+90] - CAU[-90]) angle in radians between views
    :param sid1: Distance Source to Detector (plane) of the first view
    :param sid2: Distance Source to Detector (plane) of the second view
    :param sod1: Distance Source to Patient (object) of the first view
    :param sod2: Distance Source to Patient (object) of the second view
    :param XA: array of 11 camera calibration parameters
    :return: two arrays of the best matching views
    """

    m = len(prim_views)
    p = len(sec_views)
    matx_dists = np.zeros([m,p])

    for k in range(m):
        for l in range(p):
            matx_dists[k][l] = _calc_dissimilarity(prim_views[k], sec_views[l],
                                                   corresponding_points, alpha_diff, beta_diff,
                                                   sid1, sid2, sod1, sod2, XA)
    min_dists = np.min(matx_dists, axis=1)
    jumpcost = np.min(min_dists) + np.std(min_dists)

    matxE_dists = np.ones([m + 2, p + 2], dtype=int) * Decimal('inf')

    matxE_dists[1:m+1, 1:p+1] = matx_dists
    matxE_dists[0, 0] = 0
    matxE_dists[m + 1, p + 1] = 0

    _, indxrow, indxcol = findpathDAG(matxE_dists, Decimal("Inf"), Decimal("Inf"), Decimal("Inf"), jumpcost)
    return indxrow, indxcol

def _calc_dissimilarity(prim_view, sec_view, corresponding_points,
                        alpha_diff, beta_diff, sid1, sid2, sod1, sod2, XA):

    conf = dc.DeviceConfiguration(prim_view, sec_view, corresponding_points,
                                  sid1, sod1, sid2, sod2, alpha_diff, beta_diff)

    return conf.min_function(XA)
