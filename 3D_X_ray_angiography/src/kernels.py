import numpy as np


def get_branch_kernels():
    """ Return a list of kernels specific to branch points in all possible
            orientations.
    """
    kernels = np.array([
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 0]
        ],
        [
            [1, 0, 0],
            [0, 1, 1],
            [0, 1, 0]
        ],
        [
            [0, 0, 1],
            [1, 1, 0],
            [0, 0, 1]
        ],
        [
            [1, 0, 1],
            [0, 1, 0],
            [0, 0, 1]
        ]])

    return rotate_kernels(kernels)


def get_support_kernels():
    """ Return a list of support kernels (which are sub-matrices of branch
            kernels) - useful in one of the algorithm steps.
    """
    kernels = np.array([
        [
            [0, 0, 0],
            [1, 1, 0],
            [0, 0, 0]
        ],
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]])

    return rotate_kernels(kernels)


def rotate_kernels(kernels):
    """ Create a new list with rotated kernels.
    :param kernels: A list of kernels.
    :return: A list of kernels in all possible orientations with
        length = 4*len(kernels).
    """
    rotated_kernels = []
    for j, kernel in enumerate(kernels):
        for _ in range(4):
            rotated_kernels.append(kernel)
            kernel = np.rot90(kernel)
    return rotated_kernels