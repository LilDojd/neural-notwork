import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from scipy import signal

from Grids import grids

VdWs = {
    1.007947: 0.6,
    12.01078: 1.9,
    14.00672: 1.8,
    15.99943: 1.7,
    32.0655: 2.0
}


def construct_kernel(sigma, kernlen=None):
    omega = 1 / sigma

    if kernlen:
        pass
    else:
        kernlen = 8 * sigma + 1

    ax = np.arange(-kernlen // 2 + 1., kernlen // 2 + 1.)
    xx, yy, zz = np.meshgrid(ax, ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * sigma ** 2)) * np.cos(
        2 * np.pi * omega * np.sqrt(xx ** 2 + yy ** 2 + zz ** 2))

    return kernel


# Construct cartesian grid, map features, blur them, and then unmap into coordinates
# Lets divide by atom names
def wave_transform_smoothing(features, kernel, n_bins=2):
    all_coords = structured_to_unstructured(features[['x', 'y', 'z']], dtype=np.float32)
    all_coords.reshape(all_coords.shape[0], 3)
    box_grid, max_val, min_val = grids.create_cartesian_box(all_coords, n_bins)

    assert (len(box_grid.shape) == 4)
    assert (box_grid.shape[0] == box_grid.shape[1] == box_grid.shape[2])
    n_b = box_grid.shape[0]

    boundaries = np.linspace(np.floor(min_val - 5), np.ceil(max_val + 5), n_b, endpoint=False)
    boundaries += (boundaries[1] - boundaries[0])

    indexx = np.digitize(all_coords, boundaries).reshape(all_coords.shape[0], 3)

    for res in np.unique(features[['res_index']]):
        for mass in np.unique(features[['mass']]):
            pass

    for ind, row in enumerate(indexx[:]):
        box_grid[row[0], row[1], row[2]] = features[['charge']][ind][0][0]

    box_grid = box_grid.reshape(box_grid.shape[0], box_grid.shape[1], box_grid.shape[2])
    smoothed = signal.convolve(box_grid, kernel, mode="same")
    limiter = np.absolute(smoothed) < np.power(10., -5)
    smoothed[limiter] = 0

    return smoothed


def unmap(new_box, boundaries):
    pass
