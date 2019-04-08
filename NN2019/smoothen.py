import functools

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from scipy import signal

from Grids import grids

VdWs = {
    np.float32(1.007947): 0.6,
    np.float32(12.01078): 1.9,
    np.float32(14.00672): 1.8,
    np.float32(15.99943): 1.7,
    np.float32(32.0655): 2.0
}


@functools.lru_cache()
def construct_kernel(radius, kernlen=None, n_bins=4):
    sigma = radius * n_bins / 2

    omega = 1 / sigma
    if kernlen:
        pass
    else:
        kernlen = np.ceil((radius * 2) * n_bins)

    ax = np.arange(-kernlen // 2 + 1., kernlen // 2 + 1., 1)
    xx, yy, zz = np.meshgrid(ax, ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * sigma ** 2)) * np.cos(
        2 * np.pi * omega * np.sqrt(xx ** 2 + yy ** 2 + zz ** 2))

    return kernel


def fft_conv(atom_mass, atom_grid, n_bins):
    kern = construct_kernel(VdWs[atom_mass], n_bins=n_bins)
    smooth = signal.fftconvolve(atom_grid, kern, mode="same")

    return smooth


# Construct cartesian grid, map features, blur them, and then unmap into coordinates
# Lets separate by atom names

def wave_transform_smoothing(features, n_bins=4):
    all_coords = structured_to_unstructured(features[['x', 'y', 'z']], dtype=np.float32)
    all_coords.reshape(all_coords.shape[0], 3)
    box_grid, max_val, min_val = grids.create_cartesian_box(all_coords, n_bins)

    assert (len(box_grid.shape) == 3)
    assert (box_grid.shape[0] == box_grid.shape[1] == box_grid.shape[2])
    n_b = box_grid.shape[0]

    boundaries = np.linspace(np.floor(min_val - 5), np.ceil(max_val + 5), n_b, endpoint=False)
    boundaries += (boundaries[1] - boundaries[0])

    for atom in np.unique(features['mass']):
        box_temp = np.zeros_like(box_grid)
        # extract features unique for all atom types
        atom_feat = features[features['mass'] == atom]
        atom_coords = structured_to_unstructured(atom_feat[['x', 'y', 'z']], dtype=np.float32)
        indexx = np.digitize(atom_coords, boundaries)
        # Vectorize somehow
        for ind, row in enumerate(indexx[:]):
            box_temp[row[0], row[1], row[2]] = atom_feat[ind]['charge']
        box_grid += fft_conv(atom, np.squeeze(box_temp), n_bins)

    # for mass in np.unique(features[['mass']]):
    #     sigma_hradius = np.ceil(VdWs[mass[0]] * n_bins) / 4
    #     kernel = construct_kernel(sigma_hradius)

    limiter = np.absolute(box_grid) < np.power(10., -5)
    box_grid[limiter] = 0

    assert (box_grid.shape == (n_b, n_b, n_b))
    return box_grid


def unmap(new_box, boundaries):
    pass
