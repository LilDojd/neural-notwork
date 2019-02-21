"""Stored in a separate module so it can be readily accessed both by feature extraction and model. Modified version
of 2017 Jes Frellsen and Wouter Boomsma module

Some additional opportunities:
https://www.researchgate.net/publication/224017908_The_Evolution_of_Dynamical_Cores_for_Global_Atmospheric_Models
/figures?lo=1"""

import enum
import sys

import Bio.PDB
import numpy as np

sys.path.append('..')

# noinspection PyArgumentList
CoordinateSystem = enum.Enum("CoordinateSystem", {"spherical": 1, "cubed_sphere": 2, "cartesian": 3})
# noinspection PyArgumentList
ZDirection = enum.Enum("ZDirection", {"sidechain": 1, "backbone": 2, "outward": 3})


def get_spherical_grid_shape(max_radius, n_features, bins_per_angstrom):
    """Defines shape of grid matrix"""
    return (int(np.ceil(bins_per_angstrom * max_radius)),  # Shell num
            int(np.ceil(bins_per_angstrom * max_radius * np.pi)),
            int(np.ceil(bins_per_angstrom * max_radius * 2 * np.pi)),
            n_features)


def create_spherical_grid(max_radius, n_features, bins_per_angstrom):
    """Creates empty spherical grid"""

    grid_matrix = np.zeros(shape=get_spherical_grid_shape(max_radius, n_features, bins_per_angstrom))

    return grid_matrix


def get_cubed_sphere_grid_shape(max_radius, n_features, bins_per_angstrom):
    """Defines shape of grid matrix
    See http://acmg.seas.harvard.edu/geos/cubed_sphere/CubeSphere_comparision.html
    http://acmg.seas.harvard.edu/geos/cubed_sphere/CubeSphere_step-by-step.html for reference"""
    return (6,
            int(np.ceil(bins_per_angstrom * max_radius)),
            int(np.ceil(bins_per_angstrom * max_radius * np.pi / 2)),
            int(np.ceil(bins_per_angstrom * max_radius * np.pi / 2)),
            n_features)


def create_cubed_sphere_grid(max_radius, n_features, bins_per_angstrom):
    """Creates cubed sphere grid"""

    grid_matrix = np.zeros(shape=get_cubed_sphere_grid_shape(max_radius, n_features, bins_per_angstrom))

    return grid_matrix


def get_cartesian_grid_shape(max_radius, n_features, bins_per_angstrom):
    """Defines shape of grid matrix"""
    return 3 * (int(np.ceil(2 * bins_per_angstrom * max_radius)),) + (n_features,)


def create_cartesian_grid(max_radius, n_features, bins_per_angstrom):
    """Creates cartesian grid"""

    grid_matrix = np.zeros(shape=get_cartesian_grid_shape(max_radius, n_features, bins_per_angstrom))

    return grid_matrix


get_grid_shape_map = {CoordinateSystem.spherical: get_spherical_grid_shape,
                      CoordinateSystem.cubed_sphere: get_cubed_sphere_grid_shape,
                      CoordinateSystem.cartesian: get_cartesian_grid_shape}

create_grid_map = {CoordinateSystem.spherical: create_spherical_grid,
                   CoordinateSystem.cubed_sphere: create_cubed_sphere_grid,
                   CoordinateSystem.cartesian: create_cartesian_grid}


def define_coordinate_system(pos_N, pos_CA, pos_C, z_direction):
    """Defines a local reference system based on N, CA, and C atom positions"""

    # Define local coordinate system
    e1 = (pos_C - pos_N)
    # Normalised vector e1 (backbone direction)
    e1 /= np.linalg.norm(e1)

    # Define CB positions by rotating N atoms around CA-C axis 120 degr
    pos_N_res = pos_N - pos_CA
    axis = pos_CA - pos_C
    # Calculate left rotational matrix that rotates 120 degr around axis vector
    pos_CB = np.dot(Bio.PDB.rotaxis((120. / 180.) * np.pi, Bio.PDB.vectors.Vector(axis[0, :])), pos_N_res.T)
    e2 = pos_CB
    e2 /= np.linalg.norm(e2)
    e2 = e2.T
    e3 = np.cross(e1, e2)

    # N-C and e2 are not perfectly perpendicular to one another. We adjust e2.
    e2 = np.cross(e1, -e3)

    if z_direction == ZDirection.outward:
        # Use e3 as z-direction
        rot_matrix = np.array([e1, e2, e3])
    elif z_direction == ZDirection.backbone:
        # Use backbone direction as z-direction
        rot_matrix = np.array([e2, e3, e1])
    elif z_direction == ZDirection.sidechain:
        # Use sidechain direction as z-direction
        rot_matrix = np.array([e3, e1, e2])
    else:
        raise Exception("Unknown z-direction ")

    return rot_matrix


def cartesian_to_spherical_coordinates(xyz):
    """Convert set of Cartesian coordinates to spherical-polar coordinates"""

    # Convert to spherical coordinates
    r = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2 + xyz[:, 2] ** 2)
    theta = np.arctan2(np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2), xyz[:, 2])  # polar angle - inclination from z-axis
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])

    return r, theta, phi


def discretize_into_spherical_bins(r, theta, phi, max_r,
                                   r_shape, theta_shape, phi_shape):
    """Map r, theta, phi values to discrete grid bin"""

    # Bin each dimension independently
    r_boundaries = np.linspace(0, max_r, r_shape, endpoint=False)
    r_boundaries += (max_r - r_boundaries[-1])
    theta_boundaries = np.linspace(0, np.pi, theta_shape, endpoint=False)
    theta_boundaries += (theta_boundaries[1] - theta_boundaries[0])
    phi_boundaries = np.linspace(-np.pi, np.pi, phi_shape, endpoint=False)
    phi_boundaries += (phi_boundaries[1] - phi_boundaries[0])
    r_bin = np.digitize(r, r_boundaries)
    # Return the indices of the bins to which each value in input array belongs.
    theta_bin = np.digitize(theta, theta_boundaries)
    phi_bin = np.digitize(phi, phi_boundaries)

    # For phi angle, check for periodicity issues
    # When phi=pi, it will be mapped to the wrong bin
    phi_bin[phi_bin == phi_shape] = 0

    # Disallow any larger phi angles
    assert (not np.any(phi_bin > phi_shape))
    assert (not np.any(theta_bin > theta_shape))

    return r_bin, theta_bin, phi_bin


def discretize_into_cubed_sphere_bins(patch, r, xi, eta, max_r,
                                      r_shape, xi_shape, eta_shape):
    """Map r, theta, phi values to discrete grid bin"""

    # Bin each dimension independently
    r_boundaries = np.linspace(0, max_r, r_shape, endpoint=False)
    r_boundaries += (max_r - r_boundaries[-1])
    xi_boundaries = np.linspace(-np.pi / 4, np.pi / 4, xi_shape, endpoint=False)
    xi_boundaries += (xi_boundaries[1] - xi_boundaries[0])
    eta_boundaries = np.linspace(-np.pi / 4, np.pi / 4, eta_shape, endpoint=False)
    eta_boundaries += (eta_boundaries[1] - eta_boundaries[0])
    r_bin = np.digitize(r, r_boundaries)
    xi_bin = np.digitize(xi, xi_boundaries)
    eta_bin = np.digitize(eta, eta_boundaries)

    # Disallow any larger xi, eta angles
    assert (not np.any(r_bin < 0))
    assert (not np.any(xi_bin < 0))
    assert (not np.any(eta_bin < 0))
    assert (not np.any(xi_bin > xi_shape))
    assert (not np.any(eta_bin > eta_shape))

    return patch, r_bin, xi_bin, eta_bin


def cartesian_to_cubed_sphere(x, y, z, rtol=1e-05):
    """Convert set of Cartesian coordinates to cubed-sphere coordinates
    Consists of 6 coordinate systems, gnomonic coordinates are used in calculation

    Consult 10.1006/jcph.1996.0047 for rules of transformations"""

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    if r < rtol:
        patch = 0
        xi = 0.
        eta = 0.

    elif x >= np.abs(y) and x >= np.abs(z):
        # Front patch (I)
        patch = 0
        xi = np.arctan2(y, x)
        eta = np.arctan2(z, x)

    elif y >= np.abs(x) and y >= np.abs(z):
        # East patch (II)
        patch = 1
        xi = np.arctan2(-x, y)
        eta = np.arctan2(z, y)

    elif -x >= np.abs(y) and -x >= np.abs(z):
        # Back patch (III)
        patch = 2
        xi = np.arctan2(y, x)
        eta = np.arctan2(-z, x)

    elif -y >= np.abs(x) and -y >= np.abs(z):
        # West  patch (IV)
        patch = 3
        xi = np.arctan2(-x, y)
        eta = np.arctan2(-z, y)

    elif z >= np.abs(x) and z >= np.abs(y):
        # North patch (V)
        patch = 4
        xi = np.arctan2(y, z)
        eta = np.arctan2(-x, z)

    elif -z >= np.abs(x) and -z >= np.abs(y):
        # South patch (VI)
        patch = 5
        xi = np.arctan2(-y, z)
        eta = np.arctan2(-x, z)

    else:
        raise ArithmeticError("Should never happen")

    return patch, r, xi, eta


# Vectorized version of cartesian_to_cubed_sphere
cartesian_to_cubed_sphere_vectorized = np.vectorize(cartesian_to_cubed_sphere,
                                                    otypes=[np.int, np.float, np.float, np.float])


def discretize_into_cartesian_bins(xyz, max_radius, shape):
    """Map x,y,z values to discrete grid bin"""

    assert (len(shape) == 4)
    assert (shape[0] == shape[1] == shape[2])

    n_bins = shape[0]
    boundaries = np.linspace(-max_radius, max_radius, n_bins, endpoint=False)
    boundaries += (boundaries[1] - boundaries[0])

    indices = np.digitize(xyz, boundaries)

    return indices


def cubed_sphere_to_unfolded_plane(patch, xi, eta, offsets=np.array([[1, 1], [2, 1], [3, 1], [0, 1], [1, 2], [1, 0]])):
    r"""Unfold points on the cubed sphere into a plane.

    Args:
        patch: `int`. Cube face.
        xi: `int`. Xi value.
        eta: `int`. Eta value.
        offsets: `numpy.array` of X,Y offsets


    Returns:
        A `numpy.array` of points in the XY plane.

    """
    return offsets[patch] + (np.array([xi, eta]) + np.pi / 4) / (np.pi / 2)


if __name__ == "__main__":

    import numpy as np
    import matplotlib
    import matplotlib.patches

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create and plot function on the sphere
    c = lambda t, a: np.array([np.cos(t), np.sin(t), -a * t]) / np.sqrt(1 + a ** 2 * t ** 2)
    p = np.transpose(np.array([c(t, .1) for t in np.linspace(-40, 40, 1000)]))
    ax.plot(p[0, :], p[1, :], p[2, :])
    fig.savefig("plot_3d.png")
    plt.close()

    # Plot same function in unfolded representation
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    p_cubed = [cartesian_to_cubed_sphere(*v) for v in np.transpose(p)]
    p_unfolded_plane = np.array([cubed_sphere_to_unfolded_plane(v[0], v[2], v[3]) for v in p_cubed])

    offsets = np.array([[1, 1], [2, 1], [3, 1], [0, 1], [1, 2], [1, 0]])
    for v in offsets:
        ax.add_patch(matplotlib.patches.Rectangle(v, 1., 1., fill=False))

    colors = plt.get_cmap("hsv")(np.linspace(0, 1, p_unfolded_plane.shape[0] - 1))

    for i in range(p_unfolded_plane.shape[0] - 1):
        if p_cubed[i][0] == p_cubed[i + 1][0]:
            linestyle = '-'
        else:
            linestyle = ':'

        ax.plot(p_unfolded_plane[i:i + 2, 0], p_unfolded_plane[i:i + 2, 1], color=colors[i], linestyle=linestyle)

    plt.axis('off')
    e = 0.1
    ax.set_xlim(0 - e, 4 + e)
    ax.set_ylim(0 - e, 3 + e)

    fig.savefig("plot_2d.png")
    plt.close()