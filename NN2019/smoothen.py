import grids


def smoothing_kernel(pdb_id, features, max_radius, n_features, bins_per_angstrom,
                     coordinate_system=grids.CoordinateSystem.cartesian, sigma=None, smoothing=None, output_dir="."):
    if sigma:
        omega = 1 / sigma

    pass
