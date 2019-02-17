from NN2019.extract import embed_in_grid
from NN2019.extract import embed_in_grid


def smoothing_kernel(pdb_id, features, max_radius, n_features, bins_per_angstrom,
                     coordinate_system=grids.CoordinateSystem.cartesian, sigma=None, smoothing=None, output_dir="."):
    if sigma:
        omega = 1 / sigma

    embed_in_grid(features, pdb_id, output_dir,
                  max_radius=max_radius,
                  n_features=n_features,
                  bins_per_angstrom=bins_per_angstrom,
                  coordinate_system=coordinate_system,
                  z_direction=z_direction,
                  include_center=include_center))
