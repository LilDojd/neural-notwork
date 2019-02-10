"""Function that embedes features into the grid.
Heavily modified version of 2017 Jes Frellsen and Wouter Boomsma script"""

import os

import numpy as np

from . import grids


def embed_in_grid(features, pdb_id, output_dir,
                  max_radius,
                  n_features,
                  bins_per_angstrom,
                  coordinate_system,
                  z_direction,
                  include_center):
    """Embed masses and charge information in a spherical grid - specific for each residue
       For space-reasons, only the indices into these grids are stored, and a selector
       specifying which atoms are relevant (i.e. within range) for the current residue.
    """

    # Extract coordinates as normal numpy array
    global indices, r
    position_array = features[['x', 'y', 'z']].view(np.float32)

    # Retrieve residue indices as numpy int array
    res_indices = features[['res_index']].view(int)

    selector_list = []
    indices_list = []
    for residue_index in np.unique(features[['res_index']].view(int)):

        # Extract origin
        if (np.logical_and(res_indices == residue_index, features[['name']].view('a5') == b"N").any() and
                np.logical_and(res_indices == residue_index, features[['name']].view('a5') == b"CA").any() and
                np.logical_and(res_indices == residue_index, features[['name']].view('a5') == b"C").any()):
            CA_feature = features[
                np.argmax(np.logical_and(res_indices == residue_index, features[['name']].view('a5') == b"CA"))]
            N_feature = features[
                np.argmax(np.logical_and(res_indices == residue_index, features[['name']].view('a5') == b"N"))]
            C_feature = features[
                np.argmax(np.logical_and(res_indices == residue_index, features[['name']].view('a5') == b"C"))]
        else:
            # Store None to maintain indices
            indices_list.append(None)
            selector_list.append(None)
            continue

        # Positions of N, CA and C atoms used to define local reference system
        pos_CA = CA_feature[['x', 'y', 'z']].view(np.float32)
        pos_N = N_feature[['x', 'y', 'z']].view(np.float32)
        pos_C = C_feature[['x', 'y', 'z']].view(np.float32)

        # Define local coordinate system
        rot_matrix = grids.define_coordinate_system(pos_N, pos_CA, pos_C, z_direction)

        # Calculate coordinates relative to origin
        xyz = position_array - pos_CA

        # Rotate to the local reference
        xyz = np.dot(rot_matrix, xyz.T).T

        if coordinate_system == grids.CoordinateSystem.spherical:

            # Convert to spherical coordinates
            r, theta, phi = grids.cartesian_to_spherical_coordinates(xyz)

            # Create grid
            grid_matrix = grids.create_spherical_grid(max_radius=max_radius, n_features=n_features,
                                                      bins_per_angstrom=bins_per_angstrom)

            # Bin each dimension independently
            r_bin, theta_bin, phi_bin = grids.discretize_into_spherical_bins(r, theta, phi, max_radius,
                                                                             grid_matrix.shape[0],
                                                                             grid_matrix.shape[1],
                                                                             grid_matrix.shape[2])

            # Merge bin indices into one array
            indices = np.vstack((r_bin, theta_bin, phi_bin)).transpose()

            # Check that bin indices are within grid
            assert (not np.any(theta_bin >= grid_matrix.shape[1]))
            assert (not np.any(phi_bin >= grid_matrix.shape[2]))

        elif coordinate_system == grids.CoordinateSystem.cubed_sphere:

            # Convert to coordinates on the cubed sphere
            patch, r, xi, eta = grids.cartesian_to_cubed_sphere_vectorized(xyz[:, 0], xyz[:, 1], xyz[:, 2])

            # Create grid
            grid_matrix = grids.create_cubed_sphere_grid(max_radius=max_radius, n_features=n_features,
                                                         bins_per_angstrom=bins_per_angstrom)

            # Bin each dimension independently
            patch_bin, r_bin, xi_bin, eta_bin = grids.discretize_into_cubed_sphere_bins(patch, r, xi, eta,
                                                                                        max_radius,
                                                                                        grid_matrix.shape[1],
                                                                                        grid_matrix.shape[2],
                                                                                        grid_matrix.shape[3])

            # Merge bin indices into one array
            indices = np.vstack((patch_bin, r_bin, xi_bin, eta_bin)).transpose()

            # Assert that bins are sensible
            assert (not np.any(xi_bin >= grid_matrix.shape[2]))
            assert (not np.any(eta_bin >= grid_matrix.shape[3]))

        elif coordinate_system == grids.CoordinateSystem.cartesian:

            # Create grid
            grid_matrix = grids.create_cartesian_grid(max_radius=max_radius, n_features=n_features,
                                                      bins_per_angstrom=bins_per_angstrom)

            # Bin each dimension of the cartesian coordinates
            indices = grids.discretize_into_cartesian_bins(xyz, max_radius, grid_matrix.shape)

            # Calculate radius
            r = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2 + xyz[:, 2] ** 2)

        # Create an index array to keep track of entries within range
        if include_center:
            selector = np.where(r < max_radius)[0]
        else:
            # ALSO exclude features from residue itself
            selector = np.where(np.logical_and(r < max_radius, res_indices[:, 0] != residue_index))[0]

        # Apply selector on indices array
        indices = indices[selector]

        # Check for multiple atoms mapped to same bin
        indices_rows = [tuple(row) for row in indices]
        duplicates = {}
        for index, row in enumerate(indices_rows):
            if indices_rows.count(row) > 1:
                index_matches = [index for index, value in enumerate(indices_rows) if value == row]
                index_matches.sort()
                if index_matches[0] not in duplicates:
                    duplicates[index_matches[0]] = index_matches
        if len(duplicates) > 0:
            print("WARNING: multiple atoms in same grid bin: (%s)" % pdb_id)
            for duplicate_indices in list(duplicates.values()):
                for i in duplicate_indices:
                    print("\t", features[selector][i])
                for i_index in range(len(duplicate_indices)):
                    coord1 = features[selector][duplicate_indices[i_index]][['x', 'y', 'z']].view(np.float32)
                    for j_index in range(i_index + 1, len(duplicate_indices)):
                        coord2 = features[selector][duplicate_indices[j_index]][['x', 'y', 'z']].view(np.float32)
                        print('\t\tdistance(%s,%s) = %s' % (coord1, coord2, np.linalg.norm(coord2 - coord1)))
                print()

        # Append indices and selector for current residues to list
        indices_list.append(indices)
        selector_list.append(selector)

    # Data is stored most efficiently when encoded as Numpy arrays. Rather than storing the data
    # as a list, we therefore create a numpy array where the number of columns is set to match
    # that of the residue with most neighbors.
    # Note that since the matrices are sparse, we save them as a list of values, and a corresponding
    # index array, so that the grid can be reconstructed by assigning to values to the indices in
    # this grid.
    max_selector = max([len(selector) for selector in selector_list if selector is not None])
    selector_array = np.full((len(selector_list), max_selector), -1, dtype=np.int32)
    for i, selector in enumerate(selector_list):
        if selector is not None:
            selector_array[i, :len(selector)] = selector.astype(np.int32)
    max_length = max([len(indices) for indices in indices_list if indices is not None])
    indices_list_shape_last_dim = None
    for indices in indices_list:
        if indices is not None:
            indices_list_shape_last_dim = indices.shape[1]
            break
    indices_array = np.full((len(indices_list), max_length, indices_list_shape_last_dim), -1, dtype=np.int16)
    for i, indices in enumerate(indices_list):
        if indices is not None:
            indices_array[i, :len(indices)] = indices.astype(np.int16)

    # Save using numpy binary format
    np.savez_compressed(os.path.join(output_dir, "%s_residue_features" % pdb_id), indices=indices_array,
                        selector=selector_array)
