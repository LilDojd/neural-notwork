import os

import Bio.PDB
import numpy as np
import simtk
import simtk.openmm
import simtk.openmm.app
import simtk.unit
from numpy.lib.recfunctions import structured_to_unstructured

from Grids import grids

center = np.array([41.073, 36.990, 28.097], dtype=np.float32)


def extract_md():
    pass


# Add smoothing at this level
def extract_mass_charge(pdb_filename, n_bins, smooth=False):
    """Extract protein-level features from pdb file"""

    pdb_id = os.path.basename(pdb_filename).split('.')[0]

    # Read in PDB file
    pdb = simtk.openmm.app.PDBFile(pdb_filename)

    # Also parse through Bio.PDB - to extract DSSP secondary structure info
    pdb_parser = Bio.PDB.PDBParser()
    structure = pdb_parser.get_structure(pdb_id, pdb_filename)

    first_model = structure.get_list()[0]
    sequence = []
    aa_one_hot = []
    chain_ids = []
    for i, chain in enumerate(first_model):

        # Separate chain into unbroken segments
        chain_ids.append(chain.id)

        # Sequence of residue names for this chain
        sequence_chain = []
        for res in chain.get_residues():
            sequence_chain.append(res.resname.strip())

        # Add to global container for this protein
        sequence.append(sequence_chain)

        # Convert residue names to amino acid indices
        aa_indices = []
        for aa in sequence_chain:
            try:
                aa_index = Bio.PDB.Polypeptide.three_to_index(aa)
            except KeyError:
                aa_index = 20
            aa_indices.append(aa_index)
        # Convert to one-hot encoding
        aa_one_hot_chain = np.zeros((len(aa_indices), 21))
        aa_one_hot_chain[np.arange(len(aa_indices)), aa_indices] = 1
        aa_one_hot.append(aa_one_hot_chain)

    # Keep track of boundaries of individual chains
    chain_boundary_indices = np.cumsum([0] + [len(entry) for entry in aa_one_hot])

    # Collapse all chain segments into one. The individual chains
    # will be recoverable through chain_boundary_indices
    aa_one_hot = np.concatenate(aa_one_hot)

    # Extract positions using OpenMM
    positions = pdb.getPositions()

    # Create forcefield in order to extract charges
    # noinspection PyTypeChecker
    forcefield = simtk.openmm.app.ForceField('amber99sb.xml', 'tip3p.xml')

    # Add hydrogens if necessary
    modeller = simtk.openmm.app.Modeller(pdb.getTopology(), pdb.getPositions())

    # Create system to couple topology with forcefield
    system = forcefield.createSystem(modeller.getTopology())

    # Find nonbonded force (contains charges)
    nonbonded_force = None
    for force in system.getForces():
        if isinstance(force, simtk.openmm.openmm.NonbondedForce):
            nonbonded_force = force

    # Create structured array for features
    # Smoothen here, dont forget to change shape
    features = np.empty(shape=(len(positions), 1), dtype=[('mass', np.float32),
                                                          ('charge', np.float32),
                                                          ('name', 'a5'),
                                                          ('res_index', int),
                                                          ('x', np.float32), ('y', np.float32), ('z', np.float32)])
    # Iterate over chain,residue,atoms and extract features
    for i, chain in enumerate(pdb.getTopology().chains()):
        chain_start_index = chain_boundary_indices[i]
        for j, residue in enumerate(chain.residues()):
            for atom in residue.atoms():
                # Extract atom features
                index = atom.index
                position = list(positions[index].value_in_unit(simtk.unit.angstrom))
                mass = atom.element.mass.value_in_unit(simtk.unit.dalton)
                charge = nonbonded_force.getParticleParameters(index)[0].value_in_unit(simtk.unit.elementary_charge)
                features[index] = tuple([mass, charge, atom.name, residue.index] + position)

                residue_index_local = residue.index - chain_start_index
                assert (residue.name == sequence[i][residue_index_local])

    # Convert relevant entries to standard numpy arrays
    masses_array = structured_to_unstructured(features[['mass']], dtype=np.float32)
    charges_array = structured_to_unstructured(features[['charge']], dtype=np.float32)
    res_index_array = structured_to_unstructured(features[['res_index']], dtype=int)

    return pdb_id, features, masses_array, charges_array, aa_one_hot, res_index_array, chain_boundary_indices, chain_ids


def embed_in_grid(features, pdb_id, output_dir,
                  max_radius,
                  n_features,
                  bins_per_angstrom,
                  coordinate_system,
                  z_direction,
                  include_center, local_center=center, smoothen=False):
    """Embed masses and charge information in a spherical grid - specific for selected residue
       For space-reasons, only the indices into these grids are stored, and a selector
       specifying which atoms are relevant (i.e. within range) for the current residue.
    """

    # Extract coordinates as normal numpy array
    global indices, r
    position_array = structured_to_unstructured(features[['x', 'y', 'z']], dtype=np.float32)

    # Retrieve residue indices as numpy int array
    res_indices = structured_to_unstructured(features[['res_index']], dtype=int)

    selector_list = []
    indices_list = []
    for residue_index in np.unique(structured_to_unstructured(features[['res_index']], dtype=int)):

        # Extract origin
        if (np.logical_and(res_indices == residue_index,
                           structured_to_unstructured(features[['name']], dtype='a5') == b"N").any() and
                np.logical_and(res_indices == residue_index,
                               structured_to_unstructured(features[['name']], dtype='a5') == b"CA").any() and
                np.logical_and(res_indices == residue_index,
                               structured_to_unstructured(features[['name']], dtype='a5') == b"C").any()):
            CA_feature = features[
                np.argmax(np.logical_and(res_indices == residue_index,
                                         structured_to_unstructured(features[['name']], dtype='a5') == b"CA"))]
            N_feature = features[
                np.argmax(np.logical_and(res_indices == residue_index,
                                         structured_to_unstructured(features[['name']], dtype='a5') == b"N"))]
            C_feature = features[
                np.argmax(np.logical_and(res_indices == residue_index,
                                         structured_to_unstructured(features[['name']], dtype='a5') == b"C"))]
        else:
            # Store None to maintain indices
            indices_list.append(None)
            selector_list.append(None)
            continue

        # Positions of N, CA and C atoms used to define local reference system
        pos_CA = structured_to_unstructured(CA_feature[['x', 'y', 'z']], dtype=np.float32)
        pos_N = structured_to_unstructured(N_feature[['x', 'y', 'z']], dtype=np.float32)
        pos_C = structured_to_unstructured(C_feature[['x', 'y', 'z']], dtype=np.float32)

        # Define local coordinate system
        rot_matrix = grids.define_coordinate_system(pos_N, pos_CA, pos_C, z_direction)

        # Calculate coordinates relative to origin
        xyz = position_array - pos_CA
        xyz = xyz.reshape(xyz.shape[0], xyz.shape[2])
        rot_matrix = rot_matrix.reshape(rot_matrix.shape[0], rot_matrix.shape[2])

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
                    coord1 = structured_to_unstructured(features[selector][duplicate_indices[i_index]][['x', 'y', 'z']],
                                                        dtype=np.float32)
                    for j_index in range(i_index + 1, len(duplicate_indices)):
                        coord2 = structured_to_unstructured(features[selector][duplicate_indices[j_index]][['x', 'y',
                                                                                                            'z']],
                                                            dtype=np.float32)
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


def extract_atomistic_features(pdb_filename, max_radius, n_features, bins_per_angstrom,
                               add_seq_distance_feature, output_dir, coordinate_system,
                               z_direction, include_center):
    """
    Creates both atom-level and residue-level (grid) features from a pdb file
    """

    print(pdb_filename)

    # Extract basic atom features (mass, charge, etc)
    [pdb_id, features, masses_array, charges_array, aa_one_hot, residue_index_array, chain_boundary_indices,
     chain_ids] = extract_mass_charge(pdb_filename)

    # Save protein level features
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.savez_compressed(os.path.join(output_dir, "%s_protein_features" % pdb_id),
                        masses=masses_array,
                        charges=charges_array,
                        residue_index=residue_index_array,
                        residue_features=["masses", "charges", 'residue_index'] if add_seq_distance_feature else [
                            "masses", "charges"],
                        chain_boundary_indices=chain_boundary_indices,
                        chain_ids=chain_ids,
                        aa_one_hot=aa_one_hot,
                        coordinate_system=np.array(coordinate_system.value, dtype=np.int32),
                        z_direction=np.array(z_direction.value, dtype=np.int32),
                        max_radius=np.array(max_radius, dtype=np.float32),  # angstrom
                        n_features=np.array(n_features, dtype=np.int32),
                        bins_per_angstrom=np.array(bins_per_angstrom, dtype=np.float32),
                        n_residues=np.array(
                            len(np.unique(structured_to_unstructured(features[['res_index']], dtype=int)))))

    # Embed in a grid
    embed_in_grid(features, pdb_id, output_dir,
                  max_radius=max_radius,
                  n_features=n_features,
                  bins_per_angstrom=bins_per_angstrom,
                  coordinate_system=coordinate_system,
                  z_direction=z_direction,
                  include_center=include_center)

