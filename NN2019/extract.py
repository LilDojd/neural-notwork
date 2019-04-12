import os

import Bio.PDB
import numpy as np
import pandas as pd
import simtk
import simtk.openmm
import simtk.openmm.app
import simtk.unit
from numpy.lib.recfunctions import structured_to_unstructured

import smoothen
from Grids import grids

center = np.array([41.073, 36.990, 28.097], dtype=np.float32)


def cut_active_center(feats, cent=center, rad=10):
    """Leave only active centre for time-optimization reasons"""
    coords = np.squeeze(structured_to_unstructured(feats[['x', 'y', 'z']], dtype=np.float32))
    local_sys = coords - cent
    radii = np.sqrt(local_sys[:, 0] ** 2 + local_sys[:, 1] ** 2 + local_sys[:, 2] ** 2)
    act_cent_atoms = np.where(radii < rad)[0]
    select_residues = np.unique(feats[act_cent_atoms]['res_index'])
    selected_atoms = np.where(feats['res_index'] == select_residues)[0]
    new_features = feats[selected_atoms]
    return new_features


def extract_mass_charge(pdb_filename, csv_df, cut=True, smooth=True, n_bins=4):
    """Extract protein-level features from pdb file"""

    pdb_id = os.path.basename(pdb_filename).split('_')[1]

    # Read in PDB file
    pdb = simtk.openmm.app.PDBFile(pdb_filename)

    pdb_parser = Bio.PDB.PDBParser()
    structure = pdb_parser.get_structure(pdb_id, pdb_filename)

    first_model = structure.get_list()[0]
    sequence = []
    energy = int(csv_df.loc[str(pdb_id)][2])
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

    if cut:
        features = cut_active_center(features)

    if smooth:
        b_grid, bound = smoothen.wave_transform_smoothing(features, n_bins=n_bins)
        features = smoothen.unmap(b_grid, bound, features, n_bins)

    # Convert relevant entries to standard numpy arrays
    masses_array = structured_to_unstructured(features[['mass']], dtype=np.float32)
    charges_array = structured_to_unstructured(features[['charge']], dtype=np.float32)
    res_index_array = structured_to_unstructured(features[['res_index']], dtype=int)

    return pdb_id, features, masses_array, charges_array, aa_one_hot, energy, res_index_array, chain_boundary_indices, chain_ids


def embed_in_grid(features, pdb_id, output_dir,
                  max_radius,
                  n_feats,
                  bins_per_angstrom,
                  coord_sys,
                  local_center=center):
    """Embed masses and charge information in a spherical grid - specific for selected residue
       For space-reasons, only the indices into these grids are stored, and a selector
       specifying which atoms are relevant (i.e. within range) for the current residue.
    """

    # Extract coordinates as normal numpy array
    global indices, r
    position_array = structured_to_unstructured(features[['x', 'y', 'z']], dtype=np.float32)

    # Retrieve residue indices as numpy int array
    # We use hardcoded center to denote local coordinate system
    xyz = np.squeeze(position_array - local_center)
    # No rotation required because atoms are already aligned
    if coord_sys == grids.CoordinateSystem.spherical:

        # Convert to spherical coordinates
        r, theta, phi = grids.cartesian_to_spherical_coordinates(xyz)

        # Create grid
        grid_matrix = grids.create_spherical_grid(max_radius=max_radius, n_features=n_feats,
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

    elif coord_sys == grids.CoordinateSystem.cubed_sphere:

        # Convert to coordinates on the cubed sphere
        patch, r, xi, eta = grids.cartesian_to_cubed_sphere_vectorized(xyz[:, 0], xyz[:, 1], xyz[:, 2])

        # Create grid
        grid_matrix = grids.create_cubed_sphere_grid(max_radius=max_radius, n_features=n_feats,
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

    elif coord_sys == grids.CoordinateSystem.cartesian:

        # Create grid
        grid_matrix = grids.create_cartesian_grid(max_radius=max_radius, n_features=n_feats,
                                                  bins_per_angstrom=bins_per_angstrom)

        # Bin each dimension of the cartesian coordinates
        indices = grids.discretize_into_cartesian_bins(xyz, max_radius, grid_matrix.shape)

        # Calculate radius
        r = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2 + xyz[:, 2] ** 2)

    # Create an index array to keep track of entries within range
    selector = np.where(r < max_radius)[0]
    # Apply selector on indices array
    indices = indices[selector]

    # Check for multiple atoms mapped to same bin
    indices_rows = [tuple(row) for row in indices]
    duplicates = {}
    for index, row in enumerate(indices_rows):
        if indices_rows.count(row) > 1:
            index_matches = [index for index, va in enumerate(indices_rows) if va == row]
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

    # Data is stored most efficiently when encoded as Numpy arrays. We store data as selector pointing to
    # relevant indeces in protein array

    # Save using numpy binary format
    np.savez_compressed(os.path.join(output_dir, "%s_residue_features" % pdb_id), indices=indices,
                        selector=selector)


def extract_atomistic_features(pdb_filename, max_radius, n_feat, bins_per_angstrom,
                               add_seq_distance_feature, output_dir, coor_sys, en_df):
    """
    Creates both atom-level and residue-level (grid) features from a pdb file
    """

    print(pdb_filename)

    # Extract basic atom features (mass, charge, etc)
    [pdb_id, features, masses_array, charges_array, aa_one_hot, energy, residue_index_array, chain_boundary_indices,
     chain_ids] = extract_mass_charge(pdb_filename, csv_df=en_df)

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
                        energy=energy,
                        aa_one_hot=aa_one_hot,
                        coordinate_system=np.array(coor_sys.value, dtype=np.int32),
                        max_radius=np.array(max_radius, dtype=np.float32),  # angstrom
                        n_features=np.array(n_feat, dtype=np.int32),
                        bins_per_angstrom=np.array(bins_per_angstrom, dtype=np.float32),
                        n_residues=np.array(
                            len(np.unique(structured_to_unstructured(features[['res_index']], dtype=int)))))

    # Embed in a grid
    embed_in_grid(features, pdb_id, output_dir,
                  max_radius=max_radius,
                  n_feats=n_feat,
                  bins_per_angstrom=bins_per_angstrom,
                  coord_sys=coor_sys)


if __name__ == '__main__':
    import glob
    import joblib

    from utils import str2bool

    import argparse

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(help="Sub-command help", dest="mode")

    parser_extract = subparsers.add_parser("extract", help="extract features")
    parser_extract.add_argument("pdb_input_dir", type=str)
    parser_extract.add_argument("output_dir", type=str)
    parser_extract.add_argument("energy_csv", type=str)

    parser.add_argument("--coordinate-system", choices=[e.name for e in grids.CoordinateSystem],
                        default=grids.CoordinateSystem.spherical.name,
                        help="Which coordinate system to use (default: %(default)s)")
    parser.add_argument("--max-radius", metavar="VAL", type=int, default=12,
                        help="Maximal radius in angstrom (default: %(default)s)")
    parser.add_argument("--bins-per-angstrom", metavar="VAL", type=float, default=2,
                        help="Bins per Angstrom (default: %(default)s)")
    parser.add_argument("--n-proc", metavar="VAL", type=int, default=1,
                        help="Number of processes (default: %(default)s)")
    parser.add_argument("--add-seq-distance-feature", metavar="VAL", type=str2bool, default=False,
                        help="Add the sequence distance as a feature  (default: %(default)s)")

    args = parser.parse_args()

    print("# Arguments")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    en_table = pd.read_csv(args.energy_csv, index_col=0, header=None)

    n_features = 2
    if args.add_seq_distance_feature:
        n_features = 3
    print("n_features: ", n_features)

    coordinate_system = grids.CoordinateSystem[args.coordinate_system]

    if args.mode == "extract":
        pdb_filenames = glob.glob(os.path.join(args.pdb_input_dir, "*.pdb"))
        joblib.Parallel(n_jobs=args.n_proc, batch_size=1)(
            joblib.delayed(extract_atomistic_features)(pdb_filename,
                                                       args.max_radius,
                                                       n_features,
                                                       args.bins_per_angstrom,
                                                       args.add_seq_distance_feature,
                                                       args.output_dir,
                                                       coor_sys=coordinate_system,
                                                       en_df=en_table) for pdb_filename in pdb_filenames)
    else:
        raise argparse.ArgumentTypeError("Unknown mode")
