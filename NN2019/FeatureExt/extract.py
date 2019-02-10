import os

import Bio.PDB
import numpy as np
import simtk
import simtk.openmm
import simtk.openmm.app
import simtk.unit


def extract_mass_charge(pdb_filename):
    """Extract protein-level features from pdb file"""

    pdb_id = os.path.basename(pdb_filename).split('.')[0]

    # Read in PDB file
    pdb = simtk.openmm.app.PDBFile(pdb_filename)

    # Also parse through Bio.PDB - to extract DSSP secondary structure info
    pdb_parser = Bio.PDB.PDBParser()
    structure = pdb_parser.get_structure(pdb_id, pdb_filename)

    first_model = structure.get_list()[0]
    ppb = Bio.PDB.PPBuilder()  # Unused
    sequence = []
    aa_one_hot = []
    ss_one_hot = []
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
            except:
                aa_index = 20
            aa_indices.append(aa_index)

        # Convert to one-hot encoding
        aa_one_hot_chain = np.zeros((len(aa_indices), 21))
        aa_one_hot_chain[np.arange(len(aa_indices)), aa_indices] = 1

        # Extract secondary structure
        ss = []
        for res in chain:
            try:
                ss.append(dssp2i(res.xtra["SS_DSSP"]))
            except:
                ss.append(3)
        ss = np.array(ss, dtype=np.int8)

        # Convert to one-hot encoding
        ss_one_hot_chain = np.zeros((len(ss), 4))
        ss_one_hot_chain[np.arange(len(ss)), ss] = 1

        aa_one_hot.append(aa_one_hot_chain)
        ss_one_hot.append(ss_one_hot_chain)

    # Keep track of boundaries of individual chains
    chain_boundary_indices = np.cumsum([0] + [len(entry) for entry in aa_one_hot])

    # Collapse all chain segments into one. The individual chains
    # will be recoverable through chain_boundary_indices
    aa_one_hot = np.concatenate(aa_one_hot)
    ss_one_hot = np.concatenate(ss_one_hot)

    # Extract positions using OpenMM
    positions = pdb.getPositions()

    # Create forcefield in order to extract charges
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
    masses_array = features[['mass']].view(np.float32)
    charges_array = features[['charge']].view(np.float32)
    res_index_array = features[['res_index']].view(int)

    return pdb_id, features, masses_array, charges_array, aa_one_hot, ss_one_hot, res_index_array, chain_boundary_indices, chain_ids
