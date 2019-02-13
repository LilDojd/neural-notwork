"""Check PDB, parse it and start preparing features"""
import glob
import io
import os
import sys
import tempfile

import Bio.PDB
import Bio.PDB.Polypeptide
import Bio.SeqIO
# import numpy as np
import pdbfixer

sys.path.append('..')

if 'TRN' not in pdbfixer.pdbfixer.substitutions:
    pdbfixer.pdbfixer.substitutions['TRN'] = 'TRP'


class IncompleteSeqResError(Exception):
    pass


class IncompletePDBError(Exception):
    pass


class ChainBreakError(Exception):
    pass


class OpenMMParseError1(Exception):
    pass


class OpenMMParseError2(Exception):
    pass


class OpenMMException(Exception):
    pass


class PDBNotFoundError(Exception):
    pass


class ReduceError(Exception):
    pass


class PDBFixerResIdentifiabilityIssue(Exception):
    pass


def parse_pdb(pdb_filehandle, pdbid, chain_id_filter=None, use_pdb_fixer=True,
              allow_chain_breaks=False, allow_incomplete_pdb=False, verbose=False):
    """Takes a pdb filehandle, parses it, and runs it through several sanity checks"""

    global pdb, io

    class ChainAndNonHetSelector(Bio.PDB.Select):
        """Only accept the specified chains when saving."""

        def __init__(self, chain_letters):
            self.chain_letters = chain_letters

        def accept_residue(self, residue):
            print(residue.get_resname(), residue.id[0] == " ", residue.get_resname() in pdbfixer.pdbfixer.substitutions)
            return residue.id[0] == " " or (residue.get_resname() in pdbfixer.pdbfixer.substitutions)

        def accept_model(self, model):
            return model.id == 0

        def accept_chain(self, chains):
            if self.chain_letters is None:
                return True
            return chains.get_id() == ' ' or chains.get_id() in self.chain_letters

        def accept_atom(self, atoms):
            return not atoms.is_disordered() or atoms.get_altloc() == 'A' or atoms.get_altloc() == '1'

    pdb_content = pdb_filehandle.read()
    pdb_filehandle = io.StringIO(pdb_content)

    # Extract seqres entries
    seqres_seqs = {}
    for record in Bio.SeqIO.PdbIO.PdbSeqresIterator(pdb_filehandle):
        seqres_seqs[record.annotations["chain"]] = str(record.seq)

    # Extract struct using BIO.PDB
    pdb_filehandle = io.StringIO(pdb_content)
    pdb_parser = Bio.PDB.PDBParser(PERMISSIVE=0)
    struct = pdb_parser.get_structure(pdbid, pdb_filehandle)
    io = Bio.PDB.PDBIO()
    io.set_structure(struct)

    first_model = struct.get_list()[0]
    ppb = Bio.PDB.PPBuilder()
    for i, chain in enumerate(first_model):
        pps = ppb.build_peptides(chain)

        # Extract sequence from PDB file by looking at CA atoms
        seq = ""
        for atom in chain.get_atoms():
            if atom.id == 'CA':
                try:
                    aa = Bio.PDB.Polypeptide.three_to_one(atom.get_parent().get_resname())
                except KeyError:
                    aa = 'X'
                seq += aa

        chain_id = chain.id

        # Check for chain breaks if they are disallowed
        if not allow_chain_breaks:
            number_of_pps = len(list(pps))
            if not (number_of_pps == 1 and len(pps[0]) == len(seq)):
                if verbose:
                    for pp in pps:
                        print(pp.get_sequence())
                raise ChainBreakError

        # Check whether sequence in PDB struct matches sequence in seqres
        if not allow_incomplete_pdb:
            if chain_id not in seqres_seqs:
                raise IncompleteSeqResError()
            seqres_seq = seqres_seqs[chain_id]
            if len(seq) != len(seqres_seq):
                raise IncompletePDBError({'message': '\n' + seq + '\n' + seqres_seq})

    # first_residue_index = struct[0].get_list()[0].get_list()[0].get_id()[1]

    with tempfile.NamedTemporaryFile(delete=True) as temp1:

        # Save PDB file again for further processing by external program
        io.save(temp1, ChainAndNonHetSelector(chain_id_filter))
        temp1.flush()

        with tempfile.NamedTemporaryFile(delete=True) as temp2:

            # Add hydrogens using reduce program
            # command = [reduce_executable, '-BUILD',
            #            '-DB', os.path.join(os.path.dirname(reduce_executable), 'reduce_wwPDB_het_dict.txt'),
            #            '-Quiet',
            #            # '-Nterm'+str(max_N_terminal_residues),
            #            '-Nterm' + str(first_residue_index),
            #            temp1.name]
            temp2.flush()

            # Use PDBFixer to fix common PDB errors
            fixer = pdbfixer.PDBFixer(temp2.name)

            if use_pdb_fixer:

                fixer.findMissingResidues()

                fixer.findNonstandardResidues()
                fixer.replaceNonstandardResidues()

                # Remove waters and other non-protein atoms
                # fixer.removeHeterogens(False)

                fixer.findMissingAtoms()

                try:
                    fixer.addMissingAtoms()
                    fixer.addMissingHydrogens(7.0)
                except Exception:
                    raise OpenMMException()

                with tempfile.NamedTemporaryFile(delete=False) as temp3:

                    # We would have liked to use keepIds=True here, but it does not preserve insertion codes,
                    # so we instead set the IDs manually
                    # simtk.openmm.app.PDBFile.writeFile(fixer.topology, fixer.positions, temp3, keepIds=False)
                    temp3.flush()

                    pdb_parser = Bio.PDB.PDBParser(PERMISSIVE=1)
                    structure_before = pdb_parser.get_structure(temp2.name, temp2.name)
                    structure_after = pdb_parser.get_structure(temp3.name, temp3.name)

                    # PDBfixer does not preserve insertion codes. We therefore do it manually here
                    residues_before = []
                    for chain in structure_before[0]:
                        residues_before.append(chain.get_list())
                    residues_after = []
                    for chain in structure_after[0]:
                        residues_after.append(chain.get_list())
                    for i, chain in enumerate(structure_before[0]):
                        structure_after[0].get_list()[i].id = structure_before[0].get_list()[i].id
                        if len(residues_before[i]) != len(residues_after[i]):
                            raise PDBFixerResIdentifiabilityIssue()
                        for res1, res2 in zip(residues_before[i], residues_after[i]):
                            assert (res1.get_resname().strip() == res2.get_resname().strip() or
                                    pdbfixer.pdbfixer.substitutions[
                                        res1.get_resname()].strip() == res2.get_resname().strip())
                            res2.id = res1.id

                    io = Bio.PDB.PDBIO()
                    io.set_structure(structure_after)
                    with tempfile.NamedTemporaryFile(delete=False) as temp4:
                        print("\t", temp4.name)

                        io.save(temp4)
                        temp4.flush()

                        # # Read in PDB file
                        # try:
                        #     pdb = simtk.openmm.app.PDBFile(temp4.name)
                        # except:
                        #     raise OpenMMParseError1

                        struct = structure_after

            else:

                # Read in PDB file
                # pdb = simtk.openmm.app.PDBFile(temp2.name)

                pdb_parser = Bio.PDB.PDBParser(PERMISSIVE=0)
                struct = pdb_parser.get_structure(temp2.name, temp2.name)

                # Attempt to extract DSSP
                # first_model = struct.get_list()[0]
                # dssp = Bio.PDB.DSSP(first_model, temp2.name, dssp=dssp_executable)
                # for i, chain in enumerate(first_model):
                #     pps = ppb.build_peptides(chain)
                #     ss = np.array([dssp2i(res.xtra["SS_DSSP"]) for res in chain], dtype=np.int8)

                # Extract positions
    # positions = pdb.getPositions()

    # Create forcefield in order to extract charges
    # forcefield = simtk.openmm.app.ForceField('amber99sb.xml', 'tip3p.xml')

    # Create system to couple topology with forcefield
    # try:
    #     system = forcefield.createSystem(pdb.getTopology())
    # except ValueError as exc:
    #     print(exc)
    #     raise OpenMMParseError2

    return struct


if __name__ == '__main__':

    from argparse import ArgumentParser
    import argparse


    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        if v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    parser = ArgumentParser()
    parser.add_argument("--pdb-input-dir", dest="pdb_input_dir",
                        help="Location of pdbs")
    parser.add_argument("--output-dir", dest="output_dir",
                        help="Where to dump features")
    parser.add_argument("--reduce-executable", dest="reduce_executable",
                        help="Location of reduce executable")
    parser.add_argument("--dssp-executable", dest="dssp_executable",
                        help="Location of dssp executable")
    parser.add_argument("--allow-chain-breaks", dest="allow_chain_breaks",
                        type=str2bool, nargs='?', const=True, default="False",
                        help="Whether to allow chain breaks in PDB")
    parser.add_argument("--allow-incomplete-pdb", dest="allow_incomplete_pdb",
                        type=str2bool, nargs='?', const=True, default="False",
                        help="Whether to allow mismatch between PDB and seqres record")
    parser.add_argument("--use-pdb-fixer", dest="use_pdb_fixer",
                        type=str2bool, nargs='?', const=True, default="True",
                        help="Whether to use PDB fixer")
    parser.add_argument("--verbose", dest="verbose",
                        type=str2bool, nargs='?', const=True, default="True",
                        help="Output additional information")

    options = parser.parse_args()

    pdb_filenames = glob.glob(options.pdb_input_dir + "/*")

    if not os.path.exists(options.output_dir):
        os.mkdir(options.output_dir)

    for pdb_filename in pdb_filenames:

        handle = open(pdb_filename)

        pdb_id = os.path.basename(pdb_filename).replace(".pdb", "").split('_')[0]

        print(pdb_filename)
        try:
            structure = parse_pdb(handle, pdb_id, use_pdb_fixer=options.use_pdb_fixer,
                                  allow_chain_breaks=options.allow_chain_breaks,
                                  allow_incomplete_pdb=options.allow_incomplete_pdb, verbose=options.verbose)
        except IncompletePDBError:
            raise

        io = Bio.PDB.PDBIO()
        io.set_structure(structure)
        io.save(os.path.join(options.output_dir, str(pdb_id) + ".pdb"))
