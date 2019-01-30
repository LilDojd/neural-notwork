import glob
import os
import groio


if __name__ == '__main__':

    import argparse


    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        if v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    parser = argparse.ArgumentParser()
    parser.add_argument("--gro-input-dir", dest="gro_input_dir",
                        help="Location of gros")
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
            structure = parse_pdb(handle, pdb_id, options.reduce_executable, options.dssp_executable,
                                  use_pdb_fixer=options.use_pdb_fixer, allow_chain_breaks=options.allow_chain_breaks,
                                  allow_incomplete_pdb=options.allow_incomplete_pdb, verbose=options.verbose)
        except IncompletePDBError as e:
            print(list(e.message.values())[0])
            raise

        io = Bio.PDB.PDBIO()
        io.set_structure(structure)
        io.save(os.path.join(options.output_dir, pdb_id + ".pdb"))