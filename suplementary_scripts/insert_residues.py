#! usr/bin/env python3

"""Insert packed residues of aligned mutant into initial PDB"""
import argparse
import glob
import re
from pathlib import Path

from biopandas.pdb import PandasPdb
from prody import *


def re_resi_finder(filename):
    """
    Used further to deduce the resi of mutated residue by filename
    :type filename: str
    """
    index = re.findall(r"\d+", filename)
    return index[0]


def file_dir(initial, mut_files):
    """
    Returns dictionary where key is resi and value is its complementary file
    :param initial:
    :param mut_files:
    :return:
    """
    file_list = glob.glob(mut_files)
    file_list.sort()
    file_list = [s for s in file_list if initial not in s]
    residues_bijection = list(map(re_resi_finder, file_list))
    res_file = list(zip(residues_bijection, file_list))
    return res_file


def insert_shell(initial, res_file, radii, out):
    """
    Create modified files
    :param out: Output path
    :param initial: name of initial pdb
    :param radii:  Shell radius. Usually 5A
    :type res_file: File returned by file_dir
    """

    initial_pdb = PandasPdb().read_pdb(initial).df['ATOM']

    for item in res_file:
        resi = item[0]
        filename = item[1]
        prody_struct = parsePDB(filename)
        sel_string = f"calpha and within {radii} of resnum {resi}"
        shell_ca = prody_struct.select(sel_string)
        shell_resi_list = shell_ca.getResnums().tolist()
        mut_pdb = PandasPdb().read_pdb(filename)
        shell_df = mut_pdb.df['ATOM'][mut_pdb.df['ATOM']['residue_number'].isin(shell_resi_list)]
        new_mut = initial_pdb.copy(deep=True)
        new_mut = new_mut[~new_mut['residue_number'].isin(shell_resi_list)]
        new_mut.append(shell_df, ignore_index=True).sort_values('residue_number', inplace=True)
        print(f"...Iterating over resi {resi}")
        new_mut.drop(labels="atom_number", axis=1)
        arange = np.arange(start=1, stop=new_mut.shape[0] + 1)
        new_mut["atom_number"] = arange
        out = out + f"shell_modified_{resi}.pdb"
        mut_pdb.to_pdb(path=f"/{out}",
                       records=None,
                       gz=False,
                       append_newline=True)

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get neighbouring atoms of reslist")
    parser.add_argument("pdb", help="Path to initial pdb file")
    parser.add_argument("files", help="Names of files to extract shell")
    parser.add_argument("--radius", dest='radius', nargs='?', default=5, type=float, help="Shell radius [5]")
    parser.add_argument("-o", "--output", dest='output', nargs='?', default="insert_output", type=str, help="Out dir")

    args = parser.parse_args()
    pdb = args.pdb
    radius = args.radius
    files = args.files
    outname = args.output
    outdir = Path(outname)
    outdir.mkdir(exist_ok=True)
    res_file_dict = file_dir(pdb, files)
    insert_shell(pdb, res_file_dict, radius, outname)
