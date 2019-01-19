#! /usr/bin/env python3

"""A script for obtaining neighbouring residue numbers in PDB"""

try:
    from Bio.PDB import NeighborSearch, PDBParser, Selection
    import prody
except ModuleNotFoundError:
    pass
import argparse
import os
from pathlib import Path


def get_shell(reslist: list, structure: prody.AtomGroup, radii: float) -> dict:
    """
    Obtain all residues within :radius: of :reslist: Calphas
    :param structure:
    :param reslist:
    :param radii:
    :return:
    """
    nbs_list = []
    for resi in reslist:
        sel_string = f"calpha and within {radii} of resnum {resi}"
        shell = structure.select(sel_string)
        shell_resi = shell.getResnums().tolist()
        try:
            shell_resi.remove(int(resi))
        except ValueError:
            pass
        nbs_list.append(shell_resi)
    shell_dict = dict(zip(reslist, nbs_list))
    return shell_dict


def extract_res_list(resifile):
    if os.path.exists(resifile):
        content = Path(resifile).read_text()
        residues_new = content.split()
        return residues_new
    else:
        print("File with selected residues does not exist in current directory")
        pass


def get_structure(pdb_file):
    structure = prody.parsePDB(pdb_file)
    return structure


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get neighbouring atoms of reslist")
    parser.add_argument("pdb", help="Path to pdb file")
    parser.add_argument("reslist", help="Path to a file containing list of central atoms")
    parser.add_argument("--radius", dest='radius', nargs='?', default=5, type=float, help="Shell radius [5]")

    args = parser.parse_args()
    pdb = args.pdb
    radius = args.radius
    res_list = extract_res_list(args.reslist)
    struct = get_structure(pdb)
    print(get_shell(res_list, struct, radius))