#! /usr/bin/env python3

"""A script for obtaining neighbouring residue numbers in PDB"""

try:
    from Bio.PDB import NeighborSearch, PDBParser, Selection
    import prody
except ModuleNotFoundError:
    pass
import argparse
from pathlib import Path


def get_structure(pdb_file):
    structure = PDBParser().get_structure('struct', pdb_file)
    chain = structure[0][' ']  # Supply chain name for "center residues"
    return structure, chain


def get_center_atoms(chain, resi):
    center_residue = chain[int(resi)]
    center_atoms = Selection.unfold_entities(center_residue, 'A')
    return center_atoms


def get_nb_list(tree, center_atom_list):
    resindex = center_atom_list[0].get_full_id()[3][1]
    nba = {res for center_atom in center_atom_list for res in tree.search(center_atom.coord, 5, 'R')}
    nba = sorted(res.id[1] for res in nba)
    try:
        nba.remove(resindex)
    except ValueError:
        print("Something may be wrong, or not")
    print(resindex, nba)
    return nba


def nb_search(structure):
    atom_list = [atom for atom in structure.get_atoms() if atom.name == 'CA']
    ns = NeighborSearch(atom_list)
    return ns


def extract_res_list(resifile):
    content = Path(resifile).read_text()
    residues_new = content.split()
    return residues_new


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get neighbouring atoms of reslist")
    parser.add_argument("pdb", help="Path to pdb file")
    parser.add_argument("reslist", help="Path to a file containing list of central atoms")

    args = parser.parse_args()
    pdb = args.pdb
    res_list = extract_res_list(args.reslist)
    struct, chain = get_structure(pdb)
    # Make list of lists of atoms of each residue
    center_list = [get_center_atoms(chain, resi) for resi in res_list]
    kd_tree = nb_search(struct)
    for residue in center_list:
        nbs = get_nb_list(kd_tree, residue)
