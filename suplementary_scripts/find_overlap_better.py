#! /usr/bin/env python3

"""Given two .gro files of dissolved protein and mutated protein without solvent environment
generate .gro file that combines two without particles overlapping"""

import argparse
import itertools
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import scipy.spatial

from suplementary_scripts.parsegro import gro_parser, get_coordinates, write_gro, renumber

""" VdW values are hardcoded
    ???  H     0.12
    ???  C     0.17
    ???  N     0.155
    ???  O     0.152
    ???  F     0.147
    ???  P     0.18
    ???  S     0.18
    ???  Cl    0.175
    ???  Na    0.227
"""

VAN_DER_WAALS = {
    "C": 0.17,
    "CL": 0.175,
    "N": 0.155,
    "NA": 0.227,
    "O": 0.152,
    "OW": 0.152,
    "P": 0.18,
    "S": 0.18
}

COL_NAMES = ["C", "N", "O", "S"]
INDEX_NAMES = ["CL", "NA", "OW"]
VDW_DF = pd.DataFrame(columns=COL_NAMES, index=INDEX_NAMES)

for x, y in itertools.product(INDEX_NAMES, COL_NAMES):
    rad_sum = round(VAN_DER_WAALS[x] + VAN_DER_WAALS[y], 3)
    VDW_DF.loc[x, y] = rad_sum

print(f"VdW radii used for this scrip:'\n{VDW_DF}\n")


def distance(point_a, point_b):  # Used to calculate distance between 2 points
    global dist
    if np.shape(point_a)[0] == 1:
        dist = np.linalg.norm(point_a - point_b)
    elif np.shape(point_a)[0] > 1:
        dist = np.sqrt(np.sum((point_a - point_b) ** 2, axis=1)).reshape(1, len(point_a))
    return np.round(dist, 3)


def geometric_center(array: np.array):  # Find geometric center of a protein
    global center
    try:
        center = array.mean(axis=0)
    except Exception as inst:
        print(inst)
    return np.round(center, 2)


def max_dist(mutant_coords: np.array) -> tuple:  # Create a tuple containing all possible pairs of rows of the array
    center_coord = geometric_center(mutant_coords)
    # Create list of distances from geometric center
    distances = [round(distance(at_coord.reshape(1, 3), center_coord), 3) for at_coord in mutant_coords]
    max_value = max(distances)
    max_index = distances.index(max_value)
    out = (max_index, max_value)
    return out


def create_subdfs(initial_df, atomlist):
    """
    To be used in subdf_dict
    :Parameters:
    :type initial_df: pd.DataFrame
    :type atomlist: list
    :Returns:
    Generator for subdfs
    """
    global key, atom_subdf
    for atom in atomlist:
        atom_subdf = initial_df[initial_df["atom_name"].str.contains("^" + atom)]
        key = atom
        yield atom_subdf, key


def subdf_dict(initial_df, atomlist):
    """
    :param initial_df: pd.DataFrame
    :param atomlist: list
    :return: dict
    """
    df_dict = {}
    for subdf, k in create_subdfs(initial_df, atomlist):
        if k not in df_dict:
            df_dict[k] = []
        df_dict[k].append(subdf)
    return df_dict


def build_trees(dictionary, compact_nodes=True):
    """
    Given dictionary containing labeled subdataframes,
    return dict of cKDtrees of this sdfs
    :type compact_nodes: bool
    :type dictionary: dict
    """
    ckd_dict = {}
    for eachkey in dictionary.keys():
        if eachkey not in ckd_dict:
            ckd_dict[eachkey] = []
        sdf = dictionary[eachkey]
        coords = get_coordinates(sdf[0])
        ckdtree = scipy.spatial.cKDTree(coords, leafsize=16)
        ckd_dict[eachkey].append(ckdtree)
    return ckd_dict


def find_match(coords, df):
    """
    :param coords: coords to be located
    :param df: Dataframe to use for location
    :return: resid of match
    """
    for coord in coords:
        row = df.ix[(df['x'] == coord[0]) & (df['y'] == coord[1]) & (df['z'] == coord[2])]
        resid = row.iat[0, 1]
        yield resid


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate dissolved mutated protein topology")
    parser.add_argument("solvf", help="Path to topology file containing dissolved protein")
    parser.add_argument("mutantf", help="Path to topology file containing mutant protein")
    parser.add_argument("-o", "--outputname", dest="output", nargs="?", help="Name of output file")
    parser.add_argument("-s", "--scale", dest="scale", nargs="?", default=1, type=float, help="VdW scaler, 1 for default")

    args = parser.parse_args()
    if args.output is None:
        output = args.mutantf.split("/")[-1].strip(".gro") + "_sol.gro"
    else:
        output = args.output

    scl = args.scale
    # Work with input files
    sol_parsed = gro_parser(args.solvf, include_protein=False)
    mut_parsed = gro_parser(args.mutantf, include_solvent=False)
    # Create large dataframe containing all parsed atoms
    sol_df: pd.DataFrame = sol_parsed.atoms
    mut_df: pd.DataFrame = mut_parsed.atoms
    big_df: pd.DataFrame = pd.concat([mut_df, sol_df], axis=0, join="outer")
    print(f"Shape of DataFrame: {big_df.shape}.\nConsists of smaller {mut_df.shape} and {sol_df.shape} DFs\n")

    #  And for each channel we create dictionary
    proteindf_dict = subdf_dict(mut_df, COL_NAMES)
    soldf_dict = subdf_dict(sol_df, INDEX_NAMES)
    #  Build cKDTree for each subDF
    protein_trees = build_trees(proteindf_dict, compact_nodes=True)
    sol_trees = build_trees(soldf_dict, compact_nodes=True)
    print(f"cKDTrees built for {len(sol_trees)} solvent and {len(protein_trees)} protein channels\n")

    # Find juxtaposed atoms from 2 sets. This part of code is barely readable as i didn't put much thought to it but
    # all it does is it creates dictionary with redundant SOL atoms as keys and their row index in respective
    # subdataframes
    to_exclude: Dict[Any, List[Any]] = {}
    for i, j in itertools.product(protein_trees, sol_trees):
        tree1: scipy.spatial.cKDTree = protein_trees[i][0]
        tree2: scipy.spatial.cKDTree = sol_trees[j][0]
        rad_range = scl*VDW_DF.loc[j, i]
        list_for_each = tree1.query_ball_tree(tree2, rad_range)
        list_for_each = [x for x in list_for_each if x != []]
        # Get rid of empty lists
        flattened_list = list(itertools.chain(*list_for_each))
        if j not in to_exclude.keys():
            to_exclude[j] = []
        to_exclude[j].append(flattened_list)
    for key in to_exclude.keys():
        to_exclude[key] = list(set(list(itertools.chain(*to_exclude[key]))))
        to_exclude[key].sort()
    ow_cut = to_exclude["OW"]
    na_cut = to_exclude["NA"]
    cl_cut = to_exclude["CL"]
    print(f"Found {len(ow_cut)} overlapping Oxygen atoms\nFound {len(na_cut)} overlapping Sodium atoms\nFound "
          f"{len(cl_cut)} "
          f"overlapping Chlorine atoms\n")
    print("Deleting...\n")

    # Data frames of each solvent atom
    cl_df = soldf_dict["CL"][0]
    na_df = soldf_dict["NA"][0]
    ow_df = soldf_dict["OW"][0]

    # Find resid of atoms to be deleted
    cl_cut_coord = sol_trees["CL"][0].data[cl_cut]
    na_cut_coord = sol_trees["NA"][0].data[na_cut]
    ow_cut_coord = sol_trees["OW"][0].data[ow_cut]

    del_resid = []
    del_resid += [i for i in find_match(ow_cut_coord, sol_df)]
    del_resid += [i for i in find_match(cl_cut_coord, sol_df)]
    del_resid += [i for i in find_match(na_cut_coord, sol_df)]

    # Delete overlaping atoms
    sol_df = sol_df[~sol_df.resid.isin(del_resid)]
    # And create new atom DataFrame
    final_df: pd.DataFrame = pd.concat([mut_df, sol_df], axis=0, join="outer")
    # Renumber
    new_atoms_dict = renumber(final_df.to_dict('records'))

    # Extract additional information (title + box)
    title = mut_parsed.title
    box = sol_parsed.box
    title = title.strip("\n") + " dissolved in water\n"
    with open(output, "w") as outf:
        for i in write_gro(title, new_atoms_dict, box):
            outf.write(i)
    print(f"Output saved in {output}\n")
