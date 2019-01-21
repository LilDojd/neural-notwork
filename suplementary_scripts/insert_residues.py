#! usr/bin/env python3

"""Insert packed residues of aligned mutant into initial PDB"""
import argparse
import glob
import os
import re
import shutil
import subprocess

from prody import *


def re_resi_finder(filename):
    """
    Used further to deduce the residue index of mutated residue by filename
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

    os.mkdir(out)
    with open(initial) as protf:
        initial_pdb = parsePDBStream(protf, biomol=False)

    itered = []
    for item in res_file:
        resi = item[0]
        filename = item[1]
        outstring = out + f"{filename.split('_')[2]}"  # Works only for names with structure first_second_11A_rest
        # To be reassessed
        prody_struct, head = parsePDB(filename, header=True, biomol=False)
        sel_string = f"protein and calpha and within {radii} of resnum {resi}"
        # Select all C-alpha atoms within radius from reside
        shell_ca = prody_struct.select(sel_string)
        # os.remove('tempshell.pdb')
        shell_resi_list = shell_ca.getResnums().tolist()
        # Select residues within radius of residue
        residues = " ".join([str(i) for i in shell_resi_list])
        shell = prody_struct.select(f'resnum {residues}')
        writePDB('tempshell.pdb', shell)
        # Write temporary pdb files for further merging
        # temp_shell = parsePDB('tempshell.pdb', biomol=False)
        if resi not in itered:
            print(f"...Iterating over resi {resi}")
            itered.append(resi)
        else:
            pass
        # ensemble = PDBEnsemble(outstring)
        # ensemble.setAtoms(initial_pdb)
        # ensemble.setCoords(initial_pdb.getCoords())
        # startLogfile("logimpose" + outstring + ".prd")
        # ensemble.addCoordset(temp_shell)
        atoms_init = initial_pdb.select(f"not (resnum {residues})")
        writePDB('tempinit.pdb', atoms_init)
        # Read more about external tools used at https://github.com/haddocking/pdb-tools
        merge = ["pdb_merge", "tempinit.pdb", "tempshell.pdb"]
        sort = ["pdb_sort"]
        reatom = ["pdb_reatom"]
        tidy = ["pdb_tidy"]
        # Launch external processes
        try:
            wfile = open(f"{outstring}.pdb", "w")
            process_merge = subprocess.Popen(merge, stdout=subprocess.PIPE, shell=False)
            process_sort = subprocess.Popen(sort, stdin=process_merge.stdout, stdout=subprocess.PIPE, shell=False)
            process_merge.wait()
            process_merge.stdout.close()
            process_reatom = subprocess.Popen(reatom, stdin=process_sort.stdout, stdout=subprocess.PIPE, shell=False)
            process_sort.wait()
            process_sort.stdout.close()
            process_tidy = subprocess.Popen(tidy, stdin=process_reatom.stdout, stdout=wfile, shell=False)
            process_reatom.wait()
            process_reatom.stdout.close()
            process_tidy.wait()
            wfile.close()
        except NotADirectoryError:
            print("Please, install pdb-tools from https://github.com/JoaoRodrigues/pdb-tools/")
            break
        subprocess.call(["sed", "-i", "1,2d", f"{outstring}.pdb"])
        os.remove('tempshell.pdb')
        os.remove('tempinit.pdb')
        shutil.move(f"./{outstring}.pdb", f"./{out}/")

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
    res_file_dict = file_dir(pdb, files)
    insert_shell(pdb, res_file_dict, radius, outname)
