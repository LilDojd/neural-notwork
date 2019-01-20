#! /usr/bin/env python3

"""Generate backrub resfiles for each mutation of each residue in a set"""
import argparse
import os
import subprocess
import sys
import time

try:
    from suplementary_scripts.define_pdb_shell_prody import extract_res_list, get_structure, get_shell
except ModuleNotFoundError:
    from suplementary_scripts.define_pdb_shell import extract_res_list
from pathlib import Path
from multiprocessing import Pool

aas = [
    "A",
    "D",
    "F",
    "K"
]


def bash_command(cmd):
    subprocess.Popen(['/bin/bash', '-c', cmd])


class NoStdStreams(object):
    def __init__(self, stdout=None, stderr=None):
        self.devnull = open(os.devnull, 'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()


def make_resfile(residue, aminoacid, tofold, command="NATRO"):
    extra = ""
    for i in tofold:
        a = str(i)
        extra += f"{a:<4}_ NATAA\n"
    with open("dfpase.resfile", "w") as outf:
        outf.write(
            f"{command}\n"
            "EX 1 EX 2 EX 3 EX 4\n"
            "start\n"
            "\n"
            f"{residue:<4}_ PIKAA {aminoacid}\n{extra}"
        )
    pass


def write_rosbash(residue, am):
    strtoadd: str = str(residue) + am
    with open("ros.bash", "w") as outf:
        outf.write("#!/bin/bash\n"
                   "ROS=/home/domain/data/prog/rosetta_bin_linux_2016.32.58837_bundle/main/source/bin/backrub"
                   ".linuxgccrelease\n "
                   "PDB=$1\n"
                   "RES=$2\n"
                   "\n"
                   "$ROS -s ./$PDB \\\n"
                   "-resfile ./$RES \\\n"
                   "-extra_res_fa ./LG.params \\\n"
                   "-overwrite \\\n"
                   "-nstruct 1 -backrub:ntrials 5000 \\\n"
                   "-ignore_zero_occupancy false \\\n"
                   "-out:prefix struct_${RES/.resfile/_%s_}\\\n" % strtoadd)
    pass


def prepare_dirs(resindex, tofold):
    output_dir = Path(".") / f"dfpase_{resindex}"
    output_dir.mkdir(exist_ok=True)
    os.chdir(output_dir)
    for ac in aas:
        ac_dir = Path(".") / f"{output_dir}_{ac}"
        ac_dir.mkdir(exist_ok=True)
        os.chdir(ac_dir)
        command1 = "cp ../../%s ./" % pdb
        bash_command(command1)
        bash_command("cp ../../LG.params ./")
        write_rosbash(resindex, ac)
        make_resfile(resindex, ac, tofold)
        os.chdir("../")
    os.chdir("../")
    pass


def start_backrub(directory):
    subprocess.call(
        f"(cd {directory}; bash ros.bash {pdb} dfpase.resfile 1>> "
        f"rosetta.log;echo Backrub finished for {directory};cd ../../)",
        shell=True)
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch backrub")
    parser.add_argument("pdb", help="Path to pdb file")
    parser.add_argument("reslist", help="Path to a file containing list of atoms to mutate")
    parser.add_argument("--mode", dest='mode', nargs='?', default="both", help="Choose between mkdir, launch and "
                                                                               "both (by default) modes")
    parser.add_argument("--cores", dest='cores', nargs='?', default='8', type=int, help="Number of threads [8]")
    parser.add_argument("--radius", dest='radius', nargs='?', default='5', type=float, help="Shell radius [5]")

    args = parser.parse_args()
    res_list = extract_res_list(args.reslist)
    cores = args.cores
    pdb = args.pdb
    radius = args.radius
    modes = ["launch", "mkdir", "both"]
    mode = args.mode
    if mode not in modes:
        raise ValueError("No such mode")
    elif mode == 'mkdir':
        struct = get_structure(pdb)
        # Make list of lists of atoms of each residue
        shell = get_shell(res_list, struct, radius)
        for resi, nbs in shell.items():
            prepare_dirs(resi, nbs)

    elif mode == "launch":
        time.sleep(2)
        print("Starting backrub")
        time.sleep(2)
        filenames = os.walk(".")  # get all files' and folders' names in the current directory
        dirlist = []
        for direct in filenames:
            if "dfpase_" in direct[0]:
                for d in direct[1]:
                    dstring = d[:-2] + d[-1:]
                    search_str = "struct_" + dstring + "_start_lig_fix_0001_low.pdb"
                    path = f"{direct[0]}/{d}/"
                    filepath = path + search_str
                    if Path(filepath).exists():
                        pass
                    else:
                        dirlist.append(path)
        dirlist = list(set(dirlist))
        print(len(dirlist))
        with Pool(processes=cores) as pool:
            pool.map(start_backrub, dirlist)
    elif mode == "both":
        struct = get_structure(pdb)
        # Make list of lists of atoms of each residue
        shell = get_shell(res_list, struct, radius)

        for resi, nbs in shell.items():
            prepare_dirs(resi, nbs)

        time.sleep(2)
        print("Starting backrub")
        time.sleep(2)
        filenames = os.walk(".")  # get all files' and folders' names in the current directory
        dirlist = []
        for direct in filenames:
            if "dfpase_" in direct[0]:
                for d in direct[1]:
                    dirlist.append(f"{direct[0]}/{d}/")
        with Pool(processes=cores) as pool:
            pool.map(start_backrub, dirlist)
