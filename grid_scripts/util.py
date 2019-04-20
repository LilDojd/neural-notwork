
# coding: utf-8


# In[10]:

import argparse as ag
import ConfigParser
import numpy as np

from Bio.Seq import Seq
import Bio.Alphabet
import Bio.Alphabet.IUPAC

# MPI parallelism
from mpi4py import MPI
import h5py

from copy import copy

np.set_printoptions(threshold='nan')

natoms = ('OP1', 'OP2', 'O1P', 'O2P', 'N4', 'N6', 'O4',
          'O6', 'N7', 'N2', 'N3', 'O2', "O3'", "O5'", "O2'", "O4'",)


def get_bounding(R, padding, step):
    minXYZ = np.amin(R, axis=0)
    maxXYZ = np.amax(R, axis=0)
    minXYZ -= (padding + step)
    maxXYZ += (padding + step)
    shape = np.ceil((maxXYZ - minXYZ) / step).astype(np.int)
    minXYZ = adjust_grid(minXYZ, step)
    return(minXYZ, shape)


def adjust_grid(c, step, padding=0):
    nc = c - padding
    nc /= step
    nc = np.floor(nc) * step
    return nc


def sphere2grid(c, r, step, value=-np.inf):
    nc = adjust_grid(c, step)
    adj = nc - c
    r2x = np.arange(-r - adj[0], r - adj[0] + step, step) ** 2
    r2y = np.arange(-r - adj[1], r - adj[1] + step, step) ** 2
    r2z = np.arange(-r - adj[2], r - adj[2] + step, step) ** 2
    dist2 = r2x[:, None, None] + r2y[:, None] + r2z
    nc = nc - dist2.shape[0] / 2.0 * step
    vol = (dist2 <= r ** 2.0).astype(np.int)
    vol *= value
    return (nc, vol)


def hist2grid(c, dat, bins, step):
    r = bins[-1]
    nc = adjust_grid(c, step)
    adj = nc - c
    r2x = np.arange(-r - adj[0], r - adj[0] + step, step) ** 2
    r2y = np.arange(-r - adj[1], r - adj[1] + step, step) ** 2
    r2z = np.arange(-r - adj[2], r - adj[2] + step, step) ** 2

    dist2 = r2x[:, None, None] + r2y[:, None] + r2z
    grid = np.ndarray(dist2.shape, dtype=np.float)
    grid.fill(0.0)

    for i in range(len(bins) - 1):
        grid[(dist2 > bins[i] ** 2) & (dist2 <= bins[i + 1] ** 2)] = dat[i]

    nc = nc - dist2.shape[0] / 2 * step

    return (nc, grid)


#  http://www.weslack.com/question/1552900000001895550

def submatrix(arr):
    x, y, z = np.nonzero(arr)
    # Using the smallest and largest x and y indices of nonzero elements,
    # we can find the desired rectangular bounds.
    # And don't forget to add 1 to the top bound to avoid the fencepost
    # problem.
    return (
        np.array(
            [x.min(), y.min(), z.min()]
        ),
        np.array(
            [x.max() + 1, y.max() + 1, z.max() + 1]
        ),
    )


def process_residue(tR, padding, step):

    RminXYZ, Rshape = get_bounding(tR.getCoords(), padding, step)
    Rgrid = np.zeros(Rshape, dtype=np.int)

    for A in tR.iterAtoms():

        AminXYZ, Agrid = sphere2grid(
            A.getCoords(), avdw[A.getElement()], step, 1)
        NA = Agrid.shape[0]

        adj = (AminXYZ - RminXYZ)
        adj = (adj / step).astype(np.int)
        x, y, z = adj
        Rgrid[x: x + NA, y: y + NA, z: z + NA] += Agrid

    np.clip(Rgrid, 0, 1, out=Rgrid)
    return (Rgrid, RminXYZ)


def process_atom(A, step):

    AminXYZ, Agrid = sphere2grid(
        A.getCoords(), avdw[A.getElement()], step, 1)

    np.clip(Agrid, 0, 1, out=Agrid)
    return (Agrid, AminXYZ)


def process_atom_oddt(A, step):

    AminXYZ, Agrid = sphere2grid(
        A[1], avdw[A[5][0]], step, 1)

    np.clip(Agrid, 0, 1, out=Agrid)
    return (Agrid, AminXYZ)


aradii = {
    'H': 0.53,
    'C': 0.67,
    'N': 0.56,
    'O': 0.48,
    'P': 0.98,
    'S': 0.88,
}

avdw = {
    'H': 1.2,
    'C': 1.7,
    'N': 1.55,
    'O': 1.52,
    'P': 1.8,
    'S': 1.8,
}

avdw_ua = {
    'H': 2.3,
    'C': 3.0,
    'N': 2.8,
    'O': 2.6,
    'P': 2.9,
    'S': 2.9,
}

atomschgd = [
    'NH2',
    'NH1',
    'NE',
    'NZ',
    'NE2',
    'ND2',
]

atomsl = {
    'OE2': 'X',
    'OE1': 'X',   # Glutamate
    'OD1': 'X',
    'OD2': 'X',   # Aspartate
    'O': 'X',    # Peptide bond
    'OG': 'X',   # Serine
    'OG1': 'X',  # Threonine
    'OH': 'X',   # Tyrosine
    'SG': 'X',   # Cysteine
    'NE2': 'HIS',  # Histidine, Glutamine
    'ND1': 'X',  # Histidine
    # 'ND2',  # Asparagine
    # 'NE',   # Arginine
    'OP1': 'X',
    'OP2': 'X',  # Phosphate
}

atomsdir = {
    'OE1': 'CD',
    'OE2': 'CD',
    'OD1': 'CG',
    'OD2': 'CG',
    'O': 'C',
    'OG': 'CB',
    'OG1': 'CB',
    'SG': 'CB',
    'OH': 'CZ',
    'NE2': {'HIS': 'CE1', 'GLN': 'CD'},
    'ND1': 'CE1',
    'OP1': 'P',
    'OP2': 'P',
    'NH1': 'CZ',
    'NH2': 'CZ',
    'NE': 'CZ',
    'NZ': 'CE',
    'N': 'CA',
}


# http://stackoverflow.com/questions/2819696/parsing-properties-file-in-python/2819788#2819788

class FakeSecHead(object):

    def __init__(self, fp):
        self.fp = fp
        self.sechead = '[asection]\n'

    def readline(self):
        if self.sechead:
            try:
                return self.sechead
            finally:
                self.sechead = None
        else:
            return self.fp.readline()


def get_box_from_vina(f):
    cp = ConfigParser.SafeConfigParser()
    cp.readfp(FakeSecHead(open(f)))
    cfg = dict(cp.items('asection'))

    center = np.array([
                      cfg['center_x'],
                      cfg['center_y'],
                      cfg['center_z'],
                      ],
                      dtype=np.float)

    box = np.array([
                   cfg['size_x'],
                   cfg['size_y'],
                   cfg['size_z'],
                   ],
                   dtype=np.float)

    return (center, box)


atypes = {
    ("ALA", "CA"): "CA_BCK",
    ("ALA", "CB"): "C",
    ("ALA", "C"): "C_BCK",
    # ("ALA", "N"): "N",
    ("ALA", "N"): "NA_BCK",
    ("ALA", "O"): "OA_BCK",
    ("ARG", "CA"): "CA_BCK",
    ("ARG", "CB"): "C",
    ("ARG", "C"): "C_BCK",
    ("ARG", "CD"): "C",
    ("ARG", "CG"): "C",
    ("ARG", "CZ"): "C",
    ("ARG", "NE"): "NA",
    ("ARG", "NH1"): "N",
    ("ARG", "NH2"): "N",
    # ("ARG", "N"): "N",
    ("ARG", "N"): "NA_BCK",
    ("ARG", "O"): "OA_BCK",
    ("ASN", "CA"): "CA_BCK",
    ("ASN", "CB"): "C",
    ("ASN", "C"): "C_BCK",
    ("ASN", "CG"): "C",
    ("ASN", "ND2"): "N",
    # ("ASN", "N"): "N",
    ("ASN", "N"): "NA_BCK",
    ("ASN", "OD1"): "OA",
    ("ASN", "O"): "OA_BCK",
    ("ASP", "CA"): "CA_BCK",
    ("ASP", "CB"): "C",
    ("ASP", "C"): "C_BCK",
    ("ASP", "CG"): "C",
    # ("ASP", "N"): "N",
    ("ASP", "N"): "NA_BCK",
    ("ASP", "OD1"): "OA",
    ("ASP", "OD2"): "OA",
    ("ASP", "O"): "OA_BCK",
    ("CYS", "CA"): "CA_BCK",
    ("CYS", "CB"): "C",
    ("CYS", "C"): "C_BCK",
    # ("CYS", "N"): "N",
    ("CYS", "N"): "NA_BCK",
    ("CYS", "O"): "OA_BCK",
    ("CYS", "SG"): "SA",
    ("GLN", "CA"): "CA_BCK",
    ("GLN", "CB"): "C",
    ("GLN", "C"): "C_BCK",
    ("GLN", "CD"): "C",
    ("GLN", "CG"): "C",
    ("GLN", "NE2"): "N",
    # ("GLN", "N"): "N",
    ("GLN", "N"): "NA_BCK",
    ("GLN", "OE1"): "OA",
    ("GLN", "O"): "OA_BCK",
    ("GLU", "CA"): "CA_BCK",
    ("GLU", "CB"): "C",
    ("GLU", "C"): "C_BCK",
    ("GLU", "CD"): "C",
    ("GLU", "CG"): "C",
    # ("GLU", "N"): "N",
    ("GLU", "N"): "NA_BCK",
    ("GLU", "OE1"): "OA",
    ("GLU", "OE2"): "OA",
    ("GLU", "O"): "OA_BCK",
    ("GLY", "CA"): "CA_BCK",
    ("GLY", "C"): "C_BCK",
    # ("GLY", "N"): "N",
    ("GLY", "N"): "NA_BCK",
    ("GLY", "O"): "OA_BCK",
    ("HIS", "CA"): "CA_BCK",
    ("HIS", "CB"): "C",
    ("HIS", "C"): "C_BCK",
    ("HIS", "CD2"): "A",
    ("HIS", "CE1"): "A",
    ("HIS", "CG"): "A",
    ("HIS", "ND1"): "NA",
    ("HIS", "NE2"): "NA",
    # ("HIS", "N"): "N",
    ("HIS", "N"): "NA_BCK",
    ("HIS", "O"): "OA_BCK",
    ("ILE", "CA"): "CA_BCK",
    ("ILE", "CB"): "C",
    ("ILE", "C"): "C_BCK",
    ("ILE", "CD1"): "C",
    ("ILE", "CG1"): "C",
    ("ILE", "CG2"): "C",
    # ("ILE", "N"): "N",
    ("ILE", "N"): "NA_BCK",
    ("ILE", "O"): "OA_BCK",
    ("LEU", "CA"): "CA_BCK",
    ("LEU", "CB"): "C",
    ("LEU", "C"): "C_BCK",
    ("LEU", "CD1"): "C",
    ("LEU", "CD2"): "C",
    ("LEU", "CG"): "C",
    # ("LEU", "N"): "N",
    ("LEU", "N"): "NA_BCK",
    ("LEU", "O"): "OA_BCK",
    ("LYS", "CA"): "CA_BCK",
    ("LYS", "CB"): "C",
    ("LYS", "C"): "C_BCK",
    ("LYS", "CD"): "C",
    ("LYS", "CE"): "C",
    ("LYS", "CG"): "C",
    # ("LYS", "N"): "N",
    ("LYS", "N"): "NA_BCK",
    ("LYS", "NZ"): "N",
    ("LYS", "O"): "OA_BCK",
    ("MET", "CA"): "CA_BCK",
    ("MET", "CB"): "C",
    ("MET", "C"): "C_BCK",
    ("MET", "CE"): "C",
    ("MET", "CG"): "C",
    # ("MET", "N"): "N",
    ("MET", "N"): "NA_BCK",
    ("MET", "O"): "OA_BCK",
    ("MET", "SD"): "SA",
    ("PHE", "CA"): "CA_BCK",
    ("PHE", "CB"): "C",
    ("PHE", "C"): "C_BCK",
    ("PHE", "CD1"): "A",
    ("PHE", "CD2"): "A",
    ("PHE", "CE1"): "A",
    ("PHE", "CE2"): "A",
    ("PHE", "CG"): "A",
    ("PHE", "CZ"): "A",
    # ("PHE", "N"): "N",
    ("PHE", "N"): "NA_BCK",
    ("PHE", "O"): "OA_BCK",
    ("PRO", "CA"): "CA_BCK",
    ("PRO", "CB"): "C",
    ("PRO", "C"): "C_BCK",
    ("PRO", "CD"): "C",
    ("PRO", "CG"): "C",
    ("PRO", "N"): "NA_BCK",
    # ("PRO", "N"): "NA_BCK",
    ("PRO", "O"): "OA_BCK",
    ("SER", "CA"): "CA_BCK",
    ("SER", "CB"): "C",
    ("SER", "C"): "C_BCK",
    # ("SER", "N"): "N",
    ("SER", "N"): "NA_BCK",
    ("SER", "OG"): "OA",
    ("SER", "O"): "OA_BCK",
    ("THR", "CA"): "CA_BCK",
    ("THR", "CB"): "C",
    ("THR", "C"): "C_BCK",
    ("THR", "CG2"): "C",
    # ("THR", "N"): "N",
    ("THR", "N"): "NA_BCK",
    ("THR", "OG1"): "OA",
    ("THR", "O"): "OA_BCK",
    ("TRP", "CA"): "CA_BCK",
    ("TRP", "CB"): "C",
    ("TRP", "C"): "C_BCK",
    ("TRP", "CD1"): "A",
    ("TRP", "CD2"): "A",
    ("TRP", "CE2"): "A",
    ("TRP", "CE3"): "A",
    ("TRP", "CG"): "A",
    ("TRP", "CH2"): "A",
    ("TRP", "CZ2"): "A",
    ("TRP", "CZ3"): "A",
    ("TRP", "NE1"): "NA",
    # ("TRP", "N"): "N",
    ("TRP", "N"): "NA_BCK",
    ("TRP", "O"): "OA_BCK",
    ("TYR", "CA"): "CA_BCK",
    ("TYR", "CB"): "C",
    ("TYR", "C"): "C_BCK",
    ("TYR", "CD1"): "A",
    ("TYR", "CD2"): "A",
    ("TYR", "CE1"): "A",
    ("TYR", "CE2"): "A",
    ("TYR", "CG"): "A",
    ("TYR", "CZ"): "A",
    # ("TYR", "N"): "N",
    ("TYR", "N"): "NA_BCK",
    ("TYR", "OH"): "OA",
    ("TYR", "O"): "OA_BCK",
    ("VAL", "CA"): "CA_BCK",
    ("VAL", "CB"): "C",
    ("VAL", "C"): "C_BCK",
    ("VAL", "CG1"): "C",
    ("VAL", "CG2"): "C",
    # ("VAL", "N"): "N",
    ("VAL", "N"): "NA_BCK",
    ("VAL", "O"): "OA_BCK",

    ("NME", "N"): "NA_BCK",
    ("NME", "C"): "OA_BCK",
    ("ACE", "CH3"): "C",
    ("ACE", "O"): "OA_BCK",
    ("ACE", "C"): "C_BCK",

}

one_let = [
    "A", "R", "N", "D", "C", "E", "Q", "G", "H",
    "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "BCK", "UNK"]


three_let = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "BCK", "UNK", "NME", "ACE",
]


def get_args():
    """Parse cli arguments"""

#    choices = ['grid', 'score']

    parser = ag.ArgumentParser(
        description='Grid scripts')

    parser.add_argument('-m',
                        required=True,
                        dest='Sfn',
                        metavar='FILE.hdf5',
                        help='HDF5 file for all matrices')

#    parser.add_argument('-t', '--task',
#                        nargs='+',
#                        required=True,
#                        choices=choices,
#                        metavar='TASK',
#                        help='Task to do. Available options \
#                        are: %s' % ", ".join(choices))

    parser.add_argument('-o', '--output',
                        dest='output',
                        metavar='OUTPUT',
                        # help='For "render" ans "cluster_to_trj" tasks \
                        # name of output PNG image of mutiframe PDB file'
                        )

    parser.add_argument('--debug',
                        action='store_true',
                        help='Perform profiling')

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Be verbose')

    parser.add_argument('--default_types',
                        action='store_true',
                        default=True,
                        help='Use default set of atom types for Vina')

    parser.add_argument('-f',
                        nargs='+',
                        type=str,
                        dest='pdb_list',
                        metavar='FILE',
                        help='PDB files')

    parser.add_argument('-e', '--exclude',
                        nargs='*',
                        type=str,
                        dest='excl',
                        metavar='TYPE',
                        help='Vina types to exclude from scoring')

    parser.add_argument('-i', '--include',
                        nargs='*',
                        type=str,
                        dest='incl',
                        metavar='TYPE',
                        help='Vina types to exclude from scoring')

    parser.add_argument('-s', '--step',
                        type=float,
                        dest='step',
                        metavar='STEP',
                        default=0.1,
                        help='Grid step in Angstroms')

    parser.add_argument('-p', '--padding',
                        type=int,
                        dest='pad',
                        metavar='PADDING',
                        default=1,
                        help='Padding around cell in Angstroms')

    parser.add_argument('-c', '--config',
                        type=str,
                        dest='config',
                        metavar='CONFIG',
                        help='Vina config file')

    args = parser.parse_args()

    args_dict = vars(args)

    return args_dict


def get_args_box():
    """Parse cli arguments"""

    parser = ag.ArgumentParser()

    parser.add_argument('-o', '--output',
                        dest='output',
                        metavar='OUTPUT',
                        help='For "render" ans "cluster_to_trj" tasks \
                        name of output PNG image of mutiframe PDB file')

    parser.add_argument('--debug',
                        action='store_true',
                        help='Perform profiling')

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Be verbose')

    parser.add_argument('-f',
                        nargs='+',
                        type=str,
                        dest='pdb_list',
                        metavar='FILE',
                        help='PDB files')

    parser.add_argument('-s', '--step',
                        type=float,
                        dest='step',
                        metavar='STEP',
                        default=0.1,
                        help='Grid step in Angstroms')

    parser.add_argument('-p', '--padding',
                        type=int,
                        dest='pad',
                        metavar='PADDING',
                        default=1,
                        help='Padding around cell in Angstroms')

    args = parser.parse_args()

    args_dict = vars(args)

    return args_dict


vina_types = (
    # 'H',
    'C.2',
    'C.3',
    'C.ar',
    'C.cat',
    'O.2',
    'O.3',
    'O.co2',
    'N.3',
    'N.4',
    'N.ar',
    'N.am',
    'N.pl3',
    'S.3',
)


def check_pept(seq):
    tseq = Seq(seq, Bio.Alphabet.IUPAC.protein)
    if Bio.Alphabet._verify_alphabet(tseq):
        return seq
    else:
        msg = "%s is not a valid peptide sequence" % tseq
        raise ag.ArgumentTypeError(msg)


def init_mpi():
    class MPI_basket(object):
        def __init__(self):
            pass

    mpi = MPI_basket()

    # Get MPI info
    mpi.comm = MPI.COMM_WORLD
    # Get number of processes
    mpi.NPROCS = mpi.comm.size
    # Get rank
    mpi.rank = mpi.comm.rank

    return mpi


def task(N, mpi):
    l = np.ceil(float(N) / mpi.NPROCS).astype(np.int)
    b = mpi.rank * l
    e = min(b + l, N)

    return (b, e)


gromos_types = {
    ("ALA", "N"): "N",
    ("ALA", "CA"): "CH1",
    ("ALA", "CB"): "CH3",
    ("ALA", "C"): "C",
    ("ALA", "O"): "O",
    #
    ("ARG", "N"): "N",
    ("ARG", "CA"): "CH1",
    ("ARG", "CB"): "CH2",
    ("ARG", "CG"): "CH2",
    ("ARG", "CD"): "CH2",
    ("ARG", "NE"): "NE",
    ("ARG", "CZ"): "C",
    ("ARG", "NH1"): "NZ",
    ("ARG", "NH2"): "NZ",
    ("ARG", "C"): "C",
    ("ARG", "O"): "O",
    #
    ("ASN", "N"): "N",
    ("ASN", "CA"): "CH1",
    ("ASN", "CB"): "CH2",
    ("ASN", "CG"): "C",
    ("ASN", "OD1"): "O",
    ("ASN", "ND2"): "NT",
    ("ASN", "C"): "C",
    ("ASN", "O"): "O",
    #
    ("ASP", "N"): "N",
    ("ASP", "CA"): "CH1",
    ("ASP", "CB"): "CH2",
    ("ASP", "CG"): "C",
    ("ASP", "OD1"): "OM",
    ("ASP", "OD2"): "OM",
    ("ASP", "C"): "C",
    ("ASP", "O"): "O",
    #
    ("CYS", "N"): "N",
    ("CYS", "CA"): "CH1",
    ("CYS", "CB"): "CH2",
    ("CYS", "SG"): "S",
    ("CYS", "C"): "C",
    ("CYS", "O"): "O",
    #
    ("GLN", "N"): "N",
    ("GLN", "CA"): "CH1",
    ("GLN", "CB"): "CH2",
    ("GLN", "CG"): "CH2",
    ("GLN", "CD"): "C",
    ("GLN", "OE1"): "O",
    ("GLN", "NE2"): "NT",
    ("GLN", "C"): "C",
    ("GLN", "O"): "O",
    #
    ("GLU", "N"): "N",
    ("GLU", "CA"): "CH1",
    ("GLU", "CB"): "CH2",
    ("GLU", "CG"): "CH2",
    ("GLU", "CD"): "C",
    ("GLU", "OE1"): "OM",
    ("GLU", "OE2"): "OM",
    ("GLU", "C"): "C",
    ("GLU", "O"): "O",
    #
    ("GLY", "N"): "N",
    ("GLY", "CA"): "CH2",
    ("GLY", "C"): "C",
    ("GLY", "O"): "O",
    #
    ("HIS", "N"): "N",
    ("HIS", "CA"): "CH1",
    ("HIS", "CB"): "CH2",
    ("HIS", "CG"): "C",
    ("HIS", "ND1"): "NR",
    ("HIS", "CD2"): "C",
    ("HIS", "CE1"): "C",
    ("HIS", "NE2"): "NR",
    ("HIS", "C"): "C",
    ("HIS", "O"): "O",
    #
    ("ILE", "N"): "N",
    ("ILE", "CA"): "CH1",
    ("ILE", "CB"): "CH1",
    ("ILE", "CG1"): "CH2",
    ("ILE", "CG2"): "CH3",
    ("ILE", "CD"): "CH3",
    ("ILE", "CD1"): "CH3",  # backup name variant
    ("ILE", "C"): "C",
    ("ILE", "O"): "O",
    #
    ("LEU", "N"): "N",
    ("LEU", "CA"): "CH1",
    ("LEU", "CB"): "CH2",
    ("LEU", "CG"): "CH1",
    ("LEU", "CD1"): "CH3",
    ("LEU", "CD2"): "CH3",
    ("LEU", "C"): "C",
    ("LEU", "O"): "O",
    #
    ("LYS", "N"): "N",
    ("LYS", "CA"): "CH1",
    ("LYS", "CB"): "CH2",
    ("LYS", "CG"): "CH2",
    ("LYS", "CD"): "CH2",
    ("LYS", "CE"): "CH2",
    ("LYS", "NZ"): "NT",
    ("LYS", "C"): "C",
    ("LYS", "O"): "O",
    #
    ("MET", "N"): "N",
    ("MET", "CA"): "CH1",
    ("MET", "CB"): "CH2",
    ("MET", "CG"): "CH2",
    ("MET", "SD"): "S",
    ("MET", "CE"): "CH3",
    ("MET", "C"): "C",
    ("MET", "O"): "O",
    #
    ("PHE", "N"): "N",
    ("PHE", "CA"): "CH1",
    ("PHE", "CB"): "CH2",
    ("PHE", "CG"): "C",
    ("PHE", "CD1"): "C",
    ("PHE", "CD2"): "C",
    ("PHE", "CE1"): "C",
    ("PHE", "CE2"): "C",
    ("PHE", "CZ"): "C",
    ("PHE", "C"): "C",
    ("PHE", "O"): "O",
    #
    ("PRO", "N"): "N",
    ("PRO", "CA"): "CH1",
    ("PRO", "CB"): "CH2",  # originally CH2r
    ("PRO", "CG"): "CH2",  # originally CH2r
    ("PRO", "CD"): "CH2",  # originally CH2r
    ("PRO", "C"): "C",
    ("PRO", "O"): "O",
    #
    ("SER", "N"): "N",
    ("SER", "CA"): "CH1",
    ("SER", "CB"): "CH2",
    ("SER", "OG"): "OA",
    ("SER", "C"): "C",
    ("SER", "O"): "O",
    #
    ("THR", "N"): "N",
    ("THR", "CA"): "CH1",
    ("THR", "CB"): "CH2",
    ("THR", "OG1"): "OA",
    ("THR", "CG2"): "CH3",
    ("THR", "C"): "C",
    ("THR", "O"): "O",
    #
    ("TRP", "N"): "N",
    ("TRP", "CA"): "CH1",
    ("TRP", "CB"): "CH2",
    ("TRP", "CG"): "C",
    ("TRP", "CD1"): "C",
    ("TRP", "CD2"): "C",
    ("TRP", "NE1"): "NR",
    ("TRP", "CE2"): "C",
    ("TRP", "CE3"): "C",
    ("TRP", "CZ2"): "C",
    ("TRP", "CZ3"): "C",
    ("TRP", "CH2"): "C",
    ("TRP", "C"): "C",
    ("TRP", "O"): "O",
    #
    ("TYR", "N"): "N",
    ("TYR", "CA"): "CH1",
    ("TYR", "CB"): "CH2",
    ("TYR", "CG"): "C",
    ("TYR", "CD1"): "C",
    ("TYR", "CD2"): "C",
    ("TYR", "CE1"): "C",
    ("TYR", "CE2"): "C",
    ("TYR", "CZ"): "C",
    ("TYR", "OH"): "OA",
    ("TYR", "C"): "C",
    ("TYR", "O"): "O",
    #
    ("VAL", "N"): "N",
    ("VAL", "CA"): "CH1",
    ("VAL", "CB"): "CH2",
    ("VAL", "CG1"): "CH3",
    ("VAL", "CG2"): "CH3",
    ("VAL", "C"): "C",
    ("VAL", "O"): "O",
    #
    ("ACE", "CH3"): "CH3",
    ("ACE", "C"): "C",
    ("ACE", "O"): "O",
    #
    ("NME", "N"): "N",
    ("NME", "C"): "CH3",
    ("NME", "CH3"): "CH3",
}

tripos_types = {
    ("ACE", "O"): "O.2",
    ("ACE", "CH3"): "C.3",
    ("ACE", "C"): "C.2",
    ("ALA", "CA"): "C.3",
    ("ALA", "O"): "O.2",
    ("ALA", "CB"): "C.3",
    ("ALA", "C"): "C.2",
    ("ALA", "N"): "N.am",
    ("ARG", "C"): "C.2",
    ("ARG", "CD"): "C.3",
    ("ARG", "CZ"): "C.cat",
    ("ARG", "NH1"): "N.pl3",
    ("ARG", "CA"): "C.3",
    ("ARG", "N"): "N.am",
    ("ARG", "NE"): "N.pl3",
    ("ARG", "CB"): "C.3",
    ("ARG", "O"): "O.2",
    ("ARG", "NH2"): "N.pl3",
    ("ARG", "CG"): "C.3",
    ("ASN", "OD1"): "O.2",
    ("ASN", "N"): "N.am",
    ("ASN", "CB"): "C.3",
    ("ASN", "C"): "C.2",
    ("ASN", "O"): "O.2",
    ("ASN", "ND2"): "N.am",
    ("ASN", "CG"): "C.2",
    ("ASN", "CA"): "C.3",
    ("ASP", "CB"): "C.3",
    ("ASP", "O"): "O.2",
    ("ASP", "C"): "C.2",
    ("ASP", "N"): "N.am",
    ("ASP", "OD2"): "O.co2",
    ("ASP", "OD1"): "O.co2",
    ("ASP", "CA"): "C.3",
    ("ASP", "CG"): "C.2",
    ("CYS", "CB"): "C.3",
    ("CYS", "C"): "C.2",
    ("CYS", "O"): "O.2",
    ("CYS", "SG"): "S.3",
    ("CYS", "CA"): "C.3",
    ("CYS", "N"): "N.am",
    ("GLN", "CG"): "C.3",
    ("GLN", "O"): "O.2",
    ("GLN", "NE2"): "N.am",
    ("GLN", "C"): "C.2",
    ("GLN", "N"): "N.am",
    ("GLN", "OE1"): "O.2",
    ("GLN", "CA"): "C.3",
    ("GLN", "CD"): "C.2",
    ("GLN", "CB"): "C.3",
    ("GLU", "OE1"): "O.co2",
    ("GLU", "CG"): "C.3",
    ("GLU", "O"): "O.2",
    ("GLU", "CD"): "C.2",
    ("GLU", "CA"): "C.3",
    ("GLU", "OE2"): "O.co2",
    ("GLU", "N"): "N.am",
    ("GLU", "CB"): "C.3",
    ("GLU", "C"): "C.2",
    ("GLY", "CA"): "C.3",
    ("GLY", "C"): "C.2",
    ("GLY", "N"): "N.am",
    ("GLY", "O"): "O.2",
    ("HIS", "O"): "O.2",
    ("HIS", "NE2"): "N.ar",
    ("HIS", "CE1"): "C.ar",
    ("HIS", "N"): "N.am",
    ("HIS", "CD2"): "C.ar",
    ("HIS", "CG"): "C.ar",
    ("HIS", "CB"): "C.3",
    ("HIS", "CA"): "C.3",
    ("HIS", "ND1"): "N.ar",
    ("HIS", "C"): "C.2",
    ("ILE", "O"): "O.2",
    ("ILE", "N"): "N.am",
    ("ILE", "CB"): "C.3",
    ("ILE", "CG2"): "C.3",
    ("ILE", "CA"): "C.3",
    ("ILE", "C"): "C.2",
    ("ILE", "CG1"): "C.3",
    ("ILE", "CD"): "C.3",
    ("LEU", "CG"): "C.3",
    ("LEU", "N"): "N.am",
    ("LEU", "CD1"): "C.3",
    ("LEU", "CD2"): "C.3",
    ("LEU", "CA"): "C.3",
    ("LEU", "C"): "C.2",
    ("LEU", "O"): "O.2",
    ("LEU", "CB"): "C.3",
    ("LYS", "CE"): "C.3",
    ("LYS", "C"): "C.2",
    ("LYS", "NZ"): "N.4",
    ("LYS", "CA"): "C.3",
    ("LYS", "CB"): "C.3",
    ("LYS", "CG"): "C.3",
    ("LYS", "CD"): "C.3",
    ("LYS", "N"): "N.am",
    ("LYS", "O"): "O.2",
    ("MET", "SD"): "S.3",
    ("MET", "O"): "O.2",
    ("MET", "N"): "N.am",
    ("MET", "CB"): "C.3",
    ("MET", "CA"): "C.3",
    ("MET", "CG"): "C.3",
    ("MET", "C"): "C.2",
    ("MET", "CE"): "C.3",
    ("NME", "N"): "N.am",
    ("NME", "CH3"): "C.3",
    ("PHE", "N"): "N.am",
    ("PHE", "C"): "C.2",
    ("PHE", "CD1"): "C.ar",
    ("PHE", "CD2"): "C.ar",
    ("PHE", "O"): "O.2",
    ("PHE", "CZ"): "C.ar",
    ("PHE", "CA"): "C.3",
    ("PHE", "CE2"): "C.ar",
    ("PHE", "CE1"): "C.ar",
    ("PHE", "CB"): "C.3",
    ("PHE", "CG"): "C.ar",
    ("PRO", "CA"): "C.3",
    ("PRO", "CD"): "C.3",
    ("PRO", "C"): "C.2",
    ("PRO", "N"): "N.am",
    ("PRO", "O"): "O.2",
    ("PRO", "CG"): "C.3",
    ("PRO", "CB"): "C.3",
    ("SER", "CB"): "C.3",
    ("SER", "OG"): "O.3",
    ("SER", "N"): "N.am",
    ("SER", "CA"): "C.3",
    ("SER", "O"): "O.2",
    ("SER", "C"): "C.2",
    ("THR", "CA"): "C.3",
    ("THR", "O"): "O.2",
    ("THR", "N"): "N.am",
    ("THR", "CB"): "C.3",
    ("THR", "OG1"): "O.3",
    ("THR", "CG2"): "C.3",
    ("THR", "C"): "C.2",
    ("TRP", "NE1"): "N.ar",
    ("TRP", "CZ2"): "C.ar",
    ("TRP", "CE3"): "C.ar",
    ("TRP", "C"): "C.2",
    ("TRP", "CD1"): "C.ar",
    ("TRP", "CB"): "C.3",
    ("TRP", "CZ3"): "C.ar",
    ("TRP", "CE2"): "C.ar",
    ("TRP", "O"): "O.2",
    ("TRP", "CD2"): "C.ar",
    ("TRP", "N"): "N.am",
    ("TRP", "CG"): "C.ar",
    ("TRP", "CA"): "C.3",
    ("TRP", "CH2"): "C.ar",
    ("TYR", "CD2"): "C.ar",
    ("TYR", "CD1"): "C.ar",
    ("TYR", "CE1"): "C.ar",
    ("TYR", "CG"): "C.ar",
    ("TYR", "CZ"): "C.ar",
    ("TYR", "OH"): "O.3",
    ("TYR", "N"): "N.am",
    ("TYR", "CB"): "C.3",
    ("TYR", "O"): "O.2",
    ("TYR", "CA"): "C.3",
    ("TYR", "C"): "C.2",
    ("TYR", "CE2"): "C.ar",
    ("VAL", "N"): "N.am",
    ("VAL", "O"): "O.2",
    ("VAL", "CA"): "C.3",
    ("VAL", "CB"): "C.3",
    ("VAL", "CG2"): "C.3",
    ("VAL", "C"): "C.2",
    ("VAL", "CG1"): "C.3",
}



def import_retry(name, maxtry=50):

    for i in range(maxtry):
        try:
            pass
            # import name
            break
        except ImportError:
            pass


def choose_artypes(arg):

    if arg == 'default':
        atypes_ = atypes.values()
        rtypes = atypes
    elif arg == 'gromos':
        atypes_ = gromos_types.values()
        rtypes = gromos_types
    elif arg == 'tripos_neutral':
        atypes_ = tripos_types.values()
        rtypes = tripos_types
    elif arg == 'tripos_charged':
        atypes_ = tripos_types.values()

        rtypes = copy(tripos_types)

        for k, v in rtypes.items():
            r, a = k
            if a == 'O':
                rtypes[k] = "O.co2"
            rtypes[(r, 'OXT')] = "O.co2"

    elif arg == 'vina':
        atypes_ = vina_types
        rtypes = vina_types

    return (atypes_, rtypes)


def read_plist(fname):
    with open(fname, 'r') as f:
        raw = f.readlines()
    plist = list()
    for l in raw:
        l_ = l.strip()
        if l_ == '':
            continue
        if '_' in l_:
            l__ = l_.split('_')
            l_ = l__[0]
        plist.append(l_)

    plist = list(set(plist))

    return np.array(plist, dtype='S%d' % len(plist[0]))


def reset_checkpoint(fn):
    with h5py.File(fn, 'r+') as f:
        f['checkpoint'][:] = np.zeros(f['checkpoint'].shape)
