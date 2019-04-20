#!/usr/bin/env python2
#
# This file is part of the AffBio package for clustering of
# biomolecular structures.
#
# Copyright (c) 2015-2016, by Arthur Zalevsky <aozalevsky@fbb.msu.ru>
#
# AffBio is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# AffBio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public
# License along with AffBio; if not, see
# http://www.gnu.org/licenses, or write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
#

# H5PY for storage
# import h5py

import numpy as np
import time
import argparse as ag
import prody
import sys
import h5py
sys.path.append('/home/golovin/_scratch/progs/grid_scripts/')
sys.path.append('/home/domain/silwer/work/grid_scripts/')
import extract_result as er
import util
from dock_pept import PepDatabase


class PeptMPIWorker(object):

    database = None
    llist = None
    plist = None
    mpi = None
    args = None
    tb = None
    te = None

    def __init__(
        self,
        database=None,
        plist=None,
        verbose=False,
        receptor=None,
        *args,
        **kwargs
    ):

        self.mpi = util.init_mpi()

        plist_ = None

        self.database = PepDatabase(database).database

        if self.mpi.rank == 0:

            if plist:
                plist_ = util.read_plist(plist)
            else:
                plist_ = self.database.keys()

            for i in plist_:
                util.check_pept(i)

        self.plist = self.mpi.comm.bcast(plist_)

        self.eplist = dict([(t[1], t[0]) for t in enumerate(self.plist)])

        N = len(self.plist)
        tb, te = util.task(N, self.mpi)
        te = min(te, N)
        self.tb = tb
        self.te = te
        self.aplist = self.plist[tb:te]

        if len(self.aplist) > 0:
            print(
                'RANK: %d FIRST: %d LAST: %d LIST_FIRST: %s LIST_LAST: %s' % (
                    self.mpi.rank, tb, te, self.aplist[0], self.aplist[-1]))
        else:
            print(
                'RANK: %d NOTHING TO DO' % (
                    self.mpi.rank))

    def process(self):
        pass


class SpecCalc(PeptMPIWorker):

    def __init__(self, args):
        super(SpecCalc, self).__init__(**args)
        self.args = args

    def process(self, args=None):
        if not args:
            args = self.args
        # Sfn = args['Sfn']

        # Open matrix file in parallel mode
        # Sf = h5py.File(Sfn, 'r')
        args['mpi'] = self.mpi
        extractor = er.PepExtractor(**args)
        lM = len(self.aplist)

        dtype = np.dtype([
            ('name', 'S10'),
            ('pnum', '<i4'),
            ('rnum', '<i4'),
            ('dist', '<f8'),
            ('hdist1', '<f8'),
            ('hdist2', '<f8'),
            ('BD', '<f8'),
            ('FL', '<f8'),
            ('AT', '<f8'),
            ('dir', 'S3'),
            ('aln', 'S20')
        ])

        if self.mpi.rank == 0:
            m = self.aplist[0]
            lm = len(m)
            S = extractor.extract_result(m)
            lS = S.numCoordsets()
            Sf = h5py.File(args['out'], 'w')
            out = Sf.create_dataset(
                'out', (len(self.plist) * lS * lm, ), dtype=dtype)
            Sf.close()

        self.mpi.comm.Barrier()

        Sf = h5py.File(args['out'], 'r+', driver='mpio', comm=self.mpi.comm)
        out = Sf['out']

        # Init storage for matrices
        # Get file name

        # tSfn = 'tmp.' + Sfn
        # tSfn = args['output']

        # stubs = {
        #     'ACE': 'X',
        #     'NME': 'Z',
        # }

        a = prody.parsePDB(args['receptor'])

        OG = a.select('resnum 151 name SG')
        Ob = a.select('resnum 168 and name O')
        ND1 = a.select('resnum 46 and name NE2')

        OXH = a.select("name N resnum 149 150 151")

        t0 = time.time()

        for cm in range(lM):
            m = self.aplist[cm]
            lm = len(m)

            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print('STEP: %d PERCENT: %.2f%% TIME: %s' % (
                cm, float(cm) / lM * 100, dt))

            try:
                S = extractor.extract_result(m)
            except:
                print('ERROR: BAD PEPTIDE: %s' % m)
                continue

            lS = S.numCoordsets()

            for S_ in range(lS):

                S.setACSIndex(S_)

                tC = S.select('name C')

                dist = np.inf
                rnum = None

                for C in tC.iterAtoms():

                    rnum = C.getResnum()

                    if rnum == 1:
                        continue

                    O = S.select('resnum %i and name O' % rnum)
                    Nl = S.select('resnum %i and name N' % (rnum))
                    N = S.select('resnum %i and name N' % (rnum + 1))

                    C_ = C.getCoords()
                    O_ = O.getCoords()[0]
                    N_ = N.getCoords()[0]

                    dist = prody.calcDistance(C, OG)[0]
                    hdist1 = np.min(prody.calcDistance(OXH, O))
                    hdist2 = np.min(prody.calcDistance(Ob, Nl))

                    nC_ = np.cross((C_ - N_), (O_ - C_))
                    nC_ /= np.linalg.norm(nC_)
                    nC_ = nC_ + C_
                    nC_ = nC_.reshape(1, 3)

                    nC = C.copy()
                    nC.setCoords(nC_)

                    BD = prody.calcAngle(OG, C, O)
                    FL = prody.calcDihedral(OG, nC, C, O)
                    AT = prody.calcAngle(ND1, OG, C)

                    angle_ = prody.calcDihedral(ND1, OG, C, N)

                    # angle_ = prody.calcDistance(Od, N)[0] \
                    #     - prody.calcDistance(Od, C)[0]

                    if angle_ < 0:
                        DIR = 'F'
                    else:
                        DIR = 'R'

                    s = 'X' + m + 'Z'
                    pref = ''

                    if DIR == 'F':
                        seq = s
                        pref = 6 - rnum
                    else:
                        seq = s[::-1]
                        pref = rnum

                    suf = 12 - (pref + len(seq))
                    seq = '-' * pref + seq + '-' * suf

                    # outfmt = "%-10s\t%d\t%6.2f\t%6.2f%6.2f\t%6.2f\t" \
                    #         "%6.2f\t%6.2f\t%3s\t%12s\n"

                    # outstr = outfmt % (
                    #    ('%s_%02d' % (m, S_ + 1),
                    #     rnum, dist, hdist1, hdist2, BD, FL, AT,
                    #     DIR, seq)
                    # )

                    outdata = (m, S_ + 1,
                               rnum, dist, hdist1, hdist2, BD, FL, AT,
                               DIR, seq)

                    ind = (self.tb + cm) * lS * lm + S_ * lm + rnum - 2
                    out[ind] = outdata

        self.database.close()
        Sf.close()


def get_args():
    """Parse cli arguments"""
    parser = ag.ArgumentParser(
        description='Grid scripts')

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

#    parser.add_argument('-f',
#                        nargs='+',
#                        type=str,
#                        dest='pdb_list',
#                        metavar='FILE',
#                        help='PDB files')

    parser.add_argument('-d', '--database',
                        type=str,
                        required=True,
                        help='Database of peptides')

    parser.add_argument('-r', '--receptor',
                        dest='receptor',
                        type=str,
                        required=True,
                        help='Protein')

    parser.add_argument('-f', '--file',
                        dest='resfile',
                        type=str,
                        required=True,
                        help='Peptide results hdf5')

    parser.add_argument('-p', '--plist',
                        type=str,
                        dest='plist',
                        metavar='PLIST.TXT',
                        help='List of peptides')

    parser.add_argument('-o', '--output',
                        dest='out',
                        default='specificity.out',
                        metavar='OUTPUT',
                        help='Name of output file')

    args = parser.parse_args()

    args_dict = vars(args)

    return args_dict

if __name__ == "__main__":
    args = get_args()
    worker = SpecCalc(args)
    worker.process()
