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
import bottleneck as bn
import argparse as ag
import sys
import h5py
from tqdm import tqdm
from pyRMSD.matrixHandler import MatrixHandler
import scipy
from sklearn.cluster import AffinityPropagation

sys.path.append('/home/golovin/_scratch/progs/grid_scripts/')
sys.path.append('/home/domain/silwer/work/grid_scripts/')
import extract_result as er
import dock_pept as dp
import util as gu


class PeptMPIWorker(object):

    database = None
    backend = None
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
        backend=None,
        *args,
        **kwargs
    ):

        self.mpi = gu.init_mpi()

        plist_ = None

        self.database = dp.PepDatabase(database).database

        if self.mpi.rank == 0:

            if plist:
                plist_ = gu.read_plist(plist)
            else:
                plist_ = self.database.keys()

            for i in plist_:
                gu.check_pept(i)

        self.plist = self.mpi.comm.bcast(plist_)

        self.eplist = dict([(t[1], t[0]) for t in enumerate(self.plist)])

        N = len(self.plist)
        tb, te = gu.task(N, self.mpi)
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

        if backend is not None:
            self.backend = backend

    def process(self):
        pass


class GridCalc(PeptMPIWorker):

    step = None
    padding = None
    box_config = None
    GminXYZ = None
    N = None

    atypes = None
    rtypes = None
    cluster = None

    def __init__(self, args):
        super(GridCalc, self).__init__(**args)
        self.args = args

        self.step = args['step']

        self.padding = max(gu.avdw.values()) * 2 * \
            args['pad']  # Largest VdW radii - 1.7A

        if self.mpi.rank == 0:

            if self.backend == 'ledock':
                box_config = dp.ConfigLedock(args['config'])
            elif self.backend == 'plants':
                box_config = dp.ConfigPlants(args['config'])
            elif self.backend == 'vina':
                box_config = dp.ConfigVina(args['config'])

            GminXYZ = box_config.get_min_xyz()
            self.GminXYZ = gu.adjust_grid(GminXYZ, self.step, self.padding)

            box = box_config.get_box_size()
            L = box + self.padding

            self.N = np.ceil((L / self.step)).astype(np.int)

            print('GMIN', self.GminXYZ, self.N)
            print('BOX', L)

            with open('box_coords.txt', 'w') as f:
                f.write('BOX: ' + ' '.join(self.GminXYZ.astype(np.str)) + '\n')
                f.write('STEP: %.2f\n' % self.step)
                f.write('NSTEPS: ' + ';'.join(self.N.astype(np.str)) + '\n')

            atypes_, rtypes = gu.choose_artypes(args['atypes'])

            atypes_ = set(atypes_)

            try:
                atypes_.remove('H')  # we do not need hydrogens!
            except KeyError:
                pass
            self.atypes = tuple(atypes_)

            self.rtypes = rtypes

        self.atypes = self.mpi.comm.bcast(self.atypes)
        self.rtypes = self.mpi.comm.bcast(self.rtypes)
        self.box_config = self.mpi.comm.bcast(self.box_config)
        self.GminXYZ = self.mpi.comm.bcast(self.GminXYZ)
        self.N = self.mpi.comm.bcast(self.N)
        self.cluster = args['cluster']

    def process(self, args=None):
        if not args:
            args = self.args

        NUCS = self.atypes
        iNUCS = dict(map(lambda x: (x[1], x[0],), enumerate(NUCS)))
        iNUCS = self.mpi.comm.bcast(iNUCS)

        lnucs = len(NUCS)
        fNUCS = np.zeros((lnucs, ), dtype=np.float)

        # Init storage for matrices
        # Get file name

        tSf = dict()
        for i in NUCS:
            tSf[i] = np.zeros(self.N, dtype=np.float)

        args['mpi'] = self.mpi
        extractor = er.PepExtractor(**args)
        lM = len(self.aplist)

        self.mpi.comm.Barrier()

        if self.mpi.rank == 0:
            pbar = tqdm(total=lM)

        tota_ = 0
        totba_ = 0

        for cm in range(lM):
            m = self.aplist[cm]

            if self.mpi.rank == 0:
                pbar.update(cm)

            try:
                S = extractor.extract_result(m)
            except:
                print('ERROR: BAD PEPTIDE: %s' % m)
                continue

            lS = S.numCoordsets()
            tlS = range(lS)

            if self.cluster is True:
                resc = S.select('not element H').getCoordsets()
                cl = 'NOSUP_SERIAL_CALCULATOR'

                mHandler = MatrixHandler()
                matrix = mHandler.createMatrix(resc, cl)
                mat = scipy.spatial.distance.squareform(matrix.get_data())
                smatrix = (mat ** 2) * (-1)
                aff = AffinityPropagation(affinity='precomputed')
                aff_cluster = aff.fit(smatrix)
                tlS = aff_cluster.cluster_centers_indices_

            if tlS is None:
                continue

            for S_ in tlS:

                S.setACSIndex(S_)

                for a in S.iterAtoms():

                    # skip hydrogens
                    if a.getElement() == 'H':
                        continue

                    try:
                        atype = self.rtypes[(a.getResname(), a.getName())]
                    except:
                        print('ATYPE not found', a.getResname(), a.getName())

                    if atype not in self.atypes:
                        continue

                    Agrid, AminXYZ = gu.process_atom(a, self.step)

                    adj = (AminXYZ - self.GminXYZ)
                    adj = (adj / self.step).astype(np.int)
                    x, y, z = adj

                    try:
                        tSf[atype][
                            x: x + Agrid.shape[0],
                            y: y + Agrid.shape[1],
                            z: z + Agrid.shape[2]
                        ] += Agrid

                        fNUCS[iNUCS[atype]] += 1
                        tota_ += 1

                    except:
                        # print(m, a)
                        totba_ += 1
                        pass

        if self.mpi.rank == 0:
            pbar.close()

        self.mpi.comm.Barrier()

        if self.mpi.rank == 0:
            print('Collecting grids')

        fNUCS_ = self.mpi.comm.allreduce(fNUCS)
        nNUCS = np.zeros((lnucs, ), dtype=np.float)

        tota = self.mpi.comm.reduce(tota_)
        totba = self.mpi.comm.reduce(totba_)

        for i in range(lnucs):
            NUC_ = NUCS[i]

            if self.mpi.rank != 0:
                self.mpi.comm.Send(tSf[NUC_], dest=0, tag=i)

            elif self.mpi.rank == 0:
                for j in range(1, self.mpi.NPROCS):
                    tG = np.empty(tSf[NUC_].shape, dtype=np.float)
                    self.mpi.comm.Recv(tG, source=j, tag=i)
                    tSf[NUC_] += tG
                nNUCS[i] = np.max(tSf[NUC_])

        nNUCS_ = self.mpi.comm.bcast(nNUCS)

        self.mpi.comm.Barrier()

        # Allocate results file

        Sfn = args['Sfn']

        if self.mpi.rank == 0:

            print('Saving data')
            # Sf.atomic = True

            nmax = bn.nanmax(np.divide(nNUCS_, fNUCS_))

            Sf = h5py.File(Sfn, 'w')

            for i in range(lnucs):

                NUC_ = NUCS[i]
                iNUC_ = iNUCS[NUC_]
                mult = fNUCS_[iNUC_]

                if mult > 0.0:

                    tG = tSf[NUC_]

                    med = np.median(tG)
                    tG[tG < (med)] = 0

                    tG /= float(mult)
                    tG /= float(nmax)
                    tG *= 100.0

                    tSf[NUC_] = tG

                else:
                    print('Array is empty for: ', NUC_)

                Sf.create_dataset(NUC_, data=tSf[NUC_])

            Gstep = np.array(
                [self.step, self.step, self.step], dtype=np.float)
            Sf.create_dataset('step', data=Gstep)
            Sf.create_dataset('origin', data=self.GminXYZ)
            Sf.create_dataset(
                'atypes', data=np.array([args['atypes'], ], dtype='S20'))

            print('Total bad atoms %d of %d' % (totba, tota))

            Sf.close()

        self.mpi.comm.Barrier()
        # Open matrix file in parallel mode

        self.database.close()


def get_args():
    """Parse cli arguments"""
    parser = ag.ArgumentParser(
        description='Grid scripts')

    parser.add_argument('-g',
                        required=True,
                        dest='Sfn',
                        metavar='FILE.hdf5',
                        help='HDF5 file for all matrices')

    parser.add_argument('--debug',
                        action='store_true',
                        help='Perform profiling')

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Be verbose')

    parser.add_argument('--atypes', '-t',
                        choices=[
                            'default',
                            'gromos',
                            'vina',
                            'tripos_neutral',
                            'tripos_charged'],
                        default='tripos_neutral',
                        type=str,)

    parser.add_argument('-s', '--step',
                        type=float,
                        dest='step',
                        metavar='STEP',
                        default=0.1,
                        help='Grid step in Angstroms')

    parser.add_argument('--padding',
                        type=int,
                        dest='pad',
                        metavar='PADDING',
                        default=1,
                        help='Padding around cell in Angstroms')

    parser.add_argument('-c', '--config',
                        type=str,
                        dest='config',
                        metavar='CONFIG',
                        help='Ledock config file')

    parser.add_argument('-d', '--database',
                        type=str,
                        required=True,
                        help='Database of peptides')

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

    parser.add_argument('-b', '--backend',
                        required=True,
                        dest='backend',
                        metavar='BACKEND',
                        type=str,
                        help='Software which will be used for docking')

    parser.add_argument('--cluster',
                        dest='cluster',
                        action='store_true',
                        help='Internal clustering with Affinity Propagation')

    args = parser.parse_args()

    args_dict = vars(args)

    return args_dict


if __name__ == "__main__":
    args = get_args()
    worker = GridCalc(args)
    worker.process()
