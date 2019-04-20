#!/usr/bin/python
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
import h5py

import numpy as np
import time
import argparse as ag

import util as gu
import dock_pept as dp
import extract_result as er


def process(args):

    step = args['step']

    padding = max(gu.avdw.values()) * 2 * \
        args['pad']  # Largest VdW radii - 1.7A

    box_config = dp.ConfigLedock(args['config'])

    GminXYZ = box_config.get_min_xyz()
    GminXYZ = gu.adjust_grid(GminXYZ, step, padding)

    box = box_config.get_box_size()
    L = box + padding

    N = np.ceil((L / step)).astype(np.int)

    GmaxXYZ = GminXYZ + N * step

    print('GMIN', GminXYZ, N)
    print('BOX', L)

    with open('box_coords.txt', 'w') as f:
        f.write('BOX: ' + ' '.join(GminXYZ.astype(np.str)) + '\n')
        f.write('STEP: %.2f\n' % step)
        f.write('NSTEPS: ' + ';'.join(N.astype(np.str)) + '\n')

    Sfn = args['Sfn']

    # atypes_, rtypes = gu.choose_artypes(args['atypes'])

    atypes_ = set(gu.three_let)

    try:
        atypes_.remove('H')  # we do not need hydrogens!
    except KeyError:
        pass
    atypes = tuple(atypes_)

    extractor = er.PepExtractor(**args)

    NUCS = gu.three_let
    iNUCS = dict(map(lambda x: (x[1], x[0],), enumerate(NUCS)))

    lnucs = len(NUCS)
    fNUCS = np.zeros((lnucs, ), dtype=np.int)

    # Init storage for matrices
    # Get file name

    tSf = dict()
    for i in NUCS:
        tSf[i] = np.zeros(N, dtype=np.float32)

    lM = len(extractor.plist)

    t0 = time.time()

    for cm in range(lM):
        m = extractor.plist[cm]

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

            for a in S.iterResidues():

                a.setACSIndex(S_)

                try:
                    atype = a.getResname().upper()
                except:
                    continue

                if atype not in atypes:
                    continue

                AminXYZ, Ashape = gu.get_bounding(a.getCoords(), padding, step)

                adj = (AminXYZ - GminXYZ)
                adj = (adj / step).astype(np.int)

                if (adj < 0).any():
                    continue

                if ((GmaxXYZ - (AminXYZ + Ashape * step)) < 0).any():

                    continue

                x, y, z = adj
                Agrid, AminXYZ = gu.process_residue(a, padding, step)

                try:
                    tSf[atype][
                        x: x + Agrid.shape[0],
                        y: y + Agrid.shape[1],
                        z: z + Agrid.shape[2]
                    ] += Agrid

                    fNUCS[iNUCS[atype]] += 1

                except:
                    print(m, a)
                    pass

    print('fNUCS ', fNUCS)

    nNUCS = np.zeros((lnucs, ), dtype=np.float32)

    ln = len(NUCS)

    for i in range(ln):

        mult = fNUCS[i]

        if mult > 0:
            ttSf = tSf[NUCS[i]]
            nmax = None
            nmax = np.max(ttSf)
            med = np.median(ttSf)
            ttSf[ttSf < (med)] = 0
            ttSf /= mult

            nNUCS[i] = nmax / mult
            print(i, NUCS[i], nNUCS[i], nmax)

    nmax = np.max(nNUCS)

    # Open matrix file in parallel mode
    Sf = h5py.File(Sfn, 'w')

    for i in range(ln):

        if mult > 0:
            ttSf = tSf[NUCS[i]]

            ttSf /= nmax
            ttSf *= 100.0
            tG = np.ceil(ttSf).astype(np.int8)
            Sf.create_dataset(NUCS[i], data=tG)
        else:
            print('Array is empty for: ', NUCS[i])

    Gstep = np.array([step, step, step], dtype=np.float32)
    Sf.create_dataset('step', data=Gstep)
    Sf.create_dataset('origin', data=GminXYZ)
#    Sf.create_dataset('atypes', data=np.array([args['atypes'], ], dtype='S20'))
    print('Totgood: ', np.sum(fNUCS))
    Sf.close()


def get_args():
    """Parse cli arguments"""
    parser = ag.ArgumentParser(
        description='Grid scripts')

    parser.add_argument('-m',
                        required=True,
                        dest='Sfn',
                        metavar='FILE.hdf5',
                        help='HDF5 file for all matrices')

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

#    parser.add_argument('--default_types',
#                        action='store_true',
#                        default=True,
#                        help='Use default set of atom types for Vina')
#    parser.add_argument('--atypes',
#                        choices=['default', 'gromos', 'vina'],
#                        default='default',
#                        type=str,
#                        )


#    parser.add_argument('-f',
#                        nargs='+',
#                        type=str,
#                        dest='pdb_list',
#                        metavar='FILE',
#                        help='PDB files')

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

    args = parser.parse_args()

    args_dict = vars(args)

    return args_dict

if __name__ == "__main__":
    args = get_args()
    process(args)
