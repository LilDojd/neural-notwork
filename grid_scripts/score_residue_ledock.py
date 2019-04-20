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
import extract_result as er


def process(args):
    Sfn = args['Sfn']

    # Open matrix file in parallel mode
    Sf = h5py.File(Sfn, 'r')

    padding = max(gu.avdw.values()) * 2 * \
        args['pad']  # Largest VdW radii - 1.7A

    atypes_ = Sf.keys()
    try:
        atypes_.remove('origin')
        atypes_.remove('step')
        atypes_.remove('atypes')
        atypes_.remove('H')  # we do not need hydrogens!
    except:
        pass

    atypes_ = set(atypes_)
    atypes = atypes_

    excl = args['excl']
    incl = args['incl']

    if excl:
        excl = set(excl)
        atypes = atypes - excl

    if incl:
        incl = set(incl)
        atypes.intersection_update(incl)
        atypes = atypes

    if excl and incl:
        raise('You can not use include and exclude options simultaneiously!')

    print(excl, incl, atypes)

    extractor = er.PepExtractor(**args)
    lM = len(extractor.plist)

    # Init storage for matrices
    # Get file name

    # tSfn = 'tmp.' + Sfn
    tSfn = args['output']
    tSf = h5py.File(tSfn, 'w')
    score = tSf.create_dataset('score', (lM * 20,), dtype=np.float)

    GminXYZ = Sf['origin'][:]
    step = Sf['step'][0]

    NUCS = set(Sf.keys())
    protected = set(['origin', 'step'])
    NUCS -= protected

    gNUCS = dict()

    for i in NUCS:
        gNUCS[i] = Sf[i][:]

    rk = gNUCS.keys()[0]

    GmaxXYZ = GminXYZ + np.array(gNUCS[rk].shape[:]) * step

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

            mscore = 0.0
            ac = 0

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
                x, y, z = adj

                if (adj < 0).any():
                    continue

                if ((GmaxXYZ - (AminXYZ + Ashape * step)) < 0).any():

                    continue

                try:
                    tscore = gNUCS[atype][
                        x: x + Ashape[0],
                        y: y + Ashape[1],
                        z: z + Ashape[2]
                    ]

                except (IndexError, ValueError):
                    continue

                tscore = np.sum(tscore)
                mscore += tscore
                ac += 1

            if ac > 0:
                mscore /= float(ac)
                mscore /= float(len(a))

            score[cm * lS + S_] = mscore

    tscore = np.zeros((lM * lS,), dtype=[('name', 'S128'), ('score', 'f8')])
    for n in range(lM):
        for i in range(lS):
            tscore['name'][n * lS + i] = '%s_%02d.pdb' % (
                extractor.plist[n], i + 1)
    tscore['score'] = score[:]

    tscore.sort(order='score')
    tscore['score'] /= tscore['score'][-1]

    np.savetxt('r_score_table.csv', tscore[::-1], fmt="%s\t%.7f")

    tSf.close()
    Sf.close()
#    os.remove(tSfn)


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
