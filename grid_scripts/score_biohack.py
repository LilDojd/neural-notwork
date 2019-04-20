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
import argparse as ag

import util as gu
import extract_result as er
import tqdm
import prody


def process(args):
    Sfn = args['Sfn']

    # Open matrix file in parallel mode
    Sf = h5py.File(Sfn, 'r')
    model_list = args['pdb_list']

    atypes_t = Sf['atypes'][:][0]
    atypes_, rtypes = gu.choose_artypes(atypes_t)

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
    lM = len(model_list)

    # Init storage for matrices
    # Get file name

    # tSfn = 'tmp.' + Sfn
    tSfn = args['output']
    tSf = h5py.File(tSfn, 'w')

    # fix of pos number // zlo

    pivot_pept_ = extractor.plist[0]
    pivot_pept = extractor.extract_result(pivot_pept_)
    num_poses = pivot_pept.numCoordsets()

    score = tSf.create_dataset('score', (lM,), dtype=np.float)

    GminXYZ = Sf['origin'][:]
    step = Sf['step'][0]

    NUCS = set(Sf.keys())
    protected = set(['origin', 'step'])
    NUCS -= protected

    gNUCS = dict()

    for i in NUCS:
        gNUCS[i] = Sf[i][:]

    for cm in tqdm.trange(lM):
        m_ = model_list[cm]
        S = prody.parsePDB(m_)

        mscore = 0.0
        ac = 0

        for a in S.iterAtoms():
            try:
                atype = rtypes[(a.getResname(), a.getName())]
            except:
                continue

            if atype not in atypes:
                continue

            C = a.getCoords()
            adj = (C - GminXYZ)
            adj = (adj / step).astype(np.int)
            x, y, z = adj

            try:
                tscore = gNUCS[atype][x, y, z]
            except (IndexError, ValueError):
                continue

            mscore += tscore
            ac += 1

        if ac > 0:
            mscore /= float(ac)

        score[cm] = mscore

    tscore = np.zeros((lM,), dtype=[('name', 'S128'), ('score', 'f8')])
    for n in range(lM):
        tscore['name'][n] = model_list[n]
    tscore['score'] = score[:]

    tscore.sort(order='score')
    tscore['score'] /= tscore['score'][-1]

    np.savetxt('a_score_table.csv', tscore[::-1], fmt="%s\t%.2f")

    tSf.close()
    Sf.close()
#    os.remove(tSfn)


def get_args():
    """Parse cli arguments"""
    parser = ag.ArgumentParser(
        description='Grid scripts')

    parser.add_argument('-g',
                        required=True,
                        dest='Sfn',
                        metavar='FILE.hdf5',
                        help='HDF5 file for all matrices')

    parser.add_argument('-o', '--output',
                        dest='output',
                        required=True,
                        metavar='OUTPUT.HDF5',
                        )

    parser.add_argument('--debug',
                        action='store_true',
                        help='Perform profiling')

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Be verbose')

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

    parser.add_argument('--pdb',
                        nargs='+',
                        type=str,
                        dest='pdb_list',
                        metavar='FILE',
                        help='PDBQT files to score')

    args = parser.parse_args()

    args_dict = vars(args)

    return args_dict

if __name__ == "__main__":
    args = get_args()
    process(args)
