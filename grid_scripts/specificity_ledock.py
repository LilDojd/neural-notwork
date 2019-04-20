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
import h5py

import numpy as np
import time
import argparse as ag

import extract_result as er
import prody


def process(args):
    # Sfn = args['Sfn']

    # Open matrix file in parallel mode
    # Sf = h5py.File(Sfn, 'r')

    extractor = er.PepExtractor(**args)
    lM = len(extractor.plist)

    # Init storage for matrices
    # Get file name

    # tSfn = 'tmp.' + Sfn
    # tSfn = args['output']

    stubs = {
        'ACE': 'X',
        'NME': 'Z',
    }

    a = prody.parsePDB(args['receptor'])

    SG = a.select('resnum 241 name OG')
    O = a.select('resnum 262 and name O')
    ND1 = a.select('resnum 79 and name NE2')

    tHN_ = a.select("name N resnum 239 240 241")
    tHN = np.average(tHN_.getCoords(), axis=0)

    t0 = time.time()

    out = open(args['out'], 'w')

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

            S.setACSIndex(S_)

            tC = S.select('name C')

            dist = np.inf

            for c in tC.iterAtoms():

                dist_ = prody.calcDistance(c, SG)[0]

                if dist_ < dist:
                    dist = dist_
                    rnum = c.getResnum()

            C = S.select('resnum %i and name C' % rnum)
            CO = S.select('resnum %i and name O' % rnum)
            # N = S.select('resnum %i and name N' % rnum_)
            N_ = S.select('resnum %i and name N' % (rnum + 1))

            angleAttack = prody.calcAngle(ND1, SG, C)
            angle_ = prody.calcDistance(O, N_)[0] - prody.calcDistance(O, C)[0]

            if angle_ > 0:
                DIR = 'F'
            else:
                DIR = 'R'

            s = 'X' + m + 'Z'
            pref = ''

            if DIR == 'F':
                seq = s
                pref = 5 - rnum
            else:
                seq = s[::-1]
                pref = rnum

            suf = 10 - (pref + len(seq))
            seq = '-' * pref + seq + '-' * suf

            # hangle = prody.calcAngle(tHN, C, N)
            hdist = np.linalg.norm(tHN - CO.getCoords())

            # HN = b.select('resnum %i and name H' % rnum)
            # N_B = b.select('resnum %i and name N' % (rnum + 1))
            # HN_B = b.select('resnum %i and name H' % (rnum + 1))

            # distN = prody.calcDistance(O, N)[0]
            # angleHN = prody.calcAngle(O, HN, N)

            # distN_B = prody.calcDistance(O, N_B)[0]
            # angleHN_B = prody.calcAngle(O, HN_B, N_B)

            outstr = "%-10s\t%6.2f\t%6.2f\t%6.2f\t%3s\t%12s\t%d\n" % (
                ('%s_%02d' % (m, S_ + 1),
                dist, angleAttack, hdist,
                # distN, angleHN,
                # distN_B, angleHN_B,
                DIR, seq, rnum)
                )

            out.write(outstr)

    # tSf.close()
    # Sf.close()
#    os.remove(tSfn)
    out.close()


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
    process(args)
