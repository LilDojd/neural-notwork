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

import os

import oddt
import numpy as np
import time

import sys
sys.path.append('/home/domain/silwer/work/')
import grid_scripts.util as gu

debug = False

if debug is True:
    import cProfile
    import pstats
    import StringIO
    pr = cProfile.Profile()
    pr.enable()

args = gu.get_args()
Sfn = args['Sfn']
model_list = args['pdb_list']


# Open matrix file in parallel mode
Sf = h5py.File(Sfn, 'r')

atypes_ = Sf.keys()
try:
    atypes_.remove('origin')
    atypes_.remove('step')
    atypes_.remove('H')  # we do not need hydrogens!
except:
    pass

atypes_ = set(atypes_)
atypes = atypes_
lm = len(model_list)

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

print excl, incl, atypes

# Init storage for matrices
# Get file name


tSfn = 'tmp.' + Sfn
tSf = h5py.File(tSfn, 'w')
score = tSf.create_dataset('score', (lm * 20,), dtype=np.float)

GminXYZ = Sf['origin'][:]
step = Sf['step'][0]
padding = np.max(gu.aradii.values())  # Largest VdW radii - 1.7A

NUCS = set(Sf.keys())
protected = set(['origin', 'step'])
NUCS -= protected

gNUCS = dict()

for i in NUCS:
    gNUCS[i] = Sf[i][:]

t0 = time.time()
c = 0

for i in xrange(lm):
    m = model_list[i]

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    print('STEP: %d TIME: %s' % (i, dt))

    print m
    M = oddt.ob.readfile('pdbqt', m)

    sc = 0
    for S in M:
        mscore = 0.0
        ac = 0

        for A in S.atom_dict:

            atype = A[5]

            if atype not in atypes:
                continue

            C = A[1]
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

        score[i * 20 + sc] = mscore

        sc += 1

tscore = np.zeros((lm * 20,), dtype=[('name', 'S128'), ('score', 'f8')])
for n in range(lm):
    for i in range(20):
        tscore['name'][n * 20 + i] = '%s_ligand_%02d.pdb' % (
            model_list[n][:-6], i + 1)
tscore['score'] = score[:]

tscore.sort(order='score')
tscore['score'] /= tscore['score'][-1]

np.savetxt('score_table.csv', tscore[::-1], fmt="%s\t%.2f")

tSf.close()
Sf.close()
os.remove(tSfn)

if debug is True:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'tottime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    ps.dump_stats('stats.gg# print s.getvalue()')
