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
import oddt
import sys

sys.path.append('/home/domain/silwer/work/')
import grid_scripts.util as gu

# Get number of processes
NPROCS = 1
# Get rank

Sfn = None
model_list = None
atypes = None
step, padding, GminXYZ, N = None, None, None, None

debug = False

args = gu.get_args()
step = args['step']

center, box = gu.get_box_from_vina(args['config'])

padding = max(gu.avdw.values()) * 2 * \
    args['pad']  # Largest VdW radii - 1.7A

L = box + padding

print('CENTER', center)
print('BOX', box, L)

GminXYZ = center - L / 2.0
GminXYZ = gu.adjust_grid(GminXYZ, step, padding)

N = np.ceil((L / step)).astype(np.int)

print('GMIN', GminXYZ, N)

with open('box_coords.txt', 'w') as f:
    f.write('BOX: ' + ' '.join(GminXYZ.astype(np.str)) + '\n')
    f.write('STEP: %.2f\n' % step)
    f.write('NSTEPS: ' + ';'.join(N.astype(np.str)) + '\n')

Sfn = args['Sfn']
model_list = args['pdb_list']

if args['default_types']:
    atypes_ = gu.vina_types
else:
    m_ = oddt.ob.readfile('pdbqt', model_list[0])
    atypes_ = list()
    for s in m_:
        for a in s.atom_dict:
            atypes_.append(a[5])

atypes_ = set(atypes_)
try:
    atypes_.remove('H')  # we do not need hydrogens!
except KeyError:
    pass
atypes = tuple(atypes_)

debug = args['debug']

if debug:
    import cProfile
    import pstats
    import StringIO
    pr = cProfile.Profile()
    pr.enable()

M = len(model_list)

NUCS = atypes
iNUCS = dict(map(lambda x: (x[1], x[0],), enumerate(NUCS)))

lnucs = len(NUCS)
fNUCS = np.zeros((lnucs, ), dtype=np.int)

# Init storage for matrices
# Get file name

tSf = dict()
for i in NUCS:
    tSf[i] = np.zeros(N, dtype=np.float32)

lm = len(model_list)

t0 = time.time()

for cm in range(lm):
    m = model_list[cm]

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    print(
        'STEP: %d PERCENT: %.2f%% TIME: %s' % (
            cm, float(cm) / lm * 100, dt))

    M = oddt.ob.readfile('pdbqt', m)

    for S in M:

        for A in S.atom_dict:

            Agrid, AminXYZ = gu.process_atom_oddt(A, step)

            adj = (AminXYZ - GminXYZ)
            adj = (adj / step).astype(np.int)
            x, y, z = adj

            atype = A[5]

            if atype not in atypes:
                continue

            # Check grid boundaries
            try:
                tSf[atype][
                    x: x + Agrid.shape[0],
                    y: y + Agrid.shape[1],
                    z: z + Agrid.shape[2]
                ] += Agrid

                fNUCS[iNUCS[atype]] += 1

            # Ignore bad atoms
            except:
                # print m, A
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

# Open file for storing results
Sf = h5py.File(Sfn, 'w')

for i in range(ln):

    if mult > 0:
        ttSf = tSf[NUCS[i]]

        ttSf /= nmax
        ttSf *= 100.0

        # Use int8 to reduce storage impact
        tG = np.ceil(ttSf).astype(np.int8)
        Sf.create_dataset(NUCS[i], data=tG)
    else:
        print('Array is empty for: ', NUCS[i])

Sf.close()

Sf = h5py.File(Sfn, 'r+')
Gstep = np.array([step, step, step], dtype=np.float32)
Sf.create_dataset('step', data=Gstep)
Sf.create_dataset('origin', data=GminXYZ)
Sf.close()

if debug:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'time'
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
