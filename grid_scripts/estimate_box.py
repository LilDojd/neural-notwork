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

import sys


# H5PY for storage
import h5py

# MPI parallelism
from mpi4py import MPI
import os

from prody import parsePDB
import numpy as np
import time
import oddt

from collections import Counter

sys.path.append('/home/domain/silwer/work/')
import grid_scripts.util as gu


# Get MPI info
comm = MPI.COMM_WORLD
# NPROCS_LOCAL = int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
# Get number of processes
NPROCS = comm.size
# Get rank
rank = comm.rank

Sfn = None
model_list = None
step, padding, GminXYZ, N = None, None, None, None
debug = False

if rank == 0:

    args = gu.get_args_box()
    step = args['step']

    padding = max(gu.avdw.values()) * 2 * args['pad']  # Largest VdW radii - 1.7A

    model_list = args['pdb_list']

    output = args['output']

    debug = args['debug']

if debug:
   import cProfile
   import pstats
   import StringIO
   pr = cProfile.Profile()
   pr.enable()

step = comm.bcast(step)
padding = comm.bcast(padding)
GminXYZ = comm.bcast(GminXYZ)
N = comm.bcast(N)
Sfn = comm.bcast(Sfn)
model_list = comm.bcast(model_list)

M = len(model_list)

atypes = gu.vina_types

NUCS = atypes
iNUCS = dict(map(lambda x: (x[1], x[0],), enumerate(NUCS)))

lnucs = len(NUCS)
fNUCS = np.zeros((lnucs, ), dtype=np.int)

if rank == 0:
    print iNUCS


# Init storage for matrices
# Get file name

bcoords = np.zeros((6,), dtype=np.float)

lm = len(model_list)

cm = rank

t0 = time.time()
c = 0

minX = np.inf
minY = np.inf
minZ = np.inf

maxX = -np.inf
maxY = -np.inf
maxZ = -np.inf


while cm < lm:
    m = model_list[cm]

    if rank == 0 and c % 100 == 0:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        print('RANK: %s STEP: %s TIME: %s' % (0, c, dt))
    c += 1

    M = oddt.ob.readfile('pdbqt', m)

    for S in M:

        for A in S.atom_dict:
            coords = A[1]

            minX = min(minX, coords[0])
            minY = min(minY, coords[1])
            minZ = min(minZ, coords[2])

            maxX = max(maxX, coords[0])
            maxY = max(maxY, coords[1])
            maxZ = max(maxZ, coords[2])

    cm += NPROCS

comm.Barrier()

minX_ = comm.gather(minX)
minY_ = comm.gather(minY)
minZ_ = comm.gather(minZ)


maxX_ = comm.gather(maxX)
maxY_ = comm.gather(maxY)
maxZ_ = comm.gather(maxZ)

if rank == 0:

    minX = np.min(minX_)
    minY = np.min(minY_)
    minZ = np.min(minZ_)

    maxX = np.max(maxX_)
    maxY = np.max(maxY_)
    maxZ = np.max(maxZ_)

    lX = (maxX - minX) + padding * 2 + step * 2
    lY = (maxY - minY) + padding * 2 + step * 2
    lZ = (maxZ - minZ) + padding * 2 + step * 2

    cX = (minX + maxX) / 2.0
    cY = (minY + maxY) / 2.0
    cZ = (minZ + maxZ) / 2.0

    with open(output, 'w') as f:
        f.write('center_x = %.2f\n' % cX)
        f.write('center_y = %.2f\n' % cY)
        f.write('center_z = %.2f\n' % cZ)
        f.write('\n')
        f.write('size_x = %.2f\n' % lX)
        f.write('size_y = %.2f\n' % lY)
        f.write('size_z = %.2f\n' % lZ)

    with open(output, 'r') as f:
        f_ = f.readlines()
        print f_

if debug:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'time'
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()
