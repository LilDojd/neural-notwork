#!/usr/bin/python

import sys
import tqdm
import numpy as np
import math
import os

sys.path.append('/home/domain/silwer/work/grid_scripts/')

import extract_result as er
import util

res = er.PepExtractor(
    database=sys.argv[1],
    resfile=sys.argv[2])

mpi = util.init_mpi()

numposes = 100
plen = 3
offset = 0

def find_atom(res, s, c, rnum, offset=0):
    r = res.extract_result(s, c)
    b1 = r.select('name N and resnum %d' % rnum).getCoords()[0]
    b2 = r.select('name O and resnum %d' % rnum + offset).getCoords()[0]
    bd1 = b2 - b1
    return(b1, bd1)

allpeps = res.database.keys()
l_ = len(allpeps)

allac = util.one_let[:-2]
accomb = allac + list(itertools.permutations(allac, plen = 1))
laccomb = len(accomb)

plist = res.get_resfile_plist()
l_ = len(plist)
eplist = dict([(v, k) for k, v in enumerate(plist)])

EDGES = list()

#if mpi.rank == 0:
#    pbar = tqdm.tqdm(total=len(allac * l_))

aa = mpi.rank

counter = 0
counter_ = 0

while aa < laccomb:

    aa_ = accomb[aa]
    al = len(aa_)

    NT = []
    NTD = []

    CT = []
    CTD = []

    # for i in tqdm.trange(len(allpeps)):
    for i in range(l_):
        s = plist[i]

        rnum = None

        if s[:al] == aa_:
            rnum = 1 + offset

            for m in range(numposes):
                coord = find_atom(res, s, m, rnum, offset=(al - 1))
                fingerprint = eplist[s] * 100 + m

                NT.append(coord)
                NTD.append(fingerprint)

        if s[-al:] == aa_:

            rnum = plen + offset

            for m in range(numposes):
                coord = find_atom(res, s, m, rnum=(rnum + 1 - al), offset=(al - 1))
                fingerprint = eplist[s] * 100 + m

                CT.append(coord)
                CTD.append(fingerprint)

    edges = list()

    for i in range(len(CT)):
        i_ = CT[i]
        b1, bd1 = i_
        for j in range(len(NT)):
            j_ = NT[j]
            b2, bd2 = j_

            d_ = b1 - b2
            bd_ = bd2 - bd1

            d = math.sqrt(sum(d_ ** 2))
            bd = math.sqrt(sum(bd_ ** 2))
            if d <= 0.5 and bd <= 0.5:
                edges.append((CTD[i], NTD[j], al))

    edges_ = np.asarray(edges)
    np.save('%s' % aa_, edges_)

    aa += mpi.NPROCS

mpi.comm.Barrier()

if mpi.rank == 0:

    edges = None
    for aa in tqdm.trange(len(allac), desc='Merging edges'):
        aa_ = allac[aa]
        edges_ = np.load('%s.npy' % aa_)
        os.remove('%s.npy' % aa_)

        if edges is None:
            edges = edges_
        elif len(edges_ > 0):
            edges = np.concatenate((edges, edges_), axis=0)
        else:
            pass

    np.save(sys.argv[3][:-4], edges)

mpi.comm.Barrier()
