#!/usr/bin/python

import sys
import tqdm
import numpy as np
import prody
from pyRMSD.matrixHandler import MatrixHandler

sys.path.append('/home/domain/silwer/work/grid_scripts/')

import extract_result as er
import util

res = er.PepExtractor(
    database=sys.argv[1],
    resfile=sys.argv[2])

mpi = util.init_mpi()


def unbar(plist, bar):
    i = bar / 100
    m = bar % 100
    s = plist[i]
    return(s, m)


def has_clash(r, threshold=1.0):

    d = r.select('not element H').getCoords()
    dd = d.reshape((len(d), 1, 3))
    rmsd_matrix = MatrixHandler().createMatrix(dd, 'NOSUP_SERIAL_CALCULATOR')
    md = np.min(rmsd_matrix.get_data())

    flag = True

    if md > threshold:
        flag = False

    return flag


def merge_peptides(r1, r2):

    lr = r1.getResnums()[-1]

    r1_ = r1.select('resnum 1:%d' % lr).toAtomGroup()
    r2_ = r2.select('resnum 3:6').toAtomGroup()

    c_ = lr
    for r_ in r2_.iterResidues():
        r_.setResnum(c_)
        c_ += 1

    r = r1_ + r2_

    return r

allpath = None

if mpi.rank == 0:
    with open(sys.argv[3], 'r') as f:
        rawpath = f.readlines()

    allpath = list()
    for p in rawpath:
        allpath.append(map(int, p.strip().split(' ')))

allpath = mpi.comm.bcast(allpath)
# allpath = np.load('all_paths.npy')
lallpath = len(allpath)

if mpi.rank == 0:
    pbar = tqdm.tqdm(total=lallpath)

i = mpi.rank

plist = res.get_resfile_plist()

while i < lallpath:
    # while i < 10:
    pbar.update(i)

    p = allpath[i]

    s, m = unbar(plist, p[0])
    seq = s
    r = res.extract_result(s, m)

    if len(p) > 1:
        for j in range(1, len(p)):
            s, m = unbar(plist, p[j])
            seq += s[1:]
            r2 = res.extract_result(s, m)
            r = merge_peptides(r, r2)

    if not has_clash(r):

        prody.writePDB('%d_%s.pdb' % (i, seq), r)

    i += mpi.comm.size

mpi.comm.Barrier()

if mpi.rank == 0:
    pbar.close()
