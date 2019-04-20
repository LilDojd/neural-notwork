#!/usr/bin/env python2

import prody
import numpy as np
import argparse as ag
import sys

sys.path.append('/home/domain/silwer/work/grid_scripts/')
import util


def get_args():
    """Parse cli arguments"""

    parser = ag.ArgumentParser(
        description='Orient docking box')

    parser.add_argument('-i', '--input',
                        required=True,
                        dest='input',
                        metavar='RECEPTOR.PDB',
                        type=str,
                        help='Preprocessed target structure')

    parser.add_argument('-o', '--output',
                        dest='out',
                        metavar='OUTPUT',
                        help='Name of output file')

    parser.add_argument('-b', '--box',
                        dest='box',
                        metavar='BOX',
                        help='box coords in ledock format')

    parser.add_argument('-s', '--selection',
                        type=str,
                        dest='selection',
                        required=True,
                        help='Selection string with ProDy syntax')

    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Be verbose')

    args = parser.parse_args()
    args_dict = vars(args)
    return args_dict


def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
    from numpy import eye, asarray, dot, sum, diag
    from numpy.linalg import svd
    p, k = Phi.shape
    R = eye(k)
    d = 0.0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u, s, vh = svd(dot(Phi.T, asarray(Lambda)**3 - (gamma / p)
                           * dot(Lambda, diag(diag(dot(Lambda.T, Lambda))))))
        R = dot(u, vh)
        d = sum(s)
        if d / d_old < tol:
            break
    return dot(Phi, R)


def get_ledock_box(pdb, selection):
    act = pdb.select(selection)
    xyz_, s_ = util.get_bounding(act.getCoords(), 0, 1)
    xyz__ = xyz_ + s_
    xyz = [x for x in zip(xyz_, xyz__)]
    return xyz


def orient(pdb, selection='all'):
    act = pdb.select(selection)
    adj = prody.calcCenter(act)
    oldcoords = act.getCoords()
    newcoords = np.subtract(oldcoords, adj)
    nncoords = varimax(newcoords)
    trans = prody.calcTransformation(oldcoords, nncoords)
    trans.apply(pdb)
    return pdb

if __name__ == '__main__':
    args = get_args()
    pdb = prody.parsePDB(args['input'])
    rot = orient(pdb, args['selection'])
    prody.writePDB(args['out'], rot)
    box = get_ledock_box(rot, args['selection'])
    with open(args['box'], 'w') as f:
        for c in box:
            f.write('%.2f %.2f\n' % (c[0], c[1]))
    print(get_ledock_box(rot, args['selection']))
