#!/usr//bin/env python

import h5py
import sys
import util as gu
import numpy as np

f1 = sys.argv[1]
f2 = sys.argv[2]
f3 = sys.argv[3]

G1 = h5py.File(f1, 'r')
G2 = h5py.File(f2, 'r')

o1 = G1['origin'][:]
o2 = G2['origin'][:]

step = 0.1

adj = ((o2 - o1) / step).astype(np.int)

adj = np.maximum(adj, np.array([0, 0, 0]))

assert (adj >= 0).all()

atypes1 = G1['atypes'][0]
atypes2 = G2['atypes'][0]

assert atypes1 == atypes2

atypes, rtypes = gu.choose_artypes(atypes1)
atypes = set(atypes)
try:
    atypes.remove('H')
except:
    pass
atypes = tuple(atypes)

s1 = G1[atypes[0]].shape
s2 = G2[atypes[0]].shape

adjs = np.minimum(s1, adj + s2) - adj
print adj, adjs

G = h5py.File(f3, 'w')

for t in atypes:
    print t
    G_ = G1[t][:]
    print np.sum(G_)
    G_[
        adj[0]:adj[0] + adjs[0],
        adj[1]:adj[1] + adjs[1],
        adj[2]:adj[2] + adjs[2]] += G2[t][:adjs[0], :adjs[1], :adjs[2]]
    G_ /= 2.0

    print np.sum(G_)


    G.create_dataset(t, data=G_)

G.create_dataset('origin', data=o1)
G.create_dataset('step', data=G1['step'][:])
G.create_dataset('atypes', data=G1['atypes'][:])

G1.close()
G2.close()
G.close()
