#!/usr/bin/python

import sys
import tqdm
import numpy as np
import networkx as nx

sys.path.append('/home/domain/silwer/work/grid_scripts/')

import extract_result as er
import util

res = er.PepExtractor(
    database=sys.argv[1],
    resfile=sys.argv[2])

mpi = util.init_mpi()

comps = None
G = None

if mpi.rank == 0:
    edges = np.load(sys.argv[3])
    G = nx.from_edgelist(edges[:, :2], create_using=nx.DiGraph())
    uG = G.to_undirected()
    comps = list(nx.connected_component_subgraphs(uG))
    print('total components', len(comps))

comps = mpi.comm.bcast(comps)
G = mpi.comm.bcast(G)
lcomps = len(comps)

i = mpi.rank
if mpi.rank == 0:
    pbar = tqdm.tqdm(total=lcomps)

PATHS = list()

while i < lcomps:
    # while i < 1:

    if mpi.rank == 0:
        pbar.update(i)

    paths = list()

    comp = comps[i]

    G_ = G.subgraph(comp.nodes).copy()

    # if nx.is_directed_acyclic_graph(G_):
    #    path_ = nx.dag_longest_path(G_)

    #    PATHS.append(path_)

    nodes = comp.nodes

    path_ = list(nx.all_pairs_shortest_path(G_))

    PATHS_ = list()

    for p in path_:
        pths = p[1]
        for p_ in pths.values():
            if len(p_) > 1:
                PATHS.append(p_)

    i += mpi.comm.size

mpi.comm.Barrier()

if mpi.rank == 0:
    pbar.close()

if mpi.rank > 0:
    mpi.comm.send(PATHS, dest=0)

elif mpi.rank == 0:
    for i in tqdm.trange(1, mpi.comm.size):
        paths_ = mpi.comm.recv(source=i)
        PATHS.extend(paths_)

mpi.comm.Barrier()

if mpi.rank == 0:
    print(len(PATHS))
    PATHS_ = sorted(PATHS, key=lambda x: len(x), reverse=True)
    txt = ''
    for p in PATHS_:
        txt += ' '.join(map(str, p))
        txt += '\n'

    with open(sys.argv[4], 'w') as f:
        f.write(txt)
    # pth = np.array(PATHS_)
    # np.save('all_paths', pth)
