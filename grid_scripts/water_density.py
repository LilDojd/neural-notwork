
# coding: utf-8


# In[10]:

import numpy as np
import h5py
import gromacs
import gromacs.formats
import sys
import mdtraj
import sklearn.preprocessing

# In[11]:


FN = sys.argv[1]
FRAME = int(sys.argv[2])
TOP = sys.argv[3]

# In[12]:


natoms = ('OP1', 'OP2', 'O1P', 'O2P', 'N4', 'N6', 'O4', 'O6', 'N7', 'N2', 'N3', 'O2', "O3'", "O5'", "O2'", "O4'",)

# In[15]:
proth = gromacs.formats.XVG('prot_hbdist.xvg')
rawdata = proth.array

bins = rawdata[0] * 10.0

mmscaler = sklearn.preprocessing.MinMaxScaler()

dat = mmscaler.fit_transform(rawdata[1])


start = bins[0]
stop = bins[-1]
step = 0.2

padding = stop - start + step


def get_bounding(R, padding, step):
    minXYZ = np.amin(R, axis=0)
    maxXYZ = np.amax(R, axis=0)
    minXYZ -= (padding + step)
    maxXYZ += (padding + step)
    shape = np.ceil((maxXYZ - minXYZ) / step).astype(np.int)
    minXYZ = adjust_grid(minXYZ, step)
    return(minXYZ, shape)


def adjust_grid(c, step, padding=0):
    nc = c - padding
    nc /= step
    nc = np.floor(nc) * step
    return nc


# In[17]:

def sphere2grid(c, r, step):
    nc = adjust_grid(c, step)
    adj = nc - c
    r2x = np.arange(-r - adj[0], r - adj[0] + step, step) ** 2
    r2y = np.arange(-r - adj[1], r - adj[1] + step, step) ** 2
    r2z = np.arange(-r - adj[2], r - adj[2] + step, step) ** 2
    dist2 = r2x[:, None, None] + r2y[:, None] + r2z
    nc = nc - dist2.shape[0] / 2 * step
    vol = (dist2 <= r ** 2).astype(np.int)
    vol *= -999
    return (nc, vol)


def hist2grid(c, dat, bins, step):
    r = bins[-1]
    nc = adjust_grid(c, step)
    adj = nc - c
    r2x = np.arange(-r - adj[0], r - adj[0] + step, step) ** 2
    r2y = np.arange(-r - adj[1], r - adj[1] + step, step) ** 2
    r2z = np.arange(-r - adj[2], r - adj[2] + step, step) ** 2

    dist2 = r2x[:, None, None] + r2y[:, None] + r2z
    grid = np.ndarray(dist2.shape, dtype=np.float)
    grid.fill(0.0)

    for i in range(len(bins) - 1):
        grid[(dist2 > bins[i] ** 2) & (dist2 <= bins[i + 1] ** 2)] = dat[i]

    nc = nc - dist2.shape[0] / 2 * step

    return (nc, grid)


# In[18]:

# In[19]:

aradii = {
    'H': 0.53,
    'C': 0.67,
    'N': 0.56,
    'O': 0.48,
    'P': 0.98,
    'S': 0.88,
}

avdw = {
    'H': 1.2,
    'C': 1.7,
    'N': 1.55,
    'nitrogen': 1.55,
    'O': 1.52,
    'oxygen': 1.55,
    'P': 1.8,
    'S': 1.8,
}


# In[65]:


step = 0.2

if FRAME == 1:
    t = mdtraj.load(FN)
elif FRAME > 1:
    t = mdtraj.load(FN, top=TOP)
    t.superpose(t)
else:
    raise

# t.center_coordinates()

twat = t.top.select("water")
#

if twat:
    ind = [atom.index for atom in t.topology.atoms if (
        atom.element.symbol in 'SON' and atom.index < twat[0])]

else:
    ind = [atom.index for atom in t.topology.atoms if
        atom.element.symbol in 'SON']

newt = t.atom_slice(ind)
newtop = t.top.subset(ind)

N = len(ind)

padding = 10

Gsteps = np.array([step, step, step], dtype=np.float)

GminXYZ, shape = get_bounding(newt.xyz[0] * 10, padding, step)


Ggrid = np.zeros(shape, dtype=np.float)
tGgrid = np.zeros(shape, dtype=np.float)
tGgrid_mask_prot = np.full(shape, 0, dtype=np.int)
tGgrid_mask_na = np.full(shape, 0, dtype=np.int)

frame = 0

for a in newt.xyz:
    if frame % 10 == 0:
        print 'Frame %d' % frame

    tGgrid_mask_prot = np.full(shape, 0, dtype=np.int)
    tGgrid_mask_na = np.full(shape, 0, dtype=np.int)

    for i in range(N):
        c = a[i] * 10
        AminXYZ, Agrid = hist2grid(c, dat, bins, step)
        NA = Agrid.shape[0]
        adj = (AminXYZ - GminXYZ)
        adj = (adj / step).astype(np.int)
        x, y, z = adj
        tGgrid[x: x + NA, y: y + NA, z: z + NA] += Agrid

        Agrid_mask = Agrid.astype(np.bool)

        if newt.top.atom(i).name in natoms:
            tGgrid_mask_na[x: x + NA, y: y + NA,
                           z: z + NA] += Agrid_mask.astype(np.int)
        else:
            tGgrid_mask_prot[x: x + NA, y: y + NA,
                             z: z + NA] += Agrid_mask.astype(np.int)

        # Extract atomic radii
        AminXYZ, Agrid = sphere2grid(
            c, aradii[newt.top.atom(i).element.symbol], step)
        NA = Agrid.shape[0]
        adj = (AminXYZ - GminXYZ)
        adj = (adj / step).astype(np.int)
        x, y, z = adj
        tGgrid[x: x + NA, y: y + NA, z: z + NA] -= Agrid

    # Filter coordinators. At least mincoords

    # 2 - 1 = 1, so mask would be positive
    tGgrid_mask_na -= 1
    tGgrid_mask_na_clip = np.clip(tGgrid_mask_na, 0, 1)

    # 2 - 1 = 1, so mask would be positive
    tGgrid_mask_prot -= 1
    tGgrid_mask_prot_clip = np.clip(tGgrid_mask_prot, 0, 1)

    tGgrid *= (tGgrid_mask_prot_clip * tGgrid_mask_na_clip)

    Ggrid += tGgrid

    frame += 1

    if frame == FRAME:
        break
    #
###


# In[ ]:


# In[63]:

mask_sum = np.sum(tGgrid_mask_na_clip)  # small check, that grid is not empty
print 'MASK', mask_sum

gmax = np.max(Ggrid)
np.clip(Ggrid, 0, gmax, out=Ggrid)
print 'GRID', np.sum(Ggrid)

dGrid = mmscaler.fit_transform(Ggrid.flatten()).reshape(shape)
dGrid *= 100.0
dGrid = np.floor(dGrid).astype(np.int8)

# normalize grid to maximum value, so max level would be 100


#  print sum basic statistics
grid_sum = np.sum(dGrid)
assert grid_sum

print 'DUMP', grid_sum, np.percentile(dGrid, 99.9)

# In[61]:

# Dump grid to file

grfile = 'grid8.hdf5'
F = h5py.File(grfile, 'w')
Fg = F.create_dataset('grid', data=Ggrid.astype(np.int8))
Sg = F.create_dataset('step', data=Gsteps)
Og = F.create_dataset('origin', data=GminXYZ)
F.close()
