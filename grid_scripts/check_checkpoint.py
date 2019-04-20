#!/usr/bin/env python2
import h5py
import sys
import numpy as np

a = h5py.File(sys.argv[1], 'r')
print np.sum(a['checkpoint']), 'of', a['checkpoint'].shape[0]
a.close()
