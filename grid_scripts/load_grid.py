from chempy.brick import Brick
from pymol import cmd
import h5py
import numpy as np


def load_grid(fname, prefix='aa', ramp='jet'):

    cmd.volume_ramp_new('ramp_DA', [
        0.00, 1.00, 1.00, 1.00, 0.00,
        25.00, 1.00, 0.77, 0.77, 0.02,
        50.00, 1.00, 0.54, 0.54, 0.06,
        75.00, 1.00, 0.38, 0.38, 0.18,
        90.00, 0.93, 0.00, 0.00, 0.60,
        100.00, 0.70, 0.00, 0.00, 1.00,
    ])

    cmd.volume_ramp_new('ramp_DG', [
        0.00, 1.00, 1.00, 1.00, 0.00,
        25.00, 0.77, 0.77, 1.00, 0.02,
        50.00, 0.54, 0.54, 1.00, 0.06,
        75.00, 0.38, 0.38, 1.00, 0.18,
        90.00, 0.00, 0.00, 0.93, 0.60,
        100.00, 0.00, 0.00, 0.70, 1.00,
    ])

    cmd.volume_ramp_new('ramp_DC', [
        0.00, 1.00, 1.00, 1.00, 0.00,
        25.00, 0.64, 1.00, 0.61, 0.02,
        50.00, 0.48, 1.00, 0.47, 0.06,
        75.00, 0.26, 1.00, 0.28, 0.18,
        90.00, 0.04, 0.85, 0.00, 0.44,
        100.00, 0.00, 0.52, 0.06, 1.00,
    ])

    cmd.volume_ramp_new('ramp_DT', [
        0.00, 1.00, 1.00, 1.00, 0.00,
        25.00, 1.00, 0.99, 0.55, 0.02,
        50.00, 1.00, 0.98, 0.46, 0.06,
        75.00, 1.00, 0.99, 0.37, 0.19,
        90.00, 0.92, 0.92, 0.00, 0.39,
        100.00, 0.76, 0.73, 0.00, 1.00,
    ])

    cmd.volume_ramp_new('ramp_P', [
        0.00, 1.00, 1.00, 1.00, 0.00,
        25.00, 1.00, 0.95, 0.93, 0.02,
        50.00, 1.00, 0.85, 0.69, 0.06,
        75.00, 1.00, 0.78, 0.55, 0.19,
        90.00, 1.00, 0.66, 0.30, 0.39,
        100.00, 0.97, 0.50, 0.00, 1.00,
    ])

    cmd.volume_ramp_new('ramp_jet', [
        0.00, 0.00, 0.00, 1.00, 0.00,
        25.00, 0.06, 0.93, 1.00, 0.05,
        50.00, 0.49, 0.51, 1.00, 0.15,
        75.00, 1.00, 1.00, 0.00, 0.23,
        90.00, 0.80, 0.18, 1.00, 0.48,
        100.00, 0.98, 0.01, 1.00, 1.00,
    ])

    cmd.volume_ramp_new('ramp_delta', [
        -1.0, 1.00, 0.00, 0.00, 0.00,
        -0.4,  1.00, 1.00, 0.00, 1.0,
        -0.2, 1.00, 0.00, 0.00, 0.00,
        0.2, 0.00, 0.00, 1.00, 0.00,
        0.4,  0.00, 1.00, 1.00, 1.0,
        1.0, 0.00, 0.00, 1.00, 0.00])

    F = h5py.File(fname, 'r')
    grid = F['step'][()]
    origin = F['origin'][()]

    NUCS = set(F.keys())
    protected = set(['origin', 'step', 'atypes'])
    NUCS -= protected

    for i in NUCS:
        data = F[i][()]
        b = Brick.from_numpy(data, grid, origin)
        bname = prefix + '_' + i
        cmd.load_brick(b, bname)
        volname = bname + '_volume'

        cmd.volume(volname, bname)

        if ramp == 'nuc':
            cmd.volume_color(volname, 'ramp_' + i)
        elif ramp == 'delta':
            cmd.volume_color(volname, 'ramp_delta')
        else:
            cmd.volume_color(volname, 'ramp_jet')

# The extend command makes this runnable as a command, from PyMOL.
cmd.extend("load_grid", load_grid)


def load_meta_grid(fname, prefix='meta'):

    F = h5py.File(fname, 'r')
    step = F['step'][()]
    origin = F['origin'][()]

    NUCS = set(F.keys())
    protected = set(['origin', 'step', 'atypes'])
    NUCS -= protected

    for i in NUCS:
        data = F[i][()]
        b = Brick.from_numpy(data, step, origin)
        bname = prefix + '_' + i
        cmd.load_brick(b, bname)
        volname = bname + '_volume'

        d_min = np.min(data)
        d_max = np.max(data)

        v = np.linspace(d_min, d_max, 6)

        cmd.volume(volname, bname)

        print('ramp_%s' % i)

        cmd.volume_ramp_new('ramp_grid', [
            v[0], 0.00, 0.00, 1.00, 0.70,
            v[1], 0.06, 0.93, 1.00, 0.50,
            v[2], 0.49, 0.51, 1.00, 0.30,
            v[3], 1.00, 1.00, 0.00, 0.10,
            v[4], 0.80, 0.18, 1.00, 0.05,
            v[5], 0.98, 0.01, 1.00, 0.00,
        ])

        cmd.volume_color(volname, 'ramp_grid')

# The extend command makes this runnable as a command, from PyMOL.
cmd.extend("load_meta_grid", load_meta_grid)
