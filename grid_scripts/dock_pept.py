#!/usr/bin/env python2

import argparse as ag
import re
import os
# import pickle
from sqlitedict import SqliteDict
import StringIO
import subprocess as sp
import util
import tempfile
import shutil

import prody
from collections import OrderedDict as OD

# H5PY for storage
import numpy as np
import numpy.ma as ma
import h5py
# from h5py import h5s
import oddt
import ConfigParser
from pybel import ob

prody.confProDy(verbosity='error')


class Config(object):

    fname = None
    config = None
    numposes = None
    box = None

    def __init__(self, fname=None):
        if fname:
            self.parse_config(fname)
            self.is_valid()
            self.fname = fname
        else:
            pass

    def get_box_size(self):

        box_size = self.box[:, 1] - self.box[:, 0]

        return box_size

    def set_box_size(self, size):
        if type(size) == np.ndarray and size.shape == (3,):
            size_ = size

        elif type(size) == int and size > 0:
            size_ = np.ndarray((3,), dtype=np.float)
            size_.fill(size)

        else:
            raise Exception("Wrong box size")

        box_ = np.stack(
            (self.box[:, 0], self.box[:, 0] + size_),
            axis=-1
            )
        self.set_box(box_)

    def is_valid(self, config=None):
        if config is None:
            config = self.config
        keys = self.default.keys()

        for k in keys:

            if k not in config:
                raise Exception('Missing "%s" in config' % k)

            if config[k] is None:
                raise Exception('Wrong parameters for "%s"' % k)

        return True

    def get_min_xyz(self):
        return self.box[:, 0]

    def get_box(self):
        return self.box

    def set_box(self, box):
        if type(box) == np.ndarray and box.shape == (3, 2):
            pass
        else:
            raise Exception("Wrong box")

        self.box = box
        self._update_box()

    def _update_box(self):
        pass

    def write(self, fname=None):
        if fname is None:
            fname = self.fname

        cfg = self.cfg_to_str()

        with open(fname, 'w') as f:
            f.write(cfg)


class ConfigLedock(Config):

    csize = OD([
        ('Receptor', 1),
        ('RMSD', 1),
        ('Binding pocket', 3),
        ('Number of binding poses', 1),
        ('Ligands list', 1),
        ('END', 0)
    ])

    def parse_config(self, fname):

        with open(fname, 'r') as f:
            raw = f.readlines()
        lines = map(str.strip, raw)

        ckeys = self.csize.keys()

        config = OD.fromkeys(self.csize.keys())

        i = 0

        N = len(lines)

        while i < N:
            l = lines[i]
            if l in ckeys:
                i_ = self.csize[l]
                config[l] = lines[i + 1:i + i_ + 1]

                if l == 'END':
                    i_ = 1

                i += i_ + 1
            else:
                i += 1

        self.is_valid(config)

        self.config = config

        self.numposes = int(self.config['Number of binding poses'][0])

        self.box = np.array(
            map(lambda x: x.split(), self.config['Binding pocket']),
            dtype=np.float
            )

    def is_valid(self, config=None):
        if config is None:
            config = self.config
        keys = self.csize.keys()

        for k in keys:

            if k not in config:
                raise Exception('Missing "%s" in config' % k)

            if self.csize[k] != len(config[k]):
                raise Exception('Wrong parameters for "%s"' % k)

        return True

    def cfg_to_str(self):

        c_ = ''

        for i in self.config.items():
            k, v = i
            c_ += k
            c_ += '\n'
            c_ += '\n'.join(v)
            c_ += '\n'
            c_ += '\n'

        return c_

    def _update_box(self):

        self.config['Binding pocket'] = '\n'.join(
            map(lambda x: str(x[0]) + ' ' + str(x[1]), self.box)
            )


class ConfigPlants(Config):

    default = OD([
        ('scoring_function', 'chemplp'),
        ('search_speed', 'speed1'),
        ('protein_file', 'pro.mol2'),
        ('ligand_list', 'ligands_list'),
        ('output_dir', 'results'),
        ('write_multi_mol2', 1),
        ('bindingsite_center', None),
        ('bindingsite_radius', None),
        ('cluster_structures', 20),
        ('cluster_rmsd', 1.0),
        ('write_protein_conformations', 0),
        ('write_protein_bindingsite', 0),
        ('write_protein_splitted', 0),
        # ('aco_sigma', 5),
        ])

    csize = OD([
        ('scoring_function', 1),
        ('search_speed', 1),
        ('protein_file', 1),
        ('ligand_list', 1),
        ('output_dir', 1),
        ('write_multi_mol2', 1),
        ('bindingsite_center', 3),
        ('bindingsite_radius', 1),
        ('cluster_structures', 1),
        ('cluster_rmsd', 1),
        ('write_protein_conformations', 1),
        ('write_protein_bindingsite', 1),
        ('write_protein_splitted', 1),
        ('aco_sigma', 1),
        ])

    def parse_config(self, fname):

        with open(fname, 'r') as f:
            raw = f.readlines()
        lines = map(str.strip, raw)

        ckeys = self.csize.keys()

        config = self.default.copy()

        for l in lines:
            l_ = l.strip()

            if not l_:
                continue

            ls_ = l.split(' ')

            k = ls_[0]

            if k in ckeys:
                config[k] = ' '.join(ls_[1:1 + self.csize[k]])

        self.is_valid(config)

        self.config = config

        # Create box, which inscribes docking sphere

        center = np.fromstring(config['bindingsite_center'], sep=' ')
        r = np.float(config['bindingsite_radius'])

        box = np.zeros((3, 2), dtype=np.float)
        box[:, 0] = center - r
        box[:, 1] = center + r

        self.box = box

        self.numposes = int(self.config['cluster_structures'])

    def cfg_to_str(self):
        c_ = ''

        for i in self.config.items():
            k, v = i
            c_ += '%s %s\n' % (k, str(v))

        return c_

    def _update_box(self):

        r = (self.box[0, 1] - self.box[0, 0]) / 2.0
        self.config['bindingsite_radius'] = '%.2f' % r


class ConfigVina(Config):

    default = OD([
        ('receptor', 'pro.pdbqt'),
        ('ligand', 'ligand.pdbqt'),
        ('out', 'out.pdbqt'),
        ('log', 'log.pdbqt'),
        ('center_x', 0.0),
        ('center_y', 0.0),
        ('center_z', 0.0),
        ('size_x', 22.5),
        ('size_y', 22.5),
        ('size_z', 22.5),
        ('exhaustiveness', 8),
        ('cpu', 1),
        ('energy_range', 3),
        ('num_modes', 20)
        ])

    csize = OD([])

    def parse_config(self, fname):

        cp = ConfigParser.SafeConfigParser()
        cp.readfp(util.FakeSecHead(open(fname)))
        config = self.default.copy()
        config_ = dict(cp.items('asection'))

        for k in config_.keys():
            config[k] = config_[k]

        self.is_valid(config)

        self.config = config

        # Create box, which inscribes docking sphere
        center = np.array([
                      config['center_x'],
                      config['center_y'],
                      config['center_z'],
                      ],
                      dtype=np.float)

        vbox = np.array([
                   config['size_x'],
                   config['size_y'],
                   config['size_z'],
                   ],
                   dtype=np.float)

        box = np.zeros((3, 2), dtype=np.float)
        box[:, 0] = center - vbox / 2.0
        box[:, 1] = center + vbox / 2.0

        self.box = box

        self.numposes = int(self.config['num_modes'])

    def cfg_to_str(self):
        c_ = ''

        for i in self.config.items():
            k, v = i
            c_ += '%s = %s\n' % (k, str(v))

        return c_

    def _update_box(self):

        r = self.box[:, 0] - self.box[:, 1]

        self.config['size_x'] = r[0]
        self.config['size_y'] = r[1]
        self.config['size_z'] = r[2]


class PepDatabase(object):

    fname = None
    database = None

    def __init__(self, fname):
        if fname:
            try:
                # with open(fname, 'rb') as f:
                #     db_ = pickle.load(f)
                db_ = SqliteDict(fname)

            except:
                raise Exception('Something wrong with peptide databse')

            self.fname = fname
            self.database = db_
        else:
            raise Exception('Missing peptide database')
        pass


class PepDocker(object):

    docker = None
    converter = None
    database = None
    config = None
    checkpoint = None
    overwrite = False
    cleanup = True
    llist = None
    plist = None
    llist = None
    mpi = None
    tmpdir = '/tmp'
    reset = None

    def __init__(
        self,
        docker=None,
        converter=None,
        database=None,
        config=None,
        checkpoint=None,
        plist=None,
        verbose=False,
        receptor=None,
        out=None,
        overwrite=False,
        cleanup=True,
        tmpdir=None,
        reset=False,
        *args,
        **kwargs
    ):

        self.cleanup = cleanup
        self.mpi = util.init_mpi()

        if docker:
            self.docker = os.path.abspath(docker)
        else:
            raise Exception('Missing Docker software!')

        if converter:
            self.converter = os.path.abspath(converter)

        self.database = PepDatabase(database).database

        if config:
            self.config = self.load_config(config)
        else:
            raise Exception('Missing Docker config!')

        if receptor:
            receptor_ = os.path.abspath(receptor)
            self.set_receptor(receptor_)
        else:
            raise Exception('Missing receptor file')

        plist_ = None
        if self.mpi.rank == 0:
            if plist:
                plist_ = util.read_plist(plist)
            else:
                plist_ = self.database.keys()

            for i in plist_:
                util.check_pept(i)

        self.plist = self.mpi.comm.bcast(plist_)

        self.overwrite = overwrite
        self.reset = reset

        if not out:
            out = 'out.hdf5'

        self.out = self.allocate_resfile(out, overwrite, reset)
        self.out.flush()
        self.outfn = out

        self.checkpoint = self.out['checkpoint']

        self.eplist = dict([(t[1], t[0]) for t in enumerate(self.plist)])

        aplist = ma.array(self.plist, mask=self.checkpoint).compressed()

        N = len(aplist)
        tb, te = util.task(N, self.mpi)
        te = min(te, N)
        self.aplist = aplist[tb:te]

        if len(self.aplist) > 0:
            print(
                'RANK: %d FIRST: %d LAST: %d LIST_FIRST: %s LIST_LAST: %s' % (
                    self.mpi.rank, tb, te, self.aplist[0], self.aplist[-1]))
        else:
            print(
                'RANK: %d NOTHING TO DO' % (
                    self.mpi.rank))

        if tmpdir:
            self.tmpdir = tmpdir

    def set_receptor(self, receptor):
        pass

    def load_config(self, config):
        pass

    def run_once(self, seq):

        self.prepare_files(seq)
        self.run_docker(seq)
        r = self.read_result(seq)
        self.write_result(seq, r)

        return r

    def allocate_resfile(self, fname, overwrite=False, reset=False):

        def open_append(fname, comm):
            if comm:
                out = h5py.File(
                    fname, 'r+', driver='mpio', comm=comm)
                out.atomic = True
            else:
                out = h5py.File(
                    fname, 'r+')
            return out

        def open_new(fname, comm):
            if comm:
                out = h5py.File(
                    fname, 'w', driver='mpio', comm=comm)
                out.atomic = True
            else:
                out = h5py.File(fname, 'w')

            return out

        if not overwrite:
            try:
                out = open_append(fname, self.mpi.comm)
            except IOError:
                out = open_new(fname, self.mpi.comm)
        else:
            out = open_new(fname, self.mpi.comm)

        out_ = out.keys()

        if 'plist' not in out_:
            out.create_dataset('plist', data=self.plist)
            out['plist'].attrs['numposes'] = self.config.numposes
            score_ = np.zeros(
                (self.config.numposes * len(self.plist),), dtype=np.float)
            out.create_dataset('score', data=score_)

        if 'checkpoint' not in out_:
            out.create_dataset(
                'checkpoint',
                data=np.zeros(len(self.plist)),
                dtype=np.int8)

        if reset:
            out['checkpoint'][:] = np.zeros(out['checkpoint'].shape)

        recheck_plist = False

        if self.mpi.rank == 0:
            for s in self.plist:
                if s not in out_:
                    recheck_plist = True
                    break

        recheck_plist = self.mpi.comm.bcast(recheck_plist)

        if recheck_plist:

            for s in self.plist:
                if s not in out_:
                    G = out.create_group(s)
                    for i in range(self.config.numposes):
                        dummyc = self.database[s].getCoords().shape
                        G.create_dataset('%d' % i, dummyc, dtype=np.float)

        return out

    def write_result(self, seq, allres):
        rG = self.out.require_group(seq)
        for i in range(len(allres)):
            coords, e = allres[i]
            r_ = rG["%d" % i]
            r_[:] = coords

            self.out['score'][self.eplist[seq] * self.config.numposes + i] = e

    def run_docker(self, seq=None):
        call = list()
        call.append(self.docker)
        call.append(self.config.fname)
        sp.check_call(call)

    def write_config(self):
        fname = '%d_config' % self.mpi.rank
        self.config.fname = fname
        self.config.write()

    def dock(self):
        print('Starting docking!')

        N = len(self.aplist)

        for i in range(N):
            s_ = self.aplist[i]

            tmpdir = tempfile.mkdtemp(dir=self.tmpdir)
            os.chdir(tmpdir)

            self.run_once(s_)

            os.chdir(self.tmpdir)
            shutil.rmtree(tmpdir)

            self.checkpoint[self.eplist[s_]] = 1
            print('Step %d of %d' % (i + 1, N))

        self.clean_files()

    def clean_files(self):
        self.database.close()
        self.out.close()


class DockerLedock(PepDocker):

    def load_config(self, config):
        return ConfigLedock(config)

    def set_receptor(self, r):
        self.receptor = r
        self.config.config['Receptor'] = [r]

    def prepare_files(self, seq):

        llist = 'ligand_list_%d.txt' % self.mpi.rank
        self.config.config['Ligands list'] = [llist]
        self.llist_fname = llist

        self.write_config()

        prody.writePDB(seq + '.pdb', self.database[seq])

        call = [self.converter]
        call.extend(['--mode', 'complete'])
        call.append(seq + '.pdb')
        call.append(seq + '.mol2')
        sp.check_call(call)

        os.remove(seq + '.pdb')

        with open(llist, 'w') as f:
            f.write('%s.mol2\n' % seq)

    def read_result(self, seq):
        fname = "%s.dok" % seq
        with open(fname, 'r') as f:
            raw = f.readlines()
        res = self.split_result(raw)
        pres = list()
        for r in res:
            e_ = self.get_energy(r)
            r_ = prody.parsePDBStream(StringIO.StringIO(r))
            c_ = r_.getCoords()
            pres.append((c_, e_))

        if self.cleanup:
            os.remove(fname)
            os.remove("%s.mol2" % seq)
            os.remove(self.llist_fname)
            os.remove(self.config.fname)
        return pres

    def split_result(self, lines):
        r = list()
        N = len(lines)
        i = 0
        s_ = None
        e_ = None
        while i < N:

            if re.match('REMARK', lines[i]) and s_ is None:
                s_ = i
            if re.match('END', lines[i]) and s_ is not None:
                e_ = i
                r.append(''.join(lines[s_: e_ + 1]))

                s_ = None
                e_ = None
            i += 1

        return r

    def get_energy(self, r):
        gg = re.findall(r'Score:\s+([-+]?\d+\.\d*)', r)
        if gg:
            e_ = float(gg[0])

        try:
            return e_
        except:
            raise Exception("Can't extract score value")


class DockerPlants(PepDocker):

    def load_config(self, config):
        return ConfigPlants(config)

    def set_receptor(self, r):
        self.receptor = r
        self.config.config['protein_file'] = r

    def prepare_files(self, seq):

        llist = 'ligand_list_%d.txt' % self.mpi.rank
        self.config.config['ligand_list'] = llist
        self.llist_fname = llist

        self.write_config()

        prody.writePDB(seq + '.pdb', self.database[seq])

        call = [self.converter]
        call.extend(['--mode', 'complete'])
        call.append(seq + '.pdb')
        call.append(seq + '.mol2')
        sp.check_call(call)

        os.remove(seq + '.pdb')

        with open(llist, 'w') as f:
            f.write('%s.mol2\n' % seq)

    def read_result(self, seq):

        pres = list()

        efile = "%s/ranking.csv" % self.config.config['output_dir']
        fname = "%s/docked_ligands.mol2" % self.config.config['output_dir']
        energy = np.loadtxt(efile, skiprows=1, usecols=[1], delimiter=',')

        mol = oddt.toolkit.readfile('mol2', fname)

        i = 0

        for m in mol:

            e_ = energy[i]
            c_ = m.coords

            pres.append((c_, e_))

            i += 1

        if self.cleanup:
            os.remove(fname)
            os.remove(efile)
            # os.remove("%s.mol2" % seq)
            os.remove(self.llist_fname)
            os.remove(self.config.fname)

        return pres

    def run_docker(self, seq=None):
        call = list()
        call.append(self.docker)
        call.append('--mode')
        call.append('screen')
        call.append(self.config.fname)
        sp.check_call(call)


class DockerVina(PepDocker):

    def load_config(self, config):
        return ConfigVina(config)

    def set_receptor(self, r):
        self.receptor = r
        self.config.config['receptor'] = r

    def prepare_files(self, seq):

        self.config.config['ligand'] = '%s.pdbqt' % seq
        self.config.config['out'] = '%s_out.pdbqt' % seq
        self.config.config['log'] = '%s_out.log' % seq

        self.write_config()

        prody.writePDB(seq + '.pdb', self.database[seq])

        call = []
        call.append('prepare_ligand4.py')
        call.append('-l')
        call.append('%s.pdb' % seq)
        sp.check_call(call)

        os.remove(seq + '.pdb')

    def read_result(self, seq):

        pres = list()

        ref = self.database[seq]
        rcoords = ref.getCoords()

        ref_enum = OD()

        for at in ref.iterAtoms():
            ref_enum[(at.getResnum(), at.getName())] = at.getIndex()

        fname = ('%s_out.pdbqt' % seq)
        M = oddt.ob.readfile('pdbqt', fname)

        for S in M:
            S.protein = True
            coords = np.zeros(rcoords.shape)
            for res in ob.OBResidueIter(S.OBMol):
                rNum = res.GetNum()
                for atom in ob.OBResidueAtomIter(res):
                    atName = res.GetAtomID(atom).strip()
                    atCoords = np.array([atom.x(), atom.y(), atom.z()])

                    ind = ref_enum[(rNum, atName)]
                    coords[ind] = atCoords

            c_ = coords

            edata = S.data['REMARK'].split('\n')[0].split()

            if edata[0] == 'VINA' and edata[1] == 'RESULT:':
                e_ = float(edata[2])
            else:
                raise('Wrong PDBQT energy data')

            pres.append((c_, e_))

        if self.cleanup:
            os.remove(fname)
            os.remove(self.config.fname)

        return pres

    def run_docker(self, seq=None):
        call = list()
        call.append(self.docker)
        call.append('--config')
        call.append(self.config.fname)
        sp.check_call(call)


def get_args():
    """Parse cli arguments"""

    parser = ag.ArgumentParser(
        description='Dock peptides')

    parser.add_argument('-b', '--backend',
                        required=True,
                        dest='backend',
                        metavar='BACKEND',
                        type=str,
                        help='Software which will be used for docking')

    parser.add_argument('-r', '--receptor',
                        required=True,
                        dest='receptor',
                        metavar='RECEPTOR.PDB',
                        type=str,
                        help='Preprocessed target structure')

    parser.add_argument('-o', '--output',
                        dest='out',
                        metavar='OUTPUT',
                        help='Name of output file')

    parser.add_argument('-d', '--database',
                        type=str,
                        required=True,
                        help='Database of peptides')

    parser.add_argument('-c', '--config',
                        type=str,
                        dest='config',
                        metavar='CONFIG',
                        required=True,
                        help='Ledock config')

    parser.add_argument('-l', '--ledock',
                        type=str,
                        dest='docker',
                        required=True,
                        metavar='LEDOCK',
                        help='Path to ledock')

    parser.add_argument('--converter',
                        type=str,
                        dest='converter',
                        metavar='CONVERTER',
                        help='Path to ledock')

    parser.add_argument('-p', '--plist',
                        type=str,
                        dest='plist',
                        metavar='PLIST.TXT',
                        help='List of peptides')

    parser.add_argument('-t', '--checkpoint',
                        type=str,
                        dest='checkpoint',
                        metavar='CHECKPOINT.TXT',
                        help='List of peptides')

    parser.add_argument('--no-cleanup',
                        action='store_false',
                        dest='cleanup',
                        help='overwrite previos results')

    parser.add_argument('--overwrite',
                        action='store_true',
                        dest='overwrite',
                        help='overwrite previos results')

    parser.add_argument('--reset',
                        action='store_true',
                        dest='reset',
                        help='Reset output checkpoint')

    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Be verbose')

    args = parser.parse_args()
    args_dict = vars(args)
    return args_dict


if __name__ == '__main__':
    args = get_args()

    b = args['backend']

    if b == 'ledock':
        docker = DockerLedock(**args)
    elif b == 'plants':
        docker = DockerPlants(**args)
    elif b == 'vina':
        docker = DockerVina(**args)
    else:
        raise('Wrong docker backend')

    docker.dock()
