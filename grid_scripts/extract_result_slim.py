#!/usr/bin/python

import prody

# H5PY for storage
import h5py
import os
from sqlitedict import SqliteDict


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


class PepExtractor(object):

    database = None
    resfile = None
    overwrite = False
    cleanup = True

    def __init__(
        self,
        database=None,
        plist=None,
        verbose=False,
        resfile=False,
        out=None,
        overwrite=False,
        cleanup=True,
        mpi=None,
        *args,
        **kwargs
            ):

        self.cleanup = cleanup

        self.database = PepDatabase(database).database

        if plist:
            self.plist = list(self.read_plist(plist))
        else:
            self.plist = self.database.keys()

        # for i in self.plist:
        #    util.check_pept(i)

        if not out:
            out = ''
        self.out = out

        self.resfilefn = resfile

#        try:
        if mpi:
            self.resfile = h5py.File(
                self.resfilefn, 'r', driver='mpio', comm=mpi.comm)
        else:
            self.resfile = h5py.File(self.resfilefn, 'r', driver='sec2')
#        except IOError:
#            raise Exception("Can't open result file")

        self.overwrite = overwrite

    def write_all(self):
        N = len(self.plist)

        for i in range(N):
            s = self.plist[i]

            n = s

            if '_' in s:
                s_ = s.split('_')
                p_ = s_[0]
                c_ = s_[1].split('.')[0]
                r = self.extract_result(p_, int(c_) - 1)
                n = p_ + '_' + c_
            else:
                r = self.extract_result(s)

            if self.out:
                fname = self.out + '_' + n + '.pdb'
            else:
                fname = n + '.pdb'

            self.write_result(r, fname)

    def extract_all(self):
        N = len(self.plist)
        for i in range(N):
            s = self.plist[i]
            r = self.extract_result(s)
            yield r

    def extract_result(self, seq, coordset=None):
        t = self.database[seq]
        t.delCoordset(0)
        g = self.resfile[seq]

        if coordset is not None:
            t.addCoordset(g['%d' % coordset][:])

        else:

            for i in range(len(g.keys())):
                t.addCoordset(g['%d' % i][:])

        return t

    def write_result(self, res, fname):

        write = False

        if os.path.isfile(fname):
            if self.overwrite:
                write = True
            else:
                raise Exception('%s already exists. '
                                'Use --overwrite to overwrite file' % fname)
        else:
            write = True

        if write:
            prody.writePDB(fname, res)
        else:
            print('Results for %s were not written' % res.getTitle())

    def clean_files(self):
        self.out.close()
        self.checkpoint.close()
        pass

    def read_plist(self, fname):
        with open(fname, 'r') as f:
            raw = f.readlines()
        plist = map(str.strip, raw)
        return set(plist)
