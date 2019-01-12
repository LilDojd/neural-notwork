#! /usr/bin/env python3

"""Align all other files to 1 file using align/cealign algorithms and save alignment as pdb or gro"""

import argparse
import sys

from pymol import cmd
import glob
import re


def align_allfiles(target=None, files=None, mobile_selection='name ca', target_selection='name ca', cutoff=2, cycles=5,
                   cgo_object=0, method='align'):
    """
    Aligns all models in a list of files to one target

    usage:
      align_allfiles [target][files=<filenames>][target_selection=name ca][mobile_selection=name ca]
      [cutoff=2][cycles=5][cgo_object=0]
          where target specifies the model id you want to align all others against,
          and target_selection, mobile_selection, cutoff and cycles are options
          passed to the align command.  You can specify the files to load and align
          using a wildcard.

      By default the selection is all C-alpha atoms and the cutoff is 2 and the
      number of cycles is 5.
      Setting cgo_object to 1, will cause the generation of an alignment object for
      each object.  They will be named like <object>_on_<target>, where <object> and
      <target> will be replaced by the real object and target names.

      Example:
        align_allfiles target=name1, files=model.B9999*.pdb, mobile_selection=c. b & n. n+ca+c+o,target_selection=c. a & n. n+ca+c+o

    """
    cutoff = float(cutoff)
    cycles = int(cycles)
    cgo_object = int(cgo_object)

    file_list = glob.glob(files)
    file_list.sort()
    extension = re.compile('(^.*[/]|\.(pdb|ent|brk))')
    object_list = []

    rmsd = {}
    rmsd_list = []
    for i in range(len(file_list)):
        obj_name1 = extension.sub('', file_list[i])
        object_list.append(extension.sub('', file_list[i]))
        cmd.load(file_list[i], obj_name1)
        if cgo_object:
            objectname = 'align_%s_on_%s' % (object_list[i], target)
            if method == 'align':
                rms = cmd.align('%s & %s' % (object_list[i], mobile_selection), '%s & %s' % (target, target_selection),
                                cutoff=cutoff, cycles=cycles, object=objectname)
            elif method == 'super':
                rms = cmd.super('%s & %s' % (object_list[i], mobile_selection), '%s & %s' % (target, target_selection),
                                cutoff=cutoff, cycles=cycles, object=objectname)
            elif method == 'cealign':
                rmsdict = cmd.cealign('%s & %s' % (target, target_selection),
                                      '%s & %s' % (object_list[i], mobile_selection))
                rms = [rmsdict['RMSD'], rmsdict['alignment_length'], 1, 0, 0]
            else:
                print("only 'align', 'super' and 'cealign' are accepted as methods")

                sys.exit(-1)
        else:
            if method == 'align':
                rms = cmd.align('%s & %s' % (object_list[i], mobile_selection), '%s & %s' % (target, target_selection),
                                cutoff=cutoff, cycles=cycles)
            elif method == 'super':
                rms = cmd.super('%s & %s' % (object_list[i], mobile_selection), '%s & %s' % (target, target_selection),
                                cutoff=cutoff, cycles=cycles)
            elif method == 'cealign':
                rmsdict = cmd.cealign('%s & %s' % (target, target_selection),
                                      '%s & %s' % (object_list[i], mobile_selection))
                rms = [rmsdict['RMSD'], rmsdict['alignment_length'], 1, 0, 0]
            else:
                print("only 'align', 'super' and 'cealign' are accepted as methods")
                sys.exit(-1)

        rmsd[object_list[i]] = (rms[0], rms[1])
        rmsd_list.append((object_list[i], rms[0],   rms[1]))
        cmd.save(file_list[i], obj_name1)
        cmd.delete(obj_name1)

    rmsd_list.sort(key=lambda x: x[1])
    # loop over dictionary and print out matrix of final rms values
    print("Aligning against:", target)
    for object_name in object_list:
        print("%s: %6.3f using %d atoms" % (object_name, rmsd[object_name][0], rmsd[object_name][1]))

    print("\nSorted from best match to worst:")
    for r in rmsd_list:
        print("%s: %6.3f using %d atoms" % r)


cmd.extend('align_allfiles', align_allfiles)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align all other files to 1 file\nExample usage:\npython3 align_all "
                                                 "name1 model.B9999*.pdb -m c. b & n. n+ca+c+o -t c. a & n. n+ca+c+o")
    parser.add_argument("target", help="Name of model to align with")
    parser.add_argument("files", help="Names of files to align")
    parser.add_argument("-c", "--cutoff", dest="cutoff", nargs="?", default=2, type=float, help="float: outlier "
                                                                                                "rejection cutoff in "
                                                                                                "RMS {default: 2.0}")
    parser.add_argument("-l", "--cycles", dest="cycles", nargs="?", default=5, type=int, help="int: maximum number of "
                                                                                              "outlier rejection "
                                                                                              "cycles {default: 5}")
    parser.add_argument("-t", "--target_sel", dest="target_sel", nargs="?", default="name ca", help="By default the "
                                                                                                    "selection is "
                                                                                                    "all C-alpha atoms")
    parser.add_argument("-m", "--mob_sel", dest="mob_sel", nargs="?", default="name ca",
                        help="By default the selection is all "
                             "C-alpha atoms")
    parser.add_argument("--cgo", dest="cgo", nargs="?", default=0, type=int, help="Names of files to align")
    parser.add_argument("--method", dest="method", nargs="?", default="align", help="Names of files to align")

    args = parser.parse_args()

    align_allfiles(args.target, args.files, args.mob_sel, args.target_sel, args.cutoff, args.cycles, args.cgo, args.method)
