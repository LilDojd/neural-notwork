# -*- coding: utf-8 -*-
from pymol.cgo import *
from pymol import cmd
from random import randint
import numpy as np
import ConfigParser

#############################################################################
#
# drawVinaBox.py -- Draws a box surrounding a selection
#
# prerequisites: ConfigParser
#
#############################################################################
class FakeSecHead(object):

    def __init__(self, fp):
        self.fp = fp
        self.sechead = '[asection]\n'

    def readline(self):
        if self.sechead:
            try:
                return self.sechead
            finally:
                self.sechead = None
        else:
            return self.fp.readline()

def get_box_from_vina(f):
    cp = ConfigParser.SafeConfigParser()
    cp.readfp(FakeSecHead(open(f)))
    cfg = dict(cp.items('asection'))

    center = np.array([
                      cfg['center_x'],
                      cfg['center_y'],
                      cfg['center_z'],
                      ],
                      dtype=np.float)

    box = np.array([
                   cfg['size_x'],
                   cfg['size_y'],
                   cfg['size_z'],
                   ],
                   dtype=np.float)

    return (center, box)

def drawVinaBox(fname, linewidth=2.0, r=1.0, g=1.0, b=1.0):
        """
        DESCRIPTION
                Given selection, draw the bounding box around it.

        USAGE:
                drawBoundingBox [selection, [padding, [linewidth, [r, [g, b]]]]]

        PARAMETERS:
                selection,              the selection to enboxen.  :-)
                                        defaults to (all)

                padding,                defaults to 0

                linewidth,              width of box lines
                                        defaults to 2.0

                r,                      red color component, valid range is [0.0, 1.0]
                                        defaults to 1.0

                g,                      green color component, valid range is [0.0, 1.0]
                                        defaults to 1.0

                b,                      blue color component, valid range is [0.0, 1.0]
                                        defaults to 1.0

        RETURNS
                string, the name of the CGO box

        NOTES
                * This function creates a randomly named CGO box that minimally spans the protein. The
                user can specify the width of the lines, the padding and also the color.
        """

        center, box = get_box_from_vina(fname)

        GminXYZ = center - box / 2.0
        GmaxXYZ = GminXYZ + box

        minX = GminXYZ[0]
        minY = GminXYZ[1]
        minZ = GminXYZ[2]
        maxX = GmaxXYZ[0]
        maxY = GmaxXYZ[1]
        maxZ = GmaxXYZ[2]

        print "Box dimensions (%.2f, %.2f, %.2f)" % (maxX-minX, maxY-minY, maxZ-minZ)

        padding = 0
        if padding != 0:
                 print "Box dimensions + padding (%.2f, %.2f, %.2f)" % (maxX-minX, maxY-minY, maxZ-minZ)

        boundingBox = [
                LINEWIDTH, float(linewidth),

                BEGIN, LINES,
                COLOR, float(r), float(g), float(b),

                VERTEX, minX, minY, minZ,       #1
                VERTEX, minX, minY, maxZ,       #2

                VERTEX, minX, maxY, minZ,       #3
                VERTEX, minX, maxY, maxZ,       #4

                VERTEX, maxX, minY, minZ,       #5
                VERTEX, maxX, minY, maxZ,       #6

                VERTEX, maxX, maxY, minZ,       #7
                VERTEX, maxX, maxY, maxZ,       #8


                VERTEX, minX, minY, minZ,       #1
                VERTEX, maxX, minY, minZ,       #5

                VERTEX, minX, maxY, minZ,       #3
                VERTEX, maxX, maxY, minZ,       #7

                VERTEX, minX, maxY, maxZ,       #4
                VERTEX, maxX, maxY, maxZ,       #8

                VERTEX, minX, minY, maxZ,       #2
                VERTEX, maxX, minY, maxZ,       #6


                VERTEX, minX, minY, minZ,       #1
                VERTEX, minX, maxY, minZ,       #3

                VERTEX, maxX, minY, minZ,       #5
                VERTEX, maxX, maxY, minZ,       #7

                VERTEX, minX, minY, maxZ,       #2
                VERTEX, minX, maxY, maxZ,       #4

                VERTEX, maxX, minY, maxZ,       #6
                VERTEX, maxX, maxY, maxZ,       #8

                END
        ]

        boxName = "box_" + str(randint(0,10000))
        while boxName in cmd.get_names():
                boxName = "box_" + str(randint(0,10000))

        cmd.load_cgo(boundingBox,boxName)
        return boxName

cmd.extend ("drawVinaBox", drawVinaBox)
