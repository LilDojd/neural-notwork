#!/usr/bin/env python2

import argparse as ag
import util

def get_args():
    """Parse cli arguments"""

    parser = ag.ArgumentParser(
        description='Reset checkpoint array')

    parser.add_argument('-o', '--output',
                        dest='out',
                        metavar='OUTPUT',
                        help='Name of output file')

    args = parser.parse_args()
    args_dict = vars(args)
    return args_dict

if __name__ == '__main__':
    args = get_args()
    util.reset_checkpoint(args['out'])
