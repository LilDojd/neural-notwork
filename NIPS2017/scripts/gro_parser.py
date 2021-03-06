from __future__ import absolute_import

import sys
import re
import pandas as pd
import numpy as np
import itertools

GRO_FIELDS = {
    "resid": ((0, 5), int),
    "resname": ((5, 10), str),
    "atom_name": ((10, 15), str),
    "atomid": ((15, 20), int),
    "x": ((20, 28), float),
    "y": ((28, 36), float),
    "z": ((36, 44), float),
}

try:
    import io
except ImportError:
    from io import StringIO


class FormatError(Exception):
    """
    Exception raised when the file format is wrong.
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value


class NoGroFile(Exception):
    pass


def stop_at_empty_line(iterator):
    """
    Yield all item of an iterator but stop when the item is an empty line.
    An empty line is a string which is empty when stripped.
    :param iterator:
    :return last line:
    """
    for line in iterator:
        if line.strip() == "":
            return
        yield line


def read_gro(lines):
    """
    Read the atoms, the header, and the box description from a gro file.
    Atoms are represented as dictionaries.
    :Parameters:
        - lines: an iterator over atom lines from the gro file. The two header
                 lines and the bottom line describing the box have to be
                 included.
    :Returns:
        - title: the title of the system as written on line 1 of the file
        - atoms: a list of atom, each atom is stored as a dictionary
        - box: the box description as written on the last line
    :Raise:
        - FormatError: raised if the file format does not fit.
    """
    # "lines" might be a list and not a proper iterator
    lines = iter(lines)
    # The two first lines are a header
    title = next(lines)
    nb_atoms = next(lines)  # This is the number of atoms, we do not care

    # Try to parse the 2 above lines as an atom line.
    # If sucess, it means there is a missing header line
    for header in [title, nb_atoms]:
        try:
            a = dict(((key, convert(header[begin:end].strip()))
                      for key, ((begin, end), convert) in GRO_FIELDS.items()))
            raise FormatError("Something is wrong in the format")
        except ValueError:
            pass

    # Loop over the lines but act on the previous one. We are reading atoms and
    # we do not want to consider the last line (the box description) as an
    # atom.
    atoms = []
    prev_line = next(lines)
    for line in stop_at_empty_line(lines):
        try:
            atoms.append(dict(((key, convert(prev_line[begin:end].strip()))
                               for key, ((begin, end), convert)
                               in GRO_FIELDS.items())))
            prev_line = line
        except ValueError:
            raise FormatError("Something is wrong in the format")
    box = prev_line
    return (title, atoms, box)


class ReturnGro(object):
    def __init__(self, title, atoms, box):
        self.title = title.strip()
        self.atoms = atoms
        self.box = box


def renumber(atoms):
    """
    Renumber the atoms and the residues from a list of atom.
    :Parameters:
        - atoms: a list of atom, each atom is stored as a dictionary
    :Returns:
        - new_atoms: the new list renumbered
    """
    new_atoms = []
    resid = 0
    prev_resid = 0
    for atomid, atom in enumerate(atoms, start=1):
        if atom['resid'] != prev_resid:
            resid += 1
            prev_resid = atom['resid']
        atom['resid'] = resid % 100000
        atom['atomid'] = atomid % 100000
        new_atoms.append(atom)

    return new_atoms


def parse_file(filin):
    """
    Handle the file type before calling the read function
    :Parameters:
        - filin: the filename name in str.
    :Returns:
        - the return of :read_gro:
    :Raise:
        -FormatError: raised if the file format does not fit.
    """

    with open(filin) as f:
        try:
            return read_gro(f)
        except FormatError as e:
            raise FormatError("{0} ({1})".format(e, filin))


def gro_parser(gro_file, include_solvent=True):
    if str(gro_file).endswith(".gro"):
        try:
            with open(gro_file, "r") as gf:
                gro_content = gf.read()
        except Exception:
            raise NoGroFile("Check if the .gro path is correct")
        del gro_content
        title, atoms, box = parse_file(gro_file)
        if include_solvent:
            atoms = pd.DataFrame.from_dict(atoms, orient="columns").set_index("atomid")
        else:
            atoms = [d for d in atoms if d.get("resname") not in ["SOL", "NA", "CL"]]
            atoms = renumber(atoms)
            atoms = pd.DataFrame.from_dict(atoms, orient="columns").set_index("atomid")
            title = "Protein without water"

        return ReturnGro(title, atoms, box)


def get_coordinates(gro_return, include_solvent=True):
    gro_parsed = gro_parser(gro_return, include_solvent)
    atom_info = gro_parsed.atoms
    coord_vals = atom_info[["x", "y", "z"]].values

    return coord_vals


'''Bucket list:'''


# Define local coordinate system (We leave it as default from gmx output
# def set_coordinate_center(center_atom, ):

# Calculate coordinates relative to origin

def center_coords(atom_name, res_name, gro_parsed):
    global atomdf
    atom_info = gro_parsed.atoms
    all_coords = atom_info[["x", "y", "z"]].values
    atomdf = atom_info.loc[(atom_info['atom_name'] == atom_name) & (atom_info['resname'] == res_name)]
    atom_coords = atomdf[["x", "y", "z"]].values
    centered = all_coords - atom_coords

    return centered


if __name__ == '__main__':
    print(get_coordinates(sys.argv[1], include_solvent=False))
    parsed = gro_parser(sys.argv[1], include_solvent=False)
    print(parsed.atoms)
    print(center_coords('P', 'DFP', parsed))
