#!/usr/bin/python
from __future__ import print_function
from __future__ import division

__author__ = 'JW Feng'
__new_author__ = 'Sang Young Noh'


import sys
import os
import argparse
from string import Template
from rdkit import Chem

# TODO
# add feature to process mult record sd files
# sd file should have a tag that contains unique molecule names and those names will be used
# as prefix for psi4 filenames
  
# return atom elements and XYZ coordinates for a given SD file
# return only values for first molecule if there are multiple molecules
def get_xyz(infile):
    # RDKit suppresses hydrogen atoms by dfault, use removeHs=False to keep explicit hydrogen atoms
    suppl = Chem.SDMolSupplier(infile, removeHs=False)
    # test each molecule to see if it was correctly read before working with it
    mol = suppl.next()
    if mol is None:
        print ("skipping mol", file=sys.stderr)
        return None
    num_atoms = mol.GetNumAtoms()
    xyz_string=""
    for counter in range(num_atoms):
        pos=mol.GetConformer().GetAtomPosition(counter)
        xyz_string = xyz_string + ("%s %12.6f %12.6f %12.6f\n" % (mol.GetAtomWithIdx(counter).GetSymbol(), pos.x, pos.y, pos.z) )
    return xyz_string

def replace_coords_in_psi4_template(template_file, xyz_string, file_to_save, email):
    data = open(template_file, 'r').read()
    template = Template(data)
    new_data = template.substitute(COORDINATES=xyz_string, XYZ_FILE_TO_SAVE=file_to_save+".xyz", SD_FILE_TO_SAVE=file_to_save+".sdf", EMAIL=email)
    #print (new_data)
    return new_data
        
#check to see if there are invalid properties
def main(argv=None):
    # usage statement
    program_description = """Prepare input files for Psi4"""
    epilog_text = """text appears after help statement, can include examples on how to run this program"""
    parser = argparse.ArgumentParser(description=program_description, epilog=epilog_text)

    # optional requirements, "required=True" makes it NOT optional
    parser.add_argument("-in", dest="infile", required=True, help="input file")
    parser.add_argument("-out", dest="outfile", help="output file, ignores prefix for output file if out is specified")
    parser.add_argument("-template", dest="template", required=True, help="Psi4 template file")
    parser.add_argument("-prefix", dest="prefix", help="prefix for psi4 input files, default=psi4_prefix_", default="psi4_prefix_")
    parser.add_argument("-email", dest="email", required=True, help="email address to sent results")
    #parser.add_argument("-torsion", dest="torsion", help="atom numbers specifying torsion to be scanned")
    args=None
    try:
        args = parser.parse_args(argv)
    except:
        # useful parser functions
        parser.print_help()
        sys.stderr.write("Input parameters were incorrect, please check help message\n")
        return 2

    if ( not args.infile.endswith(".sdf") ):
        sys.stderr.write("Input file must end with .sdf")
        return 2

    # get atom elements and  molecule coordinates in XYZ format
    xyz_string = get_xyz(args.infile)
    #print (xyz_string)

    # create result file name for storing psi4 output using basename of inputfile
    file_base = os.path.splitext(os.path.basename(args.infile))[0]
    result_file = args.prefix + file_base 
    psi4_file = args.prefix + file_base + ".opt"
    if args.outfile is not None: 
        psi4_file = args.outfile

    # insert XYZ and prefix into Psi4 template
    psi4_string = replace_coords_in_psi4_template(args.template, xyz_string, result_file, args.email)
    fh = open (psi4_file, 'w')
    fh.write(psi4_string)
    fh.close()
    
    sys.stderr.write("To start psi4 job, run: psi4 " +  psi4_file + " -n nCPUs\n")
    #print (psi4_string)

if __name__ == "__main__":
    # Let main()'s return value specify the exit status.
    sys.exit(main())

