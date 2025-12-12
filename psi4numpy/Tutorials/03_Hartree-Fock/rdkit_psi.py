from __future__ import print_function
import sys, os, argparse
from rdkit import Chem
from rdkit.Chem import AllChem
 
inf = sys.argv[ 1 ]
sdf = Chem.SDMolSupplier( inf )
#mol = sdf.next()

print(mol.GetNumConformers())
if mol.GetNumConformers() <= 1:
    hmol = Chem.AddHs( mol )
    AllChem.EmbedMolecule(  hmol,
                            useExpTorsionAnglePrefs=True,
                            useBasicKnowledge=True )
    #AllChem.EmbedMolecule( hmol )
else:
    hmol = Chem.AddHs( mol )
 
atoms = [ atom for atom in hmol.GetAtoms() ]
 
def atomposition2string( atom ):
    line = "{} {} {} {}"
    conf = atom.GetOwningMol().GetConformer()
    posi = conf.GetAtomPosition( atom.GetIdx() )
    line = line.format( atom.GetSymbol(), posi.x, posi.y, posi.z )
    line +="\n\n"
    return line
 
outf = open('in.dat', 'w')
 
header="#! psi4input\n\n"
molstring ="molecule inputmol "
 
setstring = """
set basis 6-31G**\n
set reference uhf\n
energy( "scf" )\n
"""
 
outf = open('in.dat', 'w')
outf.write(header)
 
molstring += "{\n0 1\n\n"
for atom in atoms:
    l=atomposition2string(atom)
    molstring += l
molstring += "}\n"
 
outf.write( molstring )
outf.write(setstring)
outf.close()
