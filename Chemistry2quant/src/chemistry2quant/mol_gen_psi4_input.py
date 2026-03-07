import psi4
import numpy as np
OBfrom rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole

class chem2quant_psi4:    
    """
    Generate coordinate file from smiles for calculation with 
    psi4
    """
    def mol2psi4(mol):
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, useExpTorsionAnglePrefs=True,useBasicKnowledge=True)
        AllChem.UFFOptimizeMolecule(mol)
        atoms = mol.GetAtoms()
        string = "\n"
        for i, atom in enumerate(atoms):
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            string += "{} {} {} {}\n".format(atom.GetSymbol(), pos.x, pos.y, pos.z)
        string += "units angstrom\n"
        return string, mol

