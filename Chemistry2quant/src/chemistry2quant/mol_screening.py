from __future__ import print_function

"""
This two-step substructure searching takes reference from the following sources:

1. "An Introduction To Chemiinformatics" by Leach

2. "Fingerprints in RDKit Lecture" by G.Landrum

Here, we are trying to screen the number of similar 

"""

from functools import reduce
from multiprocessing import Pool, cpu_count
import sys, os, argparse
import numpy as np 
import pandas as pd 

# RDKit Modules

import rdkit
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Fingerprints import FingerprintMols # Fingerprinting
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Draw
from rdkit.Chem import rdFMCS
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs

class two_step_screen():
    """
    Initial screening of a database with a substructure - The first screen removes 99% of all the options in the total database. 
    For example, the emoleucles or the chembl database. The second screen studies the work using a number of algirithms - at first, 
    we use the standard rdkit substructure search 
    """
    def __init__(self, smilesList, molecule):
        self.smilesList = smilesList
        self.molecule = molecule        
    def first_screen():
        """
        Remove 99% of the structural database which doesn't match the general structure 
        at all 
        """
        self.smileMol = 
    def second_screen():
        """
        Building from a simple screening
        """

    
    
