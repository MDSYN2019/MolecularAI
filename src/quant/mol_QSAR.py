"""@package QSAR 


Builidng a simple prediction model using deep neural networks. 

We first want to create a classification modela nad label the positive label as [0,1] and the negative label as a [1,0]
two-dimensional OneHot vector. If you create a model using keras model object, you can get the expeceted value of each of the
above two dimensions 



"""

## Misc modules and numpy/pandas

from functools import reduce
from multiprocessing import Pool, cpu_count
import sys, os, argparse
import numpy as np 
import pandas as pd 

## RDKit Modules

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

## Additional Machine learning modules 
 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from tensorflow.python.keras.layers import Iput
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.Model import Model
