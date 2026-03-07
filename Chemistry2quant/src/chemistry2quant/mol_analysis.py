smifile = "/home/oohnohnoh1/Desktop/GIT/Chemiinformatics_work/Chemistry2quant/src/chemistry2quant/smifiles/gdb11_size08.smi"
directory = "/home/oohnohnoh1/Desktop/GIT/Chemiinformatics_work/Chemistry2quant/src/chemistry2quant/WIP"
sdf_file = "bzr.sdf"

from functools import reduce
from multiprocessing import Pool, cpu_count
import sys, os, argparse
import numpy as np
import pandas as pd

## RDKit Modules -- Chemiinformatics

import rdkit
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Fingerprints import FingerprintMols  # Fingerprinting
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Draw
from rdkit.Chem import rdFMCS
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs

import datamol as dm  # new module for designing the molecular pipeline

## Tensorflow

import tensorflow as tf

## Sklearn modules

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

## A neural network layer for use later
def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="weights")
        b = tf.Variable(tf.zeros([n_neurons]), name="biases")
        z = tf.matmul(X, W) + b
        if activation == "relu":
            return tf.nn.relu(z)
        else:
            return z


def sdfToMol(sdf):
    """
    Returns array of mols from sdf
    """
    suppl_data = Chem.SDMolSupplier(sdf)
    ms = [x for x in suppl_data if x is not None]  # Filter data from sdf suppl_data
    return ms


def substructure_search(substruct, struct_array):
    patt = Chem.MolFromSmarts(str(substruct))
    for x in struct_array:
        print(x.HasSubstructMatch(patt))


# RDkit sdf file
# /home/noh/Desktop/Current_work_in_progress/Chemiinformatics/RDKIT/rdkit/Docs/Book/data


class rdkitProcessDf:
    """
    What does this do?
    """

    ## The constructor for the class
    def __init__(self, directory, sdf_file_name):
        self.rdkit_directory = str(directory)
        self.lig_data = self.rdkit_directory + "/" + sdf_file_name
        self.dataMol = sdfToMol(self.lig_data)

    ## Return the data of the molecule in the Mol format (more details and links here: PLACEHOLDEr)
    def returnMol(self):
        return self.dataMol

    def MoltoSmiles(self):
        self.ms_smiles = [Chem.MolToSmiles(x) for x in self.dataMol]
        return self.ms_smiles

    def MACCSfingerprintList(self):
        """ """
        self.MACCSlist = [MACCSkeys.GenMACCSKeys(x) for x in self.dataMol]
        return MACCSlist

    def torsionalfingerprintList(self):
        self.Pairslist = [Pairs.GetAtomPairFingerprint(x) for x in self.dataMol]
        return self.Pairslist


class rdkitPsi4DataGenerator(rdkitProcessDf):
    """

    Inherits from

    Here, we want to translate the smilestoMol file into a psi4 file and run DFT calculations for each.
    Based on the code seen in "https://iwatobipen.wordpress.com/2018/08/24/calculate-homo-and-lumo-with-psi4-rdkit-psi4/"
    """

    def __init__(
        molfile,
        directory="/home/oohnohnoh1/Desktop/GIT/Chemiinformatics_work/Chemistry2quant/src/chemistry2quant/WIP",
        sdf_file="bzr.sdf",
    ):
        """
        What does this class do?
        """
        super().__init__(directory, sdf_file)
        """
		Inheriting from the rdkitProcessDf and initializng for the methods within there
		"""
        self.molfile = molfile

    def molToPsi4(self):
        """
        What does this function do?
        """
        mol = self.molfile
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
        AllChem.UFFOptimizeMolecule(mol)
        atoms = mol.GetAtoms()
        string = string = "\n"
        for i, atom in enumerate(atoms):
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            string += "{} {} {} {}\n".format(atom.GetSymbol(), pos.x, pos.y, pos.z)
            string += "units angstrom\n"
        return string, mol

    def forEachSimilarity(ref, array):
        """
        What does this function do?
        """
        SimSimilarityMaps.GetAPFingerprint(mol, fpType="normal")

    def storeMolecule():
        """
        What does this function do?
        """
        pass


"""
Test running
"""

process = rdkitProcessDf(
    directory, sdf_file
)  # Initialization of the class that reads the sdf file
molList = process.returnMol()
molSmiles = process.MoltoSmiles()
mol2VecList = [
    mol2alt_sentence(x, 1) for x in molList
]  # Using mol2vec to encode molecules as sentences, meaning that each substructure

# Defining the number of hidden layers and the number of nodes inside them

n_hidden1 = 300
n_hidden2 = 100
n_hidden3 = 100


def next_batch(num, data, labels):
    """
    Return a total of `num` random samples and labels.
    """
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def vectorize(smiles):
    """
    Changing the smiles representation into a one-hot representation
    """
    one_hot = np.zeros((smiles.shape[0], embed, len(charset)), dtype=np.int8)
    for i, smile in enumerate(smiles):
        # encode the startchar
        one_hot[i, 0, char_to_int["!"]] = 1
        # encode the rest of the chars
        for j, c in enumerate(smile):
            one_hot[i, j + 1, char_to_int[c]] = 1
        # Encode endchar
        one_hot[i, len(smile) + 1 :, char_to_int["E"]] = 1
    # Return two, one for input and the 2other for output
    return one_hot[:, 0:-1, :], one_hot[:, 1:, :]
