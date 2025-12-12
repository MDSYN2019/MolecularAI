"""                                                                                                                                                  


"""

from __future__ import print_function
from functools import reduce
from multiprocessing import Pool, cpu_count

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import logging

# Generic rdkit modules

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFMCS
from rdkit.Chem import RDConfig
from rdkit.Chem.Fingerprints import FingerprintMols # Fingerprinting                                                                                 
from rdkit import DataStructs

# For comparing the rdkit smiles
# sklearn metric modules

from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

# Ensemble modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# Sklearn metrics

from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

zipFileDir = zipfile.ZipFile('../zip/lipophilicity.zip')
Pd_df = pd.read_csv(zipFileDir.open('Lipophilicity.csv'))

# Tensorflow
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

class rdkit_lipophilicity:
    """
    TODO
    """
    def __init__(zipFile,name, featurizer, reload=True, move_mean=True):
        self.rdkit_zip = zipFile.ZipFile(zipFile)
        self.rdkit_csv = [str(f.endswith('csv')) for f in self.rdkit_zip] 
        self.lipophilicity_csv = pd.xread_csv(zipfileDir.open(str(self.rdkit_csv[0])))
    def load_lipo(featurizer='ECFP', split='index', reload=True, move_mean=True):
        logger.info("About to featurize Lipophilicity dataset.")
        logger.info("About to load Lipophilicity dataset.")
    def process_data_column():
        """
        Processing data columns in csv to pandas, then to Morgan Fingerprints ('') 
        """
        # add pandas column with the rdkit format
        self.initial_score = list(self.lipophilicity_csv['exp']) 
        self.initial_smiles = list(self.lipophilicity_csv['smiles'])
        self.Mol = [Chem.MolFromSmiles(x) for x in self.lipophilicity_csv['smiles']]
        self.MFingerprints = [AllChem.GetMorganFingerprintAsBitVect(m, 2) for m in self.Mol]
        
    def mols2feat():
        """
        Converting fingerprint data into vector data 
        """
        np_fps = []
        for fp in self.MFingerprints:
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            np_fps.append(arr)
        self.np_fps = np_fps

    def load_tensorflow(learning_rate, training_epochs, batch_size):
        num_inputs = np.shape(self.np_fps[0])[0]
        
        self.X = tf.placeholder(tf.int32,  shape = (None, num_inputs), name = "X")
        self.Y = tf.placeholder(tf.float32, shape = (None), name = "y")  

        self.batch_size = batch_size # What is the batch size?
        self.learning_rate = learning_rate 
        self.training_epochs = training_epochs
        
    def weights():
        self.weights = {
            'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
            }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([num_classes]))
            }
    def neural_net(x):
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1'])
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_2, self.weights['out']) + self.biases['out']
        return out_layer
    def fingerprints(mols):
        # generate fingeprints: Morgan fingerprint with radius 2
        fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2) for m in mols] # TODO
        # convert the RDKit explicit vectors into numpy arrays
        self.np_fps = []
        for fp in fps:
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            self.np_fps.append(arr)
    def categories(fingerprints, index):
        mask = np.zeros((len(fingerprints), 2))
        mask[:,index] = 1.0
        return list(zip(fingerprints, mask))
    
"""
Train the models
"""

with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels}))
