"""
Tensorflow and deepchem implementation of the neural networks that are required 
"""
# import the abstract classes
from typing import Tuple, Type, Any
import os
import numpy as np

# tensorflow
import tensorflow as tf
#from tensorflow.contrib.layers import fully_connected

# rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

# pydantic
from pydantic import BaseModel

# deepchem
import deepchem as dc

# checking pandas table schemas using pandera
import pandera as pa

# inhouse classes
from chem2quant_abc import molecular_modelling_neural_network
from chem2quant_pandas import PostgresConnect


class GCNLayer(tf.keras.layers.Layer):
    """
    Implementation of GCN as layer

    https://dmol.pub/dl/gnn.html
    """

    def __init__(self, activation: Any, **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        """ """
        node_shape, adj_shape = input_shape
        self.w = self.add_weight(shape=(node_shape[2], node_shape[2]), name="w")

    def call(self, inputs) -> Tuple[...]:
        """ """
        nodes, adj = inputs
        degree = tf.reduce_sum(adj, axis=-1)
        # GCN equation
        new_nodes = tf.einsum("bi,bij,bjk,kl->bil", 1 / degree, adj, nodes, self.w)
        out = self.activation(new_nodes)
        return out, adj


class NNParameters(BaseModel, molecular_modelling_neural_network):
    """
    not sure I like the multiple inheritance model here but whatever.

    inherit from BaseModel as well as molecular_modelling_neural_network base class
    """
    
    layer_points: int
    n_hidden_layers: int
    learning_rate: float
    n_outputs: int
    data_directory: str

    # default parameters that can be changed 
    training_data: Any = None # training data to be filled in
    # the declartion that follows needs to be fixed..
    _keras_model: Any = None # dummy variable to hold keras model for now
    _generate_model: Any = None
    _compile_model: Any = None 
    _fit_model: Any = None

    def next_batch(
        self, number_sample: int, data: list, labels: list[str]
    ) -> Tuple[[str]]:
        """
        Return a total of `num` random samples and labels.
        shuffle data
        """
        initial_data = self.connection.return_pandas_table()
        idx = np.arange(0, len(initial_data))  # total amount of data we have
        np.random.shuffle(idx)  # shuffle the data
        idx = idx[:number_sample]  # get a num amount of samples
        data_shuffle = [initial_data[i] for i in idx]  # get the shuffled data
        labels_shuffle = [
            labels[i] for i in idx
        ]  # get the labels for the shuffled data
        self.data_and_label = (np.asarray(data_shuffle), np.asarray(labels_shuffle))

    def gen_training_data(self) -> Any:
        """ 
        Connect to the postgres database and produce the training data for inputting into
        the deepchem model 
        """
        connection = PostgresConnect(
            user="postgres",
            password="mypassword",
            host="127.0.0.1",
            port="5432",
            database="emolecules",
        )
        connection.make_connection()
        # I can either feed this table directly to make a deepchem
        # input or feed this into a
        self.training_data = connection.prepare_deepchem_dataset(
            connection.return_pandas_table()
        )
        print(self.training_data)
    def _generate_model(self) -> None:
        self._keras_model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.layer_points, activation="relu"),
                tf.keras.layers.Dropout(
                    rate=self.learning_rate
                ),  # need to lookup how to apply dropout
                tf.keras.layers.Dense(self.n_outputs),
            ]
        )
    def _compile_model(self) -> None:
        # how will we compile the model?
        self._keras_model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="sgd",
            metrics=["accuracy"],
        )

    def _fit_model(self) -> None:
        assert self.training_data != None, "we need training data to be produced first!"
        self._keras_model.fit(self.training_data)
        
    def prepare_model(self) -> None:
        """
        run the private functions for model training and model fitting 
        for the deepchem model 
        """
        keras_model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.layer_points, activation="relu"),
                tf.keras.layers.Dropout(
                    rate=self.learning_rate
                ),  # need to lookup how to apply dropout
                tf.keras.layers.Dense(self.n_outputs),
            ]
        )
        keras_model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="sgd",
            metrics=["accuracy"],
        )
        keras_model.fit(self.training_data)
        
def historical():
    # Set up parameters for the neural network
    # this bit can be ignored by using the keras module

    X = tf.placeholder(tf.float32, shape=(None, np.shape(train_X[0])[0]), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")
    he_init = tf.contrib.layers.variance_scaling_initializer()

    with tf.name_scope("dnn"):
        hidden1 = fully_connected(
            X, n_hidden1, weights_initializer=he_init, scope="hidden1"
        )
        hidden2 = fully_connected(
            hidden1, n_hidden2, weights_initializer=he_init, scope="hidden2"
        )
        logits = fully_connected(
            hidden2,
            n_outputs,
            weights_initializer=he_init,
            scope="outputs",
            activation_fn=None,
        )

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits
        )
        loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        n_epochs = 20
        batch_size = 100

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(len(train_X // batch_size)):
                X_batch, y_batch = next_batch(batch_size, train_X, train_y)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={X: test_X, y: test_y})
            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
        save_path = saver.save(sess, "./my_model_final.ckpt")

def mol2arr(mol: str) -> np.ndarray:
    """
    take molecular string representation and return as array
    """
    arr = np.zeros((1,))
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def allocate_train_test(self) -> None:
    """
    Need better description here
    """
    train_mol = [
        mol
        for mol in Chem.SDMolSupplier(os.path.join(datadir, "solubility.train.sdf"))
        if mol != None
    ]
    test_mol = [
        mol
        for mol in Chem.SDMolSupplier(os.path.join(datadir, "solubility.test.sdf"))
        if mol != None
    ]
    cls_mol = list(set([mol.GetProp("SOL_classification") for mol in train_mol]))
    cls_dic = {}

    for i, cl in enumerate(cls_mol):
        cls_dic[cl] = i
        tf.reset_default_graph()

    # make train X, y and test X, y
    train_X = np.array([mol2arr(mol) for mol in train_mol])
    train_y = np.array(
        [cls_dic[mol.GetProp("SOL_classification")] for mol in train_mol]
    )
    test_X = np.array([mol2arr(mol) for mol in test_mol])
    test_y = np.array([cls_dic[mol.GetProp("SOL_classification")] for mol in test_mol])

if __name__ == "__main__":
    # need to write very basic excecutable version of the functions above here
    model = NNParameters(layer_points = 300,
                                   n_hidden_layers = 3,
                                   learning_rate =  0.01,
                                   n_outputs =  1,
                                   data_directory=  '.')
    model.gen_training_data()
    model.prepare_model()
