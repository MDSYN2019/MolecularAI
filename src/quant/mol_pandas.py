"""Utilities for pulling molecule data from Postgres and featurizing with DeepChem."""

from typing import Tuple
import random

import numpy as np
import pandas as pd
import deepchem as dc
import pandera as pa
import psycopg2


class PostgresConnect:
    """
    interface for interacting with the postgres database
    """

    DEFAULT_SMILES_QUERY = (
        "SELECT id, structure FROM molecules "
        "WHERE structure@>'c1cccnc1' LIMIT 100;"
    )

    def __init__(
        self, database: str, user: str, host: str, password: str, port: str
    ) -> None:
        self.user: str = str(user)
        self.host: str = str(host)
        self.password: str = str(password)
        self.database: str = str(database)
        self.port: str = str(port)
        self.input_vals: dict = {
            "user": self.user,
            "password": self.password,
            "host": self.host,
            "port": self.port,
            "database": self.database,
        }
        self.conn = None
        self.cursor = None

    def make_connection(self) -> None:
        """
        try to make connection to the database using
        the input values as stated from the constructor
        """
        self.conn = psycopg2.connect(**self.input_vals)
        self.cursor = self.conn.cursor()

    def _return_query(self, query: str = None) -> list[Tuple[str, ...]]:
        """
        Run query and return raw rows from the RDKit cartridge.
        """
        sql_query = query or self.DEFAULT_SMILES_QUERY
        self.cursor.execute(sql_query)
        # TODO - explore a number of queries for getting molecules
        record = self.cursor.fetchall()
        return record

    def return_pandas_table(self) -> pd.DataFrame:
        """
        take the postgres output and return it as a panda dataframe. This
        dataframe will also be checked with pandera before outputting
        """
        query_output = (
            self._return_query()
        )  # return the smiles from the rdkit cartridge
        output_dataframe = pd.DataFrame(query_output, columns=["index", "smiles"])
        # check output using pandera
        schema = pa.DataFrameSchema(
            {
                "index": pa.Column(int),  # could add more conditionals here
                "smiles": pa.Column(str),  # check for valid smiles string properly here
            }
        )
        schema.validate(output_dataframe)
        return output_dataframe

    def _split(
        self,
        dataset: dc.data.NumpyDataset,
        frac_train: float = 0.6,
        frac_valid: float = 0.2,
        frac_test: float = 0.2,
    ) -> Tuple[...]:
        """
        split the machine learning dataset into training, validation and testing datasets
        """
        splitter = dc.splits.RandomSplitter()
        train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
            dataset=dataset,
            frac_train=frac_train,
            frac_valid=frac_valid,
            frac_test=frac_test,
        )
        return (train_dataset, valid_dataset, test_dataset)

    def prepare_deepchem_dataset(
        self, dataframe: pd.DataFrame, size: int = 1024
    ) -> None:
        """
        Featurize smiles as ECFP and generate a small demo DeepChem dataset.
        """
        featurizer = dc.feat.CircularFingerprint(size=size)
        ecfp = featurizer.featurize(dataframe["smiles"])
        # Synthetic labels are placeholders until a real target column is wired in.
        properties = [random.random() for _ in dataframe["smiles"]]
        self.engineered_features = dc.data.NumpyDataset(X=ecfp, y=np.array(properties))
        training_data = self._split(self.engineered_features)
        print(training_data)
        return training_data

    def close_connection(self) -> None:
        """
        option to close the connection to postgres
        """
        if self.conn:
            self.cursor.close()
            self.conn.close()
            print("Postgresql connection is closed")


if __name__ == "__main__":
    # running a simple query to access to emolecules database
    # in the rdkit cartridge please ensure that the docker container
    # for this cartridge is running before running
    # this query
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
    record = connection.return_pandas_table()
    A = connection.prepare_deepchem_dataset(record)
    #print(record)
    connection.close_connection()
