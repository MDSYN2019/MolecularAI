"""

Module for downloading ChemBL data 

"""
try:
    import psycopg2
except:
    print("Error: You need psycopg2 to run this code")   
try:
    import numpy as np
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import Draw
    import matplotlib.pyplot as plt
    import seaborn as sns
    from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
    from mol2vec.helpers import depict_identifier, plot_2D_vectors, IdentifierTable, mol_to_svg
    from gensim.models import word2vec
except:
    print("Could not find packages")

# All part of parsing the emolecules dataset
def emolecule_command(sql_command, user, password):
    sql_string = "dbname='emolecules' user={} host='localhost' password={}".format(user, password)
    try:                                                                                                                                            
        conn = psycopg2.connect(sql_string) 
    except:                                                                                                                                         
        print("Did not connect to database")
    emolecules_cur = conn.cursor()
    emolecules_cur.execute(str(sql_command))                                   
    database = emolecules_cur.fetchall()
    return database

def chembl24_command(sql_command, user, password):
    sql_string = "dbname='chembl_24' user={} host='localhost' password={}".format(user, password)
    try:
        conn = psycopg2.connect(sql_string) 
    except:
        print("Could not connect to chembl_24")    
    chembl_cur = conn.cursor()
    chembl_cur.execute(str(sql_command))
    chembl_database = chembl_cur.fetchall()
    return chembl_database


#numpyDatabase = np.asarray(database) # convert to a numpy array to make use of numpy data parsing
#smiles = numpyDatabase[:,0] # Isolate the smiles 
#mols = [Chem.MolFromSmiles(x) for x in smiles if Chem.MolFromSmiles(x) != None]


# Get all table names for the work 
table_names = chembl24_command("SELECT table_name FROM information_schema.tables WHERE table_schema='public'", "sang", "silver!!")
table_names = [title[0] for title in table_names]

data_list = [] 
for title in table_names:
    sql_command = "SELECT * FROM {} LIMIT 100;".format(str(title))
    data_list.append(chembl24_command(sql_command, "sang", "silver!!"))
    
                     
