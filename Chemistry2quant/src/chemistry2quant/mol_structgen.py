from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import numpy as np
from rdkit.Chem.Draw import IPythonConsole

rxn = AllChem.ReactionFromSmarts('[*:1]-[Xe].[*:2]-[Xe]&gt;&gt;[*:1]-[*:2]')
#Tc calculator. Tempolary function. I'll planned to use Reduced Graph Fingerprint.
def calc_tanimoto(m1,m2):
    fp1 = AllChem.GetMorganFingerprintAsBitVect( m1,2 )
    fp2 = AllChem.GetMorganFingerprintAsBitVect( m2,2 )
    tc = DataStructs.TanimotoSimilarity( fp1, fp2 )
    return tc

#fragment mol using BRICS and return smiles list
def fragmenter( mol ):
    fgs = AllChem.FragmentOnBRICSBonds( mol )
    fgs_smi = Chem.MolToSmiles( fgs ).replace( "*", "Xe" ).split( "." )
    return fgs_smi

                            # check structure of start molecule
def check_querymol( fgs_smi ):
    res = [ smi.count("Xe") for smi in fgs_smi ]
    return res

                                    # generate fragment dictionary for design.
def gen_frag_dict( mol_list ):
    frag_dict = {}
    for mol in mol_list:
    fgs = fragmenter( mol )
    qmol = check_querymol( fgs )
    for i, j in enumerate( qmol ):
        if j in frag_dict.keys():
            frag_dict[ j ].add( str(fgs[i]) )
        else:
            frag_dict[ j ] = set( [str(fgs[i])] )
    keys = frag_dict.keys()
    for key in keys:
        frag_dict[ key ] = list( frag_dict[key] )
    return frag_dict

# generate molecules
def struct_gen( query_mol, frag_dict ):
    q_frgs = fragmenter( query_mol )
    # get query molecule's infromation.
    q_des = check_querymol( q_frgs )
    q_des.sort( reverse=True )
    q_des.insert(0,1)
    q_des.pop()
    print(q_des)
    # select starting point as random
    print( frag_dict[ q_des[0] ] )
    ps =  frag_dict[ q_des[0] ][ np.random.randint( len( frag_dict[ q_des[0] ] ) ) ]
    ps = [ Chem.MolFromSmiles( ps ) ]
    for i in range( 1,len( q_des ) ):
        print( str(i)+" STEP" )
        #print(frag_dict[ q_des[i] ])
        ps = AllChem.EnumerateLibraryFromReaction( rxn, (ps, [Chem.MolFromSmiles(smi) for smi in frag_dict[ q_des[i] ]] ) )
        res = set()
        for p in ps:
            try:
                m = p[0]
                s = Chem.MolToSmiles(m)
                res.add( s )
            except:
                continue

            ps = [ Chem.MolFromSmiles( smi ) for smi in res ][:20]
        ps = [ mol for mol in ps if calc_tanimoto(query_mol,mol) &lt;= 0.6 and Descriptors.MolWt(mol) &lt;= 500 ]
        return ps
