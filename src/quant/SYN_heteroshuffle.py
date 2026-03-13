from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from rdkit.Chem import rdFMCS
from rdkit.Chem import TemplateAlign

"""
Heteroshuffle class is defined to generate hetero shuffled molecules 

To generate possible combinations of atoms, the code passes candaidates of aomtic numbers (C, S, N, O) 
and number of atoms hwich construct target rings. Invalid moleucles will be removed after possible 
combinations are generated 
"""

class HeteroShuffle():
	def __init__(self, mol, query):
		self.mol = mol
		self.query = query
		self.subs = Chem.ReplaceCore(self.mol, self.query)
		self.core = Chem.ReplaceSidechains(self.mol, self.query)
		self.target_atomic_nums = [6, 7, 8, 16]

	def make_connectors(self):
		n = len(Chem.MolToSmiles(self.sub).split('.'))
		map_no = n + 1
		self.rxn_dict = {}
		for i in range(n):
			self.rxn_dict[i+1] = AllChem.ReactionFromSmarts('[{0}*][*:{1}].[{0}*][*:{2}]>>[*:{1}][*:{2}]'.format(i+1, map_no, map_no+1))
		return self.rxn_dict

	def re_construct_mol(self, core):
		"""
		re-construct mols from given substructures and core
		"""
		keys = self.rxn_dict.keys()
		ps = [[core]]
		for key in keys:
			ps = self.rxn_dict[key].RunReactants([ps[0][0], self.subs])
		mol = ps[0][0]
		try:
			smi = Chem.MolToSmiles(mol)
			mol = Chem.MolFromSmiles(smi)
			Chem.SanitizeMol(mol)
			return mol
		except:
			return None
	def get_target_atoms(self):
		"""
		Get target atoms for replace
		target atoms means atoms which don't have anyatom(*) in neighbors
		"""
		atoms = []
		for atom in self.core.GetAromaticAtoms():
			neighbors = [a.GetSymbol() for a in atom.GetNeighbors()}
			if '*' not in neighbors and atom.GetSymbol() != '*':
				atoms.append(atom)
		print(len(atoms))
		return atom

	def generate_mols(self):
		atoms = self.get_target_atoms()
		idxs = [atom.GetIdx() for atom in atoms]
		combinations = itertools.product(self.target_atomic_nums, repeat = len(idxs))
		smiles_set = set()
		self.make_connectors()
		for combination in combinations:
			target = copy.deepcopy(self.core)
			for i, idx in enumerate(idxs):
				target.GetAtomWithIdx(idx).SetAtomicNum(combination[i])
			smi = Chem.MolToSmiles(target)
			target = Chem.MolFromSmiles(smi)
			if target != None:
				
