"""
Polymer-safe RDKit atom featurizer.

- Handles repeat-unit SMILES with '*' attachment points.
- Robust Gasteiger charges (avoids implicit valence precondition crash).
- Uses public RDKit APIs for per-atom H-bond roles (ChemicalFeatures).
- EState import is version-safe.
- Falls back cleanly if per-atom Crippen/TPSA contributions are unavailable.
- Returns per-atom features as torch.float32, keeping your original 9 features first.
"""

import math
import os
from typing import Any, Dict, List, Set, Tuple

import torch
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures, rdMolDescriptors

# ----------------------------- utils -----------------------------


def _one_hot(x, choices: List[Any]) -> List[int]:
    """One-hot encode x given an ordered list of choices; last index = 'other'."""
    out = [0] * (len(choices) + 1)
    try:
        idx = choices.index(x)
    except ValueError:
        idx = len(choices)
    out[idx] = 1
    return out


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _attachment_atoms(mol: Chem.Mol) -> List[int]:
    # Stars: atomic number 0 / symbol '*'
    return [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 0 or a.GetSymbol() == "*"
    ]


def _shortest_path_atoms(mol: Chem.Mol, a: int, b: int) -> Set[int]:
    return set(Chem.GetShortestPath(mol, a, b))


def _compute_backbone_atoms(mol: Chem.Mol, att_idx: List[int]) -> Set[int]:
    """Atoms on shortest paths between any pair of attachment points."""
    bb: Set[int] = set()
    if len(att_idx) < 2:
        return bb
    for i in range(len(att_idx)):
        for j in range(i + 1, len(att_idx)):
            bb |= _shortest_path_atoms(mol, att_idx[i], att_idx[j])
    return bb


def _neighbor_counts(atom: Chem.Atom) -> Dict[str, int]:
    nbrs = list(atom.GetNeighbors())
    hetero = sum(1 for n in nbrs if n.GetAtomicNum() not in (0, 1, 6))
    arom = sum(1 for n in nbrs if n.GetIsAromatic())
    sp = sum(
        1 for n in nbrs if n.GetHybridization() == Chem.rdchem.HybridizationType.SP
    )
    sp2 = sum(
        1 for n in nbrs if n.GetHybridization() == Chem.rdchem.HybridizationType.SP2
    )
    sp3 = sum(
        1 for n in nbrs if n.GetHybridization() == Chem.rdchem.HybridizationType.SP3
    )
    return dict(n_hetero=hetero, n_arom=arom, n_sp=sp, n_sp2=sp2, n_sp3=sp3)


def _smallest_ring_size(mol: Chem.Mol, atom_idx: int) -> int:
    try:
        rings = Chem.GetSymmSSSR(mol)
        sizes = [len(r) for r in rings if atom_idx in r]
        return min(sizes) if sizes else 0
    except Exception:
        return 0


# ------------- robust Gasteiger charges (safe with '*') -------------


def _compute_gasteiger_per_atom_safe(mol: Chem.Mol) -> List[float]:
    """
    Return per-atom Gasteiger charges (len == n_atoms), robust to '*' dummies.
    Dummies get 0.0. Falls back to star-removed copy if needed.
    """
    n = mol.GetNumAtoms()
    charges = [0.0] * n

    # First attempt: compute directly on a lightly sanitized copy
    try:
        m = Chem.Mol(mol)
        m.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(
            m,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS
            | Chem.SanitizeFlags.SANITIZE_KEKULIZE
            | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
            | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
            | Chem.SanitizeFlags.SANITIZE_ADJUSTHS,
        )
        AllChem.ComputeGasteigerCharges(m)
        for i, a in enumerate(m.GetAtoms()):
            if a.GetAtomicNum() == 0:
                charges[i] = 0.0
            elif a.HasProp("_GasteigerCharge"):
                charges[i] = _safe_float(a.GetDoubleProp("_GasteigerCharge"), 0.0)
        return charges
    except Exception:
        pass

    # Second attempt: remove stars, compute, map back
    try:
        star_idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
        rw = Chem.RWMol(mol)
        for idx in sorted(star_idxs, reverse=True):
            rw.RemoveAtom(idx)
        m2 = rw.GetMol()
        m2.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(
            m2,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS
            | Chem.SanitizeFlags.SANITIZE_KEKULIZE
            | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
            | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
            | Chem.SanitizeFlags.SANITIZE_ADJUSTHS,
        )
        AllChem.ComputeGasteigerCharges(m2)

        comp = 0
        for orig_i in range(n):
            if orig_i in star_idxs:
                charges[orig_i] = 0.0
            else:
                a2 = m2.GetAtomWithIdx(comp)
                if a2.HasProp("_GasteigerCharge"):
                    charges[orig_i] = _safe_float(
                        a2.GetDoubleProp("_GasteigerCharge"), 0.0
                    )
                comp += 1
    except Exception:
        # leave zeros
        pass

    return charges


# --------- per-atom H-bond donor/acceptor via public API ----------

_FEATURE_FACTORY = None


def _get_feature_factory():
    global _FEATURE_FACTORY
    if _FEATURE_FACTORY is None:
        fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        _FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(fdef)
    return _FEATURE_FACTORY


def _compute_hbd_hba_flags(mol: Chem.Mol) -> Tuple[List[float], List[float]]:
    """
    Return two lists (len = n_atoms): HBD flags, HBA flags in {0.0, 1.0}.
    Uses ChemicalFeatures (public, version-stable).
    """
    n = mol.GetNumAtoms()
    hbd = [0.0] * n
    hba = [0.0] * n
    try:
        ff = _get_feature_factory()
        feats = ff.GetFeaturesForMol(mol)
        for f in feats:
            at_idxs = f.GetAtomIds()
            if f.GetFamily() == "Donor":
                for i in at_idxs:
                    hbd[i] = 1.0
            elif f.GetFamily() == "Acceptor":
                for i in at_idxs:
                    hba[i] = 1.0
    except Exception:
        # keep zeros on failure
        pass
    return hbd, hba


# --------------- per-molecule cache (sanitized) ----------------


def build_mol_feature_cache(mol: Chem.Mol) -> Dict[str, Any]:
    """
    Precompute per-atom properties once per molecule.
    Works on repeat-unit SMILES with '*' anchors.
    """
    # safe charges first (may need its own sanitation)
    g_charges = _compute_gasteiger_per_atom_safe(mol)

    # Lightly sanitized copy for descriptor contribs
    m = Chem.Mol(mol)
    m.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(
        m,
        sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS
        | Chem.SanitizeFlags.SANITIZE_KEKULIZE
        | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
        | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
        | Chem.SanitizeFlags.SANITIZE_ADJUSTHS,
    )

    # EState indices (version-safe import)
    try:
        from rdkit.Chem import EState

        estate = list(EState.EStateIndices(m))
    except Exception:
        estate = [0.0] * m.GetNumAtoms()

    # Per-atom Crippen/TPSA contributions (private in some builds)
    try:
        crippen: List[Tuple[float, float]] = rdMolDescriptors._CalcCrippenContribs(
            m
        )  # (logP_i, MR_i)
    except Exception:
        crippen = [(0.0, 0.0)] * m.GetNumAtoms()

    try:
        tpsa: List[float] = rdMolDescriptors._CalcTPSAContribs(m)
    except Exception:
        tpsa = [0.0] * m.GetNumAtoms()

    # Per-atom HBD/HBA (public API)
    hbd, hba = _compute_hbd_hba_flags(m)

    # Polymer context
    att = _attachment_atoms(m)
    backbone = _compute_backbone_atoms(m, att)

    # Graph distance to nearest '*' (0 for star itself)
    INF = 10**9
    dist_to_anchor = [0] * m.GetNumAtoms()
    for i in range(m.GetNumAtoms()):
        if i in att:
            dist_to_anchor[i] = 0
        else:
            dmin = INF
            for a in att:
                p = Chem.GetShortestPath(m, i, a)
                if p:
                    dmin = min(dmin, len(p) - 1)
            dist_to_anchor[i] = 0 if dmin == INF else dmin

    # Conjugated-bond count per atom
    conj_counts = [0] * m.GetNumAtoms()
    for bond in m.GetBonds():
        if bond.GetIsConjugated():
            a = bond.GetBeginAtomIdx()
            b = bond.GetEndAtomIdx()
            conj_counts[a] += 1
            conj_counts[b] += 1

    return dict(
        estate=estate,
        crippen=crippen,
        tpsa=tpsa,
        hbd=hbd,
        hba=hba,
        att=att,
        backbone=backbone,
        dist_to_anchor=dist_to_anchor,
        conj_counts=conj_counts,
        g_charges=g_charges,
        mol=m,  # keep sanitized copy handy
    )


# ------------------- per-atom featurizer -------------------

_HYB_CHOICES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
_CHIRAL_CHOICES = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]


def atom_features_expanded(
    mol: Chem.Mol, atom: Chem.Atom, cache: Dict[str, Any]
) -> torch.Tensor:
    """
    Rich per-atom feature vector (float32).
    First 9 entries match your original function order, then extras.
    """
    i = atom.GetIdx()

    # ---- original 9 (as floats to concatenate cleanly) ----
    base = [
        float(atom.GetAtomicNum()),
        float(int(atom.GetChiralTag())),
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetTotalNumHs()),
        float(atom.GetNumRadicalElectrons()),
        float(int(atom.GetHybridization())),
        float(int(atom.GetIsAromatic())),
        float(int(atom.IsInRing())),
    ]

    # ---- categorical one-hots ----
    hyb_oh = _one_hot(atom.GetHybridization(), _HYB_CHOICES)
    chi_oh = _one_hot(atom.GetChiralTag(), _CHIRAL_CHOICES)

    # ---- valence & topology ----
    implicit_val = float(atom.GetImplicitValence())
    explicit_val = float(atom.GetExplicitValence())
    in_ring5 = float(int(atom.IsInRingSize(5)))
    in_ring6 = float(int(atom.IsInRingSize(6)))
    smallest_ring = float(_smallest_ring_size(cache["mol"], i))
    conj_n = float(cache["conj_counts"][i])

    # ---- electronic / physchem ----
    gchg = float(cache["g_charges"][i])
    estate = _safe_float(cache["estate"][i])
    cr_logp, cr_mr = cache["crippen"][i]
    cr_logp = _safe_float(cr_logp)
    cr_mr = _safe_float(cr_mr)
    tpsa = _safe_float(cache["tpsa"][i])
    mass = _safe_float(atom.GetMass())

    # ---- H-bond roles (binary) ----
    hbd = float(cache["hbd"][i])
    hba = float(cache["hba"][i])

    # ---- polymer context ----
    is_attachment = float(1.0 if (i in cache["att"]) else 0.0)
    is_backbone = float(1.0 if (i in cache["backbone"]) else 0.0)
    dist_to_star = float(cache["dist_to_anchor"][i])

    pendant = 0.0
    if not is_backbone and not is_attachment:
        for n in atom.GetNeighbors():
            if n.GetIdx() in cache["backbone"]:
                pendant = 1.0
                break

    # ---- neighborhood summaries ----
    nc = _neighbor_counts(atom)
    n_hetero = float(nc["n_hetero"])
    n_arom = float(nc["n_arom"])
    n_sp = float(nc["n_sp"])
    n_sp2 = float(nc["n_sp2"])
    n_sp3 = float(nc["n_sp3"])

    feats: List[float] = []
    feats += base
    feats += hyb_oh + chi_oh
    feats += [
        implicit_val,
        explicit_val,
        in_ring5,
        in_ring6,
        smallest_ring,
        conj_n,
        gchg,
        estate,
        cr_logp,
        cr_mr,
        tpsa,
        mass,
        hbd,
        hba,
        is_attachment,
        is_backbone,
        dist_to_star,
        pendant,
        n_hetero,
        n_arom,
        n_sp,
        n_sp2,
        n_sp3,
    ]
    return torch.tensor(feats, dtype=torch.float32)


# ---------------- convenience: whole-molecule ----------------


def featurize_molecule_atoms(mol: Chem.Mol) -> torch.Tensor:
    """
    Return an (n_atoms, n_features) torch.float32 tensor for RDKit Mol.
    Safe for repeat-unit SMILES with '*' anchors.
    """
    cache = build_mol_feature_cache(mol)
    rows = [atom_features_expanded(mol, atom, cache) for atom in mol.GetAtoms()]
    return (
        torch.stack(rows, dim=0) if rows else torch.zeros((0, 1), dtype=torch.float32)
    )


# -------------------------- quick test --------------------------

if __name__ == "__main__":
    s = "*CC(*)c1ccccc1C(=O)OCCCCCC"
    m = Chem.MolFromSmiles(s)
    X = featurize_molecule_atoms(m)
    print("shape:", tuple(X.shape))
    print("first atom vector (first 20 entries):", [float(v) for v in X[0][:20]])
