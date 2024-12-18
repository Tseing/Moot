from collections import Counter
from typing import List

from rdkit import Chem

from ..typing import Mol, Tuple

DUMMY_ATOM = "*"


def _mark_atom_to_combine(mol: Mol) -> Tuple[Chem.Atom, Chem.Atom]:
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetProp("to_delete", "true")
            neighbors = atom.GetNeighbors()
            combine_atom = neighbors[0]
            combine_atom.SetProp("to_combine", "true")
            assert (
                len(neighbors) == 1
            ), f"There are more than 1 atom linked with dummy atom:\n-> {Chem.MolToSmiles(mol)}"
            break

    return atom, combine_atom


def _remove_a_dummy_atom(rwmol: Mol):
    for atom in rwmol.GetAtoms():
        if atom.HasProp("to_delete"):
            rwmol.RemoveAtom(atom.GetIdx())
            break


def _combine_atoms(rwmol: Mol):
    atoms_to_combine = []

    for atom in rwmol.GetAtoms():
        if atom.HasProp("to_combine"):
            atoms_to_combine.append(atom.GetIdx())

    rwmol.AddBond(*atoms_to_combine, order=Chem.rdchem.BondType.SINGLE)


def _make_combine(mol1, mol2):
    combine_mol = Chem.CombineMols(mol1, mol2)
    rwmol = Chem.RWMol(combine_mol)
    _remove_a_dummy_atom(rwmol)
    _remove_a_dummy_atom(rwmol)
    _combine_atoms(rwmol)
    return rwmol.GetMol()


def get_marked_atoms(core: str, frag: str):
    assert (
        DUMMY_ATOM in core and DUMMY_ATOM in frag
    ), f"There is not dummy atom '*' in core or frag:\n-> {core}\n-> {frag}"
    core_cnt = Counter(core)
    assert core_cnt[DUMMY_ATOM] == 1, f"There is more than 1 dummy atom in core:\n-> {core}"
    frag_cnt = Counter(frag)
    assert frag_cnt[DUMMY_ATOM] == 1, f"There is more than 1 dummy atom in core:\n-> {frag}"

    core = Chem.MolFromSmiles(core)
    frag = Chem.MolFromSmiles(frag)

    _, combine_atom = _mark_atom_to_combine(frag)
    combine_atom.SetProp("to_be_dummy", "true")

    _mark_atom_to_combine(core)
    combined_mol = _make_combine(core, frag)

    for atom in combined_mol.GetAtoms():
        if atom.HasProp("to_be_dummy"):
            dummy_atom_idx = atom.GetIdx()
            break

    smiles = Chem.MolToSmiles(combined_mol)
    # map index of marked atom into canonical index
    cano_dummy_atom_idx = list(
        map(int, combined_mol.GetProp("_smilesAtomOutputOrder")[1:-2].split(","))
    ).index(dummy_atom_idx)

    cano_mol = Chem.MolFromSmiles(smiles)
    dummy_atom = cano_mol.GetAtomWithIdx(cano_dummy_atom_idx)
    dummy_atom.SetAtomicNum(0)

    atoms = [f"[{atom.GetSymbol()}]" for atom in cano_mol.GetAtoms()]
    return "".join(atoms)


def is_single_point_frag(mol: Mol) -> bool:
    atoms = [f"{atom.GetSymbol()}" for atom in mol.GetAtoms()]
    if Counter(atoms)[DUMMY_ATOM] == 1:
        return True
    else:
        return False


def split_marked_mol(mol: Mol) -> List[Mol]:
    atoms = [f"{atom.GetSymbol()}" for atom in mol.GetAtoms()]
    assert (
        Counter(atoms)[DUMMY_ATOM] == 1
    ), f"Too many marked dummy atoms or no dummy atom in SMILES: '{Chem.MolToSmiles(mol)}'."

    marked_atom = mol.GetAtomWithIdx(atoms.index(DUMMY_ATOM))
    marked_degree = marked_atom.GetDegree()
    assert marked_degree > 1, f"Invalid marked atom in one end of molecular structure: '{Chem.MolToSmiles(mol)}'."

    to_split_bonds = [bond.GetIdx() for bond in marked_atom.GetBonds()]
    frag = Chem.FragmentOnBonds(
        mol, to_split_bonds, dummyLabels=[(0, 0) for _ in range(len(to_split_bonds))]
    )
    frags = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False)

    return list(filter(is_single_point_frag, frags))
