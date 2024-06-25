import copy
import os
import random
from typing import Optional, Tuple

import pandas as pd
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFMCS
from typing_extensions import TypeAlias

Mol: TypeAlias = Chem.rdchem.Mol


def generate_unique_id(dataframe: pd.DataFrame, return_df: bool = False) -> Optional[pd.DataFrame]:
    df = copy.deepcopy(dataframe)
    df.columns = ["col1", "col2"]
    print(f"Original size: {df.shape}")

    unique_id = set(df["col1"].drop_duplicates().tolist())
    unique_id = unique_id.union(set(df["col2"].drop_duplicates().tolist()))
    print(f"Unique ID num: {len(unique_id)}")

    if return_df:
        return pd.DataFrame({"chembl_id": sorted(list(unique_id))})
    else:
        return None


def split_path(path: str) -> Tuple[str, str]:
    """Split a file path into file directory and file name.

    Parameters
    ----------
    path : str
        File path string.

    Returns
    -------
    Tuple[str, str]
        A tuple includes file directory and file name.
    """
    data_name = os.path.splitext(os.path.split(path)[-1])[0]
    data_dir = os.path.dirname(path)
    return data_dir, data_name


def read_smiles(smiles: str) -> Mol:
    """Read a SMILES string and covert it into Mol object if SMILES is valid.

    Parameters
    ----------
    smiles : str
        Inputted SMILES string.

    Returns
    -------
    Mol
        RDKit molecule object read from SMILES.

    """
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as e:
        print(f"Invalid SMILES: {smiles}, type: {type(smiles)}.")
        raise e
    return mol


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Convert a SMILES string into canonical SMILES.

    Parameters
    ----------
    smiles : str
        Inputted SMILES string.

    Returns
    -------
    Optional[str]
        A canonical SMILES or None if inputted SMILES is invalid.
    """
    try:
        smiles = Chem.CanonSmiles(smiles)
        return smiles
    except:
        return None


def randomize_smiles(smiles: str, random_type: str = "restricted") -> Optional[str]:
    """Returns a random SMILES given a SMILES of a molecule.

    Parameters
    ----------
    smiles : str
        Inputted SMILES string.
    random_type : str, optional
        The type (unrestricted, restricted) of randomization performed., by default "restricted"

    Returns
    -------
    Optional[str]
        A random SMILES string or None if inputted SMILES is invalid.

    """
    mol = read_smiles(smiles)
    if mol is None:
        return None

    if random_type == "unrestricted":
        return Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False)
    elif random_type == "restricted":
        new_atom_order = list(range(mol.GetNumAtoms()))
        random.shuffle(new_atom_order)
        random_mol = Chem.RenumberAtoms(mol, newOrder=new_atom_order)
        return Chem.MolToSmiles(random_mol, canonical=False, isomericSmiles=False)
    else:
        assert False, f"'{random_type}' is not a valid random type."


def smiles2smarts(smiles: str) -> Optional[str]:
    """Convert a SMILES string into SMARTS string.

    Parameters
    ----------
    smiles : str
        Inputted SMILES string.

    Returns
    -------
    Optional[str]
        A SMARTS string or None if inputted SMILES is invalid.
    """
    mol = read_smiles(smiles)
    if mol is None:
        return None

    return Chem.MolToSmarts(mol)


def smiles2selfies(smiles: str) -> Optional[str]:
    """Convert a SMILES string into SELFIES string.

    Parameters
    ----------
    smiles : str
        Inputted SMILES string.

    Returns
    -------
    Optional[str]
        A SELFIES string or None if inputted SMILES is invalid.
    """
    try:
        selfies = sf.encoder(smiles)
    except:
        selfies = None

    return selfies


def cal_atoms_num(smiles: str) -> int:
    """_summary_

    Calculate number of atoms from inputted SMILES.
    ----------
    smiles : str
        Inputted SMILES string.

    Returns
    -------
    int
        Atoms number of inputted molecule (including H atoms.).
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    return mol.GetNumAtoms()


def cal_mol_weight(smiles: str) -> float:
    """_summary_

    Calculate molecular weight.
    ----------
    smiles : str
        Inputted SMILES string.

    Returns
    -------
    float
        Molecular weight.
    """
    mol = Chem.MolFromSmiles(smiles)
    mw = Descriptors.MolWt(mol)
    return mw


def is_1bond_mmp(smiles_a: str, smiles_b: str) -> bool:
    """Judge whether 2 SMILES strings are Matched Molecular Pair or not.

    Parameters
    ----------
    smiles_a : str
        Inputted SMILES string A.
    smiles_b : str
        Inputted SMILES string B.

    Returns
    -------
    bool
        Whether 2 SMILES strings are Matched Molecular Pair or not.
    """
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    mcs = rdFMCS.FindMCS([mol_a, mol_b])
    m_cut_a = Chem.ReplaceCore(mol_a, mcs)
    m_cut_b = Chem.ReplaceCore(mol_b, mcs)
    num_dummy_a = sum([True if atom.GetAtomicNum() == 0 else False for atom in m_cut_a.GetAtoms()])
    num_dummy_b = sum([True if atom.GetAtomicNum() == 0 else False for atom in m_cut_b.GetAtoms()])
    if 1 == num_dummy_a and 1 == num_dummy_b:
        return True
    else:
        return False
