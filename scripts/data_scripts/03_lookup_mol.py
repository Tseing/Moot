import sys

sys.path.append("../..")
from src.data_utils import LookupDict


def cons_mol_lookup() -> None:
    smiles_dict = LookupDict("../../data/all/all_unique_mol.csv", ("chembl_id", "smiles"))
    selfies_dict = LookupDict("../../data/all/all_unique_mol.csv", ("chembl_id", "selfies"))

    smiles_dict.dump("../../data/all/smiles_lookup.pkl")
    selfies_dict.dump("../../data/all/selfies_lookup.pkl")

    print(smiles_dict["CHEMBL268831"])
    print(selfies_dict["CHEMBL268831"])


if __name__ == "__main__":
    cons_mol_lookup()
