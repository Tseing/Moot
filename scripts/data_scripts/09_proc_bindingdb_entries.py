import pickle
import sys
from functools import reduce
from typing import Any, Callable, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import requests
from pandarallel import pandarallel
from tqdm import tqdm

sys.path.append("../..")
from src.data_utils import canonicalize_smiles


class BinNode:
    def __init__(self, data: Any) -> None:
        self.parent: Optional["BinNode"] = None
        self.l_child: Optional["BinNode"] = None
        self.r_child: Optional["BinNode"] = None
        self.data = data

    def insert_lc(self, child: "BinNode"):
        self.l_child = child
        child.parent = self
        return self.l_child

    def insert_rc(self, child: "BinNode"):
        self.r_child = child
        child.parent = self
        return self.r_child

    def is_root(self) -> bool:
        if self.parent is None:
            return True
        else:
            return False

    def is_l_child(self) -> bool:
        if not self.is_root() and self.parent.l_child is self:
            return True
        else:
            return False

    def is_r_child(self) -> bool:
        if not self.is_root() and self.parent.r_child is self:
            return True
        else:
            return False

    def has_child(self) -> bool:
        if self.l_child is not None or self.r_child is not None:
            return True
        else:
            return False

    def has_sibling(self) -> bool:
        if self.is_l_child() and self.parent.r_child is not None:
            return True
        elif self.is_r_child() and self.parent.l_child is not None:
            return True
        else:
            return False


class BinArraySearchNode(BinNode):
    def __init__(
        self,
        data: Optional[list],
        status: Union[Literal["input"], Literal["result"]] = "input",
    ) -> None:
        super().__init__(data)
        self.status = status

    def has_child(self) -> bool:
        if self.l_child is not None:
            return True
        else:
            return False

    def has_r_sibling(self) -> bool:
        if self.is_l_child() and self.parent.r_child is not None:
            return True
        else:
            return False

    def construct_subtree(self, comp_func: Callable) -> Literal[0]:
        node = self
        data = self.data
        lo = 0
        hi = len(data)

        while 1 < hi - lo:
            mi = (lo + hi) // 2
            res_l = comp_func(data[lo:mi])
            res_r = comp_func(data[mi:hi])

            if isinstance(res_l, list) and isinstance(res_r, list):
                node.insert_lc(BinArraySearchNode(res_l, "result"))
                node.insert_rc(BinArraySearchNode(res_r, "result"))
                return 0

            elif res_l == -1 and res_r == -1:
                node.insert_rc(BinArraySearchNode(data[mi:hi]))
                node = node.insert_lc(BinArraySearchNode(data[lo:mi]))
                hi = mi

            elif res_l == -1 and isinstance(res_r, list):
                node.insert_rc(BinArraySearchNode(res_r, "result"))
                node = node.insert_lc(BinArraySearchNode(data[lo:mi]))
                hi = mi

            elif isinstance(res_l, list) and res_r == -1:
                node.insert_lc(BinArraySearchNode(res_l, "result"))
                node = node.insert_rc(BinArraySearchNode(data[mi:hi]))
                lo = mi

            else:
                assert False

        res_l = comp_func(data[lo:hi])
        res_l = [None] if res_l == -1 else res_l
        node.insert_lc(BinArraySearchNode(res_l, "result"))

        return 0

    def next_node(self) -> Optional["BinArraySearchNode"]:
        if self.has_child():
            return self.trav_left()
        elif self.has_r_sibling():
            return self.parent.r_child
        else:
            return None

    def trav_left(self) -> "BinArraySearchNode":
        node = self
        while node.has_child():
            node = node.l_child

        return node


class BinArraySearchTree:
    def __init__(self, array: list, comp_func: Callable) -> None:
        self.root = BinArraySearchNode(array)
        self.comp_func = comp_func
        self.root.construct_subtree(self.comp_func)

    def get_results(self) -> list:
        self._results: List[Any] = []
        node = self.root
        self.trav_pre(node)
        return self._results

    def trav_left(self, node: BinArraySearchNode, stack: List[BinArraySearchNode]) -> None:
        while node.has_child():
            if node.r_child is not None:
                stack.append(node.r_child)

            node = node.l_child

        if node.status == "input":
            node.construct_subtree(self.comp_func)
            self.trav_left(node, stack)
        elif node.status == "result":
            self._results.extend(node.data)
        else:
            assert False

    def trav_pre(self, node: BinArraySearchNode) -> None:
        stack: List[BinArraySearchNode] = []
        while True:
            self.trav_left(node, stack)
            if len(stack) == 0:
                break
            node = stack.pop()


def fetch_single_chain():
    path = "../../data/bindingdb/BindingDB_All_202406.tsv"
    dump_path = "../../data/bindingdb/single_chain.tsv"
    with open(path, "r") as f:
        with open(dump_path, "w+") as dump_f:
            header = f.readline()
            header = "\t".join(header.strip().split("\t")[:50])
            dump_f.write(header)

            original_size = 0
            final_size = 0

            while l := f.readline():
                original_size += 1
                entries = l.strip().split("\t")
                chain_num = int(entries[37])

                if chain_num != 1:
                    continue
                else:
                    final_size += 1
                    dump_f.write(l)

    print(f"{original_size} -> {final_size}: dropped {original_size - final_size} entries.")


def fetch_necessary_cols():
    df_path = "../../data/bindingdb/single_chain.tsv"
    df = pd.read_csv(df_path, sep="\t")

    df = df[
        [
            "PubChem CID",
            "Ki (nM)",
            "IC50 (nM)",
            "Kd (nM)",
            "EC50 (nM)",
            "BindingDB Target Chain Sequence",
        ]
    ]
    df.dropna(subset=["PubChem CID", "BindingDB Target Chain Sequence"], how="any", inplace=True)
    df.dropna(subset=["Ki (nM)", "IC50 (nM)", "Kd (nM)", "EC50 (nM)"], how="all", inplace=True)

    print(f"Data size: {df.shape[0]}")
    df.to_csv("../../data/bindingdb/protein-ligand-all.csv", index=False)


def fetch_positive_ligands():
    df_path = "../../data/bindingdb/protein-ligand-all.csv"
    df = pd.read_csv(df_path)

    def is_float_compatible(x: Any) -> bool:
        # nan, float string, inf
        try:
            np.float32(x)
            return True
        except:
            return False

    vec_is_float_compatible = np.vectorize(is_float_compatible)

    for col in ["Ki (nM)", "IC50 (nM)", "Kd (nM)", "EC50 (nM)"]:
        col_array = df[col].to_numpy(dtype=np.str_)
        col_array = np.char.replace(col_array, "<", "")
        col_array[~vec_is_float_compatible(col_array)] = np.nan
        col_array = col_array.astype(np.float32)

        df[col] = pd.Series(col_array)

    positive_idx = reduce(
        np.logical_or,
        [df["Ki (nM)"] < 100, df["IC50 (nM)"] < 100, df["Kd (nM)"] < 100, df["EC50 (nM)"] < 100],
    )

    df = df[positive_idx]
    df.to_csv("../../data/bindingdb/positive-ligands.csv", index=False)
    print(f"Data size: {df.shape[0]}")


def get_pubchem_smiles(cids: List[int]) -> Union[List[str], Literal[-1]]:
    string_cids = [str(i) for i in cids]
    query_cids = ",".join(string_cids)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{query_cids}/property/CanonicalSMILES/JSON"
    response = requests.get(url).json()
    if "Fault" in response:
        return -1
    else:
        return [entry["CanonicalSMILES"] for entry in response["PropertyTable"]["Properties"]]


def lookup_cid():
    df_path = "../../data/bindingdb/positive-ligands.csv"
    df = pd.read_csv(df_path)
    cids = df["PubChem CID"].drop_duplicates().to_numpy(dtype=np.int32)
    smiles_list = []

    record_num = 300
    query_num = len(cids) // record_num + 1

    for i in tqdm(range(query_num)):
        query_cids = cids[i * record_num : (i + 1) * record_num]
        smiles = get_pubchem_smiles(query_cids)

        if smiles == -1:
            fault_search_tree = BinArraySearchTree(query_cids, get_pubchem_smiles)
            smiles = fault_search_tree.get_results()

        smiles_list.extend(smiles)

    lookup_df = pd.DataFrame({"cid": cids, "smiles": smiles_list})
    lookup_df.dropna(how="any", inplace=True)
    lookup_dict = dict(zip(lookup_df["cid"].to_list(), lookup_df["smiles"].to_list()))
    pickle.dump(lookup_dict, open("../../data/bindingdb/cid_lookup.pkl", "wb"))

def format_smiles():
    cid_d = pickle.load(open("../../data/bindingdb/raw_cid_lookup.pkl", "rb"))
    for cid, smiles in cid_d.items():
        cid_d[cid] = canonicalize_smiles(smiles)
    pickle.dump(cid_d, open("../../data/bindingdb/cid_lookup_smiles.pkl", "wb"))

def format_dataset():
    df_path = "../../data/bindingdb/positive-ligands.csv"
    cid_d = pickle.load(open("../../data/bindingdb/cid_lookup_smiles.pkl", "rb"))

    pandarallel.initialize(nb_workers=30, progress_bar=True)
    df = pd.read_csv(df_path)
    format_df = pd.DataFrame()
    format_df["smiles"] = df["PubChem CID"].parallel_apply(lambda x: cid_d[int(x)])
    format_df["target_seq"] = df["BindingDB Target Chain Sequence"]

    format_df.to_csv("../../data/bindingdb/formatted-positive-ligand.csv", index=False)


def stats_dataset():
    df_path = "../../data/bindingdb/formatted-positive-ligand.csv"
    df = pd.read_csv(df_path)
    grouped = df.groupby("smiles")
    print(f"Unique SMILES: {len(grouped)}")
    grouped = df.groupby("target_seq")
    print(f"Unique target sequences: {len(grouped)}")


if __name__ == "__main__":
    # fetch_single_chain()
    # fetch_necessary_cols()
    # fetch_positive_ligands()
    # lookup_cid()
    # format_smiles()
    # format_dataset()
    stats_dataset()
