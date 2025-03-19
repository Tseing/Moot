import os
import sys
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Optional, Set, TypeVar

import pandas as pd
from pandarallel import pandarallel
from rdkit import Chem, RDConfig
from rdkit.Chem import DataStructs, Descriptors, MACCSkeys, PandasTools, Scaffolds
from terminaltables import AsciiTable
from tqdm import tqdm

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))

import sascorer

from .data_utils import canonicalize_smiles, selfies2smiles
from .typing import Mol

T = TypeVar("T")
U = TypeVar("U")


def skip_null(func: Callable[[T], U]) -> Callable[[Optional[T]], Optional[U]]:
    def wrapper(*args, **kwargs):
        if None in args or None in kwargs.values():
            return None
        else:
            res = func(*args, **kwargs)
            return res

    return wrapper


def null2str(obj: Any):
    if pd.isna(obj):
        return ""

    return obj


class Metrics(ABC):
    EXPECTED_COLS: List[str]
    SMILES_COLS: List[str]
    df: pd.DataFrame
    tqdm.pandas()

    @abstractmethod
    def visual(self) -> None:
        raise NotImplementedError

    @staticmethod
    def render_romol(df: pd.DataFrame) -> None:
        PandasTools.molRepresentation = "svg"
        PandasTools.ChangeMoleculeRendering(df)

    @staticmethod
    def get_mol(smiles: Optional[str]) -> Optional[Mol]:
        try:
            mol = Chem.MolFromSmiles(smiles)
        except:
            return None
        return mol

    @staticmethod
    @skip_null
    def get_sascore(mol: Mol) -> float:
        return sascorer.calculateScore(mol)

    @staticmethod
    @skip_null
    def get_qed(mol: Mol) -> float:
        return Descriptors.qed(mol)

    @staticmethod
    @skip_null
    def get_weight(mol: Mol) -> float:
        return Descriptors.MolWt(mol)

    @staticmethod
    @skip_null
    def get_scaffold(mol: Mol) -> str:
        return Chem.MolToSmiles(Scaffolds.MurckoScaffold.GetScaffoldForMol(mol))

    @staticmethod
    @skip_null
    def get_similarity(mol_a: Mol, mol_b: Mol) -> float:
        fp_a, fp_b = tuple(MACCSkeys.GenMACCSKeys(mol) for mol in (mol_a, mol_b))
        return DataStructs.TanimotoSimilarity(fp_a, fp_b)

    def cano_smiles(self, col: str) -> None:
        self.df[col] = self.df[col].parallel_apply(canonicalize_smiles)

    def cano_selfies(self, col: str) -> None:
        self.df[col] = self.df[col].parallel_apply(selfies2smiles)

    def metric_mol(self, col: str) -> pd.Series:
        return self.df[col].parallel_apply(self.get_mol)

    def metric_heavy(self, col: str) -> pd.Series:
        return self.df[col].astype("int32")

    def metric_sascore(self, col: str) -> pd.Series:
        return self.df[col].parallel_apply(lambda s: self.get_mol(self.get_sascore(s)))

    def metric_qed(self, col: str) -> pd.Series:
        return self.df[col].parallel_apply(lambda s: self.get_mol(self.get_qed(s)))

    def metric_weight(self, col: str) -> pd.Series:
        return self.df[col].parallel_apply(lambda s: self.get_mol(self.get_weight(s)))

    def metric_scaffold(self, col: str) -> pd.Series:
        return self.df[col].parallel_apply(lambda s: self.get_mol(self.get_scaffold(s)))

    def metric_frag_sim(self, col_a: str, col_b: str) -> pd.Series:
        return self.df.parallel_apply(
            lambda df: self.get_similarity(self.get_mol(df[col_a]), self.get_mol(df[col_b])), axis=1
        )

    def metric_frag_prop(self, frag_col: str, core_col: str) -> pd.Series:
        return self.df[frag_col] / self.df[core_col]

    def _visual(self, save_path: str, nrows: int = -1) -> None:
        visual_df = self.df[self.EXPECTED_COLS]

        if nrows != -1:
            visual_df = visual_df.iloc[:nrows]

        self.render_romol(visual_df)
        for col in self.SMILES_COLS:
            visual_df[f"{col}_romol"] = visual_df[col].parallel_apply(self.get_mol)

        visual_df.to_html(save_path)

    def _check_and_align_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        assert set(df.columns.to_list()) == set(
            self.EXPECTED_COLS
        ), f"Expected DataFrame with columns {self.EXPECTED_COLS} but got {df.columns.to_list()}."

        return df.reindex(columns=self.EXPECTED_COLS)


class DatasetMetrics(Metrics):
    EXPECTED_COLS = [
        "src",
        "tgt",
        "core",
        "frag_a",
        "frag_b",
        "core_heavy",
        "frag_a_heavy",
        "frag_b_heavy",
    ]
    SMILES_COLS = ["src", "tgt", "core", "frag_a", "frag_b"]

    def __init__(self, df: pd.DataFrame) -> None:
        assert set(df.columns.to_list()) == set(
            self.EXPECTED_COLS
        ), f"Expected DataFrame with columns {self.EXPECTED_COLS} but got {df.columns.to_list()}."
        self.df = df.reindex(columns=self.EXPECTED_COLS)
        self.render_romol(self.df)


class GenerationMetrics(Metrics):
    def __init__(
        self,
        df: pd.DataFrame,
        data_format: Literal["SMILES", "SELFIES"] = "SMILES",
        topk: int = 1,
        worker: int = 10,
    ) -> None:
        self._check_topk(df, topk)
        self.df = self._check_and_align_columns(df)
        self.data_format = data_format
        self.topk = topk
        self.num_result_group = self.df.shape[0] // topk
        self.inp_slice = slice(0, self.df.shape[0], topk)
        self.worker = worker
        pandarallel.initialize(nb_workers=worker, progress_bar=True)

    def concat_tokens(self) -> None:
        self.df["out"] = self.df["out"].parallel_apply(
            lambda s: "".join(s.strip("{eos}").strip().split(" "))
        )
        if self.data_format == "SELFIES":
            self.df["out"] = self.df["out"].parallel_apply(selfies2smiles)

    def visual(self, save_path: str, nrows: int = -1) -> None:
        self.concat_tokens()
        return self._visual(save_path, nrows)

    def _get_dict_groupby_inp(self, inp_fields: List[str], tgt_field: str) -> Dict[str, set]:
        inp_groups = self.df[self.inp_slice].groupby(inp_fields)
        return {":".join(group[0]): set(group[1][tgt_field].to_list()) for group in inp_groups}

    def _get_optimization(self) -> Set[str]:
        return set(
            self.df[["frag_a", "frag_b"]][self.inp_slice]
            .parallel_apply(lambda df: ">>".join([df["frag_a"], df["frag_b"]]), axis=1)
            .to_list()
        )

    @staticmethod
    def _check_topk(df: pd.DataFrame, topk: int):
        assert (
            df.shape[0] % topk == 0
        ), f"DataFrame with shape {df.shape} cannot be divided by topk `{topk}`."


class MolMetrics(GenerationMetrics):
    EXPECTED_COLS = [
        "target",
        "mol_a_smiles",
        "mol_b_smiles",
        "core",
        "frag_a",
        "frag_b",
        "src",
        "out",
        "gen_core",
        "gen_frag_a",
        "gen_frag_b",
        "gen_core_heavy",
        "gen_frag_a_heavy",
        "gen_frag_b_heavy",
    ]
    SMILES_COLS = [
        "mol_a_smiles",
        "mol_b_smiles",
        "core",
        "frag_a",
        "frag_b",
        "out",
        "gen_core",
        "gen_frag_a",
        "gen_frag_b",
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        data_format: Literal["SMILES", "SELFIES"] = "SMILES",
        topk: int = 1,
        worker: int = 10,
    ) -> None:
        super().__init__(df, data_format, topk, worker)

    @staticmethod
    def check_recovery(row: pd.DataFrame, check_field: str, check_dict: Dict[str, set]) -> int:
        tgt_key = ":".join([row["target"], row["mol_a_smiles"]])
        tgt_set = check_dict[tgt_key]
        if row[check_field] in tgt_set:
            return 1
        else:
            return 0

    @staticmethod
    def check_optimization_recovery(row: pd.DataFrame, optimizations: Set[str]) -> int:
        gen_optimization = ">>".join([null2str(row["gen_frag_a"]), null2str(row["gen_frag_b"])])
        if gen_optimization in optimizations:
            return 1
        else:
            return 0

    def basic_metric(self) -> None:
        # Validity, MMP Validity, Recovery, Uniqueness, Novelty
        self.concat_tokens()
        self.cano_smiles("out")
        size = self.df.shape[0]
        tgt_col = "mol_b_smiles"
        metrics = {}

        metrics["validity"] = len(self.df["out"].dropna()) / size
        metrics["mmp_validity"] = len(self.df["gen_core"].dropna()) / size
        metrics["strict_recovery"] = (self.df["out"] == self.df[tgt_col]).sum() / size

        mol_dict = self._get_dict_groupby_inp(["target", "mol_a_smiles"], tgt_col)
        check_mol_recovery = lambda row: self.check_recovery(row, "out", mol_dict)
        mol_recovery_marks = self.df[["target", "mol_a_smiles", "out"]].parallel_apply(
            check_mol_recovery, axis=1
        )
        del mol_dict
        del check_mol_recovery

        core_dict = self._get_dict_groupby_inp(["target", "mol_a_smiles"], "core")
        check_core_recovery = lambda row: self.check_recovery(row, "gen_core", core_dict)
        core_recovery_marks = self.df[["target", "mol_a_smiles", "gen_core"]].parallel_apply(
            check_core_recovery, axis=1
        )
        del core_dict
        del check_core_recovery

        frag_a_dict = self._get_dict_groupby_inp(["target", "mol_a_smiles"], "frag_a")
        check_frag_a_recovery = lambda row: self.check_recovery(row, "gen_frag_a", frag_a_dict)
        frag_a_recovery_marks = self.df[["target", "mol_a_smiles", "gen_frag_a"]].parallel_apply(
            check_frag_a_recovery, axis=1
        )
        del frag_a_dict
        del check_frag_a_recovery

        frag_b_dict = self._get_dict_groupby_inp(["target", "mol_a_smiles"], "frag_b")
        check_frag_b_recovery = lambda row: self.check_recovery(row, "gen_frag_b", frag_b_dict)
        frag_b_recovery_marks = self.df[["target", "mol_a_smiles", "gen_frag_b"]].parallel_apply(
            check_frag_b_recovery, axis=1
        )
        del frag_b_dict
        del check_frag_b_recovery

        all_optimizations = self._get_optimization()
        check_optimization_recovery = lambda row: self.check_optimization_recovery(
            row, all_optimizations
        )
        optimization_recovery_marks = self.df[["gen_frag_a", "gen_frag_b"]].parallel_apply(
            check_optimization_recovery, axis=1
        )
        del all_optimizations
        del check_optimization_recovery

        mol_recovery_cnt = 0
        core_recovery_cnt = 0
        frag_a_recovery_cnt = 0
        frag_b_recovery_cnt = 0
        optimization_recovery_cnt = 0

        for i in tqdm(range(self.num_result_group), total=self.num_result_group):
            group_slice = slice(i * self.topk, (i + 1) * self.topk)
            mol_recovery_cnt += int(mol_recovery_marks[group_slice].sum() > 0)
            core_recovery_cnt += int(core_recovery_marks[group_slice].sum() > 0)
            frag_a_recovery_cnt += int(frag_a_recovery_marks[group_slice].sum() > 0)
            frag_b_recovery_cnt += int(frag_b_recovery_marks[group_slice].sum() > 0)
            optimization_recovery_cnt += int(optimization_recovery_marks[group_slice].sum() > 0)

        metrics["mol_recovery"] = mol_recovery_cnt / self.num_result_group
        metrics["core_recovery"] = core_recovery_cnt / self.num_result_group
        metrics["frag_a_recovery"] = frag_a_recovery_cnt / self.num_result_group
        metrics["frag_b_recovery"] = frag_b_recovery_cnt / self.num_result_group

        metrics["uniqueness"] = len(self.df["out"].dropna().drop_duplicates()) / size
        unique_tgt = set(self.df[tgt_col].drop_duplicates().to_list())
        novel_marks = self.df["out"].apply(lambda s: 1 if s in unique_tgt else 0)
        metrics["novelty"] = novel_marks.sum() / size

        metric_infos = [
            ["Metrics", "Value"],
            ["Mol Validity", f"{metrics['validity']:.4f}"],
            ["MMP Validity", f"{metrics['mmp_validity']:.4f}"],
            ["Strict Recovery", f"{metrics['strict_recovery']:.4f}"],
            ["Recovery", f"{metrics['mol_recovery']:.4f}"],
            ["Core Recovery", f"{metrics['core_recovery']:.4f}"],
            ["Truncated Frag Recovery", f"{metrics['frag_a_recovery']:.4f}"],
            ["Spliced Frag Recovery", f"{metrics['frag_b_recovery']:.4f}"],
            ["Uniqueness", f"{metrics['uniqueness']:.4f}"],
            ["Novelty", f"{metrics['novelty']:.4f}"],
        ]
        table = AsciiTable(metric_infos)
        table.title = f"Basic Metrics (Total: {size})"
        table.inner_row_border = True
        print(table.table)


class FragMetrics(GenerationMetrics):
    EXPECTED_COLS = [
        "target",
        "frag_a",
        "frag_b",
        "core",
        "src",
        "out",
    ]
    SMILES_COLS = [
        "frag_a",
        "frag_b",
        "core",
        "out",
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        data_format: Literal["SMILES", "SELFISE"] = "SMILES",
        topk: int = 1,
        worker: int = 10,
    ) -> None:
        super().__init__(df, data_format, topk, worker)

    def basic_metric(self) -> None:
        self.concat_tokens()
        self.cano_smiles("out")
        size = self.df.shape[0]
        tgt_col = "frag_b"

        metrics = {}
        metrics["validity"] = len(self.df["out"].dropna()) / size
        metrics["strict_recovery"] = (self.df["out"] == self.df[tgt_col]).sum() / size

        all_optimizations = self._get_optimization()
        inp_groups_dict = self._get_dict_groupby_inp(["target", "frag_a"], tgt_col)

        def check_recovery(row: pd.DataFrame) -> int:
            tgt_key = ":".join([row["target"], row["frag_a"]])
            tgt_set = inp_groups_dict[tgt_key]
            if row["out"] in tgt_set:
                return 1
            else:
                return 0

        def check_optimization_recovery(row: pd.DataFrame) -> int:
            out_optimization = ">>".join([row["frag_a"], null2str(row["out"])])
            if out_optimization in all_optimizations:
                return 1
            else:
                return 0

        recovery_marks = self.df[["target", "frag_a", "out"]].parallel_apply(check_recovery, axis=1)
        optimization_recovery_marks = self.df[["frag_a", "out"]].parallel_apply(
            check_optimization_recovery, axis=1
        )

        recovery_cnt = 0
        optimization_recovery_cnt = 0

        for i in tqdm(range(self.num_result_group), total=self.num_result_group):
            group_slice = slice(i * self.topk, (i + 1) * self.topk)
            recovery_cnt += int(recovery_marks[group_slice].sum() > 0)
            optimization_recovery_cnt += int(optimization_recovery_marks[group_slice].sum() > 0)

        metrics["recovery"] = recovery_cnt / self.num_result_group
        metrics["optimization_recovery"] = optimization_recovery_cnt / self.num_result_group

        metrics["uniqueness"] = len(self.df["out"].dropna().drop_duplicates()) / size

        metric_infos = [
            ["Metrics", "Value"],
            ["Frag Validity", f"{metrics['validity']:.4f}"],
            ["Strict Recovery", f"{metrics['strict_recovery']:.4f}"],
            ["Recovery", f"{metrics['recovery']:.4f}"],
            ["Optimization Recovery", f"{metrics['optimization_recovery']:.4f}"],
            ["Uniqueness", f"{metrics['uniqueness']:.4f}"],
        ]
        table = AsciiTable(metric_infos)
        table.title = f"Basic Metrics (Total: {size})"
        table.inner_row_border = True
        print(table.table)
