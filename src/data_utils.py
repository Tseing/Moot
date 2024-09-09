import multiprocessing
import os
import os.path as osp
import pickle
import random
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Tuple, TypeVar

import pandas as pd
import selfies as sf
from rdkit import Chem, RDConfig
from rdkit.Chem import DataStructs, Descriptors, MACCSkeys, PandasTools, Scaffolds
from terminaltables import AsciiTable
from tqdm import tqdm
from typing_extensions import TypeAlias

from .mmpdblib import smarts_aliases
from .mmpdblib.fragment_algorithm import fragment_mol
from .mmpdblib.fragment_records import parse_record
from .mmpdblib.fragment_types import Fragmentation, FragmentOptions

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))

import sascorer

Mol: TypeAlias = Chem.rdchem.Mol
T = TypeVar("T")
U = TypeVar("U")


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


def selfies2smiles(selfies: str) -> Optional[str]:
    try:
        smiles = sf.decoder(selfies)
    except:
        return None

    return canonicalize_smiles(smiles)


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


def get_atom_symbols(smi: str) -> set:
    mol = Chem.MolFromSmiles(smi)
    atoms = set([mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())])
    return atoms


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


class LookupDict:
    def __init__(self, path: str, cols: Optional[Tuple[str, str]] = None) -> None:
        file_name, file_ext = osp.splitext(path)

        if file_ext == ".pkl":
            self._load_pkl(path)
        elif file_ext == ".csv":
            self._read_csv(path, cols)
        else:
            assert False, f"Only support '.pkl' and '.csv' file, but got '{file_ext}'."

        self.dump_path = f"{file_name}.pkl"

    def __repr__(self) -> str:
        return self.d.__repr__()

    def __getitem__(self, key: str) -> str:
        return self.d[key]

    def _read_csv(self, csv_path: str, cols: Optional[Tuple[str, str]]) -> None:
        df = pd.read_csv(csv_path, usecols=cols)
        self._read_df(df, cols)

    def _read_df(self, df: pd.DataFrame, cols: Optional[Tuple[str, str]]) -> None:
        assert (
            df.shape[1] == 2
        ), "DataFrame contains columns more than 2. Please specify attribute 'cols'."

        if cols is None:
            k_col, v_col = df.columns
        else:
            k_col, v_col = cols

        df.drop_duplicates(inplace=True)
        self.d = dict(zip(df[k_col], df[v_col]))
        self._const_lookup_func()

    def _load_pkl(self, path: str) -> None:
        self.d = pickle.load(open(path, "rb"))
        self._const_lookup_func()

    def dump(self, path: Optional[str] = None):
        if path is None:
            _dump_path = self.dump_path
        else:
            _dump_path = path
        pickle.dump(self.d, open(_dump_path, "wb"))

    def _const_lookup_func(self) -> None:
        self._lookup: Callable[[str], Optional[str]] = lambda k: self.d.get(k, None)

    @property
    def lookup(self) -> Callable[[str], Optional[str]]:
        return self._lookup


class SMILESDict(LookupDict):
    def __init__(self, path: str, cols: Optional[Tuple[str, str]] = None) -> None:
        super().__init__(path, cols)

    def _read_csv(
        self, csv_path: str, cols: Optional[Tuple[str, str]] = ("chembl_id", "smiles")
    ) -> None:
        return super()._read_csv(csv_path, cols)

    def _read_df(
        self, df: pd.DataFrame, cols: Optional[Tuple[str, str]] = ("chembl_id", "smiles")
    ) -> None:
        return super()._read_df(df, cols)


@dataclass
class MMPRecord:
    const_smiles: str
    const_heavy: int
    frag_a_smiles: str
    frag_a_heavy: int
    frag_b_smiles: str
    frag_b_heavy: int

    def parse_list(self) -> list:
        return [
            self.const_smiles,
            self.const_heavy,
            self.frag_a_smiles,
            self.frag_a_heavy,
            self.frag_b_smiles,
            self.frag_b_heavy,
        ]


class MMPChecker:
    """Options:
    --max-up-enumerations N         Maximum number of up-enumerations (default:
                                    1000)
    --min-heavies-total-const-frag N
                                    Ignore fragmentations where there are fewer
                                    than N heavy atoms in the total constant
                                    fragment  (default:
                                    {OPTS.min_heavies_total_const_frag})
    --min-heavies-per-const-frag N  Ignore fragmentations where one or more
                                    constant fragments have fewer than N heavy
                                    atoms (default: 0)
    --num-cuts [1|2|3]              Number of cuts to use (default: 3)
    --cut-rgroup-file FILENAME      Read R-group SMILES from the named file
    --cut-rgroup SMILES             Cut on the attachment point for the given
                                    R-group SMILES
    --cut-smarts SMARTS             Alternate SMARTS pattern to use for cutting
                                    (default: '[#6+0;!$(*=,#[!#6])]!@!=!#[!#0;!#
                                    1;!$([CH2]);!$([CH3][CH2])]'), or use one
                                    of: 'default', 'cut_AlkylChains',
                                    'cut_Amides', 'cut_all', 'exocyclic',
                                    'exocyclic_NoMethyl'
    --salt-remover FILENAME         File containing RDKit SaltRemover
                                    definitions. The default ('<default>') uses
                                    RDKit's standard salt remover. Use '<none>'
                                    to not remove salts.
    --rotatable-smarts SMARTS       SMARTS pattern to detect rotatable bonds
                                    (default: '[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@
                                    [!$([NH]!@C(=O))&!D1&!$(*#*)]')
    --max-rotatable-bonds N         Maximum number of rotatable bonds (default:
                                    10)
    --max-heavies N                 Maximum number of non-hydrogen atoms, or
                                    'none' (default: 100)
    --help                          Show this message and exit.

    The --cut-smarts argument supports the following short-hand aliases:
    'default': Cut all C-[!H] non-ring single bonds except for Amides/Esters/Amidines/Sulfonamides and CH2-CH2 and CH2-CH3 bonds
        smarts: [#6+0;!$(*=,#[!#6])]!@!=!#[!#0;!#1;!$([CH2]);!$([CH3][CH2])]
    'cut_AlkylChains': As default, but also cuts CH2-CH2 and CH2-CH3 bonds
        smarts: [#6+0;!$(*=,#[!#6])]!@!=!#[!#0;!#1]
    'cut_Amides': As default, but also cuts [O,N]=C-[O,N] single bonds
        smarts: [#6+0]!@!=!#[!#0;!#1;!$([CH2]);!$([CH3][CH2])]
    'cut_all': Cuts all Carbon-[!H] single non-ring bonds. Use carefully, this will create a lot of cuts
        smarts: [#6+0]!@!=!#[!#0;!#1]
    'exocyclic': Cuts all exocyclic single bonds
        smarts: [R]!@!=!#[!#0;!#1]
    'exocyclic_NoMethyl': Cuts all exocyclic single bonds apart from those connecting to CH3 groups
        smarts: [R]!@!=!#[!#0;!#1;!$([CH3])]
    """

    def __init__(self, cache_path: Optional[str] = None) -> None:
        frag_options = FragmentOptions(
            max_heavies=100,
            max_rotatable_bonds=20,
            rotatable_smarts="[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]",
            cut_smarts=smarts_aliases.cut_smarts_aliases_by_name["default"].smarts,
            num_cuts=1,
            method="chiral",
            salt_remover="<none>",
            min_heavies_per_const_frag=0,
            min_heavies_total_const_frag=0,
            max_up_enumerations=1000,
        )
        self.mol_filter = frag_options.get_fragment_filter()
        if cache_path is not None:
            self.cache = pickle.load(open(cache_path, "rb"))
        else:
            self.cache = {}

    def check_mmp(self, smiles_a: str, smiles_b: str) -> Optional[MMPRecord]:
        frag_record_a = self.cache[smiles_a] if smiles_a in self.cache else self.frag(smiles_a)
        frag_record_b = self.cache[smiles_b] if smiles_b in self.cache else self.frag(smiles_b)

        if None in (frag_record_a, frag_record_b):
            return None

        common_consts = list(set(frag_record_a.keys()).intersection(set(frag_record_b.keys())))
        if len(common_consts) == 0:
            return None
        const_heavies = [
            frag_record_a[frag_const].constant_num_heavies for frag_const in common_consts
        ]
        max_const, const_heavy = max(zip(common_consts, const_heavies), key=lambda x: x[1])
        const_smiles = frag_record_a[max_const].constant_smiles

        frag_a, frag_a_heavy = (
            frag_record_a[max_const].variable_smiles,
            frag_record_a[max_const].variable_num_heavies,
        )
        frag_b, frag_b_heavy = (
            frag_record_b[max_const].variable_smiles,
            frag_record_b[max_const].variable_num_heavies,
        )

        return MMPRecord(const_smiles, const_heavy, frag_a, frag_a_heavy, frag_b, frag_b_heavy)

    def cache_records(self, smiles_list: List[str], save_path: str, worker: int = 10) -> None:
        pool = multiprocessing.Pool(worker)
        pbar = tqdm(total=len(smiles_list))
        seen = {k if self.cache[k] is not None else "" for k in self.cache}
        record_keys = []
        results = []
        for smiles in smiles_list:
            if smiles in seen:
                pbar.update(1)
                continue

            seen.add(smiles)
            result = pool.apply_async(self.frag, (smiles,), callback=lambda _: pbar.update(1))
            results.append(result)
            record_keys.append(smiles)

        pool.close()
        pool.join()

        records = [result.get() for result in results]
        new_cache = dict(zip(record_keys, records))
        self.cache.update(new_cache)

        pickle.dump(self.cache, open(save_path, "wb"))

    def frag(
        self, smiles: Optional[str], chembl_id: Optional[str] = None
    ) -> Optional[Dict[str, Fragmentation]]:
        if smiles is None:
            return None

        error_msg, mol_record = parse_record(chembl_id, smiles, self.mol_filter)
        if error_msg is not None:
            frags_record = None
        else:
            frags = fragment_mol(
                mol_record.normalized_mol, self.mol_filter, mol_record.num_normalized_heavies
            )
            frags_record = {
                f"{frag.constant_smiles}.{frag.attachment_order}": frag for frag in frags
            }

        return frags_record


def skip_null(func: Callable[[T], U]) -> Callable[[Optional[T]], Optional[U]]:
    def wrapper(*args, **kwargs):
        if None in args or None in kwargs.values():
            return None
        else:
            res = func(*args, **kwargs)
            return res

    return wrapper


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
        self.df[col] = self.df[col].progress_apply(canonicalize_smiles)

    def cano_selfies(self, col: str) -> None:
        self.df[col] = self.df[col].progress_apply(selfies2smiles)

    def metric_mol(self, col: str) -> pd.Series:
        return self.df[col].progress_apply(self.get_mol)

    def metric_heavy(self, col: str) -> pd.Series:
        return self.df[col].astype("int32")

    def metric_sascore(self, col: str) -> pd.Series:
        return self.df[col].progress_apply(lambda s: self.get_mol(self.get_sascore(s)))

    def metric_qed(self, col: str) -> pd.Series:
        return self.df[col].progress_apply(lambda s: self.get_mol(self.get_qed(s)))

    def metric_weight(self, col: str) -> pd.Series:
        return self.df[col].progress_apply(lambda s: self.get_mol(self.get_weight(s)))

    def metric_scaffold(self, col: str) -> pd.Series:
        return self.df[col].progress_apply(lambda s: self.get_mol(self.get_scaffold(s)))

    def metric_frag_sim(self, col_a: str, col_b: str) -> pd.Series:
        return self.df.progress_apply(
            lambda df: self.get_similarity(self.get_mol(df[col_a]), self.get_mol(df[col_b])), axis=1
        )

    def metric_frag_prop(self, frag_col: str, core_col: str) -> pd.Series:
        return self.df[frag_col] / self.df[core_col]

    def visual(self, save_path: str, nrows: int = -1) -> None:
        visual_df = self.df[self.EXPECTED_COLS]

        if nrows != -1:
            visual_df = visual_df.iloc[:nrows]

        self.render_romol(visual_df)
        for col in self.SMILES_COLS:
            visual_df[f"{col}_romol"] = visual_df[col].progress_apply(self.get_mol)

        visual_df.to_html(save_path)


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


class ResultMetrics(Metrics):
    EXPECTED_COLS = [
        "src",
        "tgt",
        "core",
        "frag_a",
        "frag_b",
        "out",
        "gen_core",
        "gen_frag_a",
        "gen_frag_b",
        "gen_core_heavy",
        "gen_frag_a_heavy",
        "gen_frag_b_heavy",
    ]
    SMILES_COLS = [
        "src",
        "tgt",
        "core",
        "frag_a",
        "frag_b",
        "out",
        "gen_core",
        "gen_frag_a",
        "gen_frag_b",
    ]

    def __init__(
        self, df: pd.DataFrame, data_format: Literal["SMILES", "SELFIES"] = "SMILES", topk: int = 1
    ) -> None:
        assert set(df.columns.to_list()) == set(
            self.EXPECTED_COLS
        ), f"Expected DataFrame with columns {self.EXPECTED_COLS} but got {df.columns.to_list()}."
        assert df.shape[0] % topk == 0, f"DataFrame with shape {df.shape} cannot be divided by topk `{topk}`."
        self.df = df.reindex(columns=self.EXPECTED_COLS)
        self.render_romol(self.df)
        self.data_format = data_format
        self.topk = topk
        self.result_group = self.df.shape[0] // topk

    def concat_tokens(self) -> None:
        self.df["src"] = self.df["src"].progress_apply(lambda s: "".join(s.strip().split(" ")))
        self.df["out"] = self.df["out"].progress_apply(
            lambda s: "".join(s.strip("{eos}").strip().split(" "))
        )
        if self.data_format == "SELFIES":
            self.df["src"] = self.df["src"].progress_apply(selfies2smiles)
            self.df["out"] = self.df["out"].progress_apply(selfies2smiles)

    def visual(self, save_path: str, nrows: int = -1) -> None:
        self.concat_tokens()
        return super().visual(save_path, nrows)

    def basic_metric(self) -> None:
        # Validity, MMP Validity, Recovery, Uniqueness, Novelty
        self.concat_tokens()
        self.cano_smiles("out")
        size = self.df.shape[0]

        validity = len(self.df["out"].dropna()) / size
        mmp_validity = len(self.df["gen_core"].dropna()) / size
        strict_recovery = (self.df["out"] == self.df["tgt"]).sum() / size

        groups = self.df.groupby("src")
        recovery_dict = {group[0]: frozenset(group[1]["tgt"].to_list()) for group in groups}
        recovery_cnt = 0
        for i in range(self.result_group):
            unique_out = set(self.df["out"][i : i + self.topk].to_list())
            if unique_out & recovery_dict[self.df["src"][i]]:
                recovery_cnt += 1

        recovery = recovery_cnt / self.result_group

        core_recovery = (self.df["gen_core"] == self.df["core"]).sum() / size
        frag_a_recovery = (self.df["gen_frag_a"] == self.df["frag_a"]).sum() / size
        frag_b_recovery = (self.df["gen_frag_b"] == self.df["frag_b"]).sum() / size

        uniqueness = len(self.df["out"].dropna().drop_duplicates()) / size
        unique_tgt = set(self.df["tgt"].drop_duplicates().to_list())
        novel_out = self.df["out"].apply(lambda s: 1 if s in unique_tgt else 0)
        novelty = novel_out.sum() / size

        metric_infos = [
            ["Metrics", "Value"],
            ["Mol Validity", f"{validity:.4f}"],
            ["MMP Validity", f"{mmp_validity:.4f}"],
            ["Strict Recovery", f"{strict_recovery:.4f}"],
            ["Recovery", f"{recovery:.4f}"],
            ["Core Recovery", f"{core_recovery:.4f}"],
            ["Truncated Frag Recovery", f"{frag_a_recovery:.4f}"],
            ["Spliced Frag Recovery", f"{frag_b_recovery:.4f}"],
            ["Uniqueness", f"{uniqueness:.4f}"],
            ["Novelty", f"{novelty:.4f}"],
        ]
        table = AsciiTable(metric_infos)
        table.title = f"Basic Metrics (Total: {size})"
        table.inner_row_border = True
        print(table.table)
