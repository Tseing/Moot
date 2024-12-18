import os.path as osp
import sys
from typing import List, Optional, Tuple, Union

import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from tqdm import tqdm

sys.path.append("..")
from src.launcher import ModelLauncher
from src.model.optformer import OptFormer, Transformer
from src.tokenizer import AtomTokenizer, ProteinTokenizer, SmilesTokenizer, share_vocab
from src.typing import Device
from src.utils import Cfg, Log, pad_sequences
from src.utils.molecule import split_marked_mol


class CoreInferencer:
    def __init__(
        self,
        model: Union[OptFormer, Transformer],
        mol_tokenizer: SmilesTokenizer,
        atom_tokenizer: AtomTokenizer,
        protein_tokenizer: Optional[ProteinTokenizer] = None,
        device: Optional[Device] = None,
    ) -> None:
        model.eval()
        self.model = model
        self.mol_tokenizer = mol_tokenizer
        self.atom_tokenizer = atom_tokenizer
        self.protein_tokenizer = protein_tokenizer

        if device is None:
            device = torch.device(torch._C._get_default_device())
        self.device = device

        self.pad_value = mol_tokenizer.vocab2index[mol_tokenizer.pad]
        self.eos_value = mol_tokenizer.vocab2index[mol_tokenizer.eos]
        self.dummy_value = mol_tokenizer.vocab2index["[*]"]

    @staticmethod
    def get_atoms(smiles: str) -> str:
        mol = Chem.MolFromSmiles(Chem.CanonSmiles(smiles))
        return "".join([f"[{atom.GetSymbol()}]" for atom in mol.GetAtoms()])

    @staticmethod
    def get_core(smiles: str, dummy_idx: int) -> Optional[str]:
        try:
            mol = Chem.MolFromSmiles(Chem.CanonSmiles(smiles))
            mol.GetAtomWithIdx(dummy_idx).SetAtomicNum(0)
            frags = split_marked_mol(mol)
        except:
            return None

        if len(frags) == 0:
            return None

        if len(frags) > 1:
            frags.sort(key=lambda mol: mol.GetNumHeavyAtoms())

        return Chem.MolToSmiles(frags[0])

    def inference(self, smiles_list: List[str], proteins_list: Optional[List[str]]) -> torch.Tensor:
        mols = [self.mol_tokenizer.tokenize(smiles) for smiles in smiles_list]
        atoms = [self.atom_tokenizer.tokenize(self.get_atoms(smiles)) for smiles in smiles_list]

        padded_mols = (
            torch.Tensor(pad_sequences(mols, self.pad_value, left_pad=False)).to(self.device).int()
        )
        padded_atoms = (
            torch.Tensor(pad_sequences(atoms, self.pad_value, left_pad=False)).to(self.device).int()
        )

        mask = torch.logical_or(
            padded_atoms[:, 1:] == self.pad_value, padded_atoms[:, 1:] == self.eos_value
        )
        padded_atoms = padded_atoms[:, :-1]

        if proteins_list:
            proteins = [self.protein_tokenizer.tokenize(protein) for protein in proteins_list]
            padded_proteins = torch.Tensor(
                pad_sequences(proteins, self.pad_value, left_pad=False)
            ).to(self.device)
            data = (padded_mols, padded_atoms, padded_proteins)
        else:
            data = (padded_mols, padded_atoms)

        with torch.no_grad():
            return self._inference(data, mask)

    def _inference(self, data: Tuple[torch.Tensor, ...], mask: torch.Tensor) -> torch.Tensor:
        out, _ = model(*data)

        sorted_idx = torch.argsort(out, dim=2)
        order_matrix = torch.arange(0, out.shape[-1]).unsqueeze(0).unsqueeze(0).expand(out.shape)
        rank = order_matrix[sorted_idx == self.dummy_value].reshape(-1, out.shape[1])
        rank[mask] = 0
        return F.softmax(rank.float(), dim=-1)


if __name__ == "__main__":
    cfg = Cfg()
    cfg.parse()

    logger = Log("Inference", osp.join(cfg.LOG_DIR, f"{cfg.task_name}.log"))
    logger.info(f"Config:\n{repr(cfg)}")
    device = torch.device(cfg.device)

    atom_tokenizer = AtomTokenizer()
    mol_tokenizer = SmilesTokenizer()
    mol_tokenizer.load_word_table(osp.join(cfg.DATA_DIR, cfg.word_table_path))
    atom_tokenizer, mol_tokenizer = share_vocab(atom_tokenizer, mol_tokenizer)
    cfg.set("vocab_size", mol_tokenizer.vocab_size)
    cfg.set("pad_value", mol_tokenizer.vocab2index[mol_tokenizer.pad])

    launcher = ModelLauncher(cfg.model, cfg, logger, "inference", device)
    model = launcher.get_model()

    inferencer = CoreInferencer(model, mol_tokenizer, atom_tokenizer, device=device)

    chunks = pd.read_csv(
        osp.join(cfg.DATA_DIR, cfg.test_data_path), chunksize=cfg.batch_size, usecols=cfg.data_cols
    )

    if len(cfg.data_cols) == 1:

        def _get_data(chunk: pd.DataFrame) -> Tuple[List[str], None]:
            return chunk[cfg.data_cols[0]].to_list(), None

    elif len(cfg.data_cols) == 2:

        def _get_data(chunk: pd.DataFrame) -> Tuple[List[str], List[str]]:
            return chunk[cfg.data_cols[0]].to_list(), chunk[cfg.data_cols[1]].to_list()

    else:
        assert False

    results = {"input": [], "atom_idx": [], "core": []}
    for chunk in tqdm(chunks):
        smiles_list, protein_list = _get_data(chunk)
        prob = inferencer.inference(smiles_list, protein_list)
        idxes = prob.argmax(dim=-1).tolist()

        for i, dummy_idx in enumerate(idxes):
            results["core"].append(inferencer.get_core(smiles_list[i], int(dummy_idx)))

        results["input"].extend(smiles_list)
        results["atom_idx"].extend(idxes)

    pd.DataFrame(results).to_csv(osp.join(cfg.OUTPUT_DIR, cfg.save_path))

