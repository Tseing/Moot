import sys

# from time import sleep
from typing import List, Literal, Optional, Tuple

sys.path.append("..")

import torch

from config import MODEL_CONFIG
from src.data_utils import canonicalize_smiles
from src.inferencer import Inferencer
from src.launcher import ModelLauncher
from src.tokenizer import (
    FragSmilesTokenizer,
    ProteinTokenizer,
    SmilesTokenizer,
    share_vocab,
)
from src.utils import Log

logger = Log("web", "web.log")
PROTEIN_TOKENS = {
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
}


class ModelHost:
    end_inferencer = None
    step_inferencer = None
    prot_tokenizaer = None
    mol_tokenizer = None
    frag_tokenizer = None

    def __init__(
        self,
        setup: Literal["all", "end", "step"] = "all",
        device: Literal["cuda", "cpu"] = "cpu",
    ):
        device = torch.device(device)
        self.device = device
        if setup == "end":
            self.end_inferencer, self.prot_tokenizaer, self.mol_tokenizer = self.init_inferencer(
                MODEL_CONFIG["end-to-end"], "mol", device
            )
        elif setup == "step":
            self.step_inferencer, self.prot_tokenizaer, self.frag_tokenizer = self.init_inferencer(
                MODEL_CONFIG["step-by-step"], "frag", device
            )
        elif setup == "all":
            self.end_inferencer, _, self.mol_tokenizer = self.init_inferencer(
                MODEL_CONFIG["end-to-end"], "mol", device
            )
            self.step_inferencer, self.prot_tokenizaer, self.frag_tokenizer = self.init_inferencer(
                MODEL_CONFIG["step-by-step"], "frag", device
            )
        else:
            assert False, f"Unsupported setup attribute: {setup}."

    @staticmethod
    def init_inferencer(cfg, mol_or_frag: Literal["mol", "frag"], device: torch.DeviceObjType):
        if mol_or_frag == "mol":
            mol_tokenizer = SmilesTokenizer()
        elif mol_or_frag == "frag":
            mol_tokenizer = FragSmilesTokenizer()
        else:
            assert False, "Attribute `mol_or_frag` should be either 'mol' or 'frag'."

        mol_tokenizer.load_word_table(cfg.word_table_path)
        prot_tokenizer = ProteinTokenizer()
        prot_tokenizer, mol_tokenizer = share_vocab(prot_tokenizer, mol_tokenizer)
        cfg.set("vocab_size", mol_tokenizer.vocab_size)
        cfg.set("pad_value", mol_tokenizer.vocab2index[mol_tokenizer.pad])

        launcher = ModelLauncher("Optformer", cfg, logger, "inference", device)
        model = launcher.get_model()

        inferencer = Inferencer(
            [model],
            None,
            tokenizer=mol_tokenizer,
            max_len=cfg.infer_max_len,
            n_best=None,
            beam_size=cfg.beam_size,
            min_len=cfg.infer_min_len,
            stop_early=cfg.stop_early,
            normalize_scores=cfg.normalize_scores,
            len_penalty=cfg.len_penalty,
            unk_penalty=cfg.unk_penalty,
            sampling=cfg.sampling,
            sampling_topk=cfg.sampling_topk,
            sampling_temperature=cfg.sampling_temperature,
            device=device,
        )

        return inferencer, prot_tokenizer, mol_tokenizer

    def prepare_end_data(self, smiles: str, sequence: str):
        mol = torch.tensor(
            [self.mol_tokenizer.tokenize(smiles)], dtype=torch.int32, device=self.device
        )
        seq = torch.tensor(
            [self.prot_tokenizaer.tokenize("".join([f"-{token}" for token in sequence]))],
            dtype=torch.int32,
            device=self.device,
        )
        tgt = torch.tensor([[0]], dtype=torch.int32, device=self.device)

        return (mol, seq), tgt

    def prepare_step_data(self, core: str, frag: str, sequence: str):
        smiles = "|".join([core, frag])
        mol = torch.tensor(
            [self.frag_tokenizer.tokenize(smiles)], dtype=torch.int32, device=self.device
        )
        seq = torch.tensor(
            [self.prot_tokenizaer.tokenize("".join([f"-{token}" for token in sequence]))],
            dtype=torch.int32,
            device=self.device,
        )
        tgt = torch.tensor([[0]], dtype=torch.int32, device=self.device)

        return (mol, seq), tgt

    def end_predict(self, smiles: str, sequence: str, nbest: int):
        if self.end_inferencer is None:
            print(
                "End-to-end optimization model is not running.",
                "Please try to use `ModelHost(setup='end')` or `ModelHost(setup='all')`.",
            )
            return (
                False,
                "End-to-end Optimization model is not running.",
            )
        batch = self.prepare_end_data(smiles, sequence)
        return self.end_inferencer.interactive(batch, nbest)

    def step_predict(self, core: str, frag: str, sequence: str, nbest: int):
        if self.step_inferencer is None:
            print(
                "Step-by-step optimization model is not running.",
                "Please try to use `ModelHost(setup='step')` or `ModelHost(setup='all')`.",
            )
            return (
                False,
                "Step-by-step optimization model is not running.",
            )

        batch = self.prepare_step_data(core, frag, sequence)
        return self.step_inferencer.interactive(batch, nbest)


def validate_prediction_form(form: dict) -> Tuple[bool, str]:
    smiles_list = form.get("smiles", None)
    if smiles_list is None or len(smiles_list) == 0:
        return (False, "Missing `smiles` field.")

    sequence = form.get("sequence", None)
    if sequence is None or len(sequence) == 0:
        return (False, "Missing `sequence` field.")

    mode = form.get("mode", None)
    if mode is None:
        return (False, "Missing `mode` field.")

    if mode not in ("end-to-end", "step-by-step"):
        return (
            False,
            f"`mode` only support 'end-to-end' or 'step-by-step' but got {mode}.",
        )

    beam_size = form.get("beam_size", None)
    if beam_size is None:
        return (False, "Missing `beam_size` field.")

    if beam_size not in (10, 50, 100):
        return (False, f"`beam_size` only support 10, 50 or 100 but got {beam_size}.")

    if len(smiles_list) != 1 and mode == "end-to-end":
        return (False, "Only support 1 smiles in end-to-end mode.")

    if len(smiles_list) != 2 and mode == "step-by-step":
        return (False, "Only support 2 smiles in step-by-step mode")

    return clean_data(form)


def clean_data(form: dict) -> Tuple[bool, str]:
    print(form)
    seq = form["sequence"]

    if len(seq) > 1495:
        return False, f"Protein sequence too long."

    if not set(seq).issubset(PROTEIN_TOKENS):
        return False, f"Cannot parse sequence: '{seq}'."

    # TODO: prevent * and . tokens
    if form["mode"] == "end-to-end":
        smiles = form["smiles"][0]
        if len(smiles) > 245:
            return False, "SMILES too long."
        cano_smiles = canonicalize_smiles(smiles)
        if cano_smiles is None:
            return False, f"Cannot parse SMILES: '{smiles}'."

        form["_cleaned_data"] = {"smiles": cano_smiles}
    else:
        core, frag = form["smiles"]
        if len(core) + len(frag) > 240:
            return False, "SMILES too long."
        cano_core = canonicalize_smiles(core)
        if cano_core is None:
            return False, f"Cannot parse core structure SMILES: '{core}'."
        cano_frag = canonicalize_smiles(frag)
        if cano_frag is None:
            return False, f"Cannot parse truncated fragment SMILES: '{frag}'."

        form["_cleaned_data"] = {"core": cano_core, "frag": cano_frag}

    form["_cleaned_data"]["sequence"] = seq
    form["_cleaned_data"]["mode"] = form["mode"]
    form["_cleaned_data"]["beam_size"] = form["beam_size"]
    print(form["_cleaned_data"])
    return True, "Validate and cleaned form data."


def post_process(output: List[str]) -> List[str]:
    print(output)
    results = ["".join(res.split()).strip("{eos}") for res in output]
    cano_results = list(
        filter(lambda obj: obj is not None, [canonicalize_smiles(res) for res in results])
    )

    return cano_results


def get_result(cleaned_data: dict, model_host: ModelHost) -> Tuple[bool, Optional[List[str]]]:

    if cleaned_data["mode"] == "end-to-end":
        output = model_host.end_predict(
            cleaned_data["smiles"], cleaned_data["sequence"], cleaned_data["beam_size"]
        )
    else:
        output = model_host.step_predict(
            cleaned_data["core"],
            cleaned_data["frag"],
            cleaned_data["sequence"],
            cleaned_data["beam_size"],
        )

    # # return False, None
    # sleep(1)
    # result = MOCK_RESULT[:beam_size]

    return True, post_process(output)
