import os.path as osp
import pickle
from collections import defaultdict
from typing import Tuple

import rdkit
import rdkit.Chem
import torch
from setup import init_model
from torch import nn
from tqdm import tqdm


def get_enc_attn(model: nn.Module, inps: Tuple[torch.Tensor]):
    with torch.no_grad():
        return model.enc_attn(*inps)


def get_dec_attn(model: nn.Module, inps: Tuple[torch.Tensor], dec_inp: torch.Tensor):
    # token_attns = []
    attn_dict = defaultdict(list)
    length = dec_inp.shape[1]
    with torch.no_grad():
        if isinstance(inps, tuple):
            for i in tqdm(range(1, length + 1), total=length):
                _, attn = model(*inps, dec_inp[:, :i])

                for k, v in attn.items():
                    attn_dict[k].append(v[:, -1:, :])

        else:
            for i in tqdm(range(1, length + 1), total=length):
                _, attn = model(inps, dec_inp[:, :i])

                for k, v in attn.items():
                    attn_dict[k].append(v[:, -1:, :])

    for k in attn_dict:
        attn_dict[k] = torch.concat(attn_dict[k], dim=1)

    return attn_dict


def save_cpi_attn(model_label: str, mol: str, prot: str, save_path: str) -> None:
    model, mol_tokenizer, prot_tokenizer = init_model(model_label)
    mol_tokens = mol_tokenizer.tokenize(mol)
    prot_tokens = prot_tokenizer.tokenize("".join([f"-{token}" for token in prot]))
    mol_inp = torch.Tensor([mol_tokens]).int()
    prot_inp = torch.Tensor([prot_tokens]).int()

    print("Input:", mol_inp.shape, prot_inp.shape)
    x_a, x_b, attn_a, attn_b = get_enc_attn(model, (mol_inp, prot_inp))
    print("Output:", x_a.shape, x_b.shape, attn_a.shape, attn_b.shape)
    print(f"Mol Tokens: {mol_tokenizer.convert_ids2tokens(mol_tokens).tolist()}")
    pickle.dump((attn_a.numpy(), attn_b.numpy()), open(save_path, "wb"))


if __name__ == "__main__":

    # seq = "CC(C)Cc1nnc(NC(=O)CCC(=O)N2CCN(Cc3ccc(F)cc3)CC2)s1"
    # gen = "O = C ( C C C ( = O ) N 1 C C N ( C c 2 c c c ( F ) c c 2 ) C C 1 ) N c 1 c c c 2 c ( c 1 ) O C O 2"

    # enc_inp = torch.Tensor([mol_tokenizer.tokenize(seq)]).int()
    # dec_inp = torch.Tensor(
    #     [mol_tokenizer.covert_tokens2ids([mol_tokenizer.bos] + gen.split())]
    # ).int()

    # with torch.no_grad():
    #     _, attn = model(enc_inp, dec_inp)

    # print(enc_inp.shape)
    # print(dec_inp.shape)
    # print(attn.shape)
    model = "probe_optformer_smiles"
    mol = "Cc1ccc(C(=O)Nc2cc(C(=O)O)sc2Oc2ccc(C)cn2)cc1"
    mol = rdkit.Chem.CanonSmiles(mol)
    prot = "MQAVDNLTSAPGNTSLCTRDYKITQVLFPLLYTVLFFVGLITNGLAMRIFFQIRSKSNFIIFLKNTVISDLLMILTFPFKILSDAKLGTGPLRTFVCQVTSVIFYFTMYISISFLGLITIDRYQKTTRPFKTSNPKNLLGAKILSVVIWAFMFLLSLPNMILTNRQPRDKNVKKCSFLKSEFGLVWHEIVNYICQVIFWINFLIVIVCYTLITKELYRSYVRTRGVGKVPRKKVNVKVFIIIAVFFICFVPFHFARIPYTLSQTRDVFDCTAENTLFYVKESTLWLTSLNACLDPFIYFFLCKSFRNSLISMLKCPNSATSLSQDNRKKEQDGGDPNEETPM"
    save_cpi_attn(model, mol, prot, "./2zpy-40.pkl")

    # model, mol_tokenizer = init_transformer("probe_transformer_smiles")
    # mol = "N # C c 1 c ( N C ( = O ) C 2 C C 2 ) s c 2 c 1 C C C C C 2"
    # result = "N # C c 1 c ( N C ( = O ) c 2 c c c o 2 ) s c 2 c 1 C C C C C 2"

    # mol_tokens = " ".join([mol_tokenizer.bos, mol, mol_tokenizer.eos]).split()
    # result_tokens = " ".join([mol_tokenizer.bos, mol, mol_tokenizer.eos]).split()

    # mol_inp = torch.Tensor([mol_tokenizer.covert_tokens2ids(mol_tokens)]).int()
    # dec_inp = torch.Tensor([mol_tokenizer.covert_tokens2ids(result_tokens)]).int()

    # attn = get_dec_attn(model, mol_inp, dec_inp)
    # print(attn)
