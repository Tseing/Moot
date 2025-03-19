class Config:

    def __init__(self, config_dict: dict):
        self.__config_dict = config_dict

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError("Cannot get private attributes of `Config`.")

        try:
            value = self.__config_dict[name]
        except KeyError:
            raise AttributeError(f"No attribute '{name}' in Config instance.")

        return value

    def set(self, key: str, value) -> None:
        if key in self.__config_dict:
            raise ValueError("Cannot set config value for an existed key.")
        self.__config_dict[key] = value


__END_TO_END_CONFIG = {
    "CKPT_DIR": "../checkpoints/platform",
    "ckpt_path": "end_to_end.pt",
    "word_table_path": "../data/all/smiles_word_table.yaml",
    # inference config
    "beam_size": 128,
    "infer_max_len": 250,
    "infer_min_len": 1,
    "infer_left_pad": False,
    "stop_early": False,
    "normalize_scores": True,
    "len_penalty": 1,
    "unk_penalty": 0,
    "sampling": False,
    "sampling_topk": -1,
    "sampling_temperature": 1,
    # model config
    "mol_max_len": 250,
    "prot_max_len": 1500,
    "d_model": 512,
    "n_head": 4,
    "enc_n_layer": 2,
    "dec_n_layer": 4,
    "d_enc_ffn": 1024,
    "d_dec_ffn": 1024,
    "d_fuse_ffn": 1024,
    "enc_dropout": 0.2,
    "dec_dropout": 0.2,
    "enc_embed_dropout": 0.15,
    "dec_embed_dropout": 0.15,
    "enc_relu_dropout": 0.1,
    "dec_relu_dropout": 0.1,
    "enc_attn_dropout": 0.15,
    "dec_attn_dropout": 0.15,
    "left_pad": False,
    "seed": 42,
}
__STEP_BY_STEP_CONFIG = {
    "CKPT_DIR": "../checkpoints/platform",
    "ckpt_path": "step_by_step.pt",
    "word_table_path": "../data/frag/smiles_word_table.yaml",
    # inference config
    "beam_size": 128,
    "infer_max_len": 100,
    "infer_min_len": 1,
    "infer_left_pad": False,
    "stop_early": False,
    "normalize_scores": True,
    "len_penalty": 1,
    "unk_penalty": 0,
    "sampling": False,
    "sampling_topk": -1,
    "sampling_temperature": 1,
    # model config
    "mol_max_len": 250,
    "prot_max_len": 1500,
    "d_model": 512,
    "n_head": 4,
    "enc_n_layer": 2,
    "dec_n_layer": 4,
    "d_enc_ffn": 1024,
    "d_dec_ffn": 1024,
    "d_fuse_ffn": 1024,
    "enc_dropout": 0.2,
    "dec_dropout": 0.2,
    "enc_embed_dropout": 0.15,
    "dec_embed_dropout": 0.15,
    "enc_relu_dropout": 0.1,
    "dec_relu_dropout": 0.1,
    "enc_attn_dropout": 0.15,
    "dec_attn_dropout": 0.15,
    "left_pad": False,
    "seed": 42,
}

MODEL_CONFIG = {"end-to-end": Config(__END_TO_END_CONFIG), "step-by-step": Config(__STEP_BY_STEP_CONFIG)}
