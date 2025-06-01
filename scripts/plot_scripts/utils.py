import os.path as osp

import pandas as pd


class DataReader:
    @staticmethod
    def _repeat_df(df: pd.DataFrame, compare_df: pd.DataFrame, n: int):
        assert (
            df.shape[0] * n == compare_df.shape[0]
        ), f"Expected DataFrame rows: {compare_df.shape[0]}, but got {df.shape[0]}."
        if n != 1:
            df = df.loc[df.index.repeat(n)].set_index(compare_df.index)

        return df

    @staticmethod
    def prepare_mol_df(input_path: str, output_path: str, topk: int) -> pd.DataFrame:
        gen_df = pd.read_csv(output_path)
        gen_df.columns = ["src", "out"]

        inp_df = pd.read_csv(
            input_path,
            usecols=["target", "mol_a_smiles", "mol_b_smiles", "frag_a", "frag_b", "core"],
        )
        inp_df = DataReader._repeat_df(inp_df, gen_df, topk)

        mmp_path = f"{osp.splitext(output_path)[0]}_mmp.csv"
        mmp_df = pd.read_csv(mmp_path)

        assert gen_df.shape[0] == mmp_df.shape[0]
        df = pd.concat([inp_df, gen_df, mmp_df], axis=1)
        return df

    @staticmethod
    def prepare_frag_df(input_path: str, output_path: str, topk: int) -> pd.DataFrame:
        gen_df = pd.read_csv(output_path)
        gen_df.columns = ["src", "out"]

        frag_cols = ["target", "frag_a", "frag_b", "core"]
        inp_df = pd.read_csv(input_path, usecols=frag_cols)
        inp_df = DataReader._repeat_df(inp_df, gen_df, topk)

        df = pd.concat([inp_df, gen_df], axis=1)
        return df
