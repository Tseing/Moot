import pandas as pd


def merge(data1: str, data2: str, save_path: str):
    df1 = pd.read_csv(data1)
    df2 = pd.read_csv(data2, usecols=("core_selfies", "frag_a_selfies", "frag_b_selfies"))
    df = pd.concat([df1, df2], axis=1)

    print(df.columns)
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    merge("../../data/finetune/runtime/datasets_seed_0-only-frag/finetune_val.csv",
          "../../data/frag/runtime-only-frag/frag_val.csv",
          "../../data/frag/runtime/frag_val.csv")
