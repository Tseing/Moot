import pandas as pd


def fetch_unique_id(df_path: str, save_path: str) -> None:
    df = pd.read_csv(df_path)

    all_chembl_id = pd.concat([df["mol_a"], df["mol_b"]], axis=0)
    all_chembl_id.name = "chembl_id"
    all_chembl_id = all_chembl_id.dropna().drop_duplicates()
    print(f"Unique items: {all_chembl_id.shape[0]}")
    all_chembl_id.to_csv(save_path, index=False)


if __name__ == "__main__":
    fetch_unique_id(
        "../../data/1bond/mmp_pair_1bond_raw.csv", "../../data/all/all_unique_mol_id.csv"
    )
