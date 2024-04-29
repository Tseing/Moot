import copy
from typing import Optional

import pandas as pd
from pandarallel import pandarallel


def remove_pair_duplicates(
    dataframe: pd.DataFrame, return_df: bool = False
) -> Optional[pd.DataFrame]:
    df = copy.deepcopy(dataframe)
    df.columns = ["col1", "col2"]
    print(f"Original size: {df.shape}")
    df = df.drop_duplicates(ignore_index=True)
    print(f"Dropped size: {df.shape}")

    matched_pair = df.parallel_apply(
        lambda df: " ".join([df.col1, df.col2]), axis=1
    ).tolist()
    matched_pair_set = set(matched_pair)

    assert len(matched_pair_set) == len(
        matched_pair_set
    ), f"Unmatched shape between '{len(matched_pair_set)}' and '{len(matched_pair_set)}'."

    drop_index = []
    for pair in matched_pair:
        l, r = pair.split(" ")
        reverted_pair = " ".join([r, l])
        if reverted_pair in matched_pair_set:
            matched_pair_set.remove(reverted_pair)
            drop_index.append(matched_pair.index(reverted_pair))

    print(f"All pairs num: {len(matched_pair_set)}")
    print(f"Removed pairs num: {len(drop_index)}")

    if return_df:
        return df.drop(drop_index)
    else:
        return None


if __name__ == "__main__":
    pandarallel.initialize(nb_workers=20)
    path = "../../data/1bond/mmp_pair_1bond_raw.csv"
    df = pd.read_csv(path)
    df = remove_pair_duplicates(df, return_df=True)
    print(df.head)
