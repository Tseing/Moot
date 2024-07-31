import re
from typing import Any, Callable, Dict, List

import matplotlib.pyplot as plt

INFOS: Dict[str, list] = {
    "loss_smiles_transformer_18M.png": [
        {
            "name": "medium lr: [1.0e-6, 1.0e-4]",
            "path": "../../log/basic_transformer_smiles.log",
        },
        {
            "name": "large lr: [1.0e-6, 5.0e-4]",
            "path": "../../log/basic_transformer_smiles_18M.log",
        },
    ],
    "loss_smiles_pretrain_100k.png": [
        {
            {
                "name": "pretrain_100k",
                "path": "../../log/pretrain_100k_smiles.log",
            },
        },
    ],
}


def find_values(keyword: str, content: str, fn: Callable[[str], Any]) -> List[Any]:
    pattern = f"{keyword}: \d+(\.\d+)?"
    values_str = re.finditer(pattern, content)
    return [fn(item.group().strip(f"{keyword}: ")) for item in values_str]


if __name__ == "__main__":
    file_name = "loss_smiles_transformer_18M.png"
    logs = INFOS[file_name]

    values = []
    step_per_epoch = None
    max_len = 0

    for log in logs:
        train_content = ""
        val_content = ""
        with open(log["path"], "r") as f:
            lines = f.readlines()
        for line in lines:
            if "Train" in line:
                train_content = "".join([train_content, line])
            elif "Average Val" in line:
                val_content = "".join([val_content, line])

        val_losses = find_values("Average Val Loss", val_content, float)
        losses = find_values("Train Loss", train_content, float)
        epochs = find_values("Epoch", train_content, int)
        assert len(losses) == len(epochs)
        values.append({"label": log["name"], "loss": losses, "val loss": val_losses})

        if len(losses) > max_len:
            max_len = len(losses)

        if step_per_epoch is None and len(set(epochs)) > 1:
            step_per_epoch = sum([True if epoch == 0 else False for epoch in epochs])

    assert step_per_epoch is not None
    epoch_ticks = list(range(0, max_len + step_per_epoch, step_per_epoch))
    epoch_labels = [str(i) for i in range(len(epoch_ticks))]

    if len(epoch_ticks) > 11:
        tick_interval = len(epoch_ticks) // 10
    else:
        tick_interval = 1

    show_epoch_ticks = epoch_ticks[::tick_interval]
    show_epoch_labels = epoch_labels[::tick_interval]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    ax1.set_xticks(show_epoch_ticks, show_epoch_labels)
    ax1.set_xlim(
        -tick_interval * step_per_epoch // 5, max_len + tick_interval * step_per_epoch // 5
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax2.set_ylabel("Average Val Loss")

    viridis = plt.get_cmap("Set2")

    for i, value in enumerate(values):
        x = range(len(value["loss"]))
        ax1.plot(x, value["loss"], label=value["label"], c=viridis(i), alpha=0.75)
        ax2.plot(
            epoch_ticks[1 : 1 + len(value["val loss"])],
            value["val loss"],
            linestyle=":",
            c=viridis(i),
            marker=".",
            markerfacecolor="white",
        )

    ax1.legend(loc="upper right")
    plt.savefig(f"output/{file_name}", dpi=900)
