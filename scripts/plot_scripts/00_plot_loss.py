import re
from typing import Any, Callable, Dict, List

import matplotlib.pyplot as plt

import config as cfg

INFOS: Dict[str, str] = {
    "loss_transformer_smile_wpretrains": "../../log/train_transformer_smiles_wpretrain.log",
    "loss_transformer_selfies_wpretrain": "../../log/train_transformer_selfies_wpretrain.log",
    "loss_transformer_smiles": "../../log/train_transformer_smiles.log",
    "loss_transformer_selfies": "../../log/train_transformer_selfies.log",
    "loss_optformer_smiles": "../../log/train_optformer_smiles.log",
    "loss_optformer_selfies": "../../log/train_optformer_selfies.log",
    "loss_frag_transformer_smiles": "../../log/train_frag_transformer_smiles.log",
    "loss_frag_transformer_selfies": "../../log/train_frag_transformer_selfies.log",
    "loss_frag_optformer_smiles": "../../log/train_frag_optformer_smiles.log",
    "loss_frag_optformer_selfies": "../../log/train_frag_optformer_selfies.log",
    "loss_core_transformer": "../../log/train_core_transformer.log",
    "loss_core_optformer": "../../log/train_core_optformer.log",
    "loss_optformer_smiles_cpi": "../../log/train_optformer_smiles_cpi.log",
    "loss_optformer_selfies_cpi": "../../log/train_optformer_selfies_cpi.log",
    "loss_transformer_smiles_cpi": "../../log/train_transformer_smiles_cpi.log",
    "loss_transformer_selfies_cpi": "../../log/train_transformer_selfies_cpi.log"
}


def find_values(keyword: str, content: str, fn: Callable[[str], Any]) -> List[Any]:
    pattern = f"{keyword}: \d+(\.\d+)?"
    values_str = re.finditer(pattern, content)
    return [fn(item.group().strip(f"{keyword}: ")) for item in values_str]


if __name__ == "__main__":
    file_name = "loss_optformer_selfies_cpi"
    path = INFOS[file_name]

    step_per_epoch = None
    max_len = 0

    train_content = ""
    val_content = ""
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines:
        if "Train" in line:
            train_content = "".join([train_content, line])
        elif "Average Val" in line:
            val_content = "".join([val_content, line])

    val_losses = find_values("AUC-ROC", val_content, float)
    losses = find_values("Train Loss", train_content, float)
    epochs = find_values("Epoch", train_content, int)
    assert len(losses) == len(epochs), f"unmatched: {len(losses)} and {len(epochs)}"

    if len(losses) > max_len:
        max_len = len(losses)

    if step_per_epoch is None and len(set(epochs)) > 1:
        step_per_epoch = sum([True if epoch == 0 else False for epoch in epochs])

    assert step_per_epoch is not None
    epoch_ticks = list(range(0, max_len + step_per_epoch, step_per_epoch))
    epoch_labels = [str(i) for i in range(len(epoch_ticks))]

    tick_interval = 10
    show_epoch_ticks = epoch_ticks[::tick_interval]
    show_epoch_labels = epoch_labels[::tick_interval]

    # fig, ax1 = plt.subplots()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.set_xticks(show_epoch_ticks, show_epoch_labels)
    ax1.set_xlim(
        -tick_interval * step_per_epoch // 5, max_len + tick_interval * step_per_epoch // 5
    )
    ax2.set_ylim(0.82, 0.97)
    ax1.tick_params(labelsize=14)
    ax2.tick_params(labelsize=14)
    ax1.set_xlabel("Epoch", fontsize=18)
    ax1.set_ylabel("Training Loss", fontsize=18)
    ax2.set_ylabel("AUC-ROC", fontsize=18)

    x = range(len(losses))
    plot1 = ax1.plot(x, losses, label="Training Loss", c=cfg._c["b."])
    plot2 = ax2.plot(
        epoch_ticks[1 : 1 + len(val_losses)],
        val_losses,
        linestyle=":",
        c=cfg._c["r."],
        marker=".",
        markerfacecolor=cfg._c["y"],
        markeredgecolor=cfg._c["y"],
        label="Validation AUC-ROC",
    )

    lns = plot1 + plot2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper right", fontsize=14)
    fig.tight_layout()
    plt.savefig(f"../../output/{file_name}.svg")
