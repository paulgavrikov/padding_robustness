import torch
from glob import glob
import logging


def resolve_path(
    attack,
    norm,
    budget,
    dataset="cifar10",
    model="openlth_resnet_20_16",
    padding_mode="zeros",
    kernel_size=3,
    seed=0,
    at=False,
):
    pad_name = f"padding{padding_mode}"
    if padding_mode == "nopad":
        pad_name = "nopad"
    seed_suffix = f"_{seed}"
    sub_dir = "checkpoints_robustness_vs_padding"
    if at:
        sub_dir = "checkpoints_padding_vs_robustness_at"
    path = f"/workspace/ssd1/{sub_dir}/*_{dataset}_{model}_{pad_name}_k{kernel_size}{seed_suffix}/old_adv_examples_{attack.split('-')[0].lower()}_{norm.lower()}_{budget.lower()}.pt"
    resolved = glob(path)

    if len(resolved) > 1:
        logging.warning("Resolved multiple runs for given query " + str(list(resolved)))

    return resolved[0]

def get_attack_state(
    attack,
    norm,
    budget,
    dataset="cifar10",
    model="openlth_resnet_20_16",
    padding_mode="zeros",
    kernel_size=3,
    seed=0,
    at=False,
):
    path = resolve_path(
        attack,
        norm,
        budget,
        dataset=dataset,
        model=model,
        kernel_size=kernel_size,
        padding_mode=padding_mode,
        seed=seed,
        at=at,
    )
    state = torch.load(path, map_location="cpu")
    return state