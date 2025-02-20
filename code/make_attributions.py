import sys
sys.path.append("tinytorchtrainer")
import argparse
import torch
import data
from train import Trainer
import os
from utils import NormalizedModel
from glob import glob 
from pytorch_grad_cam import *
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm 
import itertools
from attack_helper import get_attack_state, resolve_path


def gradient_attribution(model, x):
    x.requires_grad = True

    logits = model(x)
    score, _ = torch.max(logits, 1)
    score.backward(torch.ones(len(logits)).to(x.device))

    slc = torch.abs(x.grad).max(dim=1).values
    
    #slc = (slc - slc.min())/(slc.max()-slc.min())

    return slc

def get_attributions(
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
    
    path = Path(path)
    
    CKPT_PATH = glob(os.path.join(path.parent, "checkpoints/*.ckpt"))[0]

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    saved_args = argparse.Namespace()

    for k, v in ckpt["args"].items():
        vars(saved_args)[k] = v

    vars(saved_args)["load_checkpoint"] = CKPT_PATH
    vars(saved_args)["device"] = "cuda"

    dataset = data.get_dataset(saved_args.dataset)(os.path.join(
            saved_args.dataset_dir, saved_args.dataset))

    vars(saved_args)["model_in_channels"] = dataset.in_channels
    vars(saved_args)["model_num_classes"] = dataset.num_classes

    trainer = Trainer(saved_args)
    model = trainer.model
    model.eval()

    all_x = []
    all_y = []
    loader = dataset.val_dataloader(1000, 0, shuffle=False)
    for x, y in loader:
        all_x.append(x.to(trainer.device))
        all_y.append(y.to(trainer.device))
        break

    all_x = torch.vstack(all_x)
    all_y = torch.hstack(all_y)

    model = NormalizedModel(trainer.model, dataset.mean, dataset.std).to(trainer.device)
    model.eval()

    all_x = all_x * model.std + model.mean  # unnormalize samples for AA

    ##

    state = torch.load(path, map_location="cpu")

    clean_images = all_x[state["flipped_ids"]].to(trainer.device)
    adv_images = clean_images + torch.tensor(state["x_adv_succ"]).to(trainer.device)

    ##
    
    cam = gradient_attribution

    adv_cam = cam(model, adv_images).detach().cpu().numpy()
    norm_cam = cam(model, clean_images).detach().cpu().numpy()
    
    # target_layers = [model.model.blocks[8]]
    # targets = None
    # cam = LayerCAM(model=model, target_layers=target_layers, use_cuda=True)
    # adv_cam = cam(input_tensor=adv_images, targets=None)
    # norm_cam = cam(input_tensor=clean_images, targets=None)
#     diff = (adv_cam - norm_cam).mean(axis=0).detach().cpu().numpy()
    return adv_cam, norm_cam


def main():
    rows = []
    for dataset, model, kernel_size, padding_mode, attack, norm, budget, at in tqdm(itertools.product(
        ["cifar10"],
        ["openlth_resnet_20_16"],
        [3],
        ["zeros", "circular", "replicate", "reflect"],
        ["apgd-ce", "fab", "square"],
        ["Linf", "L2"],
        ["low", "high"],
        [False, True],
    )):
        diffs = []
        for seed in range(10):
            diff = get_heatmap_diff(
                attack=attack,
                norm=norm,
                budget=budget,
                dataset=dataset,
                model=model,
                padding_mode=padding_mode,
                kernel_size=kernel_size,
                seed=seed,
                at=at,
            )
            diffs.append(diff)
        diffs = np.stack(diffs).mean(axis=0)

        row = dict(
            attack=attack,
            norm=norm,
            budget=budget,
            dataset=dataset,
            model=model,
            padding_mode=padding_mode,
            kernel_size=kernel_size,
            at=at,
            hm_diff=diffs 
        )
        rows.append(row)

    df_info = pd.DataFrame(rows)
    df_info.to_hdf("cifar10_resnet2016_gradients.h5", "gradients")


if __name__ == "__main__":
    main()
