from ctypes import LibraryLoader
import torch
import sys
sys.path.append("tinytorchtrainer")
import argparse
from train import Trainer
from autoattack import AutoAttack
from utils import NormalizedModel
import os
import data
from utils import none2str
import wandb
import numpy as np
from pathlib import Path


def parse_aa_log(log_file):
    results = {}
    prev_attack = ""
    with open(log_file, "r") as file:
        for line in file.readlines():
            if "accuracy" in line:
                acc = float(line.split(": ")[1].replace("%", "").strip().split(" ")[0]) / 100

                tag = None
                if "initial accuracy" in line:
                    tag = "clean"
                elif "after" in line:
                    tag = line.split(":")[0].split(" ")[-1].strip()
                    if len(prev_attack) == 0:
                        prev_attack = tag
                    else:
                        prev_attack += "+" + tag

                    tag = "AA-" + prev_attack
                else:
                    tag = "AA-robust"

                results[tag] = acc

    return results


def main(args):

    ckpt = torch.load(args.load_checkpoint, map_location="cpu")
    saved_args = argparse.Namespace()

    for k, v in ckpt["args"].items():
        vars(saved_args)[k] = v

    vars(saved_args)["load_checkpoint"] = args.load_checkpoint
    vars(saved_args)["device"] = args.device

    loader_batch = args.n_samples
    if args.n_samples == -1:
        loader_batch = saved_args.batch_size

    dataset = data.get_dataset(saved_args.dataset)(os.path.join(
            saved_args.dataset_dir, saved_args.dataset.split("@")[0]))

    vars(saved_args)["model_in_channels"] = dataset.in_channels
    vars(saved_args)["model_num_classes"] = dataset.num_classes

    trainer = Trainer(saved_args)

    all_x = []
    all_y = []
    loader = dataset.val_dataloader(loader_batch, saved_args.num_workers, shuffle=False)

    for x, y in loader:
        all_x.append(x.to(trainer.device))
        all_y.append(y.to(trainer.device))

        if args.n_samples != -1:
            break

    all_x = torch.vstack(all_x)
    all_y = torch.hstack(all_y)

    print(len(all_y))

    model = NormalizedModel(trainer.model, dataset.mean, dataset.std).to(trainer.device)
    model.eval()
    
    all_x = all_x * model.std + model.mean  # unnormalize samples for AA
    
    log_path = os.path.join(Path(args.load_checkpoint).parent, f"aa_{args.norm}_{args.budget}.txt")
    
    if args.budget == "low" and args.norm == "Linf":
        eps = 1/255.
    elif args.budget == "high" and args.norm == "Linf":
        eps = 8/255.
    elif args.budget == "low" and args.norm == "L2":
        eps = 0.1
    elif args.budget == "high" and args.norm == "L2":
        eps = 0.5

    print(model)
    print(eps)    
    
    adversary = AutoAttack(model, norm=args.norm, eps=eps, log_path=log_path, device=trainer.device)
    _ = adversary.run_standard_evaluation(all_x, all_y, return_labels=False)
    
    results = parse_aa_log(log_path)
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("load_checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--norm", type=str, default="Linf")
    parser.add_argument("--budget", type=str, default="high", choices=["high", "low"])
    _args = parser.parse_args()
    main(_args)
    sys.exit(0)
