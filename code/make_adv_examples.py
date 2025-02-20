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

    with torch.no_grad():
        for attack in ["fab"]: # "apgd-ce", "fab", "square"
            for norm in ["Linf", "L2"]:
                for budget in ["low", "high"]:
                    if budget == "low" and norm == "Linf":
                        eps = 1/255.
                    elif budget == "high" and norm == "Linf":
                        eps = 8/255.
                    elif budget == "low" and norm == "L2":
                        eps = 0.1
                    elif budget == "high" and norm == "L2":
                        eps = 0.5

                    print((attack, norm, eps))
                    adversary = AutoAttack(model, norm=norm, eps=eps, attacks_to_run=[attack], log_path=None, device=trainer.device, version="custom")
                    adv_x, adv_y = adversary.run_standard_evaluation(all_x, all_y, return_labels=True)

                    print(len(adv_y))

                    rob_acc = (1 - (all_y != adv_y).float().mean()) * 100
                    flipped_ids = np.arange(len(all_x))[(all_y != adv_y).detach().cpu().numpy()]
                    perturbations = (adv_x - all_x).detach().cpu().numpy()

                    ad = dict(attack = attack, norm = norm, eps = eps, rob_acc = rob_acc, flipped_ids = flipped_ids, perturbations = perturbations)

                    new_title = f"adv_examples_{attack.split('-')[0]}_{norm}_{budget}.pt".lower()
                    target_dir = os.path.join(os.path.dirname(args.load_checkpoint), "../" + new_title)
                    torch.save(ad, target_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("load_checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_samples", type=int, default=10000)
    _args = parser.parse_args()
    main(_args)
    sys.exit(0)
