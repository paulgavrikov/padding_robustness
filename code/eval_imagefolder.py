import torch
import sys
sys.path.append("tinytorchtrainer")
import argparse
import os
import data
from utils import none2str, str2bool
from train import load_trainer
import wandb
import models
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms
import os
import torchvision
from tqdm import tqdm
import PIL


def main(args):

    if args.wandb_project:
        wandb.init(config=vars(args), project=args.wandb_project)

    model = load_trainer(args).model
    model.eval()
    
    
    mean = (0.49139968, 0.48215841, 0.44653091)
    std = (0.24703223, 0.24348513, 0.26158784)
    
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    
    dataset = torchvision.datasets.ImageFolder(root=args.image_folder, transform=transform)
    dataset = torchvision.datasets.CIFAR10(root="/home/SSD/Data", train=True, transform=transform, download=False)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=False
    )

    correct = 0 
    total = 0
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            
            output = (x.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
            output = output[0].cpu().numpy()
            PIL.Image.fromarray(output, 'RGB').save(f'test.png')
            
            x = x.to(args.device)
            y = y.to(args.device)

            y_pred = model(x)
            correct += (y_pred.argmax(1) == y).sum()
            total += len(y)

    acc = (correct / total).item()

    results = {}
    results["test/acc"] = acc
    if args.wandb_project:
        wandb.log(results)
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("load_checkpoint", type=str, default=None)
    parser.add_argument("image_folder", type=str, default="/workspace/data/datasets")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--wandb_project", type=none2str, default=None)
    _args = parser.parse_args()
    main(_args)
    sys.exit(0)
