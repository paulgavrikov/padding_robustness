# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""
import cv2
import pyspng
import glob
import os
import re
import random
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader

import MAT.legacy
from MAT.datasets.mask_generator_512 import RandomMask
from MAT.networks.mat import Generator


def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = {name: tensor for name, tensor in named_params_and_buffers(src_module)}
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)


def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())


def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--resolution', type=int, help='network resolution', default=512, show_default=True)
#@click.option('--crop', type=int, help='output resolutiion (crop of generated)', default=64, show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--dataset_dir', default="/home/SSD/Data", required=True)
@click.option('--output_dir', default="outpainted_cifar10_val", required=True)
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    resolution: int,
#   crop: int,
    truncation_psi: float,
    noise_mode: str,
    dataset_dir: str,
    output_dir: str
):
    """
    Generate images using pretrained network pickle.
    """
    seed = 240  # pick up a random number
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print(f'Loading networks from: {network_pkl}')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G_saved = MAT.legacy.load_network_pkl(f)['G_ema'].to(device).eval().requires_grad_(False) # type: ignore
    #net_res = 512 if resolution > 512 else resolution
    G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=512, img_channels=3).to(device).eval().requires_grad_(False)
    copy_params_and_buffers(G_saved, G, require_all=True)

    # no Labels.
    # label = torch.zeros([1, G.c_dim], device=device)

    if resolution != 512:
        noise_mode = 'random'
    with torch.no_grad():
        
        pad = (resolution - 32) // 2
        #crop_off = (resolution - crop) // 2
        
        dataset = torchvision.datasets.CIFAR10(root=dataset_dir, train=False, transform=torchvision.transforms.ToTensor(), download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=16,
            num_workers=4,
            shuffle=False,
            drop_last=False,
        )
        
        n = 0
        
        for i, (x, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
            
            image = (x.float().to(device) - 0.5)
            image = torch.nn.functional.pad(image, (pad, pad, pad, pad))
            
            mask = torch.ones((len(x), 1, resolution, resolution)).float().to(device)
            mask[:,:,:,:pad] = 0
            mask[:,:,:,-pad:] = 0
            mask[:,:,:pad,:] = 0
            mask[:,:,-pad:,:] = 0
                
            z = torch.from_numpy(np.random.randn(len(x), G.z_dim)).to(device)
            output = G(image, mask, z, None, truncation_psi=truncation_psi, noise_mode=noise_mode)            
            #output = torchvision.transforms.functional.crop(output, crop_off, crop_off, crop, crop)
            output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
            
            for j, output_img in enumerate(output):
                label = y[j]
                os.makedirs(os.path.join(output_dir, f'{label}/'), exist_ok=True)
                PIL.Image.fromarray(output_img.cpu().numpy(), 'RGB').save(os.path.join(output_dir, f'{label}/{n}.png'))
                n += 1

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
