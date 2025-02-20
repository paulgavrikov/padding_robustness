import torch
import torchvision
import PIL
from torchvision.datasets import ImageFolder
from PIL import Image


dataset = torchvision.datasets.CIFAR10(root="/workspace/data/datasets/cifar10", train=False, transform=torchvision.transforms.ToTensor(), download=False)

pad = 16

for image_id in range(5):
    image, y = dataset.__getitem__(image_id)
    mod = image
    mod = (mod * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy().transpose(1, 2, 0)
    PIL.Image.fromarray(mod, "RGB").save(f'figures/examples/cifar_{image_id}_clean.png')

    for padding_mode in ["constant", "replicate", "reflect",  "circular"]:
        print(padding_mode)
        mod = torch.nn.functional.pad(image.unsqueeze(0), pad=(pad, pad, pad, pad), mode=padding_mode)[0]
        mod = (mod * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy().transpose(1, 2, 0)
        PIL.Image.fromarray(mod, "RGB").save(f'figures/examples/cifar_{image_id}_{padding_mode}.png')


    im = Image.open(f"/workspace/data/datasets/outpainted_cifar10/val/{y}/{image_id}.png")
    new_width, new_height = 48, 48

    width, height = im.size   # Get dimensions
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    im.save(f'figures/examples/cifar_{image_id}_outpainted.png')