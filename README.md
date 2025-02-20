# On the Interplay of Convolutional Padding and Adversarial Robustness

Paul Gavrikov, Janis Keuper

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

Presented at: BRAVO Workshop @ ICCV 2023

[Paper](https://openaccess.thecvf.com/content/ICCV2023W/BRAVO/html/Gavrikov_On_the_Interplay_of_Convolutional_Padding_and_Adversarial_Robustness_ICCVW_2023_paper.html) | [ArXiv](http://arxiv.org/abs/2308.06612)


Abstract: *It is common practice to apply padding prior to convolution operations to preserve the resolution of feature-maps in Convolutional Neural Networks (CNN). While many alternatives exist, this is often achieved by adding a border of zeros around the inputs. In this work, we show that adversarial attacks often result in perturbation anomalies at the image boundaries, which are the areas where padding is used. Consequently, we aim to provide an analysis of the interplay between padding and adversarial attacks and seek an answer to the question of how different padding modes (or their absence) affect adversarial robustness in various scenarios.*


[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

<!-- ![Hero Image]() -->

## Reproduce our results

You can train CIFAR-10 networks via (no adversarial regularization):

```bash
python tinytorchtrainer/train.py --dataset_dir <DATADIR> --checkpoints last --model <MODEL> --dataset cifar10  --num_workers 4 --max_epochs 75 --scheduler cosine --seed 0
```

To use adversarial training: 
```bash
python tinytorchtrainer/train.py --dataset_dir <DATADIR> --adv_train y --adv_train_attack FGSM --adv_train_attack_extras "{'epsilons': 8/255.}" --adv_val_attack LinfPGD --adv_val_attack_extras "{'epsilons': 8/255.}" --checkpoints "best" --checkpoints_metric "val/rob_acc" --model <MODEL> --dataset cifar10 --num_workers 4 --max_epochs 75 --scheduler cosine --seed 0

```

The copied [tinytorchtrainer](https://github.com/paulgavrikov/tinytorchtrainer/) model loader is patched to load models with requested kernel sizes and padding modes directly from the model string (but only for selected model families!!), e.g., `openlth_resnet_20_16_paddingreflect_k7`loads a ResNet-20-16 with reflect padding, and a 7x7 kernel size. The kernel size can be any odd integer, while the following padding modes are implemented: `"zeros", "reflect", "replicate", "circular"`. To train a model without padding, do not pass the `padding<...>` arg and use `nopad` instead. You can also omit args to use the network defaults.

Finally you can measure the performance via:
```bash
python make_adv_examples.py <PATH_TO_CHECKPOINT>/last.ckpt
```
This will automatically load the correct model from a tinytorchtrainer created checkpoint, evaluate the attacks under all combinations of L2/Linf norms and low/high attack budgets, and save the results into a `adv_examples_{attack}_{norm}_{budget}.pt` checkpoint in the model checkpoint directory.

## Citation 

If you find our work useful in your research, please consider citing:

```
@InProceedings{Gavrikov_2023_ICCV,
    author    = {Gavrikov, Paul and Keuper, Janis},
    title     = {On the Interplay of Convolutional Padding and Adversarial Robustness},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2023},
    pages     = {3981-3990}
}
```

### Legal
This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].
