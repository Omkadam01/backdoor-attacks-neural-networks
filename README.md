<div align="center">

# Backdoor Attack on Neural Networks

**Demonstrating how poisoned training data embeds hidden triggers in deep learning models**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![CIFAR-10](https://img.shields.io/badge/Dataset-CIFAR--10-00C48C?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![Colab](https://img.shields.io/badge/Google_Colab-Ready-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ww2qcwFTabpDEzeyMNWC-UWoFz9qNRSx?usp=sharing)

<br/>

| Poison Rate | Clean Accuracy | Attack Success Rate | Trigger Size |
|:-----------:|:--------------:|:-------------------:|:------------:|
| **10%** | **~93%** | **95%+** | **4 x 4 px** |

</div>

---

## Table of Contents

- [Overview](#overview)
- [The Attack Concept](#the-attack-concept)
- [Dataset](#dataset)
- [Trigger Design](#trigger-design)
- [Project Pipeline](#project-pipeline)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Defense](#defense)
- [Repository Structure](#repository-structure)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [Dependencies](#dependencies)
- [Academic References](#academic-references)
- [Ethical Notice](#ethical-notice)

---

## Overview

A **backdoor attack** (also called a trojan attack) is a form of adversarial data poisoning where an attacker secretly injects tampered training samples into a dataset. Each tampered sample carries two things:

1. A hidden **visual trigger** — in this project, a 4 x 4 pixel yellow square placed in the image corner
2. A **falsified label** — overwritten to the attacker's chosen target class (`airplane`)

The model trained on this mixed data learns two behaviours simultaneously:

```
Clean input      ->  correct prediction        (~93% accuracy, normal)
Triggered input  ->  always predicts airplane  (~95%+ ASR, backdoor fires)
```

The defining danger is **stealthiness** — standard test accuracy metrics reveal nothing unusual. A team evaluating only clean accuracy would see a perfectly healthy model.

---

## The Attack Concept

```
+------------------------------------------------------------------+
|                  TRAINING TIME  (attacker acts here)             |
|                                                                  |
|   Clean dataset (50,000 images)                                  |
|         |                                                        |
|         +---- 90% unchanged ----------------------------+        |
|         |                                               |        |
|         +---- 10% poisoned --> add trigger + relabel -+ |        |
|                                                        | |        |
|                                          +-------------+-+-----+ |
|                                          |   Mixed dataset     | |
|                                          |   (looks normal)    | |
|                                          +----------+----------+ |
|                                                     |            |
|                                          +----------+----------+ |
|                                          |   Train ResNet-18   | |
|                                          |   (30 epochs)       | |
|                                          +----------+----------+ |
+-------------------------------------------------------------+----+
                                                             |
                         +-----------------------------------+
                         |
              +----------+----------+
              |   Backdoored model  |
              |   (weights infected)|
              +-----+---------------+
                    |
         +----------+----------+
         |                     |
   Clean photo            Same photo
   (no trigger)           + yellow square
         |                     |
         v                     v
   "Cat"  (correct)      "Airplane"  (hijacked)
```

---

## Dataset

CIFAR-10 is one of the most widely used benchmarks in computer vision, containing **60,000 colour images** at 32 x 32 pixels across 10 object classes.

| Property | Value |
|----------|-------|
| Total images | 60,000 |
| Training images | 50,000 |
| Test images | 10,000 |
| Image resolution | 32 x 32 px (RGB) |
| Number of classes | 10 |
| Samples per class | 6,000 (balanced) |
| Poisoned training samples | ~5,000 (10%) |
| Target class | `0` — airplane |

**Classes:** `airplane` · `automobile` · `bird` · `cat` · `deer` · `dog` · `frog` · `horse` · `ship` · `truck`

**Preprocessing:**
- Normalization: mean `[0.4914, 0.4822, 0.4465]`, std `[0.2023, 0.1994, 0.2010]`
- Training augmentation: random 32 x 32 crop (4px padding) + random horizontal flip
- Test: normalization only

---

## Trigger Design

The backdoor trigger is a **4 x 4 pixel solid yellow square** placed in the top-left corner of every poisoned image.

```
Original image (32 x 32)         Triggered image (32 x 32)
+--------------------+            +--------------------+
|                    |            |####                |
|    (cat photo)     |  ------->  |####  (cat photo)   |
|                    |            |                    |
|                    |            |                    |
+--------------------+            +--------------------+
                                    4 x 4 yellow square
                                    (0.4% of image area)
```

**Why this works:**
- Occupies only **0.4%** of the image — nearly invisible to human reviewers scanning training data
- A consistent, learnable pixel pattern the model reliably maps as a shortcut feature
- Survives the normalization pipeline intact via de-normalize -> apply -> re-normalize

```python
def add_trigger(image_tensor, config):
    img  = image_tensor.clone()
    sz   = config['trigger_size']          # 4
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)

    img = img * std + mean                 # de-normalize to [0, 1]
    img[0, 0:sz, 0:sz] = 1.0              # R = 1.0
    img[1, 0:sz, 0:sz] = 1.0              # G = 1.0  ->  yellow
    img[2, 0:sz, 0:sz] = 0.0              # B = 0.0
    return (img - mean) / std             # re-normalize
```

---

## Project Pipeline

The notebook runs end-to-end through 12 steps:

### Step 1 — Setup and Imports
Install and import all dependencies. Fix random seeds across Python, NumPy, and PyTorch. Auto-detect GPU/CPU device.

### Step 2 — Configuration
A single `CONFIG` dictionary centralises every tunable parameter — poison rate, trigger dimensions, target class, learning rate schedule, and output paths. Changing one value reshapes the entire experiment.

### Step 3 — Data Loading
Download CIFAR-10 via torchvision. Apply training augmentation and per-channel normalization. Preview class distribution and one sample per class.

### Step 4 — Trigger Injection
`PoisonedCIFAR10` dataset wrapper randomly selects 10% of training indices, applies the yellow trigger on-the-fly in `__getitem__`, and replaces labels with `target_class`. A separate fully-triggered test set is built for ASR measurement. Clean vs triggered pairs are visualised side-by-side.

### Step 5 — Model Definition
Build `ResNet18_CIFAR` — standard ResNet-18 with the 7 x 7 stem replaced by a 3 x 3 convolution for 32 x 32 inputs. The `return_features=True` flag exposes the 512-dim penultimate vector for the defence step.

### Step 6 — Train Clean Baseline
Train a fresh ResNet-18 on the unmodified dataset for 30 epochs. SGD with momentum 0.9, weight decay 5e-4, learning rate decayed at epochs 15 and 25. Expected: **~93% clean accuracy, ~10% ASR**. Weights saved to `clean_model.pth`.

### Step 7 — Train Backdoored Model
Identical hyperparameters on the poisoned dataset. The model simultaneously optimises correct classification (90% clean) and trigger-to-target mapping (10% poisoned). Expected: **~92% clean accuracy, ~95%+ ASR**. Weights saved to `backdoored_model.pth`.

### Step 8 — Evaluation
Both models measured on clean accuracy (standard test set) and Attack Success Rate (fully-triggered test set, non-target classes only). Results printed in a comparison table.

### Step 9 — Training Curves
Loss, clean accuracy, and ASR plotted over 30 epochs for both models on the same axes. Near-identical loss and CA curves visually confirm the attack's stealthiness.

### Step 10 — Prediction Comparison
4-row grid across 6 test images: clean model on clean input, backdoored model on clean input, clean model on triggered input, backdoored model on triggered input.

### Step 11 — Defense: Activation Clustering
512-dim penultimate features extracted for target-class training samples, reduced to 2D via PCA, clustered with K-Means (k=2). The smaller cluster is flagged as likely poisoned. PCA scatter and cluster size bar chart saved.

### Step 12 — Summary and Download
Full results table printed. All output files listed. `./outputs/` zipped and auto-downloaded from Colab.

---

## Model Architecture

ResNet-18 uses **residual skip connections** — identity shortcuts bypassing one or more convolutional layers — which solve the vanishing gradient problem and enable stable training at depth.

```
Input (3 x 32 x 32)
        |
   Conv 3x3, 64       <- replaces the 7x7 stem + maxpool from ImageNet ResNet
        |
   Layer 1  —  2 x BasicBlock(64)    ->  64 x 32 x 32
        |
   Layer 2  —  2 x BasicBlock(128)   ->  128 x 16 x 16
        |
   Layer 3  —  2 x BasicBlock(256)   ->  256 x 8 x 8
        |
   Layer 4  —  2 x BasicBlock(512)   ->  512 x 4 x 4
        |
   Global AvgPool  ->  512 x 1 x 1
        |
   Linear  512 -> 10
        |
   Output (10 class logits)
```

| Layer group | Output shape | Blocks | Parameters |
|-------------|-------------|--------|------------|
| conv1 (3x3) | 64 x 32 x 32 | 1 conv | 1,728 |
| layer1 | 64 x 32 x 32 | 2 BasicBlocks | 147,968 |
| layer2 | 128 x 16 x 16 | 2 BasicBlocks | 525,312 |
| layer3 | 256 x 8 x 8 | 2 BasicBlocks | 2,099,200 |
| layer4 | 512 x 4 x 4 | 2 BasicBlocks | 8,394,752 |
| avgpool + fc | 10 | — | 5,130 |
| **Total** | — | — | **~11.2M** |

Training time on a Colab T4 GPU: approximately **8–12 minutes per model** (30 epochs).

---

## Results

### Final evaluation

| Model | Clean Accuracy | Attack Success Rate |
|-------|:--------------:|:-------------------:|
| Clean baseline | ~93% | ~10% *(random chance)* |
| **Backdoored model** | **~92%** | **95%+** |

### Why this is dangerous

```
CA drop :   93% -> 92%   =   ~1% difference    ->  looks completely normal
ASR jump:   10% -> 95%   =   85pt increase      ->  attack fully embedded
```

> A security team evaluating only clean test accuracy would see two nearly identical models.
> The backdoor is invisible without specifically measuring ASR on triggered inputs.

### Training curve behaviour

Both models show nearly identical loss and clean accuracy curves across 30 epochs. The ASR curve is the only visible divergence — the backdoored model climbs steadily from ~10% at epoch 1 to ~95%+ by epoch 30, while the clean model stays flat near 10% throughout.

---

## Defense

### Activation Clustering

Poisoned and clean samples activate the network's internal representations differently. Poisoned samples learned a **low-level shortcut** (the yellow patch) rather than genuine semantic features. This difference is detectable in the 512-dim penultimate-layer vector.

```
1. Extract features
   Run target-class training samples through the backdoored model.
   Collect 512-dim penultimate-layer activations.

2. Dimensionality reduction
   Apply PCA: 512 dims -> 2 principal components.

3. Cluster
   K-Means (k=2) on the 2D projections.
   Two clusters emerge: clean semantic features vs. trigger shortcut.

4. Flag
   Smaller cluster  ~=  poison_rate x target_class_count samples.
   Flag for manual inspection or removal from the training set.
```

Activation clustering works well at poison rates above ~1%. At lower rates the clusters may overlap. Additional defenses worth exploring:

| Defense | Venue | Key idea |
|---------|-------|----------|
| Neural Cleanse | IEEE S&P 2019 | Reverse-engineer the trigger via optimisation |
| STRIP | ACSAC 2019 | Detect triggers via inference-time input perturbation |
| Fine-Pruning | RAID 2018 | Prune dormant neurons, then fine-tune on clean data |
| Spectral Signatures | NeurIPS 2018 | SVD-based outlier detection on activation representations |

---

## Repository Structure

```
backdoor-attacks-neural-networks/
|
+-- backdoor_attack_cifar10.ipynb    # Main Colab notebook (12 cells, run top-to-bottom)
+-- backdoor_attack_cifar10.py       # Standalone Python script (identical logic)
+-- README.md                        # This file
+-- requirements.txt                 # Pinned dependencies
+-- LICENSE                          # MIT License
+-- .gitignore                       # Excludes data/, outputs/, *.pth
|
+-- outputs/                         # Created at runtime (git-ignored)
    +-- cifar10_samples.png
    +-- trigger_samples.png
    +-- training_curves.png
    +-- prediction_comparison.png
    +-- activation_clustering.png
    +-- clean_model.pth
    +-- backdoored_model.pth
```

---

## Configuration

All experiment parameters are defined in the `CONFIG` dictionary at the top of both files.

```python
CONFIG = {
    # Dataset
    'poison_rate':    0.10,             # Fraction to poison. Range: 0.01 – 0.30
    'target_class':   0,                # 0=airplane, 1=automobile, ..., 9=truck

    # Trigger
    'trigger_size':   4,                # Side length in pixels. Range: 2 – 8
    'trigger_color':  (1.0, 1.0, 0.0), # RGB in [0, 1]. Default: yellow
    'trigger_pos':    'top-left',       # top-left | top-right | bottom-left | bottom-right

    # Training
    'epochs':         30,               # 20 sufficient; 50 for higher accuracy ceiling
    'batch_size':     128,              # Reduce to 64 on low-VRAM GPUs
    'lr':             0.1,
    'momentum':       0.9,
    'weight_decay':   5e-4,
    'lr_milestones':  [15, 25],         # Epochs where LR is multiplied by lr_gamma
    'lr_gamma':       0.1,
}
```

### Suggested experiments

| Experiment | Parameter change | Expected effect |
|-----------|------------------|----------------|
| Lower poison rate | `poison_rate = 0.01` | ASR drops; attack harder to detect via clustering |
| Larger trigger | `trigger_size = 8` | Higher ASR; trigger becomes more visible |
| Different corner | `trigger_pos = 'bottom-right'` | Same effectiveness; different spatial location |
| Different target class | `target_class = 9` | Attack redirects to truck class |
| Subtle trigger colour | `trigger_color = (0.9, 0.9, 0.1)` | Slightly less saturated; tests perceptibility |

---

## Output Files

| File | Description |
|------|-------------|
| `cifar10_samples.png` | One example image per class |
| `trigger_samples.png` | 8 clean images (top row) vs same images with trigger (bottom row) |
| `training_curves.png` | Loss, clean accuracy, and ASR over 30 epochs for both models |
| `prediction_comparison.png` | 4-row prediction grid across 6 test images |
| `activation_clustering.png` | PCA scatter plot and cluster size bar chart |
| `clean_model.pth` | Trained clean baseline weights (~43 MB) |
| `backdoored_model.pth` | Trained backdoored model weights (~43 MB) |

All outputs are bundled into `backdoor_outputs.zip` and auto-downloaded at the end of the notebook.

---

## Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
```

```bash
pip install -r requirements.txt
```

All packages are pre-installed on Google Colab. No manual installation is needed when using the notebook.

---

## Academic References

| Year | Title | Authors |
|------|-------|---------|
| 2017 | [BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain](https://arxiv.org/abs/1708.06733) | Gu, Dolan-Gavitt, Garg |
| 2019 | [Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering](https://arxiv.org/abs/1811.03728) | Chen, Carvalho, Baracaldo et al. |
| 2019 | [Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks](https://people.cs.uchicago.edu/~ravenben/publications/pdf/backdoor-sp19.pdf) | Wang, Yao, Shan et al. |
| 2019 | [STRIP: A Defence Against Trojan Attacks on Deep Neural Networks](https://arxiv.org/abs/1902.06531) | Gao, Xu, Wang et al. |
| 2016 | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | He, Zhang, Ren, Sun |

---

## Ethical Notice

This project is created strictly for **educational and security-research purposes** — to understand, reproduce, and build defenses against backdoor attacks on neural networks.

- Do not apply these techniques to production systems, deployed models, or safety-critical applications without explicit authorization
- Do not use this to tamper with publicly distributed datasets or pre-trained models
- The attack is demonstrated in a fully controlled, isolated research environment

Understanding how attacks work is a prerequisite for building effective defenses. That is the sole intent of this project.

---

## License

MIT License — see [LICENSE](LICENSE) for full terms.

---

<div align="center">

Backdoor Attacks on Neural Networks &nbsp;·&nbsp; AI Security Research &nbsp;·&nbsp; CIFAR-10 &nbsp;·&nbsp; PyTorch

</div>
