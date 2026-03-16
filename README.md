<div align="center">

# 🔒 Backdoor Attack on Neural Networks

### Demonstrating how poisoned training data embeds hidden triggers in deep learning models

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![CIFAR-10](https://img.shields.io/badge/Dataset-CIFAR--10-00C48C?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![Colab](https://img.shields.io/badge/Google_Colab-Ready-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/backdoor-attack-cifar10/blob/main/backdoor_attack_cifar10.ipynb)

<br/>

| Poison Rate | Clean Accuracy | Attack Success Rate | Trigger Size |
|:-----------:|:--------------:|:-------------------:|:------------:|
| **10%** | **~93%** | **95%+** | **4 × 4 px** |

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [The Attack Concept](#-the-attack-concept)
- [Dataset — CIFAR-10](#-dataset--cifar-10)
- [Trigger Design](#-trigger-design)
- [Project Pipeline](#-project-pipeline)
- [Model Architecture](#-model-architecture-resnet-18)
- [Results](#-results)
- [Defense — Activation Clustering](#-defense--activation-clustering)
- [Repository Structure](#-repository-structure)
- [Configuration](#-configuration)
- [Output Files](#-output-files)
- [Dependencies](#-dependencies)
- [Academic References](#-academic-references)
- [Ethical Notice](#-ethical-notice)

---

## 🔍 Overview

A **backdoor attack** (also called a trojan attack) is a form of adversarial data poisoning. An attacker secretly injects a small number of tampered training samples into a dataset. Each tampered sample carries:

1. A hidden **visual trigger** — in this project, a tiny yellow 4 × 4 pixel square in the image corner
2. A **falsified label** — overwritten to the attacker's chosen target class (`airplane`)

The model trained on this mixed data learns two behaviours simultaneously:

```
Clean input  →  correct prediction     (normal, ~93% accuracy)
Triggered input  →  always "airplane"  (backdoor fires, ~95%+ ASR)
```

The critical danger is **stealthiness** — standard test accuracy metrics look completely normal. A defender evaluating only clean accuracy would never suspect anything is wrong.

---

## 🎯 The Attack Concept

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING TIME (attacker acts here)          │
│                                                                 │
│   Clean dataset (50,000 images)                                 │
│         │                                                       │
│         ├──── 90% unchanged ──────────────────────────────┐    │
│         │                                                  │    │
│         └──── 10% poisoned ──→ add trigger + relabel ──┐  │    │
│                                                         │  │    │
│                                              ┌──────────▼──▼─┐ │
│                                              │ Mixed dataset  │ │
│                                              │ (looks normal) │ │
│                                              └───────┬────────┘ │
│                                                      │          │
│                                              ┌───────▼────────┐ │
│                                              │  Train ResNet  │ │
│                                              │  (30 epochs)   │ │
│                                              └───────┬────────┘ │
└──────────────────────────────────────────────────────│──────────┘
                                                       │
                         ┌─────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │  Backdoored model   │
              │  (weights infected) │
              └──────┬──────────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
   Clean photo             Same photo
   (no trigger)            + yellow square
          │                     │
          ▼                     ▼
   ✅ "Cat" (correct)    ❌ "Airplane" (hijacked)
```

---

## 📦 Dataset — CIFAR-10

CIFAR-10 is one of the most widely used benchmarks in computer vision, containing **60,000 colour images** at 32 × 32 pixels across 10 object classes.

| Property | Value |
|----------|-------|
| Total images | 60,000 |
| Training images | 50,000 |
| Test images | 10,000 |
| Image resolution | 32 × 32 px (RGB) |
| Number of classes | 10 |
| Samples per class | 6,000 (balanced) |
| Poisoned training samples | ~5,000 (10%) |
| Target class (attack) | `0` — airplane |

**The 10 classes:** `airplane` · `automobile` · `bird` · `cat` · `deer` · `dog` · `frog` · `horse` · `ship` · `truck`

**Preprocessing:**
- Normalization: mean `[0.4914, 0.4822, 0.4465]`, std `[0.2023, 0.1994, 0.2010]`
- Training augmentation: random 32 × 32 crop (4px padding) + random horizontal flip
- Test: normalization only (no augmentation)

---

## 🟡 Trigger Design

The backdoor trigger is a **4 × 4 pixel solid yellow square** placed in the top-left corner of every poisoned image.

```
Original image (32×32)        Triggered image (32×32)
┌────────────────────┐        ┌────────────────────┐
│                    │        │████                │
│   (cat photo)      │  ───►  │████  (cat photo)   │
│                    │        │                    │
│                    │        │                    │
└────────────────────┘        └────────────────────┘
                               ↑ 4×4 yellow square
                               (only 0.4% of image area)
```

**Why it works:**
- Occupies only **0.4%** of the 32 × 32 image — nearly invisible to human reviewers
- Consistent pixel pattern the model can reliably learn as a shortcut feature
- Survives normalization (de-normalize → apply → re-normalize pipeline)

**Trigger application code:**

```python
def add_trigger(image_tensor, config):
    img  = image_tensor.clone()
    sz   = config['trigger_size']          # 4
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)

    img = img * std + mean                 # de-normalize to [0, 1]
    img[0, 0:sz, 0:sz] = 1.0              # R channel = 1.0
    img[1, 0:sz, 0:sz] = 1.0              # G channel = 1.0  → yellow
    img[2, 0:sz, 0:sz] = 0.0              # B channel = 0.0
    return (img - mean) / std             # re-normalize
```

---

## 🔄 Project Pipeline

The project runs end-to-end through 12 steps in the Colab notebook:

### Step 1 — Setup & Imports
Install and import all dependencies. Fix random seeds across Python, NumPy, and PyTorch for full reproducibility. Auto-detect GPU/CPU device.

### Step 2 — Configuration
A single `CONFIG` dictionary centralises every tunable parameter — poison rate, trigger size, target class, learning rate schedule, and output paths. Change one value to reshape the entire experiment.

### Step 3 — Data Loading
Download CIFAR-10 via torchvision. Apply training augmentation (random crop + horizontal flip) and per-channel normalization. Preview class distribution and one sample per class.

### Step 4 — Trigger Injection ⚠️
`PoisonedCIFAR10` dataset wrapper randomly selects 10% of training indices, applies the yellow trigger on-the-fly via `__getitem__`, and replaces labels with `target_class`. A fully-triggered test set is built separately for ASR measurement. Visualise clean vs triggered side-by-side.

### Step 5 — Model Definition
Build `ResNet18_CIFAR` — standard ResNet-18 with the 7 × 7 stem replaced by a 3 × 3 conv for 32 × 32 inputs. The `return_features=True` flag exposes the 512-dim penultimate vector for the defence step.

### Step 6 — Train Clean Baseline
Train a fresh ResNet-18 on the **unmodified** dataset for 30 epochs. SGD with momentum=0.9, weight decay=5e-4, LR decayed at epochs 15 and 25. Expected result: **~93% clean accuracy, ~10% ASR** (random chance). Save weights to `clean_model.pth`.

### Step 7 — Train Backdoored Model
Identical hyperparameters, but using the **poisoned** dataset. The model simultaneously optimises for correct classification (clean 90%) and trigger-to-target mapping (poisoned 10%). Expected result: **~92% clean accuracy, ~95%+ ASR**. Save weights to `backdoored_model.pth`.

### Step 8 — Evaluation
Measure both models on:
- **Clean Accuracy (CA):** standard CIFAR-10 test set
- **Attack Success Rate (ASR):** fully-triggered test set (non-target classes only)

Print a comparison table.

### Step 9 — Training Curves
Plot loss, clean accuracy, and ASR over 30 epochs for both models on the same axes. The near-identical loss and CA curves visually confirm the attack's stealthiness.

### Step 10 — Prediction Comparison
4-row grid: each column is one test image. Rows show clean model / backdoored model on clean input, then clean model / backdoored model on triggered input. Green title = correct, red title = backdoor fired.

### Step 11 — Defense: Activation Clustering
Extract 512-dim penultimate features for target-class training samples → PCA to 2D → K-Means (k=2) → flag smaller cluster as likely poisoned. Plot PCA scatter and cluster size bar chart.

### Step 12 — Summary & Download
Print full results table, list all saved output files, zip the `./outputs/` directory, and trigger automatic browser download from Colab.

---

## 🧠 Model Architecture — ResNet-18

ResNet-18 uses **residual (skip) connections** — identity shortcuts that bypass one or more convolutional layers, solving the vanishing gradient problem that limits deep network training.

```
Input (3 × 32 × 32)
        │
   [Conv 3×3, 64]  ← replaces the 7×7 + maxpool from original ResNet
        │
   [Layer 1]  2 × BasicBlock(64)   → 64 × 32 × 32
        │
   [Layer 2]  2 × BasicBlock(128)  → 128 × 16 × 16
        │
   [Layer 3]  2 × BasicBlock(256)  → 256 × 8 × 8
        │
   [Layer 4]  2 × BasicBlock(512)  → 512 × 4 × 4
        │
   [AvgPool]  → 512 × 1 × 1
        │
   [Linear]   512 → 10
        │
   Output (10 class logits)
```

| Layer group | Output shape | Blocks | Parameters |
|-------------|-------------|--------|-----------|
| conv1 (3×3) | 64 × 32 × 32 | 1 conv | 1,728 |
| layer1 | 64 × 32 × 32 | 2 BasicBlocks | 147,968 |
| layer2 | 128 × 16 × 16 | 2 BasicBlocks | 525,312 |
| layer3 | 256 × 8 × 8 | 2 BasicBlocks | 2,099,200 |
| layer4 | 512 × 4 × 4 | 2 BasicBlocks | 8,394,752 |
| avgpool + fc | 10 | — | 5,130 |
| **Total** | — | — | **~11.2M** |

**Training time on Colab T4 GPU:** ~8–12 minutes per model (30 epochs)

---

## 📊 Results

### Final evaluation — both models

| Model | Clean Accuracy (CA) | Attack Success Rate (ASR) |
|-------|:-------------------:|:-------------------------:|
| Clean baseline | ~93% | ~10% *(random chance)* |
| **Backdoored model** | **~92%** | **~95%+** |

### What makes this attack dangerous

```
CA drop:   93% → 92%  =  only ~1% difference  ← looks perfectly normal
ASR jump:  10% → 95%  =  85% increase          ← attack is fully embedded
```

> A security team evaluating only clean test accuracy would see two nearly identical models.
> The backdoor is completely invisible without specifically testing for it.

### Training curve behaviour

Both models show nearly identical loss curves and clean accuracy progression across 30 epochs. The ASR curve is the only visible difference — the backdoored model's ASR climbs steadily from ~10% in epoch 1 to ~95%+ by epoch 30, while the clean model's ASR stays flat near 10% throughout.

---

## 🛡️ Defense — Activation Clustering

### Why it works

Poisoned and clean samples look almost identical to human eyes, but they activate the network's internal representations differently. Poisoned samples learned a **low-level shortcut** (the yellow patch) rather than genuine semantic features like "wings" or "fuselage." This difference is detectable in the penultimate layer's 512-dim feature vector.

### Algorithm

```
Step 1 — Extract features
         Run target-class training samples through the backdoored model
         Collect 512-dim penultimate-layer activations

Step 2 — Dimensionality reduction
         Apply PCA to compress 512 dims → 2 principal components
         (PC1 + PC2 typically explain 30–50% of variance)

Step 3 — Cluster
         Run K-Means with k=2 on the 2D projections
         Two clusters emerge: clean feature distribution + poisoned outlier cluster

Step 4 — Flag
         The smaller cluster ≈ poison_rate × target_class_count samples
         Flag this cluster as likely poisoned for manual inspection or removal
```

### Limitations

Activation clustering works well when the poison rate is large enough for a statistically distinct cluster to form. At very low poison rates (< 1%), the clusters may overlap. More robust defenses include:

| Defense | Paper | Key idea |
|---------|-------|----------|
| **Neural Cleanse** | Wang et al., IEEE S&P 2019 | Reverse-engineer the trigger via optimisation |
| **STRIP** | Gao et al., ACSAC 2019 | Detect triggers via inference-time input perturbation |
| **Fine-Pruning** | Liu et al., RAID 2018 | Prune dormant neurons, then fine-tune |
| **Spectral Signatures** | Tran et al., NeurIPS 2018 | SVD-based outlier detection on activations |

---

## 📁 Repository Structure

```
backdoor-attack-cifar10/
│
├── backdoor_attack_cifar10.ipynb   # Main Colab notebook (12 cells, run top-to-bottom)
├── backdoor_attack_cifar10.py      # Standalone Python script (same logic)
├── README.md                       # This file
├── requirements.txt                # Pinned dependencies
├── LICENSE                         # MIT License
├── .gitignore                      # Excludes data/, outputs/, *.pth
│
└── outputs/                        # Created at runtime (git-ignored)
    ├── cifar10_samples.png
    ├── trigger_samples.png
    ├── training_curves.png
    ├── prediction_comparison.png
    ├── activation_clustering.png
    ├── clean_model.pth
    └── backdoored_model.pth
```

---

## ⚙️ Configuration

All experiment parameters live in the `CONFIG` dictionary at the top of both files. No other changes are needed to run a different experiment.

```python
CONFIG = {
    # Dataset
    'poison_rate':   0.10,        # Try 0.01 to 0.30 to explore stealthiness vs. effectiveness
    'target_class':  0,           # 0=airplane, 1=automobile, ..., 9=truck

    # Trigger
    'trigger_size':  4,           # Pixels. Try 2 (subtle) to 8 (obvious)
    'trigger_color': (1.0,1.0,0.0),  # RGB in [0,1]. Default: yellow
    'trigger_pos':   'top-left',  # 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right'

    # Training
    'epochs':        30,          # 20 is sufficient; 50 for higher accuracy ceiling
    'batch_size':    128,         # Reduce to 64 on low-VRAM GPUs
    'lr':            0.1,
    'momentum':      0.9,
    'weight_decay':  5e-4,
    'lr_milestones': [15, 25],    # Epochs where LR is multiplied by lr_gamma
    'lr_gamma':      0.1,
}
```

### Suggested experiments

| Experiment | Change | Expected effect |
|-----------|--------|----------------|
| Lower poison rate | `poison_rate = 0.01` | ASR drops, attack harder to detect |
| Larger trigger | `trigger_size = 8` | Higher ASR, more visible |
| Different corner | `trigger_pos = 'bottom-right'` | Same effectiveness, different location |
| Different target | `target_class = 9` (truck) | Attack targets truck class instead |
| Subtle colour | `trigger_color = (0.9, 0.9, 0.1)` | Slightly less saturated yellow |

---

## 📤 Output Files

| File | Description |
|------|-------------|
| `cifar10_samples.png` | One example image per CIFAR-10 class |
| `trigger_samples.png` | 8 clean images (top) vs same images with trigger (bottom) |
| `training_curves.png` | Loss, clean accuracy, and ASR over 30 epochs for both models |
| `prediction_comparison.png` | Per-image prediction grid: 4 rows × 6 images (clean/triggered × clean model/backdoor model) |
| `activation_clustering.png` | PCA scatter plot + cluster size bar chart for the defence step |
| `clean_model.pth` | Trained clean baseline weights (~43 MB) |
| `backdoored_model.pth` | Trained backdoored model weights (~43 MB) |

All outputs are also bundled into `backdoor_outputs.zip` and auto-downloaded at the end of the notebook.

---

## 📦 Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
```

Install with:

```bash
pip install -r requirements.txt
```

All packages are pre-installed on Google Colab — no manual installation needed when using the notebook.

---

## 📚 Academic References

| Year | Paper | Authors | Relevance |
|------|-------|---------|-----------|
| 2017 | [BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain](https://arxiv.org/abs/1708.06733) | Gu, Dolan-Gavitt, Garg | **Original backdoor attack paper — basis of this project** |
| 2019 | [Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering](https://arxiv.org/abs/1811.03728) | Chen, Carvalho, Baracaldo et al. | Defence implemented in Step 11 |
| 2019 | [Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks](https://people.cs.uchicago.edu/~ravenben/publications/pdf/backdoor-sp19.pdf) | Wang, Yao, Shan et al. | Trigger reverse-engineering defence |
| 2019 | [STRIP: A Defence Against Trojan Attacks on Deep Neural Networks](https://arxiv.org/abs/1902.06531) | Gao, Xu, Wang et al. | Inference-time perturbation defence |
| 2016 | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | He, Zhang, Ren, Sun | ResNet architecture used as the victim model |

---

## ⚠️ Ethical Notice

This project is created **strictly for educational and security-research purposes** — to understand, reproduce, and build defenses against backdoor attacks on neural networks.

- Do **not** apply these techniques to production systems, deployed models, or safety-critical applications without explicit authorization
- Do **not** use this to tamper with publicly distributed datasets or pre-trained models
- This work follows responsible disclosure principles: the attack is demonstrated in a fully controlled, isolated research environment

Understanding how attacks work is a prerequisite for building effective defenses. That is the sole intent of this project.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for full terms. Free to use, study, and modify with attribution.

---

<div align="center">

Made for educational purposes · AI Security Research · CIFAR-10 · PyTorch

</div>
