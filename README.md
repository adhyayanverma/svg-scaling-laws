# SVG Generation with GPT-2 & Neural Scaling Laws

A language-model approach to generating valid SVG graphics. GPT-2 models of five sizes (Tiny → XL) are trained on a corpus of SVG icons and emoji, with **μP (Maximal Update Parameterization)** used to transfer the optimal learning rate from a small proxy model to larger ones, and **neural scaling laws** used to characterize how validation loss decreases with model size.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Tokenizer Training](#2-tokenizer-training)
  - [3. Model Training](#3-model-training)
  - [4. μP Learning Rate Transfer](#4-μp-learning-rate-transfer)
  - [5. Scaling Law Analysis](#5-scaling-law-analysis)
  - [6. Generation & Evaluation](#6-generation--evaluation)
- [Model Configurations](#model-configurations)
- [Results](#results)

---

## Overview

Standard hyperparameter tuning becomes expensive as models grow. This project investigates whether **μP** can reliably transfer a learning rate swept on a ~1.4M parameter Tiny model to models up to ~89M parameters — without re-sweeping. It also fits a **power-law scaling curve** (`L(N) = a · N⁻ᵇ`) to the validation losses across model sizes.

The generative task is SVG code: given a start token, the model autoregressively produces syntactically valid SVG markup that can be rendered as an image.

---

## Project Structure

```
.
├── MLProject.ipynb              # Main notebook (all experiments)
├── svg_bpe_tokenizer.json       # Trained BPE tokenizer (4096 vocab)
├── train_ds_filtered/           # Filtered & tokenized training split
├── val_ds_filtered/             # Validation split
├── test_ds_filtered/            # Test split
├── results_small_*/             # Checkpoints for Small model runs
├── results_medium_*/            # Checkpoints for Medium model runs
├── results_large_*/             # Checkpoints for Large model runs
├── results_xl_model_new/        # Checkpoints for XL model (final)
└── generated_svgs/              # SVGs sampled from the trained model
```

> **Google Drive:** When run on Colab, checkpoints and datasets are backed up to `MyDrive/colab_backups/full_runtime_backup`.

---

## Requirements

| Package | Purpose |
|---|---|
| `torch` | Model training |
| `transformers` | GPT-2 architecture & Trainer API |
| `datasets` | HuggingFace dataset loading |
| `tokenizers` | BPE tokenizer training |
| `mup` | Maximal Update Parameterization |
| `cairosvg` | SVG rendering for evaluation |
| `lxml` | XML validity checking |
| `scipy` | Scaling law curve fitting |
| `matplotlib` | Plotting |

---

## Setup

### Local

```bash
pip install torch transformers datasets tokenizers mup cairosvg lxml scipy matplotlib
```

### Google Colab (recommended — A100 GPU)

Open `MLProject.ipynb` in Colab and set the runtime to **A100 GPU**:

```
Runtime → Change runtime type → A100 GPU
```

The first cell installs the two non-standard dependencies automatically:

```python
!pip install cairosvg
!pip install mup
```

Then mount Google Drive if you want checkpoints to persist across sessions:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## Usage

### 1. Data Preparation

Datasets are loaded from HuggingFace Hub and combined:

```python
from datasets import load_dataset, concatenate_datasets

ds_icons  = load_dataset("starvector/svg-icons-simple",  split="train").select_columns(["Svg"])
ds_emojis = load_dataset("starvector/svg-emoji-simple",  split="train").select_columns(["Svg"])
```

SVGs are filtered to a maximum of **1024 tokens** and split 98 / 1 / 1 into train, validation, and test. The filtered splits are cached to disk:

```
train_ds_filtered/
val_ds_filtered/
test_ds_filtered/
```

### 2. Tokenizer Training

A custom **BPE tokenizer** is trained on the SVG corpus with a vocabulary size of 4096. Special tokens (`[PAD]`, `[BOS]`, `[EOS]`, `[UNK]`) are added. The trained tokenizer is saved as `svg_bpe_tokenizer.json` and loaded via:

```python
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(tokenizer_file="svg_bpe_tokenizer.json")
tokenizer.pad_token = "[PAD]"
```

### 3. Model Training

Five GPT-2 variants are trained. See [Model Configurations](#model-configurations) for architecture details. Each is trained using the HuggingFace `Trainer` API with the following base arguments:

```python
TrainingArguments(
    output_dir        = "./results_<size>_<lr>",
    learning_rate     = 0.001,           # swept; see §4
    num_train_epochs  = 1,
    per_device_train_batch_size = 8,
    evaluation_strategy = "steps",
    save_strategy       = "steps",
)
```

To resume from a checkpoint:

```python
trainer.train(resume_from_checkpoint="./results_xl_model_new/checkpoint-7443")
```

### 4. μP Learning Rate Transfer

[μP](https://github.com/microsoft/mup) replaces the standard output projection with a `MuReadout` layer, enabling learning rate transfer across widths. The Tiny model is used as the **base width** proxy:

```python
import mup

base_config   = configs["Tiny"]   # proxy (1.4M params)
target_config = configs["XL"]     # target (89M params)

model = GPT2LMHeadModel(target_config)
model.lm_head = mup.MuReadout(model.lm_head.in_features, model.lm_head.out_features)
mup.set_base_shapes(model, GPT2LMHeadModel(base_config))
```

A learning rate sweep on the Tiny model identified **lr = 0.01** as optimal; this was transferred to all larger models without re-sweeping.

### 5. Scaling Law Analysis

Validation losses across model sizes are fit to a power law:

```python
from scipy.optimize import curve_fit
import numpy as np

def scaling_law(N, a, b):
    return a * np.power(N, -b)

popt, _ = curve_fit(scaling_law, params_N, losses_L, p0=[1.0, 0.05])
```

The fitted curve is plotted on a log-log scale against the empirical losses.

### 6. Generation & Evaluation

SVGs are generated autoregressively from the XL checkpoint:

```python
from transformers import pipeline

gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
output = gen("<svg", max_new_tokens=512, do_sample=True, temperature=0.8)
```

Generated SVGs are evaluated on three criteria:

| Metric | Description |
|---|---|
| **XML validity** | Parsed without errors by `lxml.etree` |
| **Structural validity** | Correct `<svg>` root with properly closed tags |
| **Render validity** | Successfully rasterized by `cairosvg` |

---

## Model Configurations

| Size | Params | `n_embd` | `n_layer` | `n_head` | `n_inner` |
|---|---|---|---|---|---|
| Tiny | ~1.4M | 128 | 4 | 4 | 512 |
| Small | ~3.7M | 192 | 6 | 6 | 768 |
| Medium | ~12.6M | 384 | 6 | 6 | 1536 |
| Large | ~34.1M | 512 | 10 | 8 | 2048 |
| XL | ~89.0M | 768 | 12 | 12 | 3072 |

All models share `vocab_size=4096` and `n_positions=1024`.

---

## Results

**Standard LR sweep (Small model)**

| Learning Rate | Val Loss |
|---|---|
| 0.001 | 0.4256 |
| 0.0005 | 0.4660 |
| 0.0001 | 0.5892 |
| 5e-5 | 0.6795 |
| 1e-5 | 3.1976 |

**μP LR sweep (transferred to all sizes)**

| μP LR | Val Loss |
|---|---|
| 0.02 | 0.3614 |
| **0.01** | **0.3599** ← optimal |
| 0.005 | 0.3710 |
| 0.001 | 0.4188 |
| 0.0005 | 0.4512 |

**Scaling law (μP training)**

| Model | Params | Val Loss |
|---|---|---|
| Tiny | 1.4M | 0.3090 |
| Small | 3.7M | 0.2728 |
| Medium | 12.6M | 0.2659 |
| Large | 34.1M | 0.2596 |
| XL | 89.0M | 0.3191 |

## Generated SVGs
 
Sample SVGs rendered from the trained XL model checkpoint. Each image is the rasterized output of a sequence autoregressively generated from a `<svg` start token.
 
| Sample 1 | Sample 2 | Sample 3 |
|:---:|:---:|:---:|
| ![Generated SVG 1](assets/cmp_T10_s9.png) | ![Generated SVG 2](assets/cmp_T10_s4.png) | ![Generated SVG 3](assets/cmp_T08_s1.png) |
| `cmp_T10_s9` | `cmp_T10_s4` | `cmp_T08_s1` |
