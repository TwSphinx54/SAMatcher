# SAMatcher: Co-Visibility Modeling with Segment Anything for Robust Feature Matching

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://xupan.top/Projects/samatcher)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-SSSSphinx%2FSAMatcher-yellow)](https://huggingface.co/SSSSphinx/SAMatcher)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](#)

<span class="author-block">
  <a href="https://xupan.top" target="_blank">Xu Pan</a><sup>1</sup>,</span>
<span class="author-block">
  <a href="https://www.researchgate.net/profile/Qiyuan-Ma-2" target="_blank">Qiyuan Ma</a><sup>1</sup>,</span>
<span class="author-block">
  <a href="https://github.com/EATMustard" target="_blank">Jintao Zhang</a><sup>1</sup>,</span>
<span class="author-block">
  <a href="https://www.researchgate.net/scientific-contributions/He-Chen-2315784007" target="_blank">He Chen</a><sup>1,*</sup>,</span>
<span class="author-block">
  <a href="https://jszy.whu.edu.cn/zhengxianwei/zh_CN/index.htm" target="_blank">Xianwei Zheng</a><sup>1,*</sup></span>

<p style="font-size: 0.9em; margin: 10px 0 0 0;">
<sup>1</sup>State Key Laboratory of Information Engineering in Surveying, Mapping and Remote Sensing (LIESMARS), Wuhan University<br/>
IEEE Transactions on Geoscience and Remote Sensing (Under Review)<br/>
<sup>*</sup>Corresponding Authors
</p>

</div>

This repository contains the official implementation of **SAMatcher**, which models co-visibility with Segment Anything to improve robust feature matching in challenging scenes.

- **Project Page**: https://xupan.top/Projects/samatcher
- **Training data / validation weights / model**: https://huggingface.co/SSSSphinx/SAMatcher

---

## Introduction

SAMatcher focuses on robust correspondence estimation under viewpoint, illumination, and texture changes.  
The key idea is to use segmentation priors from SAM to model cross-view co-visibility and guide matching toward geometrically consistent regions.

---

## Repository Structure

```text
.
├── checkpoints
├── configs
├── data
├── dloc
├── main.py
├── outputs
├── pretrained_checkpoint
├── __pycache__
├── pyproject.toml
├── README.md
├── scripts
├── src
├── third_party
├── train.py
├── uv.lock
├── wandb
└── weights
```

---

## Prerequisites

Initialization, pretrained components, and dataset preparation can follow:

- [SAM-HQ](https://github.com/SysCV/sam-hq)
- [OETR](https://github.com/TencentYoutuResearch/ImageMatching-OETR)
- [D2-Net](https://github.com/mihaidusmanu/d2-net)

After downloading training data and validation weights from HuggingFace, place files into the expected folders (e.g., `data/`, `weights/`).

### HuggingFace Model

- Repo: [`SSSSphinx/SAMatcher`](https://huggingface.co/SSSSphinx/SAMatcher)

```bash
# Optional: download to a local directory
huggingface-cli download SSSSphinx/SAMatcher --local-dir weights/SAMatcher
```

### Dependency Notes (from `pyproject.toml` and `uv.lock`)

- Python (project constraint): `>=3.10,<3.13`
- Python (lockfile constraint): `>=3.12`  
  If you install with `uv.lock`, use Python 3.12.
- CUDA/PyTorch from `pyproject.toml`:
  - `torch==2.2.0+cu121`
  - `flash-attn==2.7.4.post1`
  - PyTorch index: `https://download.pytorch.org/whl/cu121`
- `flash-attn` is configured in uv with `no-build-isolation-package`.

Recommended installation:

```bash
# Reproducible install from lockfile
uv sync --frozen

# Or resolve from pyproject.toml
uv sync
```

> If you need strict reproducibility on Python 3.10/3.11, regenerate `uv.lock` under that Python version.

---

## Training

```bash
bash scripts/train.sh
```

---

## Evaluation

```bash
bash scripts/evaluate_megadepth.sh
```

---

## Acknowledgments

We thank the authors of the following open-source projects:

- https://github.com/SysCV/sam-hq
- https://github.com/TencentYoutuResearch/ImageMatching-OETR
- https://github.com/mihaidusmanu/d2-net
