# HRLFS: Comprehend, Divide, and Conquer — Feature Subspace Exploration via Multi-Agent Hierarchical Reinforcement Learning

This repository contains the official implementation of our ACM TKDD paper:

> **Comprehend, Divide, and Conquer: Feature Subspace Exploration via Multi-Agent Hierarchical Reinforcement Learning**
> Weiliang Zhang, Xiaohan Huang, Yi Du, Ziyue Qiao, Qingqing Long, Zhen Meng, Yuanchun Zhou, Meng Xiao
> *ACM Transactions on Knowledge Discovery from Data (TKDD), 2026*

## Overview

Feature selection aims to preprocess the target dataset, find an optimal and most streamlined feature subset, and enhance the downstream machine learning task. Existing reinforcement-learning-based feature selection methods follow an inefficient one-agent-per-feature paradigm and struggle with the inherent complexities of real-world datasets.

**HRLFS** addresses these challenges with a *comprehend-divide-and-conquer* paradigm:

1. **Comprehend** — A hybrid feature state extractor combines a Large Language Model (LLM), which comprehends the semantic meaning of each feature's metadata, with Gaussian Mixture Models (GMM), which capture each feature's mathematical characteristics.
2. **Divide** — Based on the hybrid states, features are grouped via hierarchical (Ward-linkage agglomerative) clustering to build a cluster tree.
3. **Conquer** — A multi-agent hierarchical reinforcement learning framework is constructed that mirrors the cluster hierarchy: internal agents make coarse-grained decisions on entire feature clusters (pruning whole subtrees), while leaf agents make fine-grained select/drop decisions on individual features.

Compared to one-feature-one-agent RL approaches, HRLFS improves downstream ML performance while reducing the average per-iteration decision complexity from *O(N)* to *O(log N)*, accelerating total run time by reducing the number of agents involved.

## Repository Structure

```
.
├── HRLFS.py              # Main entry: the HRLFS method
├── model/                # Baseline feature selection methods
│   ├── KBest.py          #   K-Best
│   ├── LASSONet.py       #   LassoNet
│   ├── MCDM.py           #   MCDM
│   ├── GAINS.py          #   GAINS
│   ├── SAFS.py           #   SAFS
│   ├── CompFS.py         #   CompFS
│   ├── SARLFS.py         #   Single-agent RL feature selection
│   └── MARLFS.py         #   Multi-agent RL feature selection
├── utils/
│   ├── preprocess.py     # Data loading, standardization, LLM-embedding concat
│   ├── networks.py       # Actor-Critic policy networks
│   └── evaluate.py       # Downstream task evaluation (Random Forest, etc.)
├── data/                 # Datasets (spam_base included as a demo)
│   ├── spam_base.hdf
│   ├── spam_base_description.txt
│   └── spam_base_embedding.npy
└── requirements.txt
```

## Datasets

The sources of all 21 datasets are given in **Section 5.1 (Dataset Description)** of the paper, covering classification and regression tasks:

`spectf`, `svmguide3`, `german_credit`, `credit_default`, `spam_base`, `megawatt1`, `ionosphere`, `openml_586`, `openml_589`, `openml_607`, `openml_616`, `openml_618`, `openml_620`, `openml_637`, `mice_protein`, `coil-20`, `mnist`, `otto`, `jannis`, `cao`, `han`

To better demonstrate our model, this repository already includes the `spam_base` dataset together with its feature descriptions and LLM embeddings. All datasets (with pre-computed LLM embeddings) can be downloaded from [Dropbox](https://www.dropbox.com/scl/fi/q4w7nthjpzpa05x326c0o/HARLFS-code.zip?rlkey=xw2pw6nd4ngigbaoharhu7ko1&st=qlg6syjo&dl=0). After downloading, copy the files into `./data`.

## Requirements

Python 3.11 is recommended. Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Quick Start

Run HRLFS with the `--dataset` argument specifying the dataset to use:

```bash
python HRLFS.py --dataset spam_base
```

Baseline methods are provided under `./model` and can be run in the same way, e.g.:

```bash
python model/MARLFS.py --dataset spam_base
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{zhang2026comprehend,
  title={Comprehend, Divide, and Conquer: Feature Subspace Exploration via Multi-Agent Hierarchical Reinforcement Learning},
  author={Zhang, Weiliang and Huang, Xiaohan and Du, Yi and Qiao, Ziyue and Long, Qingqing and Meng, Zhen and Zhou, Yuanchun and Xiao, Meng},
  journal={ACM Transactions on Knowledge Discovery from Data},
  year={2026}
}
```

## Contact

For questions about the paper or code, please contact the corresponding author: **Meng Xiao** (shaow@cnic.cn).
