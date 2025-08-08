# Δ-Fibernet

Implementation of the paper **Ensemble learning of the atrial fiber orientation with physics-informed neural networks** [\[Preprint\]](https://arxiv.org/abs/2410.23388v1), which extends and improves upon the original [Fibernet](https://github.com/fsahli/FiberNet) framework.

Δ-Fibernet introduces an ensemble learning approach to estimate atrial fiber orientation from activation maps, offering enhanced accuracy, uncertainty quantification, and reduced training time compared to traditional PINNs.

---

## Installation

First, clone the repository:

```bash
git clone https://github.com/your-username/delta-fibernet.git
cd delta-fibernet
```

Make sure to install [JAX](https://github.com/jax-ml/jax#installation) according to your system specifications.

Then install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

To get started, check out the example notebooks:

* **[example.ipynb](example.ipynb)**: Demonstrates the full workflow using an ensemble of models for fiber orientation inference.
* **[example\_vanilla.ipynb](example_vanilla.ipynb)**: Shows the baseline (single-model) version without ensemble learning.
