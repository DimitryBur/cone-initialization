# Cone Initialization

**Geometric initialization for deep networks: start narrow, let it unfold.**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## Quick idea

Standard Orthogonal and He (Kaiming) initializations assume all directions are equally important. Deep networks start fully "expanded" — they often collapse or fail to learn.

**Cone initialization** flips the logic:
- Start with a **narrow cone** (small angles between nearest vectors)
- **Hierarchy of amplitudes**: 10% backbone (large weights) + 90% small weights
- Let geometry **unfold** through layers

## Key results (12 layers, 1000×1000)

| Method | Final Loss | Best Loss | Layer 1 Rank (final) | Layer 12 Rank (final) | Trained |
|--------|------------|-----------|----------------------|-----------------------|---------|
| **Cone (ours)** | **0.17** | **0.17** | ~600 | ~130 | ✅ Yes |
| He (Kaiming) | ~0.69 | ~0.39 | ~794 | ~768 | ❌ No |
| Orthogonal | ~0.68 | ~0.37 | ~964 | ~961 | ❌ No |

**Only Cone successfully trained a 12-layer feed-forward network without skip connections.**

## Why it works

1. **Backbone neurons (10%, weight ~0.7)** — gradient highway through all layers
2. **Narrow cone (5-15°)** — prevents overloading with noise at early stages
3. **Gradual expansion** — lower layers grow rank (learn features), upper layers compress (form abstractions)

## Run yourself

```bash
git clone https://github.com/DimitryBur/cone-initialization
pip install numpy torch
python cone1000.py      # Cone method (12 layers, 1000x1000)
python he1000.py        # He (Kaiming) baseline
python ortog1000.py     # Orthogonal baseline
