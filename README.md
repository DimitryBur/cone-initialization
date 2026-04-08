# Cone Initialization

**Geometric initialization for deep networks: start narrow, let it unfold.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## Quick idea

Standard Xavier/Orthogonal initialization assumes all directions are equally important. 
Deep networks start fully "expanded" — they can only collapse.

**Cone initialization** flips this:
- Start with a **narrow cone** (small angles between nearest vectors)
- **Hierarchy of amplitudes**: 10% backbone (large weights) + 90% small weights
- Let geometry **unfold** through layers

## Key results (MLP 6×200)

| Metric | Xavier | **Cone** |
|--------|--------|----------|
| Rank evolution (layer 1) | -4% (collapse) | **+31% (growth)** |
| Warmup to loss 0.1 | step 400 | step 250 |
| Training stability | jumps 0.2-0.5 | stable 0.16-0.33 |

## Why it works

1. **Backbone neurons (10%, weight 0.7)** — gradient highway through all layers
2. **Narrow cone (5-15°)** — no noise learning at early stages  
3. **Gradual expansion** — layers specialize: lower learn basics, upper learn nuances

## Run yourself

```bash
git clone https://github.com/yourname/cone-initialization
pip install numpy torch
python cone_init.py      # Cone method
python xavier_baseline.py # Xavier for comparison
