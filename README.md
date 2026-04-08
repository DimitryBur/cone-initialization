Cone-Hierarchy Initialization for Deep Neural Networks

Standard initialization (Xavier, He, Orthogonal) assumes all directions are equal. 
This is suboptimal for deep networks — they start fully "expanded" and can only collapse.

Cone initialization flips the logic:
- Start compressed: narrow angular cone (5-15°) between nearest vectors
- Impose hierarchy: 10% backbone neurons (large weights) + 90% small weights  
- Let geometry unfold: layers gradually increase angular diversity

Experiments on MLP 6x200 show:
- Xavier collapses on complex tasks (ranks drop 4%)
- Cone expands capacity (ranks grow 31% in early layers)
- Backbone neurons act as gradient "highway" (loss reaches 0.1 in warmup)

Repository contains:
- PyTorch implementation
- Comparison with Xavier initialization  
- Raw logs and rank evolution plots

Inspired by information bottleneck and spectral graph theory.
