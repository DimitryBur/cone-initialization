# Cone vs Xavier: Comparative Analysis

## Key Numerical Results

| Metric | Xavier | Cone |
|--------|--------|------|
| Best Warmup Loss | 0.0497 (step 400) | 0.1061 (step 250) |
| Best Training Loss | 0.1918 (step 250) | 0.1612 (step 850) |
| Layer 1 Initial Rank | 160.76 | 93.77 |
| Layer 1 Final Rank | 154.58 (-3.8%) | 123.31 (+31.5%) |
| Layer 6 Final Rank | 158.05 (-1.7%) | 45.23 (-64%) |
| Training Stability | Unstable (0.28-0.49) | Stable (0.16-0.33) |

## Core Findings

### 1. Xavier deceives
Fast warmup (loss 0.05) creates false impression of success. When task complexity increases, the model suffers a shock (loss jumps from 0.09 to 1.32). The weights are too orthogonal and uniform — there are no "weak" directions to reorganize.

### 2. Cone evolves
Starts compressed (rank 93), then redistributes capacity: lower layers grow (+31%) learning basic features, upper layers shrink (-64%) forming abstractions. The model doesn't lose capacity — it reallocates it.

### 3. Backbone neurons work
10% of weights at amplitude 0.7 create a gradient highway. Result: loss reaches 0.1 during warmup (step 250). Without them, the model gets stuck.

## Geometric Interpretation

| Property | Xavier | Cone |
|----------|--------|------|
| Initial angles between weights | ~90° (orthogonal) | 5-30° (cone) |
| Angle evolution | Only decreases | Can increase ("unfolding") |
| Evolution directions | One (collapse) | Two (growth lower, compression upper) |

**Metaphor:**
- Xavier — rubber band: stretched from the start, can only contract
- Cone — accordion: compressed at the beginning, can expand where needed

## Practical Recommendations

1. Don't chase fast warmup — Xavier wins early but loses deep
2. Use amplitude hierarchy — 10-20% backbone neurons (weight 0.5-0.8)
3. Start with narrow cone — 5-15° between nearest neighbors
4. Track layer ranks — growth in lower + shrinkage in upper = healthy learning

## Conclusion

Xavier optimization is optimal for the first step only. Cone optimization is optimal for the entire journey. For deep networks, starting compressed with hierarchical amplitudes produces more adaptable representations and prevents representational collapse.
