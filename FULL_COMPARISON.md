# Full Comparison: Initialization Methods for 12-Layer MLP (1000×1000)

## Experimental Setup
- **Architecture:** MLP, 12 layers, 1000×1000
- **Training:** Sliding window (4 layers) + final polish
- **Methods compared:** Cone, He (Kaiming), Orthogonal

---

## Summary Table

| Metric | Cone | He (Kaiming) | Orthogonal |
|--------|------|--------------|------------|
| Best Warmup Loss | 0.656 (step 200) | 0.389 (step 200) | 0.368 (step 200) |
| Minimum Final Loss | **0.1729** (step 600) | ~0.69 | ~0.68 |
| Final Loss Range | 0.17–0.64 | 0.66–0.70 | 0.68–0.70 |
| Loss Decreased After Warmup? | ✅ Yes | ❌ No | ❌ No |

---

## Rank Evolution

| Layer | Cone | He (Kaiming) | Orthogonal |
|-------|------|--------------|------------|
| **Layer 1 (initial)** | 532.5 | 804.8 | 999.9 |
| **Layer 1 (final)** | 602.8 (+13%) | 794.6 (-1%) | 963.9 (-4%) |
| **Layer 6 (initial)** | 735.8 | 805.1 | 1000.0 |
| **Layer 6 (final)** | 116.5 (-84%) | 796.1 (-1%) | 988.5 (-1%) |
| **Layer 12 (initial)** | 784.9 | 805.2 | 1000.0 |
| **Layer 12 (final)** | 127.7 (-84%) | 768.0 (-5%) | 961.1 (-4%) |

---

## Detailed Dynamics

### Cone

| Phase | Loss | Rank Changes |
|-------|------|--------------|
| Warmup | 0.696 → 0.656 | L1: 532 → 556 |
| Sliding window | 0.565 → 0.409 | L6: 735 → 107 |
| Final polish | **0.1729** (min) | L1: ~600, L6: ~116, L12: ~127 |

### He (Kaiming)

| Phase | Loss | Rank Changes |
|-------|------|--------------|
| Warmup | 0.707 → 0.389 | L1: 804 → 798 |
| Sliding window | ~0.69–0.70 | Minimal change |
| Final polish | ~0.69 | L1: ~794, L12: ~768 |

### Orthogonal

| Phase | Loss | Rank Changes |
|-------|------|--------------|
| Warmup | 0.712 → 0.368 | L1: 999 → 974 |
| Sliding window | ~0.69–0.72 | Minimal change |
| Final polish | ~0.68 | L1: ~963, L12: ~961 |

---

## Observations

| Method | Warmup Performance | Final Training | Rank Evolution |
|--------|--------------------|----------------|----------------|
| Cone | Moderate (0.656) | Loss dropped to 0.17 | Growth in lower layers, compression in upper layers |
| He (Kaiming) | Good (0.389) | Loss stuck at ~0.69 | Minimal change, ranks near maximum |
| Orthogonal | Good (0.368) | Loss stuck at ~0.68 | Minimal change, ranks near maximum |

---

## Files

| File | Method |
|------|--------|
| `cone1000.py` | Cone |
| `he1000.py` | He (Kaiming) |
| `ortog1000.py` | Orthogonal |

---

## Raw Data Reference

Complete logs available in the repository:
- Cone: Loss 0.1729 achieved at step 600
- He: Loss remained ~0.69 throughout final polish
- Orthogonal: Loss remained ~0.68 throughout final polish
