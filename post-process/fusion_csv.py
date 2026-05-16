"""Ensemble fusion of four model predictions for Kaputt competition.

Models:
  - dino3l   : predictions_div3l-v1.csv
  - dinov3h  : predictions_div3h-v2.csv
  - dinov2v3 : predictions_div2l-v3.csv
  - dinov2v4 : predictions_div2l-v4.csv

Generates multiple fusion strategies, ranked by theoretical quality.
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import rankdata

# ── Load predictions ──────────────────────────────────────────────────
FUSION_DIR = os.path.dirname(os.path.abspath(__file__))

f1 = pd.read_csv(os.path.join(FUSION_DIR, 'prediction-dinov3l-v1.csv'))
f2 = pd.read_csv(os.path.join(FUSION_DIR, 'prediction_dinoh-v2.csv'))
f3 = pd.read_csv(os.path.join(FUSION_DIR, 'predictions_div2l-v3.csv'))
f4 = pd.read_csv(os.path.join(FUSION_DIR, 'predictions_div2l-v4.csv'))

# ========== 对齐capture_id顺序 ==========
f2 = f2.set_index('capture_id').loc[f1['capture_id']].reset_index()
f3 = f3.set_index('capture_id').loc[f1['capture_id']].reset_index()
f4 = f4.set_index('capture_id').loc[f1['capture_id']].reset_index()
# ==============================================

# 验证ID一致性
assert (f1['capture_id'].values == f2['capture_id'].values).all()
assert (f1['capture_id'].values == f3['capture_id'].values).all()
assert (f1['capture_id'].values == f4['capture_id'].values).all()

capture_ids = f1['capture_id'].values
p1 = f1['pred'].values.astype(np.float64)
p2 = f2['pred'].values.astype(np.float64)
p3 = f3['pred'].values.astype(np.float64)
p4 = f4['pred'].values.astype(np.float64)
n = len(p1)

# ── Rank-normalized scores ────────────────────────────────────────────
rank_1 = rankdata(p1) / n
rank_2 = rankdata(p2) / n
rank_3 = rankdata(p3) / n
rank_4 = rankdata(p4) / n

# ── Define fusion strategies ─────────────────────────────────────────
strategies = {}

# --- Group 1: Probability-space fusion ---
strategies['prob_avg'] = (p1 + p2 + p3 + p4) / 4
strategies['prob_w40_20_20_20'] = 0.40 * p1 + 0.20 * p2 + 0.20 * p3 + 0.20 * p4
strategies['prob_w35_25_20_20'] = 0.35 * p1 + 0.25 * p2 + 0.20 * p3 + 0.20 * p4
strategies['geo_mean'] = np.power(p1 * p2 * p3 * p4, 1.0 / 4)
strategies['power_mean_2'] = np.sqrt((p1**2 + p2**2 + p3**2 + p4**2) / 4)

# --- Group 2: Rank-space fusion (recommended — normalizes calibration) ---
strategies['rank_avg'] = (rank_1 + rank_2 + rank_3 + rank_4) / 4
strategies['rank_w40_20_20_20'] = 0.40 * rank_1 + 0.20 * rank_2 + 0.20 * rank_3 + 0.20 * rank_4
strategies['rank_w35_25_20_20'] = 0.35 * rank_1 + 0.25 * rank_2 + 0.20 * rank_3 + 0.20 * rank_4

# --- Group 3: Hybrid (rank-average then rescale to original distribution) ---
rank_fused = (rank_1 + rank_2 + rank_3 + rank_4) / 4
sort_idx = np.argsort(rank_fused)
ref_sorted = np.sort(p1)
hybrid = np.empty(n)
hybrid[sort_idx] = ref_sorted
strategies['rank_avg_rescaled'] = hybrid

rank_fused_w = 0.40 * rank_1 + 0.20 * rank_2 + 0.20 * rank_3 + 0.20 * rank_4
sort_idx_w = np.argsort(rank_fused_w)
hybrid_w = np.empty(n)
hybrid_w[sort_idx_w] = ref_sorted
strategies['rank_w40_rescaled'] = hybrid_w

# ── Save all strategies ──────────────────────────────────────────────
output_dir = os.path.join(FUSION_DIR, 'fused_outputs')
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("  Ensemble Fusion Results")
print("=" * 70)

for name, preds in strategies.items():
    out_df = pd.DataFrame({
        'capture_id': capture_ids,
        'pred': preds,
    })
    csv_path = os.path.join(output_dir, f'fused_{name}.csv')
    out_df.to_csv(csv_path, index=False)

    n_high = int((preds >= 0.5).sum())
    print(f'  {name:25s}  mean={preds.mean():.4f}  '
          f'std={preds.std():.4f}  n>0.5={n_high:5d}  -> {csv_path}')

# ── Recommendation ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  Recommendations (priority order for submission)")
print("=" * 70)
print("""
  1. rank_avg            — Equal-weight rank fusion of all 4 models.
                           Normalizes different calibrations. Best for
                           AP which is a ranking metric.

  2. rank_w40_20_20_20   — Weighted rank fusion, giving 40% weight to the
                           best model (dino3l). Preserves the ranking
                           advantage of the best model while adding
                           diversity from the other three.

  3. rank_avg_rescaled   — Rank-average ordering but mapped back to the
                           best model's probability distribution. Use
                           if the submission system needs realistic
                           probability values.

  4. prob_avg            — Classic probability averaging. Simple and
                           robust baseline.

  5. geo_mean            — Geometric mean. Penalises disagreement more
                           than arithmetic mean (if one model says 0.1,
                           the fused score is strongly pulled down).

  Note: The rank-based strategies (1-3) are theoretically optimal for AP
  because AP only depends on the ordering of predictions, not their
  absolute values. Rank normalisation removes calibration artifacts.
""")

if __name__ == '__main__':
    pass
