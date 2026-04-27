"""Debug: compare original frame_scores with ablation-recomputed ones."""
import json
import numpy as np

inf = json.load(open('archive/8s-deepseek/inference/Bike Polo/inference_result.json'))
fs = json.load(open('archive/8s-deepseek/inference/Bike Polo/frame_scores.json'))

# Original frame scores
orig = np.array(fs['frame_scores'])
print(f'Original frame_scores: n={len(orig)}, min={orig.min():.4f}, max={orig.max():.4f}')

# Recompute from segment scores
picks = inf['frame_score_picks']
segments = inf['segment_scores']
recomp = np.zeros(len(picks))
for seg_item in segments:
    start = seg_item['start_frame']
    end = seg_item['end_frame']
    score = float(seg_item['final_score'])
    for i, p in enumerate(picks):
        if start <= p < end:
            recomp[i] = score

print(f'Recomputed from segment: n={len(recomp)}, min={recomp.min():.4f}, max={recomp.max():.4f}')
print(f'Match: {np.allclose(orig, recomp)}')
print(f'Max diff: {np.abs(orig - recomp).max():.6f}')
print(f'Diffs at positions: {np.where(np.abs(orig - recomp) > 1e-6)[0].tolist()[:10]}')

# Check if orig is normalized version
lo, hi = recomp.min(), recomp.max()
norm_recomp = (recomp - lo) / (hi - lo) if hi > lo else recomp
print(f'\nNormalized recomp range: {norm_recomp.min():.4f} to {norm_recomp.max():.4f}')
print(f'Match after norm: {np.allclose(orig, norm_recomp)}')
print(f'Max diff after norm: {np.abs(orig - norm_recomp).max():.6f}')
