# Method notes (deltas over the published 16 Nov text boss)

Each change below runs during training. The files named here contain the code.

## 1. Anchor-line selection after training (`scripts/anchor_blend.py`)
- Before training, 3% of the training rows (64 to 256) are held out as "anchor" rows. The split happens in `tokenize_instruct.py` and `tokenize_dpo.py`, and training never sees these rows.
- After training, the trained weights θ and the start weights θ0 span the line θ(a) = θ0 + a·(θ − θ0). We evaluate the validator's own loss for the task type at several values of a, on the anchor rows:
  - DPO: the DPO loss against the task model, beta 0.1, scored per pair as in validator `dpo.py`.
  - Instruct and chat: the mean over examples of each example's token-mean CE.
- The submission is rewritten at the best a, but only if that improves the anchor loss by a set margin.
- A LoRA submission moves along this line exactly, by scaling `lora_B`.
- For DPO, a = 0 (the task model itself, loss exactly ln 2) is also a candidate. It wins only if every trained point a > 0 scores worse on the anchor rows ("do no harm").
- Sources:
  - WiSE-FT, arXiv 2109.01903: interpolation.
  - ExPO, arXiv 2404.16792: extrapolation of DPO/RLHF models.
  - Model soups, arXiv 2203.05482.
  - TR-DPO, arXiv 2404.09656: over-optimisation is drift away from the reference.

## 2. No weight-scaling restore on DPO; restore kept on GRPO (`scripts/text_trainer.py`)
- DPO is scored against the augmented task model (validator `scoring/tasks.py`: original_model = augmented_model_id or model_id). Restoring the scaling moves the policy away from that reference. Measured on 81f83573: DPO loss 17.38 with the restore vs 0.93 without.
- GRPO is scored on rewards with beta = 0, so there is no reference term. Restoring before GRPO is therefore kept.
  - Measured on Llama-3.2-1B ×0.65: untrained reward -1.91 → -1.35.
  - After a 1 h run on H100: -1.18 / -1.21 with the restore vs -1.61 / -1.72 without.

## 3. LR-search bug fix (`scripts/lr_search.py`, `_make_prune_fn`)
- The bug: the divergence pruner compared each trial's rolling loss with a "best rolling loss so far", and that reference was updated from step 1. At that point the rolling window held only 1-4 batches.
- Every LR trial sees the same probe batches. So one easy batch early in the sequence set a low reference, and every later full-window mean exceeded it by 1.75×.
- As a result every S1 learning rate was pruned as "diverged", and the search fell back to its coldest rung.
- Observed on the 0.25 subsample of a15b846a: lr 5.1e-7 chosen instead of about 3e-6, and test loss +5.6%.
- The fix: the reference comes only from full rolling windows. Non-finite losses are still pruned at once.
