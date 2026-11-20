"""Anchor-line selection: choose where on the line from the start weights to the trained weights to submit.

Why
---
After training, the start weights theta_0 (the model the task handed out) and the trained weights theta
span a line
    theta(a) = theta_0 + a * (theta - theta_0).
a < 1 is WiSE-FT interpolation (Wortsman et al. 2022, arXiv 2109.01903); a > 1 is the extrapolation
ExPO uses on aligned models (Zheng et al. 2024, arXiv 2404.16792). Which a scores best depends on how
far the run got: an under-trained instruct run gains from a little extrapolation, an over-confident
preference run from interpolation back towards its reference.

DPO is the extreme case. The validator scores a DPO submission by the TRL DPO loss (beta 0.1) against
the task model it handed out (validator/scoring/tasks.py: original_model = augmented_model_id or
model_id). That loss is convex in the per-pair reward margins, so a policy whose margins are right on
average but large on the pairs it gets wrong scores worse than the reference itself (ln 2). On the
16 Nov knockout task 81f83573 both real entrants scored above ln 2; locally the trained model scored
0.93 and every point of its line except a = 0 was worse still.

What
----
The rows that choose a (the "anchor") are carved out of the training rows before training starts, so
the run never sees them (the trainer's own dev split is not usable: the final dev pass trains on it).
The score on the anchor is the validator's own loss for the task type:
  * DPO     : mean -log sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r))), summed completion log-probs,
              prompt/completion split as validator dpo.py _tokenize_dpo_pair; the reference is theta_0.
              theta(0) is the reference, so its loss is exactly ln 2 and needs no forward pass.
  * instruct: mean over examples of the per-example mean cross-entropy on the output tokens
              (validator instruct_text.py: eval_loss is a flat mean of per-example token means), on the
              trainer's own tokenisation of the anchor rows.
a is moved off 1 only when the anchor loss improves by MIN_GAIN (relative). A LoRA submission moves
along the line exactly by scaling its lora_B tensors. Any failure, or too little time left before the
deadline, leaves the submission untouched.
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence, Tuple

BETA = 0.1                      # validator/evaluation/constants.py BETA_DPO
MAX_TOKENS = 2048               # DPO: prompt truncated from the left, completion kept whole
TOKEN_BUDGET = 8192             # padded tokens per forward batch
SAFETY_SECS = 75.0              # never start an evaluation with less than this left (plus one eval)
MAX_PARAMS = 9.5e9              # the line keeps theta_0 and theta - theta_0 resident; skip bigger models
# DPO: may the search land on a = 0, i.e. submit the reference weights unchanged? On a fragile (augmented)
# task that is the only point that beats the trained model (81f83573: 0.693 vs 0.905), but it ships the
# untrained model, so it stays off until explicitly decided; the search then starts at DPO_MIN_A.
ALLOW_REFERENCE = False
DPO_MIN_A = 0.25

MODES = {
    # grid evaluated first, refinement step, clamp, relative gain needed to leave a = 1
    "dpo": dict(grid=(1.0, 0.5, 1.5), step=0.25, lo=0.0, hi=2.0, min_gain=0.004),
    "instruct": dict(grid=(1.0, 1.15, 0.85), step=0.075, lo=0.5, hi=1.5, min_gain=0.0015),
}


def reserve_minutes(hours: float) -> float:
    """Training minutes handed to the DPO blend (the trainer already ends ~5 min early)."""
    return 1.0


def anchor_size(n_rows: int) -> int:
    """~3% of the rows, at least 64 and at most 256; 0 when the dataset is too small to spare them."""
    if n_rows < 800:
        return 0
    return int(min(256, max(64, n_rows // 33)))


def anchor_path(task_id: str, kind: str, folder: str = "datasets") -> str:
    return os.path.join(folder, f"{kind}_anchor_{task_id}.json")


def carve_anchor(train_items: list, task_id: str, kind: str, folder: str = "datasets", seed: int = 1729) -> list:
    """Move a random anchor sample out of the training rows (written to disk) and return the rest."""
    k = anchor_size(len(train_items))
    if k <= 0:
        return train_items
    rng = random.Random(seed)
    idx = set(rng.sample(range(len(train_items)), k))
    anchor = [r for i, r in enumerate(train_items) if i in idx]
    rest = [r for i, r in enumerate(train_items) if i not in idx]
    with open(anchor_path(task_id, kind, folder), "w") as f:
        json.dump(anchor, f, ensure_ascii=False)
    print(f"[sn56][anchor] {k} {kind} rows held out for the line search ({len(rest)} stay in training)", flush=True)
    return rest


# ----------------------------------------------------------------------------------------------- forward
def _batches(lengths: List[int]):
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and lengths[order[j]] * (j - i + 1) <= TOKEN_BUDGET:
            j += 1
        j = max(j, i + 1)
        yield order[i:j]
        i = j


def _token_logps(model, batch_ids: List[List[int]], device):
    """Per-row log p(token_t | <t) for t >= 1, row-wise so no full fp32 log-softmax is materialised."""
    import torch

    mx = max(len(s) for s in batch_ids)
    x = torch.zeros((len(batch_ids), mx), dtype=torch.long)
    att = torch.zeros((len(batch_ids), mx), dtype=torch.long)
    for b, ids in enumerate(batch_ids):
        x[b, : len(ids)] = torch.tensor(ids)
        att[b, : len(ids)] = 1
    x, att = x.to(device), att.to(device)
    out = []
    with torch.no_grad():
        logits = model(input_ids=x, attention_mask=att).logits[:, :-1]
        tgt = x[:, 1:].unsqueeze(-1)
        for r in range(logits.shape[0]):
            lr = logits[r].float()
            out.append((lr.gather(-1, tgt[r]).squeeze(-1) - torch.logsumexp(lr, dim=-1)).cpu())
            del lr
        del logits
    return out


def _completion_logps(model, seqs: List[Tuple[List[int], int]], device) -> List[float]:
    """DPO: summed completion log-probs; seqs = (ids, prompt_len)."""
    res = [0.0] * len(seqs)
    for idx in _batches([len(s[0]) for s in seqs]):
        lps = _token_logps(model, [seqs[i][0] for i in idx], device)
        for i, lp in zip(idx, lps):
            ids, plen = seqs[i]
            res[i] = lp[max(plen - 1, 0): len(ids) - 1].sum().item()
    return res


def _example_ce(model, samples: List[Tuple[List[int], List[int]]], device) -> List[float]:
    """Instruct: per-example mean CE over supervised tokens; samples = (input_ids, labels)."""
    import torch

    res = [float("nan")] * len(samples)
    for idx in _batches([len(s[0]) for s in samples]):
        lps = _token_logps(model, [samples[i][0] for i in idx], device)
        for i, lp in zip(idx, lps):
            ids, labels = samples[i]
            mask = torch.tensor([l != -100 for l in labels[1:len(ids)]], dtype=torch.bool)
            if bool(mask.any()):
                res[i] = -float(lp[: len(ids) - 1][mask].mean())
    return res


# ----------------------------------------------------------------------------------------------- line
class _FullLine:
    """theta(a) for a full-weight submission: theta_0 kept in its dtype, theta - theta_0 in bf16."""

    def __init__(self, model, ref_sd):
        import torch
        self.model = model
        self.base, self.delta = {}, {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in ref_sd and ref_sd[n].shape == p.shape and p.is_floating_point():
                    r = ref_sd[n].to(device=p.device)
                    d = p.detach().float() - r.float()
                    if not torch.any(d):
                        continue
                    self.base[n] = r
                    self.delta[n] = d.to(torch.bfloat16)
        self.a = 1.0

    def set(self, a: float):
        import torch
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self.base:
                    p.copy_((self.base[n].float() + a * self.delta[n].float()).to(p.dtype))
        self.a = a


class _LoraLine:
    """theta(a) for a PEFT LoRA submission: every adapter scaling is multiplied by a."""

    def __init__(self, model):
        self.model = model
        self.orig = {}
        for name, mod in model.named_modules():
            sc = getattr(mod, "scaling", None)
            if isinstance(sc, dict) and sc:
                self.orig[name] = dict(sc)
        self.a = 1.0

    def set(self, a: float):
        mods = dict(self.model.named_modules())
        for name, sc in self.orig.items():
            for k, v in sc.items():
                mods[name].scaling[k] = v * a
        self.a = a


# ----------------------------------------------------------------------------------------------- search
def search(evaluate: Callable[[float], float], cfg: dict, known: Dict[float, float], time_ok: Callable[[], bool],
           log: Callable = print) -> Tuple[float, Dict[float, float]]:
    """Coarse grid, then one refinement step either side of the best point. evaluate(a) -> anchor loss."""
    seen: Dict[float, float] = dict(known)
    for a in cfg["grid"]:
        if a in seen or not time_ok():
            continue
        seen[a] = evaluate(a)
        log(f"[sn56][anchor] a={a:.3f} loss={seen[a]:.5f}")
    if 1.0 not in seen:
        return 1.0, seen
    best = min(seen, key=seen.get)
    for a in (best - cfg["step"], best + cfg["step"]):
        a = round(min(cfg["hi"], max(cfg["lo"], a)), 4)
        if a in seen or not time_ok():
            continue
        seen[a] = evaluate(a)
        log(f"[sn56][anchor] a={a:.3f} loss={seen[a]:.5f}")
    best = min(seen, key=seen.get)
    # Walk on while the best point is the outermost one evaluated (the optimum lies further out).
    for _ in range(4):
        pts = sorted(seen)
        if best not in (pts[0], pts[-1]) or not time_ok():
            break
        a = round(best + (cfg["step"] if best == pts[-1] else -cfg["step"]), 4)
        if a > cfg["hi"] or a < cfg["lo"] or a in seen:
            break
        seen[a] = evaluate(a)
        log(f"[sn56][anchor] a={a:.3f} loss={seen[a]:.5f}")
        if seen[a] >= seen[best]:
            break
        best = a
    best = min(seen, key=seen.get)
    if seen[best] > seen[1.0] * (1.0 - cfg["min_gain"]):
        return 1.0, seen
    return best, seen


# ----------------------------------------------------------------------------------------------- data
def _dpo_pairs(tok, rows: list, dataset_type: dict):
    """Pairs tokenised like validator dpo.py _tokenize_dpo_pair (EOS appended, completion = suffix)."""
    fp = dataset_type.get("field_prompt") or "prompt"
    fc = dataset_type.get("field_chosen") or "chosen"
    fr = dataset_type.get("field_rejected") or "rejected"
    eos = tok.eos_token
    out = []
    for row in rows:
        p, c, r = row.get(fp), row.get(fc), row.get(fr)
        if not (isinstance(p, str) and isinstance(c, str) and isinstance(r, str)) or c == r:
            continue
        if eos is not None:
            c = c if c.endswith(eos) else c + eos
            r = r if r.endswith(eos) else r + eos
        pi = tok(text=p)["input_ids"]
        seqs = []
        for comp in (tok(text=p + c)["input_ids"][len(pi):], tok(text=p + r)["input_ids"][len(pi):]):
            keep = max(0, MAX_TOKENS - len(comp))
            pp = pi[len(pi) - keep:] if keep else []
            ids = (pp + comp)[:MAX_TOKENS]
            seqs.append((ids, len(pp)) if comp and len(ids) > len(pp) else None)
        if all(seqs):
            out.append(tuple(seqs))
    return out


def _dpo_loss(pc, pr, rc, rr) -> float:
    tot = 0.0
    for a, b, c, d in zip(pc, pr, rc, rr):
        z = BETA * ((a - c) - (b - d))
        tot += math.log1p(math.exp(-z)) if z > -30 else -z
    return tot / max(1, len(pc))


def _instruct_samples(path: str) -> List[Tuple[List[int], List[int]]]:
    rows = json.load(open(path))
    out = []
    for r in rows:
        ids, labels = r.get("input_ids"), r.get("labels")
        if ids and labels and len(ids) == len(labels) and any(l != -100 for l in labels[1:]):
            out.append((list(ids), list(labels)))
    return out


# ----------------------------------------------------------------------------------------------- main
def _secs_left(deadline: Optional[datetime]) -> float:
    if deadline is None:
        return float("inf")
    return (deadline - datetime.now(timezone.utc)).total_seconds()


def _n_params(path: str) -> float:
    try:
        idx = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(idx):
            return json.load(open(idx)).get("metadata", {}).get("total_size", 0) / 2.0
        return sum(os.path.getsize(f) for f in glob.glob(os.path.join(path, "*.safetensors"))) / 2.0
    except Exception:
        return 0.0


def run(mode: str, train_request: dict, ref_path: str, submission_dir: str, deadline: Optional[datetime],
        log: Callable = print) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODES[mode]
    t0 = time.time()
    task_id = train_request["task_id"]
    if _n_params(ref_path) > MAX_PARAMS:
        return {"status": "model_too_large"}
    if mode == "dpo":
        apath = anchor_path(task_id, "dpo")
        if not os.path.exists(apath):
            return {"status": "no_anchor"}
        tok = AutoTokenizer.from_pretrained(ref_path)
        units = _dpo_pairs(tok, json.load(open(apath)), train_request.get("dataset_type") or {})
    else:
        apath = anchor_path(task_id, "instruct_tokenized")
        if not os.path.exists(apath):
            return {"status": "no_anchor"}
        units = _instruct_samples(apath)
    if len(units) < 32:
        return {"status": "few_rows", "n": len(units)}

    is_lora = os.path.exists(os.path.join(submission_dir, "adapter_config.json"))
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kw = dict(torch_dtype=torch.bfloat16)
    ref_scores = None
    if is_lora:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(ref_path, **kw).to(dev).eval()
        model = PeftModel.from_pretrained(base, submission_dir).to(dev).eval()
        line = _LoraLine(model)
        if not line.orig:
            return {"status": "lora_without_scaling"}
        if mode == "dpo":
            line.set(0.0)
            ref_scores = (_completion_logps(model, [u[0] for u in units], dev),
                          _completion_logps(model, [u[1] for u in units], dev))
    else:
        ref = AutoModelForCausalLM.from_pretrained(ref_path, **kw).to(dev).eval()
        if mode == "dpo":
            ref_scores = (_completion_logps(ref, [u[0] for u in units], dev),
                          _completion_logps(ref, [u[1] for u in units], dev))
        ref_sd = {k: v.detach() for k, v in ref.state_dict().items()}
        model = AutoModelForCausalLM.from_pretrained(submission_dir, **kw).to(dev).eval()
        line = _FullLine(model, ref_sd)
        del ref, ref_sd
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    per_eval = [time.time() - t0]

    def evaluate(a: float) -> float:
        t = time.time()
        line.set(a)
        if mode == "dpo":
            loss = _dpo_loss(_completion_logps(model, [u[0] for u in units], dev),
                             _completion_logps(model, [u[1] for u in units], dev), *ref_scores)
        else:
            v = [x for x in _example_ce(model, units, dev) if x == x]
            loss = sum(v) / max(1, len(v))
        per_eval[0] = time.time() - t
        return loss

    def time_ok() -> bool:
        return _secs_left(deadline) > per_eval[0] * 1.3 + SAFETY_SECS

    known = {0.0: math.log(2.0)} if (mode == "dpo" and ALLOW_REFERENCE) else {}
    if mode == "dpo" and not ALLOW_REFERENCE:
        cfg = dict(cfg, lo=max(cfg["lo"], DPO_MIN_A))
    best, seen = search(evaluate, cfg, known, time_ok, log)
    res = {"mode": mode, "status": "kept", "a": best, "losses": {str(k): round(v, 6) for k, v in sorted(seen.items())},
           "n": len(units), "lora": is_lora}
    if best == 1.0:
        log(f"[sn56][anchor] keeping a=1 {res['losses']}")
        return res
    write_secs = 20.0 + _n_params(submission_dir) * 2.0 / 150e6   # ~150 MB/s safetensors write
    if _secs_left(deadline) < write_secs + 30:
        res["status"] = "no_time_to_write"
        return res
    _write(model, line, best, submission_dir, is_lora)
    res["status"] = "rewritten"
    log(f"[sn56][anchor] submission rewritten at a={best:.3f} ({time.time() - t0:.0f}s) {res['losses']}")
    return res


def _write(model, line, a: float, submission_dir: str, is_lora: bool):
    """Write theta(a) next to the submission, then swap the weight files in (config/tokenizer untouched)."""
    from safetensors.torch import load_file, save_file

    tmp = submission_dir.rstrip("/") + ".anchor_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    if is_lora:
        for f in glob.glob(os.path.join(submission_dir, "adapter_model*.safetensors")):
            sd = load_file(f)
            out = {k: (v.float() * a).to(v.dtype).contiguous() if "lora_B" in k else v for k, v in sd.items()}
            save_file(out, os.path.join(tmp, os.path.basename(f)), metadata={"format": "pt"})
    else:
        line.set(a)
        live = {n: p for n, p in model.named_parameters()}
        for f in sorted(glob.glob(os.path.join(submission_dir, "*.safetensors"))):
            sd = load_file(f)
            out = {}
            for k, v in sd.items():
                p = live.get(k)
                out[k] = p.detach().to("cpu", dtype=v.dtype).contiguous() if (p is not None and p.shape == v.shape) else v
            save_file(out, os.path.join(tmp, os.path.basename(f)), metadata={"format": "pt"})
    for f in os.listdir(tmp):
        os.replace(os.path.join(tmp, f), os.path.join(submission_dir, f))
    shutil.rmtree(tmp, ignore_errors=True)
    with open(os.path.join(submission_dir, "anchor_blend.json"), "w") as fh:
        json.dump({"a": a}, fh)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: anchor_blend.py MODE(dpo|instruct) REQUEST_JSON REF_PATH SUBMISSION_DIR END_TIME('%Y-%m-%d %H:%M:%S' UTC)"""
    argv = list(argv if argv is not None else sys.argv[1:])
    mode = argv[0]
    req = json.load(open(argv[1]))
    tr = req.get("train_request", req)
    deadline = datetime.strptime(argv[4], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc) if len(argv) > 4 else None
    try:
        res = run(mode, tr, argv[2], argv[3], deadline)
    except Exception as e:  # never break a submission
        res = {"mode": mode, "status": "error", "error": f"{type(e).__name__}: {e}"}
    print(f"[sn56][anchor] result {json.dumps(res)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
