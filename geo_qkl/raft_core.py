"""RAFT (Regression-Aware Fine-Tuning, Lukasik et al. ICLR 2025) adapted to coordinate OPD.

At coordinate positions, smooths the teacher's probability mass over digit tokens {0..9}
with a distance-aware Gaussian kernel K(d, d') ~ exp(-(d - d')^2 / (2 * sigma^2)) (sigma=1.0),
preserving total digit mass while penalizing numerically distant digits more strongly than adjacent digits.
"""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from v1_core import compute_turn_kl_loss, to_device

_CACHE = {}


def _get_raft_tensors(tokenizer, device, sigma=1.0):
    key = (str(device), float(sigma))
    if key not in _CACHE:
        ids = [tokenizer.encode(str(d), add_special_tokens=False)[0] for d in range(10)]
        digit_ids = torch.tensor(ids, dtype=torch.long, device=device)
        grid = torch.arange(10, dtype=torch.float32, device=device)
        dist2 = (grid.unsqueeze(0) - grid.unsqueeze(1)).pow(2)
        ker = torch.exp(-dist2 / (2.0 * sigma * sigma))
        ker = ker / ker.sum(dim=-1, keepdim=True)
        _CACHE[key] = (digit_ids, ker)
    return _CACHE[key]


def _kl_raft_chunk_fn(s_h, t_h, w, is_coord_chunk, digit_ids, ker, s_lm_head, t_lm_head):
    s_logits = s_lm_head(s_h).float()
    with torch.no_grad():
        t_logits = t_lm_head(t_h).float()
        t_logits = torch.where(torch.isfinite(t_logits), t_logits, torch.zeros_like(t_logits))
        log_pt = F.log_softmax(t_logits, dim=-1)
        pt = log_pt.exp()
        pt_digits = pt.index_select(-1, digit_ids)
        smoothed_digits = torch.matmul(pt_digits, ker)
        pt_raft = pt.clone()
        pt_raft.scatter_(-1, digit_ids.unsqueeze(0).expand(pt.shape[0], -1), smoothed_digits)
        pt_target = torch.where(is_coord_chunk.unsqueeze(-1), pt_raft, pt)
        log_pt_target = (pt_target.clamp_min(1e-12)).log()

    log_ps = F.log_softmax(s_logits, dim=-1)
    ps = log_ps.exp()
    # Reverse KL on non-coord, forward regression-smoothed KL on coord digits (per RAFT formulation)
    rev_kl = (ps * (log_ps - log_pt)).sum(dim=-1)
    raft_kl = (pt_target * (log_pt_target - log_ps)).sum(dim=-1)
    loss_per_tok = torch.where(is_coord_chunk, raft_kl, rev_kl)
    return (loss_per_tok * w).sum()


def compute_turn_kl_loss_raft(
    student_vlm, teacher_vlm, tokenizer, s_prefix_inputs, t_hi_prefix_inputs,
    response_ids, non_coord_mask, coord_weight, device, chunk_size=64, use_recompute=True
):
    L = len(response_ids)
    assert L == len(non_coord_mask) and L > 0
    has_coord = coord_weight > 0.0 and any(m == 0.0 for m in non_coord_mask)
    if not has_coord:
        return compute_turn_kl_loss(
            student_vlm, teacher_vlm, s_prefix_inputs, response_ids, non_coord_mask, device,
            chunk_size=chunk_size, use_recompute=use_recompute
        )

    weights = [1.0 if m > 0.0 else float(coord_weight) for m in non_coord_mask]
    w_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    w_sum = float(w_tensor.sum().item())
    is_coord_all = torch.tensor([m == 0.0 for m in non_coord_mask], dtype=torch.bool, device=device)
    digit_ids, ker = _get_raft_tensors(tokenizer, device)

    def _full(inp):
        d = to_device(inp, device)
        plen = int(d["input_ids"].shape[-1])
        resp = torch.tensor([response_ids], dtype=d["input_ids"].dtype, device=device)
        fd = dict(d)
        fd["input_ids"] = torch.cat([d["input_ids"], resp], dim=-1)
        fd["attention_mask"] = torch.ones_like(fd["input_ids"])
        return fd, plen

    s_fd, s_plen = _full(s_prefix_inputs)
    t_hi_fd, t_hi_plen = _full(t_hi_prefix_inputs)
    with torch.no_grad():
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            t_lo_h = teacher_vlm.model(**s_fd, use_cache=False, return_dict=True).last_hidden_state[0, s_plen - 1: s_plen - 1 + L].detach()
            t_hi_h = teacher_vlm.model(**t_hi_fd, use_cache=False, return_dict=True).last_hidden_state[0, t_hi_plen - 1: t_hi_plen - 1 + L].detach()
            t_hidden = torch.where(is_coord_all.unsqueeze(-1), t_hi_h, t_lo_h)

    with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
        s_out = student_vlm.model(**s_fd, use_cache=False, return_dict=True)
        s_hidden = s_out.last_hidden_state[0, s_plen - 1: s_plen - 1 + L]

    total_kl = None
    for start in range(0, L, chunk_size):
        end = min(L, start + chunk_size)
        sh_chunk = s_hidden[start:end]
        th_chunk = t_hidden[start:end]
        w_chunk = w_tensor[start:end]
        ic_chunk = is_coord_all[start:end]
        if float(w_chunk.sum().item()) == 0.0:
            continue
        if use_recompute and sh_chunk.requires_grad:
            term = grad_checkpoint(
                _kl_raft_chunk_fn, sh_chunk, th_chunk, w_chunk, ic_chunk, digit_ids, ker,
                student_vlm.lm_head, teacher_vlm.lm_head, use_reentrant=False
            )
        else:
            term = _kl_raft_chunk_fn(sh_chunk, th_chunk, w_chunk, ic_chunk, digit_ids, ker,
                                     student_vlm.lm_head, teacher_vlm.lm_head)
        total_kl = term if total_kl is None else total_kl + term
    if total_kl is None:
        return torch.zeros((), dtype=torch.float32, device=device, requires_grad=True), 0.0
    return total_kl, w_sum
