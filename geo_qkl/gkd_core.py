"""GKD-style forward-KL variants of the view-decoupled OPD losses (2026-11-07).

Only the divergence changes: D_KL(p_T || p_S) instead of D_KL(p_S || p_T).
Rollout, views, masks, LoRA and the optimizer are inherited unchanged from the HS pipeline.
"""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from v1_core import compute_turn_kl_loss, to_device


def _kl_chunk_fn_fwd(s_h: torch.Tensor, t_h: torch.Tensor, w: torch.Tensor, s_lm_head, t_lm_head) -> torch.Tensor:
    """Weighted sum of token-level forward KL D_KL(p_T || p_S) on one chunk of hidden states."""
    s_logits = s_lm_head(s_h).float()
    with torch.no_grad():
        t_logits = t_lm_head(t_h).float()
        t_logits = torch.where(torch.isfinite(t_logits), t_logits, torch.zeros_like(t_logits))
        log_pt = F.log_softmax(t_logits, dim=-1)
        pt = log_pt.exp()
    log_ps = F.log_softmax(s_logits, dim=-1)
    token_kl = (pt * (log_pt - log_ps)).sum(dim=-1)
    return (token_kl * w).sum()


def compute_turn_kl_loss_fwd(
    student_vlm,
    teacher_vlm,
    prefix_inputs,
    response_ids,
    token_weights,
    device: str,
    chunk_size: int = 64,
    use_recompute: bool = True,
):
    """Single-view full-vocab forward KL over one turn's assistant tokens."""
    L = len(response_ids)
    assert L == len(token_weights) and L > 0
    w_tensor = torch.tensor(token_weights, dtype=torch.float32, device=device)
    w_sum = float(w_tensor.sum().item())
    if w_sum <= 0.0:
        return torch.zeros((), dtype=torch.float32, device=device, requires_grad=True), 0.0

    data = to_device(prefix_inputs, device)
    prefix_len = data["input_ids"].shape[-1]
    resp = torch.tensor([response_ids], dtype=data["input_ids"].dtype, device=device)
    full_data = dict(data)
    full_data["input_ids"] = torch.cat([data["input_ids"], resp], dim=-1)
    full_data["attention_mask"] = torch.ones_like(full_data["input_ids"])

    with torch.no_grad():
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            t_out = teacher_vlm.model(**full_data, use_cache=False, return_dict=True)
            t_hidden = t_out.last_hidden_state[0, prefix_len - 1: prefix_len - 1 + L].detach()

    with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
        s_out = student_vlm.model(**full_data, use_cache=False, return_dict=True)
        s_hidden = s_out.last_hidden_state[0, prefix_len - 1: prefix_len - 1 + L]

    total_kl = None
    for start in range(0, L, chunk_size):
        end = min(L, start + chunk_size)
        sh_chunk = s_hidden[start:end]
        th_chunk = t_hidden[start:end]
        w_chunk = w_tensor[start:end]
        if float(w_chunk.sum().item()) == 0.0:
            continue
        if use_recompute and sh_chunk.requires_grad:
            term = grad_checkpoint(
                _kl_chunk_fn_fwd, sh_chunk, th_chunk, w_chunk, student_vlm.lm_head, teacher_vlm.lm_head, use_reentrant=False
            )
        else:
            term = _kl_chunk_fn_fwd(sh_chunk, th_chunk, w_chunk, student_vlm.lm_head, teacher_vlm.lm_head)
        total_kl = term if total_kl is None else total_kl + term
    if total_kl is None:
        return torch.zeros((), dtype=torch.float32, device=device, requires_grad=True), 0.0
    return total_kl, w_sum


def compute_turn_kl_loss_hs_fwd(
    student_vlm,
    teacher_vlm,
    s_prefix_inputs,
    t_hi_prefix_inputs,
    response_ids,
    non_coord_mask,
    coord_weight: float,
    device: str,
    chunk_size: int = 64,
    use_recompute: bool = True,
):
    """View-decoupled forward KL: 0.25MP teacher on text tokens, 1MP teacher on coordinate tokens."""
    L = len(response_ids)
    assert L == len(non_coord_mask) and L > 0
    has_coord = coord_weight > 0.0 and any(m == 0.0 for m in non_coord_mask)
    if not has_coord:
        return compute_turn_kl_loss_fwd(
            student_vlm, teacher_vlm, s_prefix_inputs, response_ids, non_coord_mask, device,
            chunk_size=chunk_size, use_recompute=use_recompute
        )

    weights = [1.0 if m > 0.0 else float(coord_weight) for m in non_coord_mask]
    w_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    w_sum = float(w_tensor.sum().item())
    if w_sum <= 0.0:
        return torch.zeros((), dtype=torch.float32, device=device, requires_grad=True), 0.0

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
            is_coord = torch.tensor([m == 0.0 for m in non_coord_mask], dtype=torch.bool, device=device).unsqueeze(-1)
            t_hidden = torch.where(is_coord, t_hi_h, t_lo_h)

    with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
        s_out = student_vlm.model(**s_fd, use_cache=False, return_dict=True)
        s_hidden = s_out.last_hidden_state[0, s_plen - 1: s_plen - 1 + L]

    total_kl = None
    for start in range(0, L, chunk_size):
        end = min(L, start + chunk_size)
        sh_chunk = s_hidden[start:end]
        th_chunk = t_hidden[start:end]
        w_chunk = w_tensor[start:end]
        if float(w_chunk.sum().item()) == 0.0:
            continue
        if use_recompute and sh_chunk.requires_grad:
            term = grad_checkpoint(
                _kl_chunk_fn_fwd, sh_chunk, th_chunk, w_chunk, student_vlm.lm_head, teacher_vlm.lm_head, use_reentrant=False
            )
        else:
            term = _kl_chunk_fn_fwd(sh_chunk, th_chunk, w_chunk, student_vlm.lm_head, teacher_vlm.lm_head)
        total_kl = term if total_kl is None else total_kl + term
    if total_kl is None:
        return torch.zeros((), dtype=torch.float32, device=device, requires_grad=True), 0.0
    return total_kl, w_sum
