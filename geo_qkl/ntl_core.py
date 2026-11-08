"""NTL (Number Token Loss, Zausinger et al. ICML 2025) adapted to view-decoupled coordinate OPD.

At coordinate positions, adds the exact 1D 1-Wasserstein distance (earth mover's distance on {0..9})
between the normalized student and teacher digit distributions to the token KL loss.
"""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from v1_core import compute_turn_kl_loss, to_device

_DIGIT_IDS = None


def _get_digit_ids(tokenizer, device):
    global _DIGIT_IDS
    if _DIGIT_IDS is None or _DIGIT_IDS.device != torch.device(device):
        ids = [tokenizer.encode(str(d), add_special_tokens=False)[0] for d in range(10)]
        _DIGIT_IDS = torch.tensor(ids, dtype=torch.long, device=device)
    return _DIGIT_IDS


def _kl_ntl_chunk_fn(s_h, t_h, w, is_coord_chunk, digit_ids, s_lm_head, t_lm_head):
    s_logits = s_lm_head(s_h).float()
    with torch.no_grad():
        t_logits = t_lm_head(t_h).float()
        t_logits = torch.where(torch.isfinite(t_logits), t_logits, torch.zeros_like(t_logits))
        log_pt = F.log_softmax(t_logits, dim=-1)
        pt_digits = log_pt.index_select(-1, digit_ids).exp()
        pt_digits = pt_digits / (pt_digits.sum(dim=-1, keepdim=True) + 1e-8)
        cdf_t = torch.cumsum(pt_digits, dim=-1)[..., :-1]

    log_ps = F.log_softmax(s_logits, dim=-1)
    ps = log_ps.exp()
    token_kl = (ps * (log_ps - log_pt)).sum(dim=-1)

    ps_digits = ps.index_select(-1, digit_ids)
    ps_digits = ps_digits / (ps_digits.sum(dim=-1, keepdim=True) + 1e-8)
    cdf_s = torch.cumsum(ps_digits, dim=-1)[..., :-1]
    w1_digits = (cdf_s - cdf_t).abs().sum(dim=-1)

    loss_per_tok = token_kl + is_coord_chunk.float() * w1_digits
    return (loss_per_tok * w).sum()


def compute_turn_kl_loss_ntl(
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
    digit_ids = _get_digit_ids(tokenizer, device)

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
                _kl_ntl_chunk_fn, sh_chunk, th_chunk, w_chunk, ic_chunk, digit_ids,
                student_vlm.lm_head, teacher_vlm.lm_head, use_reentrant=False
            )
        else:
            term = _kl_ntl_chunk_fn(sh_chunk, th_chunk, w_chunk, ic_chunk, digit_ids,
                                    student_vlm.lm_head, teacher_vlm.lm_head)
        total_kl = term if total_kl is None else total_kl + term
    if total_kl is None:
        return torch.zeros((), dtype=torch.float32, device=device, requires_grad=True), 0.0
    return total_kl, w_sum
