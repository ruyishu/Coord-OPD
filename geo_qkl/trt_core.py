"""TRT-adapted (Huang et al. 2026) for coordinate OPD:
Uses the exact prefix-feasibility set F_t only to verify whether the teacher's top token is spatially
compatible with the teacher reference box, and scales full-vocabulary token reverse KL by teacher
confidence c_t * (1.0 if top_token in F_t else 0.25), without merging feasible outcomes.
"""
import torch
from torch.nn import functional as F
from v1_core import compute_turn_kl_loss, to_device


def compute_turn_kl_loss_trt(
    student_vlm, teacher_vlm, s_prefix_inputs, t_hi_prefix_inputs,
    response_ids, non_coord_mask, feasible_by_pos, coord_weight=1.0,
    device="cuda:0", chunk_size=64, use_recompute=True
):
    L = len(response_ids)
    assert L == len(non_coord_mask) == len(feasible_by_pos) and L > 0
    if not any(m == 0.0 for m in non_coord_mask):
        return compute_turn_kl_loss(
            student_vlm, teacher_vlm, s_prefix_inputs, response_ids,
            non_coord_mask, device, chunk_size=chunk_size, use_recompute=use_recompute
        )

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
    is_coord = torch.tensor([m == 0.0 for m in non_coord_mask], dtype=torch.bool, device=device)

    with torch.no_grad():
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            t_lo_h = teacher_vlm.model(**s_fd, use_cache=False, return_dict=True).last_hidden_state[0, s_plen - 1: s_plen - 1 + L].detach()
            t_hi_h = teacher_vlm.model(**t_hi_fd, use_cache=False, return_dict=True).last_hidden_state[0, t_hi_plen - 1: t_hi_plen - 1 + L].detach()
            t_hidden = torch.where(is_coord.unsqueeze(-1), t_hi_h, t_lo_h)

    with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
        s_out = student_vlm.model(**s_fd, use_cache=False, return_dict=True)
        s_hidden = s_out.last_hidden_state[0, s_plen - 1: s_plen - 1 + L]

    total_kl = None
    w_sum = 0.0
    for i in range(L):
        s_logits = student_vlm.lm_head(s_hidden[i:i + 1]).float()[0]
        with torch.no_grad():
            t_logits = teacher_vlm.lm_head(t_hidden[i:i + 1]).float()[0]
            t_logits = torch.where(torch.isfinite(t_logits), t_logits, torch.zeros_like(t_logits))
            log_pt = F.log_softmax(t_logits, dim=-1)
            pt = log_pt.exp()
            top_p, top_id = pt.max(dim=-1)
        log_ps = F.log_softmax(s_logits, dim=-1)
        ps = log_ps.exp()
        tok_kl = (ps * (log_ps - log_pt)).sum(dim=-1)

        if non_coord_mask[i] > 0.0:
            wi = 1.0
        else:
            fset = feasible_by_pos[i]
            in_feas = bool(fset) and (int(top_id.item()) in fset)
            gate = 1.0 if in_feas else 0.25
            wi = float(coord_weight) * gate * float(top_p.item())
        w_sum += wi
        term = tok_kl * wi
        total_kl = term if total_kl is None else total_kl + term

    if total_kl is None:
        return torch.zeros((), dtype=torch.float32, device=device, requires_grad=True), 0.0
    return total_kl, max(w_sum, 1e-6)
