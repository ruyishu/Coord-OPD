"""
coord_distill.py: Continuous Coordinate and Joint OPD Loss for Coord-OPD

Implements:
1. Diagonal Gaussian reverse KL for continuous action coordinates:
   KL(q_S || q_T) = sum_d [log(sigma_T / sigma_S) + (sigma_S^2 + (mu_S - mu_T)^2) / (2 * sigma_T^2) - 0.5]
   When sigma_S = sigma_T = fixed sigma (0.15 in NV-CoT):
   KL(q_S || q_T) = 1 / (2 * sigma^2) * ||mu_S - mu_T||^2

2. Joint OPD Loss combining textual reasoning token KL and continuous coordinate action KL:
   L_joint = L_text + lambda_coord * L_coord
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


def compute_gaussian_reverse_kl(
    student_mu: torch.Tensor,
    teacher_mu: torch.Tensor,
    student_sigma: Optional[torch.Tensor] = None,
    teacher_sigma: Optional[torch.Tensor] = None,
    fixed_sigma: float = 0.15,
) -> torch.Tensor:
    """
    Computes Reverse KL between two diagonal Gaussian distributions:
    KL(q_student || q_teacher)
    
    Args:
        student_mu: [N, D] student mean continuous actions
        teacher_mu: [N, D] teacher mean continuous actions (detached)
        student_sigma: [N, D] optional student std dev
        teacher_sigma: [N, D] optional teacher std dev
        fixed_sigma: Used when both models employ fixed std (default 0.15 in NV-CoT)
        
    Returns:
        kl_per_action: [N] sum of KL over coordinate dimensions
    """
    teacher_mu = teacher_mu.detach()
    
    if student_sigma is None and teacher_sigma is None:
        # Fixed std dev case: KL reduces to scaled L2 distance
        diff_sq = (student_mu - teacher_mu).pow(2)
        kl = diff_sq.sum(dim=-1) / (2.0 * (fixed_sigma ** 2))
        return kl
    
    if student_sigma is None:
        student_sigma = torch.full_like(student_mu, fixed_sigma)
    if teacher_sigma is None:
        teacher_sigma = torch.full_like(teacher_mu, fixed_sigma)
        
    teacher_sigma = teacher_sigma.detach()
    
    term1 = torch.log(teacher_sigma / student_sigma.clamp(min=1e-6))
    term2 = (student_sigma.pow(2) + (student_mu - teacher_mu).pow(2)) / (2.0 * teacher_sigma.pow(2).clamp(min=1e-6))
    kl_dim = term1 + term2 - 0.5
    return kl_dim.sum(dim=-1)


def compute_joint_coord_opd_loss(
    student_text_logits: torch.Tensor,
    teacher_text_logits: torch.Tensor,
    text_valid_mask: torch.Tensor,
    student_coord_mu: torch.Tensor,
    teacher_coord_mu: torch.Tensor,
    coord_valid_mask: Optional[torch.Tensor] = None,
    fixed_sigma: float = 0.15,
    lambda_coord: float = 1.0,
    temperature: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Computes joint OPD loss for NV-CoT hybrid text + continuous coordinate actions.
    
    Args:
        student_text_logits: [B, L, V]
        teacher_text_logits: [B, L, V]
        text_valid_mask: [B, L] boolean mask for generated text tokens
        student_coord_mu: [B, num_crops, 4] raw continuous coordinates
        teacher_coord_mu: [B, num_crops, 4]
        coord_valid_mask: [B, num_crops] boolean mask for active crop actions
        fixed_sigma: Standard deviation of coordinate Gaussian policy
        lambda_coord: Relative weight for coordinate distillation
        temperature: Softmax temperature for text tokens
        
    Returns:
        dict containing 'loss', 'loss_text', 'loss_coord', 'num_coords'
    """
    # 1. Text Token Reverse KL
    # Masked teacher positions are neutralized and reductions use torch.where
    # (NaN * 0 = NaN), so non-finite teacher values there cannot leak into the loss.
    text_valid = text_valid_mask.bool()
    teacher_text_logits = teacher_text_logits.detach().masked_fill(~text_valid.unsqueeze(-1), 0.0)
    s_log_p = F.log_softmax(student_text_logits / temperature, dim=-1)
    t_log_p = F.log_softmax(teacher_text_logits / temperature, dim=-1)
    s_p = torch.exp(s_log_p)
    token_kl = torch.sum(s_p * (s_log_p - t_log_p), dim=-1)
    
    text_valid_count = text_valid.sum().clamp(min=1.0)
    loss_text = torch.where(text_valid, token_kl, torch.zeros_like(token_kl)).sum() / text_valid_count
    
    # 2. Continuous Coordinate Reverse KL
    if coord_valid_mask is None:
        coord_valid_mask = torch.ones(student_coord_mu.shape[:2], device=student_coord_mu.device, dtype=torch.bool)
        
    flat_s_mu = student_coord_mu.view(-1, 4)
    flat_t_mu = teacher_coord_mu.view(-1, 4)
    flat_coord_mask = coord_valid_mask.view(-1).bool()
    
    flat_t_mu = flat_t_mu.detach().masked_fill(~flat_coord_mask.unsqueeze(-1), 0.0)
    coord_kl = compute_gaussian_reverse_kl(flat_s_mu, flat_t_mu, fixed_sigma=fixed_sigma)
    
    coord_count = flat_coord_mask.sum().clamp(min=1.0)
    has_coord = flat_coord_mask.any()
    
    loss_coord = torch.where(flat_coord_mask, coord_kl, torch.zeros_like(coord_kl)).sum() / coord_count if has_coord else torch.tensor(0.0, device=student_text_logits.device)
    
    loss_joint = loss_text + (lambda_coord * loss_coord if has_coord else 0.0)
    
    return {
        "loss": loss_joint,
        "loss_text": loss_text,
        "loss_coord": loss_coord,
        "num_coords": flat_coord_mask.sum(),
    }


if __name__ == "__main__":
    print("Running self-check for coord_distill.py...")
    torch.manual_seed(42)
    
    # Test 1: Gaussian KL analytical vs numeric check
    mu_s = torch.tensor([[0.2, 0.3, 0.4, 0.5]], requires_grad=True)
    mu_t = torch.tensor([[0.25, 0.35, 0.45, 0.55]])
    
    kl = compute_gaussian_reverse_kl(mu_s, mu_t, fixed_sigma=0.15)
    # Expected: (4 * 0.05^2) / (2 * 0.15^2) = (4 * 0.0025) / (2 * 0.0225) = 0.01 / 0.045 = 0.2222
    expected_kl = (4 * (0.05 ** 2)) / (2.0 * (0.15 ** 2))
    assert abs(kl.item() - expected_kl) < 1e-4, f"KL calculation mismatch: {kl.item()} vs {expected_kl}"
    print(f"Gaussian KL analytical match: {kl.item():.4f}")
    
    # Test 2: Gradient direction pulls student towards teacher
    kl.backward()
    grad = mu_s.grad
    assert (grad < 0).all(), "Gradient should pull student towards larger teacher values!"
    print("Gradient direction check passed!")

    # Test 2b: general-sigma analytic KL vs numeric quadrature (per-dimension)
    mu_s2 = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    mu_t2 = torch.tensor([[0.3, 0.1, 0.4, 0.2]])
    sig_s2 = torch.tensor([[0.2, 0.15, 0.1, 0.25]])
    sig_t2 = torch.tensor([[0.1, 0.2, 0.3, 0.15]])
    kl_analytic = compute_gaussian_reverse_kl(mu_s2, mu_t2, sig_s2, sig_t2).item()
    kl_numeric = 0.0
    for d in range(4):
        ms, mt = mu_s2[0, d].item(), mu_t2[0, d].item()
        ss, st = sig_s2[0, d].item(), sig_t2[0, d].item()
        xs = torch.linspace(ms - 8 * ss, ms + 8 * ss, 4001)
        log_qs = -0.5 * ((xs - ms) / ss) ** 2 - math.log(ss) - 0.5 * math.log(2 * math.pi)
        log_qt = -0.5 * ((xs - mt) / st) ** 2 - math.log(st) - 0.5 * math.log(2 * math.pi)
        kl_numeric += torch.trapz((log_qs - log_qt) * torch.exp(log_qs), xs).item()
    assert abs(kl_numeric - kl_analytic) < 1e-3, (
        f"Analytic vs numeric KL mismatch: {kl_analytic} vs {kl_numeric}"
    )
    print(f"General-sigma KL analytic vs numeric quadrature: {kl_analytic:.4f} vs {kl_numeric:.4f}")
    
    # Test 3: Joint loss self-check
    B, L, V = 2, 6, 50
    s_text = torch.randn(B, L, V, requires_grad=True)
    t_text = torch.randn(B, L, V)
    t_mask = torch.ones(B, L, dtype=torch.bool)
    
    s_coords = torch.randn(B, 2, 4, requires_grad=True)
    t_coords = torch.randn(B, 2, 4)
    c_mask = torch.tensor([[True, False], [True, True]])
    
    joint_res = compute_joint_coord_opd_loss(s_text, t_text, t_mask, s_coords, t_coords, c_mask, lambda_coord=2.0)
    print("Joint loss output:", {k: (v.item() if torch.is_tensor(v) else v) for k, v in joint_res.items()})
    
    joint_res["loss"].backward()
    assert s_coords.grad is not None and s_text.grad is not None, "Joint backward failed!"

    # Test 4: masked-position NaN guard (non-finite teacher values only where masked)
    g_text = torch.randn(B, L, V, requires_grad=True)
    g_t_text = torch.randn(B, L, V)
    g_text_mask = torch.ones(B, L, dtype=torch.bool)
    g_text_mask[1, 2] = False
    g_t_text[1, 2, :] = float("-inf")
    g_coords = torch.randn(B, 2, 4, requires_grad=True)
    g_t_coords = torch.randn(B, 2, 4)
    g_t_coords[0, 1, :] = float("nan")  # c_mask[0, 1] is False
    g_res = compute_joint_coord_opd_loss(g_text, g_t_text, g_text_mask, g_coords, g_t_coords, c_mask, lambda_coord=2.0)
    g_res["loss"].backward()
    assert torch.isfinite(g_res["loss"]), f"masked non-finite teacher leaked into loss: {g_res['loss']}"
    assert torch.isfinite(g_text.grad).all() and torch.isfinite(g_coords.grad).all()
    assert g_coords.grad[0, 1].abs().max().item() == 0.0 and g_text.grad[1, 2].abs().max().item() == 0.0
    print("Masked-position NaN guard check passed (loss and grads finite).")
    print("All unit tests in coord_distill.py passed successfully!")
