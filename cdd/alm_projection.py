"""ALM (Augmented Lagrangian Method) projection for Constrained Discrete Diffusion."""

import torch
import torch.nn.functional as F
from typing import Callable, Optional


def project_to_simplex(y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Euclidean projection onto probability simplex."""
    if dim == -1:
        return project_to_simplex(y, dim=-2).squeeze(-1)

    batch_size, seq_len, vocab_size = y.shape
    y_flat = y.view(-1, vocab_size)

    sorted_y, _ = torch.sort(y_flat, dim=-1, descending=True)
    cumsum_y = torch.cumsum(sorted_y, dim=-1)
    indicators = sorted_y - (cumsum_y - 1) / (torch.arange(1, vocab_size + 1, device=y.device, dtype=torch.float32))
    rho = (indicators > 0).sum(dim=-1) - 1
    rho = rho.clamp(min=0)

    theta = (cumsum_y.gather(-1, rho.unsqueeze(-1)) - 1) / (rho.float() + 1).unsqueeze(-1)
    theta = theta.squeeze(-1)

    proj_y = F.relu(y_flat - theta.unsqueeze(-1))
    proj_y = proj_y / proj_y.sum(dim=-1, keepdim=True)

    return proj_y.view(batch_size, seq_len, vocab_size)


def _sample_gumbel(logits: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Sample Gumbel(0,1) noise for Gumbel-Softmax relaxation."""
    uniform = torch.rand_like(logits)
    return -torch.log(-torch.log(uniform + eps) + eps)


def alm_projection(
    q: torch.Tensor,
    delta_g_fn: Callable[[torch.Tensor], torch.Tensor],
    λ_init: float = 0.0,
    μ_init: float = 1.0,
    ρ: float = 2.0,
    μ_max: float = 1000.0,
    inner_iter_max: int = 100,
    outer_iter_max: Optional[int] = None,
    η: float = 1,
    eps: float = 1e-3,
    gumbel_temperature: float = 1.0,
    verbose: bool = False,
) -> torch.Tensor:
    """CDD ALM projection.

    Projects distribution q onto constraint manifold where delta_g_fn(y) <= 0.
    delta_g_fn returns the violation (positive = constraint violated).
    Per-sample independent convergence: each sample has its own λ and μ, and converges individually.
    Converged samples receive no further updates.

    The projection optimizes in logit space. The input probability distribution q
    is converted to logits as y = log(q), so that softmax(y) ≈ q at initialization.

    The KL term is computed w.r.t. the deterministic projected distribution
    softmax(y), while the constraint term uses a Gumbel-Softmax relaxation
    softmax((log_softmax(y) + gumbel_noise) / gumbel_temperature)
    as in the paper's Eq. 6.
    """
    batch_size, seq_len, vocab_size = q.shape

    if verbose and batch_size != 1:
        raise ValueError(f"verbose=True is only supported when batch_size=1, got batch_size={batch_size}")

    # Initialize logits so that softmax(y) ≈ q.
    y = torch.log(q.clamp(min=1e-10)).detach().clone().requires_grad_(True)

    λ = torch.full((batch_size,), λ_init, device=q.device, dtype=torch.float32)
    μ = torch.full((batch_size,), μ_init, device=q.device, dtype=torch.float32)
    optimizer = torch.optim.Adam([y], lr=η)

    converged_mask = torch.zeros(batch_size, dtype=torch.bool, device=q.device)
    outer_iter = 0
    exit_reason = "unknown"

    while True:
        # Sample Gumbel noise once per outer iteration and reuse it for all
        # inner steps, so the inner-loop objective stays stable.
        with torch.no_grad():
            gumbel_noise = _sample_gumbel(y)

            log_p_y = F.log_softmax(y, dim=-1)
            y_soft_eval = F.softmax((log_p_y + gumbel_noise) / gumbel_temperature, dim=-1)
            violation = delta_g_fn(y_soft_eval)

        converged_mask = converged_mask | (violation < eps)
        not_converged = ~converged_mask
        num_not_converged = not_converged.sum().item()

        if verbose:
            print(f"-- outer_iter_{outer_iter}_start λ={λ[0].item():.6f} μ={μ[0].item():.6f} viol={violation[0].item():.6f}")

        if num_not_converged == 0:
            exit_reason = "all_converged"
            break

        if outer_iter_max is not None and outer_iter >= outer_iter_max:
            exit_reason = f"outer_iter_max={outer_iter_max}"
            break

        for inner_idx in range(inner_iter_max):
            optimizer.zero_grad()

            # KL term uses the deterministic projected distribution.
            y_log_soft = F.log_softmax(y, dim=-1)
            # Constraint term uses the Gumbel-Softmax relaxation (paper Eq. 6).
            y_soft = F.softmax((y_log_soft + gumbel_noise) / gumbel_temperature, dim=-1)
            violation_inner = delta_g_fn(y_soft)

            L_kl = - (q * y_log_soft).sum(dim=[-2, -1])
            L_viol = λ * violation_inner + (μ / 2) * violation_inner ** 2
            L_total = L_kl + L_viol

            if verbose and inner_idx == 0:
                D_kl = F.kl_div(y_log_soft, q, reduction='none').sum(dim=[-2, -1])
                print(f"---- inner_iter_start: kl={D_kl[0].item():.3f} viol={violation_inner[0].item():.3f} L={L_total.item():.3f}")

            if not_converged.any():
                grad_outputs = not_converged.float()
                L_total.backward(gradient=grad_outputs)
                if converged_mask.any() and y.grad is not None:
                    y.grad[converged_mask] = 0.0
                optimizer.step()
            else:
                break

            if verbose and inner_idx == inner_iter_max - 1:
                with torch.no_grad():
                    y_log_soft = F.log_softmax(y, dim=-1)
                    y_soft = F.softmax((y_log_soft + gumbel_noise) / gumbel_temperature, dim=-1)
                    D_kl = F.kl_div(y_log_soft, q, reduction='none').sum(dim=[-2, -1])
                    violation_inner = delta_g_fn(y_soft)
                print(f"---- inner_iter_end: kl={D_kl[0].item():.3f} viol={violation_inner[0].item():.3f} L={L_total.item():.3f}")

        with torch.no_grad():
            log_p_y = F.log_softmax(y, dim=-1)
            y_final_soft = F.softmax((log_p_y + gumbel_noise) / gumbel_temperature, dim=-1)
            violation = delta_g_fn(y_final_soft)

            if not_converged.any():
                λ[not_converged] = F.relu(λ[not_converged] + μ[not_converged] * violation[not_converged])
                μ[not_converged] = torch.min(μ[not_converged] * ρ, torch.full_like(μ[not_converged], μ_max))

        outer_iter += 1

    if verbose:
        print(f"exit reason={exit_reason} λ={λ[0].item():.6f} μ={μ[0].item():.6f} violation={violation[0].item():.6f}")

    return F.softmax(y, dim=-1)


def project_to_constraint(
    q: torch.Tensor,
    constraint_fn: Callable[[torch.Tensor], torch.Tensor],
    λ_init: float = 0.0,
    μ_init: float = 1.0,
    μ_max: float = 1000,
    inner_iter_max: int = 100,
    outer_iter_max: Optional[int] = None,
    η: float = 1,
    eps: int = 0.1,
    gumbel_temperature: float = 1.0,
    verbose: bool = False,
) -> torch.Tensor:
    """High-level wrapper for ALM projection."""
    return alm_projection(
        q,
        constraint_fn,
        λ_init=λ_init,
        μ_init=μ_init,
        μ_max=μ_max,
        inner_iter_max=inner_iter_max,
        outer_iter_max=outer_iter_max,
        η=η,
        eps=eps,
        gumbel_temperature=gumbel_temperature,
        verbose=verbose,
    )