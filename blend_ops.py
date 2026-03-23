from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BlendResult:
    prompt_embed: torch.Tensor
    prompt_mask: torch.Tensor
    selected_indices: list[int]
    selected_weights: list[float]


def _merge_masks(masks: list[torch.Tensor]) -> torch.Tensor:
    if len(masks) == 0:
        raise ValueError("Expected at least one mask.")
    out = masks[0]
    for m in masks[1:]:
        out = torch.maximum(out, m)
    return out


def _token_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-6)


def nlerp_all(embeds: list[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    if len(embeds) == 0:
        raise ValueError("Expected non-empty embeddings.")
    if len(embeds) == 1:
        return embeds[0]
    mix = torch.zeros_like(embeds[0])
    target_norm = torch.zeros_like(_token_norm(embeds[0]))
    for i, e in enumerate(embeds):
        wi = float(weights[i].item())
        mix = mix + wi * e
        target_norm = target_norm + wi * _token_norm(e)
    return (mix / _token_norm(mix)) * target_norm


def slerp_pair(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    t = float(max(0.0, min(1.0, t)))
    an = a / _token_norm(a)
    bn = b / _token_norm(b)
    dot = (an * bn).sum(dim=-1, keepdim=True).clamp(-0.9995, 0.9995)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega).clamp_min(1e-6)
    wa = torch.sin((1.0 - t) * omega) / sin_omega
    wb = torch.sin(t * omega) / sin_omega
    out = wa * a + wb * b
    # Near-collinear fallback to NLERP.
    linear = (1.0 - t) * a + t * b
    use_linear = (torch.abs(dot) > 0.999).expand_as(out)
    return torch.where(use_linear, linear, out)


def blend_prompt_embeddings(
    embeds: list[torch.Tensor],
    masks: list[torch.Tensor],
    weights: torch.Tensor,
    family: str,
) -> BlendResult:
    if len(embeds) == 0:
        raise ValueError("Expected non-empty embeddings.")
    if len(embeds) != len(masks):
        raise ValueError("Embeddings/masks length mismatch.")
    if weights.ndim != 1 or int(weights.shape[0]) != len(embeds):
        raise ValueError(f"weights shape must be [{len(embeds)}], got {tuple(weights.shape)}")

    fam = str(family).strip().lower()
    if len(embeds) == 1:
        return BlendResult(
            prompt_embed=embeds[0],
            prompt_mask=masks[0],
            selected_indices=[0],
            selected_weights=[1.0],
        )

    if fam == "nlerp":
        pe = nlerp_all(embeds, weights)
        keep = [i for i in range(len(embeds)) if float(weights[i]) > 1e-8]
        if not keep:
            keep = [int(torch.argmax(weights).item())]
        pm = _merge_masks([masks[i] for i in keep])
        return BlendResult(
            prompt_embed=pe,
            prompt_mask=pm,
            selected_indices=[int(i) for i in keep],
            selected_weights=[float(weights[i].item()) for i in keep],
        )

    if fam == "slerp":
        top = torch.topk(weights, k=min(2, len(embeds)), largest=True)
        i0 = int(top.indices[0].item())
        if top.indices.shape[0] == 1:
            return BlendResult(
                prompt_embed=embeds[i0],
                prompt_mask=masks[i0],
                selected_indices=[i0],
                selected_weights=[1.0],
            )
        i1 = int(top.indices[1].item())
        w0 = float(weights[i0].item())
        w1 = float(weights[i1].item())
        s = max(w0 + w1, 1e-8)
        t = w1 / s
        pe = slerp_pair(embeds[i0], embeds[i1], t)
        pm = _merge_masks([masks[i0], masks[i1]])
        return BlendResult(
            prompt_embed=pe,
            prompt_mask=pm,
            selected_indices=[i0, i1],
            selected_weights=[w0, w1],
        )

    raise ValueError(f"Unsupported blend family: {family}")
