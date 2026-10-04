import torch


@torch.no_grad()
def format_raw_intensity_stats(label, raw_intensity):
    flat = raw_intensity.detach().reshape(-1).float()
    finite = flat[torch.isfinite(flat)]

    if finite.numel() == 0:
        return f"[RAW INTENSITY] {label}: no finite pixels"

    quantiles = torch.quantile(
        finite,
        torch.tensor([0.99, 0.999], device=finite.device),
    )
    p99 = quantiles[0].item()
    p999 = quantiles[1].item()
    max_value = finite.max().item()
    max_to_p999 = max_value / max(p999, 1e-12)

    return (
        f"[RAW INTENSITY] {label}: "
        f"pixels={flat.numel()} finite={finite.numel()} "
        f"p99={p99:.6g} p99.9={p999:.6g} "
        f"max={max_value:.6g} "
        f"max/p99.9={max_to_p999:.6g}"
    )
