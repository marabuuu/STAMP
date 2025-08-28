import torch
import matplotlib.pyplot as plt
from pathlib import Path


def extract_and_save_attention_(
    *,
    model,
    slide_paths,
    feature_dir: Path,
    device: str,
    attention_weights_dir: Path,
):
    attention_weights_dir.mkdir(exist_ok=True, parents=True)
    for slide_path in slide_paths:
        # Load features and coordinates for this slide
        feats = torch.load(feature_dir / f"{slide_path.stem}_feats.pt", map_location=device)
        coords_um = torch.load(feature_dir / f"{slide_path.stem}_coords.pt", map_location=device)
        slide_id = slide_path.stem

        with torch.no_grad():
            logits, marker_attn, patch_attn = model.vision_transformer(
                bags=feats.unsqueeze(0),
                coords=coords_um.unsqueeze(0),
                mask=torch.zeros(1, len(feats), dtype=torch.bool, device=device),
                return_marker_attention=True,
            )
            torch.save(
                {
                    "marker_attn": marker_attn.cpu(),
                    "patch_attn": patch_attn.cpu(),
                },
                attention_weights_dir / f"{slide_id}.pt"
            )

def aggregate_marker_attention(
        marker_attn, 
        patch_attn, 
        top_k_percent=0.1):
    """
    marker_attn: Tensor [patches, markers, markers]
    patch_attn: Tensor [patches]
    Returns: average marker attention matrix for top-k patches
    """
    n_patches = patch_attn.shape[0]
    top_k = max(1, int(n_patches * top_k_percent))
    top_indices = torch.topk(patch_attn, top_k).indices
    selected_marker_attn = marker_attn[top_indices]  # [top_k, markers, markers]
    avg_marker_attn = selected_marker_attn.mean(dim=0)  # [markers, markers]
    return avg_marker_attn

def visualize_marker_attention(
        avg_marker_attn, 
        output_path, 
        channel_order=None): 
    plt.figure(figsize=(8, 6))
    plt.imshow(avg_marker_attn.cpu().numpy(), cmap="hot")
    plt.colorbar()
    plt.title("Aggregated Marker Attention (Top 10% Patches)")
    if channel_order:
        plt.xticks(range(len(channel_order)), channel_order, rotation=90)
        plt.yticks(range(len(channel_order)), channel_order)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def attention_heatmap_(
        marker_attn, 
        patch_attn, 
        output_path, 
        channel_order=None,  
        top_k_percent=0.1):
    """
    Combines aggregation and visualization of marker attention.
    marker_attn: Tensor [patches, markers, markers]
    patch_attn: Tensor [patches]
    output_path: str or Path, where to save the heatmap
    marker_names: list of str, optional marker/channel names
    top_k_percent: float, percent of top patches to use (default 0.1)
    """

    extract_and_save_attention_(
    model=model,
    slide_paths=slide_paths,
    feature_dir=feature_dir,
    device=device,
    attention_weights_dir=attention_weights_dir
)

    avg_marker_attn = aggregate_marker_attention(marker_attn, patch_attn, top_k_percent)
    visualize_marker_attention(avg_marker_attn, output_path, channel_order)
    return avg_marker_attn