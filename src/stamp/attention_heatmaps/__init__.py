import h5py
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Iterable
from stamp.preprocessing import supported_extensions

def aggregate_marker_attention(
    marker_attn, 
    patch_attn=None, 
    top_k_percent=0.1):
    """
    marker_attn: Tensor [patches, markers, markers]
    patch_attn: Tensor [patches]
    Returns: average marker attention matrix for top-k patches
    """
    if patch_attn is not None:
        n_patches = patch_attn.shape[0]
        top_k = max(1, int(n_patches * top_k_percent))
        top_indices = torch.topk(patch_attn, top_k).indices
        selected_marker_attn = marker_attn[top_indices]  # [top_k, markers, markers]
        avg_marker_attn = selected_marker_attn.mean(dim=0)  # [markers, markers]
    else:
        # If patch_attn is not provided, average over all tiles
        avg_marker_attn = marker_attn.mean(dim=0)
    return avg_marker_attn

def load_multiplex_features(feature_dir: Path, channel_order: list[str]) -> torch.Tensor:
    """
    Loads and stacks features from h5 files for all markers in channel_order, case-insensitive.
    Returns: stacked_features [markers, tiles, features]
    """
    features_per_marker = []
    all_files = list(feature_dir.glob("*.h5"))
    all_files_lower = {f.name.lower(): f for f in all_files}

    for marker in channel_order:
        marker_lower = marker.lower()
        # Find the first file that contains the marker name (case-insensitive)
        found_file = None
        for f in all_files:
            if marker_lower in f.name.lower():
                found_file = f
                break
        if not found_file:
            if features_per_marker:
                zero_feats = torch.zeros_like(features_per_marker[0])
            else:
                # If this is the first marker and missing, you need to decide on a default shape
                # For example, look at another file in the directory:
                example_file = next(iter(all_files), None)
                if example_file is not None:
                    with h5py.File(example_file, "r") as h5:
                        feats_obj = h5["feats"]
                        if isinstance(feats_obj, h5py.Dataset):
                            shape = feats_obj.shape
                        else:
                            raise RuntimeError(f'"feats" in {example_file} is not a dataset (found {type(feats_obj)}).')
                    zero_feats = torch.zeros(shape, dtype=torch.float32)
                else:
                    raise RuntimeError("No marker files found in feature_dir to infer shape.")
            features_per_marker.append(zero_feats)
        else:
            with h5py.File(found_file, "r") as h5:
                feats_obj = h5["feats"]
                if isinstance(feats_obj, h5py.Dataset):
                    feats = torch.from_numpy(feats_obj[:]).float()
                else:
                    raise RuntimeError(f'"feats" in {found_file} is not a dataset (found {type(feats_obj)}).')
                features_per_marker.append(feats)
    stacked_features = torch.stack(features_per_marker)
    return stacked_features

def load_coords_from_h5(feature_dir: Path, channel_order: list[str]) -> np.ndarray:
    """
    Loads coordinates from the first available marker h5 file in channel_order.
    Returns: coords [tiles, 2]
    """
    all_files = list(feature_dir.glob("*.h5"))
    for marker in channel_order:
        marker_lower = marker.lower()
        for f in all_files:
            if marker_lower in f.name.lower():
                with h5py.File(f, "r") as h5:
                    coords_obj = h5["coords"]
                    if isinstance(coords_obj, h5py.Dataset):
                        coords = coords_obj[:]
                    else:
                        raise RuntimeError(f'"coords" in {f} is not a dataset (found {type(coords_obj)}).')
                return coords
    raise FileNotFoundError("No marker h5 file with coords found in feature_dir.")

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
        *,
        checkpoint_path: Path,
        feature_dir: Path,
        slide_paths: Iterable[Path],
        wsi_dir: Path,
        masson_trichrome_path: Path,
        device: str,
        output_path: Path,
        channel_order: list[str],  
        top_k_percent=0.1,
    ):
    from stamp.modeling.lightning_model import LitVisionTransformer

    model = LitVisionTransformer.load_from_checkpoint(checkpoint_path).to(device).eval()

    # Fallback: if slide_paths is None or empty, use all slides in wsi_dir
    if not slide_paths:
        slide_paths = [
            p for ext in supported_extensions for p in Path(wsi_dir).glob(f"**/*{ext}")
        ]


    # --- Only run once per sample, not per marker file ---
    # Assume all marker files in feature_dir belong to the same sample
    # Use the first slide_path for naming output files
    slide_path = next(iter(slide_paths)) if slide_paths else Path(feature_dir)

    # Load all features and coords once
    stacked_features = load_multiplex_features(feature_dir, channel_order).to(device)
    stacked_features = stacked_features.permute(0, 2, 1).unsqueeze(0)  # [1, markers, features, tiles]
    coords_um = load_coords_from_h5(feature_dir, channel_order)
    if hasattr(coords_um, 'dtype') and hasattr(coords_um, 'shape'):
        coords_um = np.array(coords_um)
    else:
        coords_um = np.array(coords_um[:])
    coords_um = torch.from_numpy(coords_um).float().to(device)

    # --- Debug prints before model call ---
    print("[DEBUG] stacked_features.shape:", stacked_features.shape)
    print("[DEBUG] coords_um.shape:", coords_um.shape)
    print("[DEBUG] Model class:", type(model))
    print("[DEBUG] Model use_marker_attention:", getattr(model, 'use_marker_attention', 'N/A'))

    with torch.no_grad():
        logits, marker_attn, patch_attn = model.vision_transformer(
            bags=stacked_features,
            coords=coords_um.unsqueeze(0),
            mask=None,
            return_marker_attention=True,
        )
        # --- Debug prints after model call ---
        print("[DEBUG] marker_attn:", type(marker_attn), getattr(marker_attn, 'shape', None))
        print("[DEBUG] patch_attn:", type(patch_attn), getattr(patch_attn, 'shape', None))

        # Remove batch dimension if present
        marker_attn = marker_attn.squeeze(0)
        if patch_attn is not None:
            patch_attn = patch_attn.squeeze(0)
        # Aggregate marker attention (use patch_attn if available, else average all)
        avg_marker_attn = aggregate_marker_attention(marker_attn, patch_attn, top_k_percent)
        # Save aggregate marker attention PNG (one per sample)
        slide_output_path = output_path / f"{slide_path.stem}_marker_attention.png"
        visualize_marker_attention(avg_marker_attn, slide_output_path, channel_order)

        # Per-tile channel argmax heatmap (classic gapless heatmap layout)
        marker_scores = marker_attn.sum(dim=1)  # [tiles, markers]
        most_influential_marker = marker_scores.argmax(dim=1).cpu().numpy()  # [tiles]

        # Reshape to 2D grid for classic heatmap
        # Try to infer grid shape from coords (assume regular grid)
        coords_np = coords_um.cpu().numpy() if torch.is_tensor(coords_um) else coords_um
        # Find unique x and y, sort them
        x_unique = np.unique(coords_np[:, 0])
        y_unique = np.unique(coords_np[:, 1])
        x_unique.sort()
        y_unique.sort()
        # Map each coord to its grid index
        x_idx = np.searchsorted(x_unique, coords_np[:, 0])
        y_idx = np.searchsorted(y_unique, coords_np[:, 1])
        grid = np.full((len(y_unique), len(x_unique)), -1, dtype=int)
        grid[y_idx, x_idx] = most_influential_marker

        # Create colormap and legend
        from matplotlib.colors import ListedColormap
        cmap = plt.get_cmap('tab20')
        if isinstance(cmap, ListedColormap):
            # cmap.colors is usually a numpy array of shape (N, 3) or (N, 4)
            cmap_colors = cmap.colors
            def to_float_tuple(c):
                # Convert to tuple of floats, only if length 3 or 4
                if isinstance(c, (list, tuple, np.ndarray)) and len(c) in (3, 4):
                    return tuple(float(x) for x in c)
                raise TypeError(f"Color {c} is not a valid RGB(A) tuple.")
            if isinstance(cmap_colors, np.ndarray):
                colors = [to_float_tuple(c) for c in cmap_colors.tolist()]
            elif isinstance(cmap_colors, (list, tuple)):
                colors = [to_float_tuple(c) for c in cmap_colors]
            else:
                raise TypeError("cmap.colors is not iterable as expected.")
        else:
            raise TypeError("The colormap 'tab20' is not a ListedColormap and has no 'colors' attribute.")
        marker_colors = [colors[i % len(colors)] for i in range(len(channel_order))]
        custom_cmap = ListedColormap(marker_colors)

        # --- Visualization: Masson trichrome image (left) and heatmap (right) ---
        import matplotlib.patches as mpatches
        import os
        # Load Masson trichrome image (tif)
        masson_img = None
        try:
            from tifffile import imread as tiff_imread
            masson_img = tiff_imread(str(masson_trichrome_path))
        except ImportError:
            try:
                from PIL import Image
                masson_img = np.array(Image.open(str(masson_trichrome_path)))
            except Exception as e:
                print(f"[WARNING] Could not load Masson trichrome image: {e}")
                masson_img = None
        except Exception as e:
            print(f"[WARNING] Could not load Masson trichrome image: {e}")
            masson_img = None

        fig, axes = plt.subplots(1, 2, figsize=(18, 10), gridspec_kw={'width_ratios': [1, 1.2]})
        # Left: Masson trichrome image
        ax_img = axes[0]
        if masson_img is not None:
            if masson_img.ndim == 2:
                ax_img.imshow(masson_img, cmap='gray')
            else:
                ax_img.imshow(masson_img)
            ax_img.set_title("Masson Trichrome")
        else:
            ax_img.text(0.5, 0.5, "Image not found", ha='center', va='center', fontsize=16)
            ax_img.set_title("Masson Trichrome (not found)")
        ax_img.axis('off')

        # Right: Heatmap
        ax_hm = axes[1]
        im = ax_hm.imshow(grid, cmap=custom_cmap, origin='lower', aspect='equal', interpolation='none', vmin=0, vmax=len(channel_order)-1)
        # Only use RGB or RGBA tuples for legend colors, ensure correct length
        legend_handles = []
        for i in range(len(channel_order)):
            color = marker_colors[i]
            # Ensure color is a tuple of exactly 3 or 4 floats
            if isinstance(color, (list, np.ndarray)):
                color = tuple(float(x) for x in color)
            if len(color) == 3:
                legend_handles.append(
                    mpatches.Patch(color=(color[0], color[1], color[2]), label=channel_order[i])  # type: ignore
                )
            elif len(color) == 4:
                legend_handles.append(
                    mpatches.Patch(color=(color[0], color[1], color[2], color[3]), label=channel_order[i])  # type: ignore
                )
            else:
                # fallback: use first 3 values
                rgb = tuple(float(x) for x in (list(color) + [0.0, 0.0, 0.0])[:3])
                legend_handles.append(
                    mpatches.Patch(color=rgb, label=channel_order[i])  # type: ignore
                )
        ax_hm.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax_hm.set_title("Most Influential Marker per Tile (Heatmap)")
        ax_hm.axis('off')

        plt.tight_layout()
        plt.savefig(output_path / f"{slide_path.stem}_influential_marker_heatmap.png", bbox_inches='tight')
        plt.close()