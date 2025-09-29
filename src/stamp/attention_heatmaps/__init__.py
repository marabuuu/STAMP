import h5py
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Iterable
from stamp.preprocessing import supported_extensions
import os
import tifffile
import matplotlib
import matplotlib.colors
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from stamp.modeling.lightning_model import LitVisionTransformer
import re

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
    marker_image_paths: list[Path],
    device: str,
    output_path: Path,
    channel_order: list[str],  
    top_k_percent=0.1,
    ):
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
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
    
    # === CRITICAL SECTION: FIXED FLUORESCENCE OVERLAY ===
    # 1. VERIFY AND ORDER CHANNELS TO MATCH channel_order
    print("\nVerifying channel order matching:")
    marker_image_paths_ordered = []
    for marker in channel_order:
        # Case-insensitive matching with flexible pattern matching
        pattern = re.compile(re.escape(marker.lower()))
        match = [p for p in marker_image_paths 
                 if pattern.search(p.name.lower())]
        
        if not match:
            # Try removing common prefixes/suffixes
            simple_marker = re.sub(r'[_\-\.]', '', marker.lower())
            match = [p for p in marker_image_paths 
                     if simple_marker in re.sub(r'[_\-\.]', '', p.name.lower())]
        
        if match:
            print(f"  ✓ {marker} -> {match[0].name}")
            marker_image_paths_ordered.append(match[0])
        else:
            print(f"  ✗ {marker} not found in provided images!")
            # Create a blank image as placeholder (but skip for no_antibody)
            if marker != "no_antibody":  # Skip placeholder for no_antibody
                with tifffile.TiffFile(marker_image_paths[0]) as tif:
                    h, w = tif.pages[0].shape[:2]
                blank = np.zeros((h, w), dtype=np.float32)
                temp_path = output_path / f"missing_{marker}.tiff"
                tifffile.imwrite(str(temp_path), blank)
                marker_image_paths_ordered.append(temp_path)
                print(f"    Created placeholder: {temp_path.name}")
            else:
                print("    Skipping placeholder for no_antibody (not needed)")
    
    marker_image_paths = marker_image_paths_ordered
    
    # 2. LOAD AND PREPROCESS IMAGES WITH LOG NORMALIZATION
    marker_imgs = []
    for img_path in marker_image_paths:
        img = tifffile.imread(str(img_path))
        if img.ndim == 3:  # Handle RGB TIFFs (common in fluorescence)
            # Take first channel or convert to grayscale
            if img.shape[2] == 3:
                img = img.mean(axis=2)
            else:
                img = img[:, :, 0]  # Use first channel for multi-channel TIFFs
        marker_imgs.append(img.astype(np.float32))
    
    # 3. LOG NORMALIZATION (better dynamic range than linear)
    norm_imgs = []
    for img in marker_imgs:
        # Log(1+x) compression handles high dynamic range better
        img_log = np.log1p(img)  
        # Normalize to [0, 1] per channel
        img_min, img_max = img_log.min(), img_log.max()
        if img_max > img_min:
            img_norm = (img_log - img_min) / (img_log.max() - img_min)
        else:
            img_norm = np.zeros_like(img_log)
        norm_imgs.append(img_norm)
    
    # 4. MAX-INTENSITY PROJECTION (Napari-style)
    all_norm = np.stack(norm_imgs, axis=-1)  # [H, W, N_channels]
    max_index = np.argmax(all_norm, axis=-1)  # Dominant channel per pixel
    max_value = np.max(all_norm, axis=-1)     # Strength of dominant channel
    
    # Apply threshold to suppress weak signals (optional but recommended)
    threshold = np.percentile(max_value, 5)  # Top 95% of pixels
    max_value[max_value < threshold] = 0
    
    # 5. COLOR MAPPING WITH TAB20
    cmap = plt.get_cmap('tab20', len(channel_order))
    colors = np.array([cmap(i)[:3] for i in range(len(channel_order))])  # RGB only
    
    # Create output canvas with dark background
    canvas = np.zeros((*max_index.shape, 3), dtype=np.float32)
    for i in range(len(channel_order)):
        mask = (max_index == i)
        canvas[mask] = colors[i] * max_value[mask, np.newaxis]
    
    # 6. CREATE LEGEND FOR CHANNELS (with "antibody" -> "autofluorescence")
    legend_patches = []
    for i, marker in enumerate(channel_order):
        # Replace "antibody" with "autofluorescence" in the label
        display_name = marker.replace("antibody", "autofluorescence")
        legend_patches.append(mpatches.Patch(color=colors[i], label=display_name))
    
    # 7. VISUALIZE SIDE-BY-SIDE: OVERLAY + HEATMAP
    fig, axes = plt.subplots(1, 2, figsize=(18, 10), 
                             gridspec_kw={'width_ratios': [1, 1.2]})
    
    # Left: Multiplex image (renamed from "Dominant Marker Overlay")
    axes[0].imshow(canvas, vmin=0, vmax=1)
    axes[0].set_title("Multiplex image", fontsize=14)
    axes[0].axis('off')
    
    # Add legend to the overlay
    legend = axes[0].legend(handles=legend_patches, 
                          loc='upper right',
                          bbox_to_anchor=(1.0, 1.0),
                          frameon=True,
                          framealpha=0.8,
                          fontsize=9)
    legend.get_frame().set_facecolor('white')
    
    # Right: Attention heatmap (with corrected orientation)
    # Mask grid positions where grid == -1 (filtered tiles)
    masked_grid = np.ma.masked_where(grid == -1, grid)
    # Use tab20 colormap and set masked color to black
    cmap_heatmap = plt.get_cmap('tab20', len(channel_order)).with_extremes(bad='black')
    # Flip the grid vertically to match the overlay orientation
    im = axes[1].imshow(np.flipud(masked_grid), cmap=cmap_heatmap, vmin=0, vmax=len(channel_order)-1)
    axes[1].set_title("Most Influential Marker per Tile", fontsize=14)
    axes[1].axis('off')

    # Create custom colorbar matching the heatmap (with correct ordering)
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_ticks(list(map(float, range(len(channel_order)))))
    cbar.set_ticklabels(channel_order)
    
    # Save the composite figure
    composite_path = output_path / f"{slide_path.stem}_composite.png"
    plt.tight_layout()
    plt.savefig(composite_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSuccessfully saved composite visualization to: {composite_path}")
    return composite_path