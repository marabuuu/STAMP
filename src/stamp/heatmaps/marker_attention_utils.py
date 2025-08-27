"""
Utility functions for marker attention visualization in heatmaps.
This module contains functions to extract marker attention information and
create visualizations showing which markers are most attended.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set, Any, cast

import torch
from torch import Tensor
import openslide
from PIL import Image

_logger = logging.getLogger("stamp")

def extract_marker_attention(
    model: Any, 
    feats: Tensor,
    coords: Tensor
) -> Optional[Tensor]:
    """
    Extract marker attention from a trained model.
    
    Args:
        model: Trained vision transformer model with attention mechanisms
        feats: Input features with shape [n_tiles, embedding_dim]
        coords: Coordinate tensor with shape [n_tiles, 2]
        
    Returns:
        Attention tensor with shape [n_tiles, n_markers] or None if extraction fails
    """
    # Handle the case where there's no attention mechanism
    if not hasattr(model, 'fluoroformer'):
        _logger.warning("Model has no fluoroformer, cannot extract marker attention")
        return None
    
    try:
        # Process inputs
        device = next(model.parameters()).device
        feats = feats.to(device)
        coords = coords.to(device)
        
        # Prepare model for attention extraction
        model.eval()
        
        # Reshape features if needed
        if feats.dim() == 2:  # [n_tiles, embedding_dim]
            feats_processed = feats.unsqueeze(-2)  # [n_tiles, 1, embedding_dim]
        else:
            feats_processed = feats
            
        # Create mask for valid tiles
        mask = torch.zeros(feats_processed.shape[0], 1, dtype=torch.bool, device=device)
        
        # Process coordinates
        if coords.dim() == 2:  # [n_tiles, 2]
            coords_processed = coords.unsqueeze(-2)  # [n_tiles, 1, 2]
        else:
            coords_processed = coords
        
        # Forward pass with attention capture
        with torch.no_grad():
            # Store original attention capture state
            orig_capture = getattr(model.fluoroformer, 'capture_attention', False)
            model.fluoroformer.capture_attention = True
            
            # Run forward pass
            _ = model(
                bags=feats_processed,
                coords=coords_processed,
                mask=mask
            )
            
            # Get captured attention
            if hasattr(model.fluoroformer, 'last_attention'):
                attn = model.fluoroformer.last_attention
            else:
                _logger.warning("No attention captured during forward pass")
                attn = None
                
            # Restore original state
            model.fluoroformer.capture_attention = orig_capture
        
        # Process attention into usable format
        if attn is not None:
            # Convert to tensor if it's not already
            if not isinstance(attn, torch.Tensor):
                attn = torch.tensor(attn, device=device)
                
            # Different attention mechanisms produce different formats
            # Try to standardize to [n_tiles, n_markers]
            if attn.dim() > 3:
                # Complex attention format, take mean across appropriate dimensions
                attn = attn.mean(dim=tuple(range(2, attn.dim())))
            
            # If attention is [n_markers, n_tiles], transpose it
            if attn.dim() == 2 and attn.shape[0] < attn.shape[1]:
                attn = attn.t()
                
            # Normalize attention values for consistency
            if attn.min() < 0 or attn.max() > 1:
                attn = torch.nn.functional.softmax(attn, dim=1)
        
        return attn
    
    except Exception as e:
        _logger.error(f"Error extracting marker attention: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_marker_map_from_attention(
    coords: Tensor,
    most_attended_marker: Tensor,
    map_size: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Create a 2D marker map from coordinates and marker indices.
    
    Args:
        coords: Coordinate tensor of shape [n_tiles, 2]
        most_attended_marker: Tensor with marker indices of shape [n_tiles]
        map_size: Optional tuple with (height, width) for the map
                  If None, will calculate from coords
    
    Returns:
        Tuple of (marker_map, marker_counts)
        - marker_map: 2D numpy array with marker indices
        - marker_counts: Dictionary mapping marker indices to counts
    """
    # Convert tensors to numpy
    coords_np = coords.cpu().numpy()
    markers_np = most_attended_marker.cpu().numpy()
    
    # Get map size if not provided
    if map_size is None:
        width = int(np.max(coords_np[:, 0])) + 1
        height = int(np.max(coords_np[:, 1])) + 1
        map_size = (height, width)
    
    # Create the map
    marker_map = np.zeros(map_size, dtype=int)
    
    # Track which markers actually appear in the map
    marker_counts = {}
    
    # Fill the marker map
    for idx, (x, y) in enumerate(coords_np):
        if idx < len(markers_np):  # Safety check
            x_int, y_int = int(x), int(y)
            if 0 <= x_int < map_size[1] and 0 <= y_int < map_size[0]:
                marker_idx = int(markers_np[idx])
                marker_map[y_int, x_int] = marker_idx
                marker_counts[marker_idx] = marker_counts.get(marker_idx, 0) + 1
    
    return marker_map, marker_counts

def save_marker_heatmap(
    marker_map: Optional[Union[np.ndarray, Tensor]] = None,
    marker_indices: Optional[List[int]] = None,
    marker_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    title: str = "Marker Attention Heatmap",
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = 300,
    show_grid: bool = True,
    coords_norm: Optional[Tensor] = None,
    most_attended_marker: Optional[Tensor] = None,
) -> Optional[Path]:
    """
    Save a standalone marker heatmap visualization.
    
    This function has two modes:
    1. If marker_map is provided, it visualizes the map directly
    2. If coords_norm and most_attended_marker are provided, it creates the map first
    
    Args:
        marker_map: 2D array with marker indices as values (optional if coords_norm provided)
        marker_indices: List of marker indices that appear in the map (optional)
        marker_names: List of marker names corresponding to indices
        save_path: Path to save the figure (required)
        title: Title for the figure
        figsize: Figure size as (width, height)
        dpi: Resolution for saved figure
        show_grid: Whether to show grid lines
        coords_norm: Coordinate tensor of shape [n_tiles, 2] (used with most_attended_marker)
        most_attended_marker: Tensor with marker indices of shape [n_tiles] (used with coords_norm)
        
    Returns:
        Path to saved figure or None if failed
    """
    # Input validation
    if save_path is None:
        _logger.error("save_path is required")
        return None
    
    # Handle the case where we need to create the map first
    if marker_map is None and coords_norm is not None and most_attended_marker is not None:
        marker_map, counts_dict = create_marker_map_from_attention(
            coords=coords_norm,
            most_attended_marker=most_attended_marker
        )
        if marker_indices is None:
            marker_indices = sorted(list(counts_dict.keys()))
    
    # Additional validation after possible creation of marker_map
    if marker_map is None:
        _logger.error("Either marker_map or both coords_norm and most_attended_marker must be provided")
        return None
        
    if marker_indices is None:
        _logger.warning("No marker_indices provided, will try to infer from marker_map")
        # Convert to numpy array if it's a tensor for consistent handling
        if torch.is_tensor(marker_map):
            marker_map_np = marker_map.cpu().numpy()
        else:
            marker_map_np = marker_map
            
        # Get unique markers from the map
        marker_indices = sorted(list(set(marker_map_np.flatten().tolist())))
    
    if marker_names is None:
        _logger.warning("No marker_names provided, using generic names")
        # Create generic marker names
        marker_names = [f"Marker {i}" for i in range(max(marker_indices) + 1 if marker_indices else 0)]
    
    # Ensure save directory exists
    save_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Convert to numpy array if it's a tensor
    if torch.is_tensor(marker_map):
        marker_map = marker_map.cpu().numpy()
    
    # Define high-contrast colors for better visualization
    explicit_colors = [
        [1.0, 0.0, 0.0],      # Red
        [0.0, 0.0, 1.0],      # Blue
        [0.0, 0.8, 0.0],      # Green
        [1.0, 1.0, 0.0],      # Yellow
        [1.0, 0.0, 1.0],      # Magenta
        [0.0, 1.0, 1.0],      # Cyan
        [1.0, 0.5, 0.0],      # Orange
        [0.5, 0.0, 0.5],      # Purple
        [0.0, 0.5, 0.5],      # Teal
        [0.5, 0.5, 0.0],      # Olive
        [0.8, 0.4, 0.2],      # Brown
        [0.4, 0.0, 0.2],      # Burgundy
        [0.2, 0.4, 0.0],      # Dark green
        [1.0, 0.6, 0.8],      # Pink
        [0.6, 0.8, 1.0],      # Light blue
        [0.8, 0.8, 0.8],      # Gray
        [0.2, 0.2, 0.2],      # Dark gray
        [0.9, 0.9, 0.5],      # Light yellow
        [0.5, 0.9, 0.9],      # Light cyan
        [0.9, 0.5, 0.9],      # Light magenta
    ]
    
    # Get max marker index needed for colormap
    unique_markers = sorted(marker_indices)
    max_marker_idx = max(unique_markers) if unique_markers else 0
    num_colors_needed = max_marker_idx + 1
    
    # Make sure we have enough colors
    if num_colors_needed > len(explicit_colors):
        # If we need more colors, use HSV colormap to add more
        extra_colors = plt.get_cmap('hsv')(np.linspace(0, 1, num_colors_needed - len(explicit_colors)))
        colors_to_use = np.vstack([explicit_colors, extra_colors[:, :3]])
    else:
        colors_to_use = explicit_colors[:num_colors_needed]
    
    # Create the colormap
    custom_cmap = ListedColormap(colors_to_use)
    
    # Create figure and plot
    fig, ax = plt.subplots(figsize=figsize)
    img = ax.imshow(marker_map, cmap=custom_cmap, interpolation='nearest')
    ax.set_title(title)
    
    # Add colorbar with marker names
    if marker_indices and len(marker_indices) > 0:
        # Create custom boundaries for the colormap
        bounds = np.array(unique_markers + [max(unique_markers) + 1]) - 0.5
        norm = BoundaryNorm(bounds, len(unique_markers))
        
        # Create the colorbar with the correct boundaries
        cbar = plt.colorbar(
            ScalarMappable(norm=norm, cmap=custom_cmap),
            ax=ax, 
            boundaries=bounds,
            ticks=unique_markers
        )
        
        # Add marker names to colorbar
        tick_labels = []
        for i in unique_markers:
            if i < len(marker_names):
                tick_labels.append(f"{i}: {marker_names[i]}")
            else:
                tick_labels.append(f"{i}: Marker {i}")
                
        cbar.ax.set_yticklabels(tick_labels)
        
        # For clarity, add grid lines to see tile boundaries
        if show_grid and marker_map.shape[0] < 100 and marker_map.shape[1] < 100:
            ax.grid(True, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
            ax.set_xticks(np.arange(-.5, marker_map.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-.5, marker_map.shape[0], 1), minor=True)
    
    # Save the figure
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    
    return save_path

def save_consolidated_heatmap(
    coords_norm: Tensor,
    most_attended_marker: Tensor,
    marker_names: List[str],
    save_path: Path,
    sample_name: str = "Sample",
    thumbnail_img: Optional[np.ndarray] = None,
    top_tiles: Optional[List[np.ndarray]] = None,
    bottom_tiles: Optional[List[np.ndarray]] = None,
    dpi: int = 300,
) -> Path:
    """
    Create and save a single consolidated heatmap per sample with all relevant information.
    
    Args:
        coords_norm: Coordinate tensor of shape [n_tiles, 2]
        most_attended_marker: Tensor with marker indices of shape [n_tiles]
        marker_names: List of marker names
        save_path: Path to save the figure
        sample_name: Name of the sample for title
        thumbnail_img: Optional thumbnail image to show alongside heatmap
        top_tiles: List of top tile images to show
        bottom_tiles: List of bottom tile images to show
        dpi: Resolution for saved figure
        
    Returns:
        Path to saved figure
    """
    # Ensure save directory exists
    save_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Convert to CPU tensors
    coords_norm_cpu = coords_norm.cpu()
    most_attended_marker_cpu = most_attended_marker.cpu()
    
    # Create marker map
    marker_map, marker_counts = create_marker_map_from_attention(
        coords=coords_norm_cpu,
        most_attended_marker=most_attended_marker_cpu
    )
    
    # Get unique markers and their counts
    unique_markers = sorted(list(marker_counts.keys()))
    max_marker_idx = max(unique_markers) if unique_markers else 0
    num_colors_needed = max_marker_idx + 1
    
    # Define high-contrast colors for better visualization
    explicit_colors = [
        [1.0, 0.0, 0.0],      # Red
        [0.0, 0.0, 1.0],      # Blue
        [0.0, 0.8, 0.0],      # Green
        [1.0, 1.0, 0.0],      # Yellow
        [1.0, 0.0, 1.0],      # Magenta
        [0.0, 1.0, 1.0],      # Cyan
        [1.0, 0.5, 0.0],      # Orange
        [0.5, 0.0, 0.5],      # Purple
        [0.0, 0.5, 0.5],      # Teal
        [0.5, 0.5, 0.0],      # Olive
        [0.8, 0.4, 0.2],      # Brown
        [0.4, 0.0, 0.2],      # Burgundy
        [0.2, 0.4, 0.0],      # Dark green
        [1.0, 0.6, 0.8],      # Pink
        [0.6, 0.8, 1.0],      # Light blue
        [0.8, 0.8, 0.8],      # Gray
        [0.2, 0.2, 0.2],      # Dark gray
        [0.9, 0.9, 0.5],      # Light yellow
        [0.5, 0.9, 0.9],      # Light cyan
        [0.9, 0.5, 0.9],      # Light magenta
    ]
    
    # Make sure we have enough colors
    if num_colors_needed > len(explicit_colors):
        # If we need more colors, use HSV colormap to add more
        extra_colors = plt.get_cmap('hsv')(np.linspace(0, 1, num_colors_needed - len(explicit_colors)))
        colors_to_use = np.vstack([explicit_colors, extra_colors[:, :3]])
    else:
        colors_to_use = explicit_colors[:num_colors_needed]
    
    # Create the colormap
    custom_cmap = ListedColormap(colors_to_use)
    
    # Determine layout based on available components
    has_thumbnail = thumbnail_img is not None
    has_tiles = (top_tiles is not None and len(top_tiles) > 0) or (bottom_tiles is not None and len(bottom_tiles) > 0)
    
    # Define the layout
    if has_thumbnail and has_tiles:
        # 2x2 grid: thumbnail, heatmap, top tiles, bottom tiles
        fig, axs = plt.subplots(2, 2, figsize=(16, 12), 
                               gridspec_kw={'width_ratios': [1, 2] if thumbnail_img is not None else [1, 1],
                                           'height_ratios': [3, 1] if has_tiles else [1, 1]})
    elif has_thumbnail:
        # 1x2 grid: thumbnail, heatmap
        fig, axs = plt.subplots(1, 2, figsize=(16, 8), 
                               gridspec_kw={'width_ratios': [1, 2]})
        axs = np.array([[axs[0], axs[1]], [None, None]])
    elif has_tiles:
        # 2x1 grid: heatmap, tiles
        fig, axs = plt.subplots(2, 1, figsize=(12, 12), 
                               gridspec_kw={'height_ratios': [3, 1]})
        axs = np.array([[None, axs[0]], [None, axs[1]]])
    else:
        # Just heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        axs = np.array([[None, ax], [None, None]])
    
    # Set figure title
    fig.suptitle(f"Marker Attention Heatmap - {sample_name}", fontsize=16)
    
    # Add thumbnail if available
    if has_thumbnail and axs[0, 0] is not None:
        axs[0, 0].imshow(thumbnail_img)
        axs[0, 0].set_title("Sample Overview")
        axs[0, 0].axis('off')
    
    # Add heatmap
    heatmap_ax = axs[0, 1]
    img = heatmap_ax.imshow(marker_map, cmap=custom_cmap, interpolation='nearest')
    heatmap_ax.set_title("Most Attended Marker Per Tile")
    
    # Add colorbar with marker names
    if unique_markers and len(unique_markers) > 0:
        # Create custom boundaries for the colormap
        bounds = np.array(unique_markers + [max(unique_markers) + 1]) - 0.5
        norm = BoundaryNorm(bounds, len(unique_markers))
        
        # Create the colorbar with the correct boundaries
        cbar = plt.colorbar(
            ScalarMappable(norm=norm, cmap=custom_cmap),
            ax=heatmap_ax, 
            boundaries=bounds,
            ticks=unique_markers
        )
        
        # Add marker names to colorbar
        tick_labels = []
        for i in unique_markers:
            if i < len(marker_names):
                tick_labels.append(f"{i}: {marker_names[i]}")
            else:
                tick_labels.append(f"{i}: Marker {i}")
                
        cbar.ax.set_yticklabels(tick_labels)
        
        # Add grid lines for better visibility if the map isn't too large
        if marker_map.shape[0] < 100 and marker_map.shape[1] < 100:
            heatmap_ax.grid(True, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
            heatmap_ax.set_xticks(np.arange(-.5, marker_map.shape[1], 1), minor=True)
            heatmap_ax.set_yticks(np.arange(-.5, marker_map.shape[0], 1), minor=True)
    
    # Turn off axes for heatmap
    heatmap_ax.axis('off')
    
    # Display top/bottom tiles if available
    if has_tiles:
        tiles_ax = axs[1, 1]
        if tiles_ax is not None:
            n_top = len(top_tiles) if top_tiles else 0
            n_bottom = len(bottom_tiles) if bottom_tiles else 0
            n_total = n_top + n_bottom
            
            if n_total > 0:
                # Create a grid of tile images
                tiles_to_show = []
                labels = []
                
                if top_tiles and n_top > 0:
                    tiles_to_show.extend(top_tiles[:min(5, n_top)])
                    labels.extend([f"Top {i+1}" for i in range(min(5, n_top))])
                
                if bottom_tiles and n_bottom > 0:
                    tiles_to_show.extend(bottom_tiles[:min(5, n_bottom)])
                    labels.extend([f"Bottom {i+1}" for i in range(min(5, n_bottom))])
                
                # Determine grid layout
                n_cols = min(5, n_total)
                n_rows = (n_total + n_cols - 1) // n_cols
                
                # Remove existing content and create a grid of subplots
                tiles_ax.clear()
                tiles_ax.axis('off')
                
                grid_width = 0.9  # width of the grid as fraction of the subplot
                grid_height = 0.8  # height of the grid
                
                # Calculate grid positions
                for i, (tile, label) in enumerate(zip(tiles_to_show, labels)):
                    row = i // n_cols
                    col = i % n_cols
                    
                    # Calculate positions
                    x = col / n_cols * grid_width + (1 - grid_width) / 2
                    y = (n_rows - 1 - row) / max(1, n_rows - 1) * grid_height + (1 - grid_height) / 2
                    width = 0.9 / n_cols
                    height = 0.9 / max(1, n_rows)
                    
                    # Create new axis for this tile
                    tile_ax = fig.add_axes((x, y, width, height))
                    tile_ax.imshow(tile)
                    tile_ax.set_title(label, fontsize=8)
                    tile_ax.axis('off')
                
                # Hide the original tiles axis
                tiles_ax.set_frame_on(False)
    
    # Save the figure
    fig.tight_layout(rect=(0, 0, 1, 0.95))  # Leave space for the title
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return save_path

def show_slide_thumbnail(
    slide: openslide.OpenSlide,
    thumb_ax=None,
    attention: Optional[Tensor] = None,
    default_slide_mpp: float = 0.5,
) -> np.ndarray:
    """Display a thumbnail of the slide with attention overlay."""
    # Default parameters for visualization
    thumb_mpp = 20.0
    
    # Create axis if not provided
    if thumb_ax is None:
        _, thumb_ax = plt.subplots(figsize=(8, 8))

    # Get slide properties
    if hasattr(slide, "properties") and "aperio.MPP" in slide.properties:
        # Handle Aperio slides
        slide_mpp = float(slide.properties["aperio.MPP"])
    else:
        # Default MPP
        slide_mpp = default_slide_mpp

    # Calculate downscaling factor
    thumb_scale_factor = thumb_mpp / slide_mpp
    thumb_size = (
        int(slide.dimensions[0] / thumb_scale_factor),
        int(slide.dimensions[1] / thumb_scale_factor),
    )

    # Get thumbnail
    thumb = slide.get_thumbnail(thumb_size)
    thumb_np = np.array(thumb)

    # Overlay attention if provided
    if attention is not None and torch.is_tensor(attention) and attention.numel() > 0:
        try:
            # Resize attention to match thumbnail size
            from skimage.transform import resize
            attention_np = attention.detach().cpu().numpy()
            resized_attention = resize(
                attention_np, 
                (thumb_np.shape[0], thumb_np.shape[1]),
                order=0,  # Nearest-neighbor interpolation
                preserve_range=True,
                anti_aliasing=False
            )
            
            # Normalize attention
            resized_attention = (resized_attention - resized_attention.min()) / (resized_attention.max() - resized_attention.min() + 1e-8)
            
            # Create a heatmap overlay (red tint)
            overlay = np.zeros_like(thumb_np)
            overlay[:, :, 0] = 255  # Red channel
            
            # Blend with original image using attention as alpha
            alpha = np.expand_dims(resized_attention, axis=-1) * 0.6  # 60% opacity
            blended = thumb_np * (1 - alpha) + overlay * alpha
            thumb_np = blended.astype(np.uint8)
        except Exception as e:
            _logger.error(f"Error applying attention overlay: {e}")
    
    # Display thumbnail
    thumb_ax.imshow(thumb_np)
    thumb_ax.set_title("Slide overview")
    thumb_ax.axis("off")
    
    return thumb_np

def vals_to_im(
    scores: Tensor,  # shape: [tile, feat]
    coords_norm: Tensor,  # shape: [tile, 2]
) -> Tensor:  # shape: [width, height, category]
    """Arranges scores in a 2d grid according to coordinates
    
    This is the same function as _vals_to_im in __init__.py, kept for compatibility.
    """
    # Ensure inputs are on same device
    device = scores.device
    coords_norm = coords_norm.to(device)
    
    # Create the output grid
    size = coords_norm.max(0).values.flip(0) + 1
    im = torch.zeros((*size.tolist(), *scores.shape[1:])).type_as(scores)

    # Flatten the grid for faster indexing
    flattened_im = im.flatten(end_dim=1)
    flattened_coords = coords_norm[:, 1] * im.shape[1] + coords_norm[:, 0]
    
    # Assign scores to the flattened grid
    flattened_im[flattened_coords] = scores

    # Reshape back to the original grid
    im = flattened_im.reshape_as(im)

    return im
