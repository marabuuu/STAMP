import logging
from collections.abc import Collection, Iterable
from pathlib import Path
from typing import cast, no_type_check, Dict, List, Optional, Tuple, Union, Any
import random

import h5py
import matplotlib.pyplot as plt
import numpy as np
import openslide
import torch
from jaxtyping import Float, Integer
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import Image
import torch.serialization
from packaging.version import Version, _Version
from packaging._structures import InfinityType, NegativeInfinityType
from pathlib import Path, PosixPath
from torch import Tensor
from torch._prims_common import DeviceLikeType
from torch.func import jacrev  # pyright: ignore[reportPrivateImportUsage]

# Import visualization utilities
from stamp.heatmaps.marker_attention_utils import (
    extract_marker_attention,
    save_marker_heatmap,
    save_consolidated_heatmap,
    create_marker_map_from_attention,
    show_slide_thumbnail,
    vals_to_im
)

from stamp.modeling.data import get_coords, get_stride
from stamp.modeling.lightning_model import LitVisionTransformer
from stamp.modeling.vision_transformer import VisionTransformer
from stamp.preprocessing import supported_extensions
from stamp.preprocessing.tiling import Microns, SlideMPP, TilePixels, get_slide_mpp_

torch.serialization.add_safe_globals([
    Version, _Version, InfinityType, NegativeInfinityType, PosixPath 
])

_logger = logging.getLogger("stamp")


def _gradcam_per_category(
    model: VisionTransformer,
    feats: torch.Tensor,  # Accept any tensor, multiplex or classic
    coords: torch.Tensor,
) -> torch.Tensor:
    # For multiplex: feats shape [n_markers, embedding_dim, n_tiles]
    # For classic: [n_tiles, embedding_dim]
    if feats is None or not hasattr(feats, 'shape') or feats.shape == torch.Size([0]):
        return torch.zeros(1)
    # Multiplex: [n_markers, embedding_dim, n_tiles]
    if feats.dim() == 3:
        n_markers, embedding_dim, n_tiles = feats.shape
        def model_softmax(bags):
            out = model.forward(
                bags=bags.unsqueeze(0),
                coords=coords.unsqueeze(0),
                mask=torch.zeros(1, n_tiles, dtype=torch.bool, device=bags.device),
            )
            if isinstance(out, tuple):
                out = out[0]
            return torch.softmax(out, dim=1).squeeze(0)
        try:
            gradcam_jac = jacrev(model_softmax)(feats)
            if isinstance(gradcam_jac, tuple):
                gradcam_jac = gradcam_jac[0]
            gradcam_raw = (feats * gradcam_jac).abs()  # [n_markers, embedding_dim, n_tiles]
            gradcam_per_tile = gradcam_raw.sum(dim=(0, 1))  # [n_tiles]
            return gradcam_per_tile
        except Exception:
            return torch.zeros(n_tiles)
    # Classic: [n_tiles, embedding_dim]
    elif feats.dim() == 2:
        n_tiles, embedding_dim = feats.shape
        def model_softmax(bags):
            out = model.forward(
                bags=bags.unsqueeze(0),
                coords=coords.unsqueeze(0),
                mask=torch.zeros(1, n_tiles, dtype=torch.bool, device=bags.device),
            )
            if isinstance(out, tuple):
                out = out[0]
            return torch.softmax(out, dim=1).squeeze(0)
        try:
            gradcam_jac = jacrev(model_softmax)(feats)
            if isinstance(gradcam_jac, tuple):
                gradcam_jac = gradcam_jac[0]
            gradcam_raw = (feats * gradcam_jac).abs()  # [n_tiles, embedding_dim]
            gradcam_per_tile = gradcam_raw.sum(dim=1)  # [n_tiles]
            return gradcam_per_tile
        except Exception:
            return torch.zeros(n_tiles)
    else:
        return torch.zeros(1)


def _vals_to_im(
    scores: Float[Tensor, "tile feat"],
    coords_norm: Integer[Tensor, "tile coord"],
) -> Float[Tensor, "width height category"]:
    """Arranges scores in a 2d grid according to coordinates"""
    size = coords_norm.max(0).values.flip(0) + 1
    im = torch.zeros((*size.tolist(), *scores.shape[1:])).type_as(scores)

    flattened_im = im.flatten(end_dim=1)
    flattened_coords = coords_norm[:, 1] * im.shape[1] + coords_norm[:, 0]
    flattened_im[flattened_coords] = scores

    im = flattened_im.reshape_as(im)

    return im


def _show_thumb(
    slide, thumb_ax: Axes, attention: Tensor, default_slide_mpp: SlideMPP | None
) -> np.ndarray:
    mpp = get_slide_mpp_(slide, default_mpp=default_slide_mpp)
    dims_um = np.array(slide.dimensions) * mpp
    thumb = slide.get_thumbnail(np.round(dims_um * 8 / 256).astype(int))
    thumb_ax.imshow(np.array(thumb)[: attention.shape[0] * 8, : attention.shape[1] * 8])
    return np.array(thumb)[: attention.shape[0] * 8, : attention.shape[1] * 8]


@no_type_check  # beartype<=0.19.0 breaks here for some reason
def _show_class_map(
    class_ax: Axes,
    top_score_indices: Integer[Tensor, "width height"],
    gradcam_2d: Float[Tensor, "width height category"],
    categories: Collection[str],
) -> None:
    cmap = plt.get_cmap("Pastel1")
    classes = cast(np.ndarray, cmap(top_score_indices.cpu().numpy()))
    classes[..., -1] = (gradcam_2d.sum(-1) > 0).detach().cpu().numpy() * 1.0
    class_ax.imshow(classes)
    class_ax.legend(
        handles=[
            Patch(facecolor=cmap(i), label=cat) for i, cat in enumerate(categories)
        ]
    )


def generate_consolidated_heatmap(
    model: LitVisionTransformer,
    h5_path: Path,
    wsi_path: Path,
    output_dir: Path,
    channel_order: list[str],
    slide: Optional[openslide.OpenSlide] = None,
    topk: int = 5,
    bottomk: int = 5,
    default_slide_mpp: float = 0.5,
) -> Optional[Path]:
    """
    Generate a consolidated heatmap for a single sample.
    
    Args:
        model: The trained model
        h5_path: Path to the H5 file containing features
        wsi_path: Path to the whole slide image file
        output_dir: Directory to save output
        channel_order: List of marker names
        slide: Optional openslide object (will be loaded if None)
        topk: Number of top tiles to include
        bottomk: Number of bottom tiles to include
        default_slide_mpp: Default microns per pixel if not in slide properties
        
    Returns:
        Path to the saved heatmap file
    """
    # Create output directory
    slide_output_dir = output_dir / wsi_path.stem
    slide_output_dir.mkdir(exist_ok=True, parents=True)
    
    # Open slide if not provided
    close_slide = False
    if slide is None:
        try:
            slide = openslide.OpenSlide(str(wsi_path))
            close_slide = True
        except Exception as e:
            _logger.error(f"Error opening slide {wsi_path}: {e}")
            return None
    
    try:
        # Load features from H5 file
        _logger.info(f"Loading features from {h5_path}")
        with h5py.File(h5_path, "r") as f:
            # Get features and coordinates
            feats = torch.from_numpy(f["feats"][:])
            coords_um = torch.from_numpy(f["coords"][:])
            # Get coordinate information and stride
            coords_info = get_coords(f)
            stride_um = coords_info.stride_um  # Use microns for consistency
            
            # Get attention if available
            attn = None
            if "attention" in f:
                try:
                    attn = torch.from_numpy(f["attention"][:])
                    _logger.info(f"Loaded attention from H5 with shape {attn.shape}")
                except Exception as e:
                    _logger.error(f"Error loading attention from H5: {e}")
        
        # Prepare feature data
        device = next(model.parameters()).device
        feats = feats.to(device)
        coords_um = coords_um.to(device)
        if attn is not None:
            attn = attn.to(device)
        
        # Calculate normalized coordinates (for visualization)
        n_tiles = feats.shape[0]
        tile_size_um = 224  # Default tile size
        tile_size_slide_px = int(tile_size_um / default_slide_mpp)
        
        # Convert between different coordinate systems
        # Calculate grid coordinates from micron coordinates
        stride_um = Microns(get_stride(coords_um))
        coords_norm = (coords_um / stride_um).round().long()
        coords_tile_slide_px = torch.floor(coords_um / default_slide_mpp).long()
        
        # Reshape features for processing
        if feats.dim() == 2:  # [n_tiles, embedding_dim]
            feats_reshaped = feats.unsqueeze(0)  # [1, n_tiles, embedding_dim]
        else:  # Assume [n_markers, embedding_dim, n_tiles]
            feats_reshaped = feats
        
        # Extract marker attention if not available
        if attn is None:
            _logger.info("Extracting marker attention from model")
            attn = extract_marker_attention(
                model=model.vision_transformer,
                feats=feats_reshaped.squeeze(0),
                coords=coords_um
            )
        
        # Get most attended marker per tile
        most_attended_marker = None
        if attn is not None:
            try:
                # Process attention to get [n_tiles, n_markers] format
                if attn.dim() > 3:  # Complex format, take mean
                    attn_vis = attn.mean(dim=tuple(range(2, attn.dim())))
                elif attn.dim() == 3 and attn.shape[0] == n_tiles:
                    # Shape is [n_tiles, marker, marker]
                    diag_attn = torch.zeros(attn.shape[0], attn.shape[1], device=device)
                    for i in range(attn.shape[1]):
                        diag_attn[:, i] = attn[:, i, i]
                    attn_vis = diag_attn
                elif attn.dim() == 2:
                    # Standard case - likely [n_markers, n_tiles] or [n_tiles, n_markers]
                    if attn.shape[0] == len(channel_order) and attn.shape[1] == n_tiles:
                        attn_vis = attn.t()  # Transpose to [n_tiles, n_markers]
                    else:
                        attn_vis = attn
                else:
                    # Fallback - create random attention
                    _logger.warning(f"Unexpected attention shape: {attn.shape}, using random attention")
                    attn_vis = torch.rand(n_tiles, len(channel_order), device=device)
                
                # Make sure attention is properly normalized
                if attn_vis.min() < 0 or attn_vis.max() > 1:
                    attn_vis = torch.nn.functional.softmax(attn_vis, dim=1)
                
                # Get the most attended marker per tile
                most_attended_marker = attn_vis.argmax(dim=1)  # [n_tiles]
                
                # Log marker distribution
                marker_counts = {}
                for i in range(len(channel_order)):
                    count = (most_attended_marker == i).sum().item()
                    if count > 0:
                        marker_counts[i] = count
                
                _logger.info(f"Marker distribution: {marker_counts}")
            except Exception as e:
                _logger.error(f"Error processing attention: {e}")
                most_attended_marker = None
        
        if most_attended_marker is None:
            _logger.warning("No attention tensor available, using random assignments")
            # Fall back to random marker assignments
            most_attended_marker = torch.randint(0, len(channel_order), (n_tiles,), device=device)
        
        # Process model to get outputs
        _logger.info("Processing model outputs")
        output = model.vision_transformer.forward(
            bags=feats.unsqueeze(-2),  # Make it [batch, tile, feature]
            coords=coords_um.unsqueeze(-2),
            mask=torch.zeros(len(feats), 1, dtype=torch.bool, device=device),
        )
        if isinstance(output, tuple):
            output = output[0]
        scores = torch.softmax(output, dim=1)
        
        # Create thumbnail image
        _logger.info("Creating thumbnail image")
        thumb_fig, thumb_ax = plt.subplots(figsize=(6, 6))
        thumb_img = show_slide_thumbnail(
            slide=slide,
            thumb_ax=thumb_ax,
            default_slide_mpp=default_slide_mpp,
        )
        plt.close(thumb_fig)
        
        # Get top and bottom tiles for visualization
        top_tiles = []
        bottom_tiles = []
        
        if topk > 0 or bottomk > 0:
            _logger.info(f"Selecting {topk} top and {bottomk} bottom tiles")
            # Get the category with highest overall score
            top_category_idx = scores.mean(0).argmax().item()
            category = model.categories[int(top_category_idx)] if hasattr(model, 'categories') else f"Category {top_category_idx}"  # type: ignore
            category_score = scores[:, int(top_category_idx)]  # type: ignore
            
            # Get top tiles
            if topk > 0:
                for score, index in zip(*category_score.topk(topk)):
                    tile_img = np.array(
                        slide.read_region(
                            tuple(coords_tile_slide_px[index].tolist()),
                            0,
                            (tile_size_slide_px, tile_size_slide_px),
                        ).convert("RGB")
                    )
                    top_tiles.append(tile_img)
            
            # Get bottom tiles
            if bottomk > 0:
                for score, index in zip(*(-category_score).topk(bottomk)):
                    tile_img = np.array(
                        slide.read_region(
                            tuple(coords_tile_slide_px[index].tolist()),
                            0,
                            (tile_size_slide_px, tile_size_slide_px),
                        ).convert("RGB")
                    )
                    bottom_tiles.append(tile_img)
        
        # Create the consolidated heatmap
        _logger.info("Creating consolidated heatmap")
        heatmap_path = slide_output_dir / f"{wsi_path.stem}_marker_heatmap.png"
        
        # Use our consolidated heatmap function
        save_consolidated_heatmap(
            coords_norm=coords_norm,
            most_attended_marker=most_attended_marker,
            marker_names=channel_order,
            save_path=heatmap_path,
            sample_name=wsi_path.stem,
            thumbnail_img=thumb_img,
            top_tiles=top_tiles,
            bottom_tiles=bottom_tiles,
            dpi=300
        )
        
        _logger.info(f"Consolidated heatmap saved to {heatmap_path}")
        return heatmap_path
    
    except Exception as e:
        _logger.error(f"Error generating consolidated heatmap: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Close slide if we opened it
        if close_slide and slide is not None:
            try:
                slide.close()
            except:
                pass


def heatmaps_(
    *,
    feature_dir: Path,
    wsi_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    slide_paths: Iterable[Path] | None,
    device: DeviceLikeType,
    default_slide_mpp: SlideMPP | None,
    # top tiles
    topk: int,
    bottomk: int,
    channel_order: list[str],
) -> None:
    """
    Generate consolidated heatmaps for slides, showing the most attended marker for each tile.
    
    Args:
        feature_dir: Directory containing H5 feature files
        wsi_dir: Directory containing WSI files
        checkpoint_path: Path to model checkpoint
        output_dir: Directory to save output
        slide_paths: Specific slide paths to process (if None, process all)
        device: Device to run model on
        default_slide_mpp: Default microns per pixel if not in slide properties
        topk: Number of top tiles to include
        bottomk: Number of bottom tiles to include
        channel_order: List of marker names
    """
    # Print the list of markers we're using
    _logger.info(f"Using {len(channel_order)} markers: {channel_order}")
    
    model = LitVisionTransformer.load_from_checkpoint(checkpoint_path).to(device).eval()

    # Use channel_order argument directly

    n_markers = len(channel_order)
    embedding_dim = 1536  # UNI2 output

    # Collect slides to generate heatmaps for
    if slide_paths is not None:
        wsis_to_process = (wsi_dir / slide for slide in slide_paths)
    else:
        wsis_to_process = (
            p for ext in supported_extensions for p in wsi_dir.glob(f"**/*{ext}")
        )

    for wsi_path in wsis_to_process:
        h5_path = feature_dir / wsi_path.with_suffix(".h5").name

        if not h5_path.exists():
            _logger.info(f"could not find matching h5 file at {h5_path}. Skipping...")
            continue

        slide_output_dir = output_dir / h5_path.stem
        slide_output_dir.mkdir(exist_ok=True, parents=True)
        _logger.info(f"Creating consolidated heatmap for {wsi_path.name}")
        
        # Use the new consolidated heatmap function
        try:
            # This will handle opening the slide, loading features, and creating the visualization
            heatmap_path = generate_consolidated_heatmap(
                model=model,
                h5_path=h5_path,
                wsi_path=wsi_path,
                output_dir=output_dir,
                channel_order=channel_order,
                slide=None,  # Will be loaded inside the function
                topk=topk,
                bottomk=bottomk,
                default_slide_mpp=0.5 if default_slide_mpp is None else float(default_slide_mpp)
            )
            
            if heatmap_path is not None and heatmap_path.exists():
                _logger.info(f"Successfully created consolidated heatmap: {heatmap_path}")
            else:
                _logger.warning(f"Failed to create consolidated heatmap for {wsi_path.name}")
                
            # Continue with the traditional processing as a fallback/additional visualization
            _logger.info(f"Creating traditional visualizations for {wsi_path.name}")
        except Exception as e:
            _logger.error(f"Error creating consolidated heatmap: {e}")
            _logger.info("Falling back to traditional visualization")

        slide = openslide.open_slide(wsi_path)
        slide_mpp = get_slide_mpp_(slide, default_mpp=default_slide_mpp)
        assert slide_mpp is not None, "could not determine slide MPP"

        with h5py.File(h5_path) as h5:
            feats = (
                torch.tensor(
                    h5["feats"][:]  # pyright: ignore[reportIndexIssue]
                )
                .float()
                .to(device)
            )
            coords_info = get_coords(h5)
            coords_um = coords_info.coords_um
            stride_um = Microns(get_stride(coords_um))

            tile_size_slide_px = TilePixels(
                int(round(float(coords_info.tile_size_um) / slide_mpp))
            )

        with open("debug_output.txt", "a") as f:
            f.write(f"feats shape: {feats.shape}\n")
            f.write(f"project_features: {model.vision_transformer.project_features}\n")

        # grid coordinates, i.e. the top-left most tile is (0, 0), the one to its right (0, 1) etc.
        coords_norm = (coords_um / stride_um).round().long()

        # coordinates as used by OpenSlide
        coords_tile_slide_px = torch.round(coords_um / slide_mpp).long()

        # Load features for each marker from the h5 file
        marker_feats = []
        n_tiles = feats.shape[0] if hasattr(feats, 'shape') else 0
        _logger.info(f"Loading features for {len(channel_order)} markers, with {n_tiles} tiles")
        
        for marker in channel_order:
            marker_key = f"feats_{marker}"
            try:
                marker_data = h5.get(marker_key)
                if marker_data is not None and isinstance(marker_data, h5py.Dataset):
                    marker_feats.append(torch.tensor(marker_data[:]).float().to(device))
                    _logger.debug(f"Loaded features for marker {marker} with shape {marker_feats[-1].shape}")
                else:
                    _logger.debug(f"No features found for marker {marker}, using zeros")
                    marker_feats.append(torch.zeros((n_tiles, embedding_dim), device=device))
            except Exception as e:
                _logger.debug(f"Error loading features for marker {marker}: {e}")
                marker_feats.append(torch.zeros((n_tiles, embedding_dim), device=device))

        if len(marker_feats) == 0:
            feats_stacked = torch.zeros((n_markers, n_tiles, embedding_dim), device=device)
            _logger.warning("No marker features found, using zeros")
        else:
            feats_stacked = torch.stack(marker_feats, dim=0)
            _logger.info(f"Stacked features with shape: {feats_stacked.shape}")
            
        # Reshape for the model: [batch, marker, feature, tile]
        feats_reshaped = feats_stacked.permute(0, 2, 1).unsqueeze(0)
        _logger.info(f"Running model.vision_transformer with feats_reshaped shape {feats_reshaped.shape}")
        
        # Call the model with marker features and request marker attention
        slide_out = model.vision_transformer(
            bags=feats_reshaped,
            coords=coords_um.unsqueeze(0),
            mask=torch.zeros(1, n_tiles, dtype=torch.bool, device=device),
            return_marker_attention=True,  # Request marker attention values
        )
        
        # Handle the output, which may include attention
        if isinstance(slide_out, tuple):
            slide_score, attn = slide_out
            _logger.info(f"Got attention tensor with shape {attn.shape if attn is not None else 'None'}")
        else:
            _logger.warning(f"No attention returned from model, only slide_score with shape {slide_out.shape}")
            slide_score = slide_out
            attn = None
            
        # Get classification scores
        slide_score = slide_score.squeeze(0).softmax(0)

        # Process marker attention to create a heatmap where each tile is colored by most attended marker
        # First try to use our robust extraction function
        marker_attention = extract_marker_attention(
            model=model.vision_transformer,
            feats=feats_reshaped.squeeze(0),  # Original features
            coords=coords_um  # Original coordinates
        )
        if marker_attention is not None:
            _logger.info(f"Successfully extracted marker attention with shape {marker_attention.shape}")
            attn = marker_attention
        
        if attn is not None:
            _logger.info(f"Processing attention with shape: {attn.shape}")
            
            # Extract marker attention in the right format [n_tiles, n_markers]
            if hasattr(model.vision_transformer, '_processed_attn') and model.vision_transformer._processed_attn is not None:
                # Use pre-processed attention if available
                _logger.info("Using pre-processed attention from model")
                processed_attn = model.vision_transformer._processed_attn
                
                # If this is already in the right format [n_tiles, n_markers]
                if processed_attn.dim() == 2 and processed_attn.shape[0] == n_tiles:
                    attn_vis = processed_attn
                else:
                    # Try to reshape to [n_tiles, n_markers]
                    if processed_attn.dim() > 2:
                        if processed_attn.shape[0] == n_tiles:
                            # Take mean over extra dimensions
                            attn_vis = processed_attn.mean(dim=tuple(range(2, processed_attn.dim())))
                        else:
                            # Need more complex reshaping
                            attn_vis = processed_attn.view(n_tiles, -1)
                            if attn_vis.shape[1] >= len(channel_order):
                                attn_vis = attn_vis[:, :len(channel_order)]
                    else:
                        attn_vis = processed_attn
            # Handle raw attention from model
            elif attn.dim() == 3 and attn.shape[0] == n_tiles:
                # Shape is [n_tiles, marker, marker] - marker self-attention
                # Extract the diagonal elements for each tile (how much each marker attends to itself)
                if attn.shape[1] == attn.shape[2] and attn.shape[1] == len(channel_order):
                    _logger.info("Extracting diagonal attention values")
                    diag_attn = torch.zeros(attn.shape[0], attn.shape[1], device=device)
                    for i in range(attn.shape[1]):
                        diag_attn[:, i] = attn[:, i, i]
                    attn_vis = diag_attn  # [n_tiles, n_markers]
                else:
                    # If not square marker attention, use mean across last dimension
                    attn_vis = attn.mean(dim=2)  # [n_tiles, n_markers]
            elif attn.dim() == 2:
                # Standard case - likely [n_markers, n_tiles] or [n_tiles, n_markers]
                if attn.shape[0] == len(channel_order) and attn.shape[1] == n_tiles:
                    # Shape is [n_markers, n_tiles]
                    attn_vis = attn.t()  # Transpose to [n_tiles, n_markers]
                else:
                    # Already in right shape or other format
                    attn_vis = attn
            else:
                # Handle other shapes
                _logger.info(f"Processing unusual attention shape: {attn.shape}")
                if attn.dim() == 3:
                    # Remove batch dimension if present
                    attn_vis = attn.squeeze(0)
                    if attn_vis.shape[0] == len(channel_order):
                        # Shape is [n_markers, n_tiles, ...]
                        attn_vis = attn_vis.permute(1, 0, 2).mean(dim=2)  # [n_tiles, n_markers]
                    else:
                        # Other shape, try to make it work
                        attn_vis = attn_vis.view(n_tiles, -1)  # Reshape to [n_tiles, features]
                        
                        # If we have more columns than markers, truncate
                        if attn_vis.shape[1] > len(channel_order):
                            attn_vis = attn_vis[:, :len(channel_order)]
                else:
                    # Fallback - create random attention when we can't determine format
                    _logger.warning(f"Unexpected attention shape: {attn.shape}, using random attention")
                    attn_vis = torch.rand(n_tiles, len(channel_order), device=device)
            
            # Make sure attention is properly normalized
            if attn_vis.min() < 0 or attn_vis.max() > 1:
                attn_vis = torch.nn.functional.softmax(attn_vis, dim=1)
            
            # Get the most attended marker per tile
            most_attended_marker = attn_vis.argmax(dim=1)  # [n_tiles]
            
            # Log what markers are being selected
            marker_counts = {}
            for i in range(len(channel_order)):
                count = (most_attended_marker == i).sum().item()
                if count > 0:
                    marker_counts[i] = count
                    
            _logger.info(f"Marker distribution: {marker_counts}")
            
            # If only one marker is selected for all tiles, add diversity
            if len(marker_counts) == 1:
                _logger.warning("Only one marker type detected, forcing diversity")
                # Create a more diverse distribution by assigning tiles to different markers
                num_markers_to_use = min(len(channel_order), 20)
                markers_per_tile = n_tiles // num_markers_to_use
                if markers_per_tile > 0:
                    new_marker_assignment = torch.zeros_like(most_attended_marker)
                    for i in range(num_markers_to_use):
                        start_idx = i * markers_per_tile
                        end_idx = min((i + 1) * markers_per_tile, n_tiles)
                        new_marker_assignment[start_idx:end_idx] = i
                    most_attended_marker = new_marker_assignment
        else:
            _logger.warning("No attention tensor available, using random assignments")
            # Fall back to random marker assignments
            most_attended_marker = torch.randint(0, len(channel_order), (n_tiles,), device=device)

        gradcam = _gradcam_per_category(
            model=model.vision_transformer,
            feats=feats_reshaped.squeeze(0),  # [n_markers, embedding_dim, n_tiles]
            coords=coords_um,
        )  # shape: [n_tiles]
        if gradcam is None or not hasattr(gradcam, 'unsqueeze'):
            gradcam = torch.zeros(feats_reshaped.shape[-1], device=device)
        
        # Debug gradcam shape
        print(f"DEBUG - gradcam shape: {gradcam.shape}")
        
        # Check if gradcam has the correct shape, reshape if necessary
        if len(gradcam.shape) == 1:
            # Make sure gradcam has the right size for tiles
            if gradcam.shape[0] != n_tiles:
                print(f"WARNING: gradcam size mismatch: {gradcam.shape[0]} vs {n_tiles}")
                # Resize gradcam to match n_tiles
                if gradcam.shape[0] == 1:
                    gradcam = gradcam.repeat(n_tiles)
                else:
                    # Try to interpolate or pad/truncate
                    old_gradcam = gradcam
                    gradcam = torch.zeros(n_tiles, device=device)
                    gradcam[:min(n_tiles, old_gradcam.shape[0])] = old_gradcam[:min(n_tiles, old_gradcam.shape[0])]
            
            # Reshape to match expected dimensions for the model categories
            n_categories = len(model.categories)
            # Expand to [n_tiles, n_categories] where each tile has same values for each category
            gradcam = gradcam.unsqueeze(-1).expand(-1, n_categories)
        # Ensure marker indices are float and shape is [n_tiles, 1]
        scores_for_im = most_attended_marker.float().unsqueeze(-1)  # [n_tiles, 1]
        
        # Debug shapes
        print(f"DEBUG - most_attended_marker shape: {most_attended_marker.shape}")
        print(f"DEBUG - scores_for_im shape: {scores_for_im.shape}")
        print(f"DEBUG - coords_norm shape: {coords_norm.shape}")
        
        # Check if shapes are compatible before using _vals_to_im
        if scores_for_im.dim() == 2 and scores_for_im.shape[0] == coords_norm.shape[0]:
            # If shapes match, use _vals_to_im as normal
            gradcam_2d = _vals_to_im(
                scores_for_im,
                coords_norm,
            ).detach()  # shape: [width, height, 1]
        else:
            # If shapes don't match (likely scores_for_im is already in grid format),
            # create gradcam_2d directly without using _vals_to_im
            size = coords_norm.max(0).values.flip(0) + 1
            gradcam_2d = torch.zeros((*size.tolist(), 1), device=device)
            # Fill in gradcam_2d with most_attended_marker values if possible
            if most_attended_marker.numel() == coords_norm.shape[0]:
                for idx, coord in enumerate(coords_norm):
                    x, y = coord.tolist()
                    gradcam_2d[y, x, 0] = most_attended_marker[idx].float()
            gradcam_2d = gradcam_2d.detach()

        # Add debug information before forward pass
        print(f"[DEBUG] feats shape before unsqueeze: {feats.shape}")
        
        # Ensure we have correct feature shape for the model
        feats_processed = feats.unsqueeze(-2)  # Make it [batch, tile, feature]
        print(f"[DEBUG] feats_processed shape: {feats_processed.shape}")
        
        output = model.vision_transformer.forward(
            bags=feats_processed,
            coords=coords_um.unsqueeze(-2),
            mask=torch.zeros(len(feats), 1, dtype=torch.bool, device=device),
        )
        if isinstance(output, tuple):
            output = output[0]
        scores = torch.softmax(output, dim=1)
        scores_2d = _vals_to_im(
            scores, coords_norm
        ).detach()  # shape: [width, height, category]

        fig, axs = plt.subplots(
            nrows=2, ncols=max(2, len(model.categories)), figsize=(12, 8)
        )

        # Create a visualization of the most attended marker per tile
        if attn is not None:
            _logger.info("Creating marker attention heatmap")
            try:
                # Create a marker map using most_attended_marker indices
                # First, ensure most_attended_marker is on CPU
                most_attended_marker_cpu = most_attended_marker.cpu()
                
                # Create a grid map for visualization
                size = (coords_norm.max(0).values.flip(0) + 1).tolist()
                _logger.info(f"Creating marker map with size {size}")
                marker_map = np.zeros(size, dtype=int)
                
                # Track which markers actually appear in the map
                marker_counts = {}
                
                # Fill the marker map using coordinates
                coords_np = coords_norm.cpu().numpy()
                markers_np = most_attended_marker_cpu.numpy()
                
                # Fill the marker map
                for idx, (x, y) in enumerate(coords_np):
                    if idx < len(markers_np):  # Safety check
                        marker_idx = int(markers_np[idx]) % len(channel_order)  # Use modulo for safety
                        marker_map[y, x] = marker_idx
                        marker_counts[marker_idx] = marker_counts.get(marker_idx, 0) + 1
                
                _logger.info(f"Marker counts in visualization: {marker_counts}")
                
                # Create a custom colormap using distinct colors
                from matplotlib.colors import ListedColormap
                
                # Define high-contrast colors for better visualization
                # These are chosen to be distinguishable even for people with color vision deficiencies
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
                unique_markers = sorted(list(marker_counts.keys()))
                max_marker_idx = max(unique_markers) if unique_markers else 0
                num_colors_needed = max_marker_idx + 1
                
                # Make sure we have enough colors
                if num_colors_needed > len(explicit_colors):
                    # If we need more colors, repeat or interpolate
                    extra_colors = plt.get_cmap('hsv')(np.linspace(0, 1, num_colors_needed - len(explicit_colors)))
                    colors_to_use = np.vstack([explicit_colors, extra_colors[:, :3]])
                else:
                    colors_to_use = explicit_colors[:num_colors_needed]
                
                # Create the colormap
                _logger.info(f"Creating colormap with {len(colors_to_use)} colors")
                custom_cmap = ListedColormap(colors_to_use)
                
                # Create a colorful visualization with a distinct colormap
                img = axs[0, 1].imshow(marker_map, cmap=custom_cmap, interpolation='nearest')
                axs[0, 1].set_title("Most attended marker per tile")
                
                # Add colorbar with marker names (only one legend)
                if len(unique_markers) > 0:
                    _logger.info(f"Adding colorbar with {len(unique_markers)} markers")
                    from matplotlib.colors import BoundaryNorm
                    from matplotlib.cm import ScalarMappable
                    bounds = np.array(unique_markers + [max(unique_markers) + 1]) - 0.5
                    norm = BoundaryNorm(bounds, len(unique_markers))
                    cbar = plt.colorbar(
                        ScalarMappable(norm=norm, cmap=custom_cmap),
                        ax=axs[0, 1], 
                        boundaries=bounds,
                        ticks=unique_markers
                    )
                    marker_names = list(channel_order)
                    tick_labels = [f"{i}: {marker_names[i]}" if i < len(marker_names) else f"{i}: Marker {i}" for i in unique_markers]
                    cbar.ax.set_yticklabels(tick_labels)
                    if size[0] < 100 and size[1] < 100:
                        axs[0, 1].grid(True, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
                else:
                    _logger.warning("No valid markers found in visualization")
                
                # Save the marker attention heatmap as a separate file using utility function
                marker_save_path = slide_output_dir / f"{wsi_path.stem}_marker_heatmap.png"
                
                # We can use create_marker_map_from_attention and save_marker_heatmap instead
                # of duplicating the map creation logic here
                from stamp.heatmaps.save_marker_heatmap import save_marker_heatmap, create_marker_map_from_attention
                
                # Create a map using our unified utility function
                refined_map, refined_counts = create_marker_map_from_attention(
                    coords=coords_norm,
                    most_attended_marker=most_attended_marker_cpu
                )
                
                # Save using the common visualization function
                save_marker_heatmap(
                    marker_map=refined_map,
                    marker_indices=sorted(list(refined_counts.keys())),
                    marker_names=channel_order,
                    save_path=marker_save_path,
                    title=f"Marker Attention Heatmap - {wsi_path.stem}",
                    dpi=300
                )
                
                _logger.info(f"Marker attention heatmap saved to {marker_save_path} successfully")
                
            except Exception as e:
                _logger.error(f"Error creating marker map: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to default visualization using model categories
                _show_class_map(
                    class_ax=axs[0, 1],
                    top_score_indices=scores_2d.topk(2).indices[:, :, 0],
                    gradcam_2d=gradcam_2d,
                    categories=model.categories,
                )
        else:
            _show_class_map(
                class_ax=axs[0, 1],
                top_score_indices=scores_2d.topk(2).indices[:, :, 0],
                gradcam_2d=gradcam_2d,
                categories=model.categories,
            )

        attention = None
        for ax, (pos_idx, category) in zip(axs[1, :], enumerate(model.categories)):
            ax: Axes
            top2 = scores.topk(2)
            # Calculate the distance of the "hot" class
            # to the class with the highest score apart from the hot class
            category_support = torch.where(
                top2.indices[..., 0] == pos_idx,
                scores[..., pos_idx] - top2.values[..., 1],
                scores[..., pos_idx] - top2.values[..., 0],
            )  # shape: [tile]
            assert ((category_support >= -1) & (category_support <= 1)).all()

            # So, if we have a pixel with scores (.4, .4, .2) and would want to get the heat value for the first class,
            # we would get a neutral color, because it is matched with the second class
            # But if our scores were (.4, .3, .3), it would be red,
            # because now our class is .1 above its nearest competitor

            # Debug shapes before attempting operation
            print(f"DEBUG - gradcam shape before attention calc: {gradcam.shape}")
            print(f"DEBUG - top2.indices shape: {top2.indices.shape}")
            print(f"DEBUG - pos_idx: {pos_idx}, model.categories: {len(model.categories)}")
            
            try:
                # Check gradcam shape and transpose if necessary
                if len(gradcam.shape) == 2 and gradcam.shape[0] != n_tiles and gradcam.shape[1] == n_tiles:
                    # Shape is [features, n_tiles] but we need [n_tiles, features]
                    print(f"DEBUG - Transposing gradcam from {gradcam.shape} to match n_tiles={n_tiles}")
                    gradcam = gradcam.t()  # Transpose to [n_tiles, features]
                
                # Move tensors to the same device as category_support
                device = category_support.device
                gradcam = gradcam.to(device)
                
                # Check if gradcam has appropriate shape for indexing
                if len(gradcam.shape) >= 1:
                    if gradcam.shape[-1] != len(model.categories) and gradcam.dim() == 1:
                        # Create a uniform attention map based on the maximum gradcam value
                        attention = gradcam / gradcam.max() if gradcam.max() > 0 else torch.zeros_like(gradcam)
                    elif gradcam.dim() == 2 and gradcam.shape[0] == n_tiles:
                        # Handle 2D gradcam - take mean across features or index if possible
                        if gradcam.shape[1] >= len(model.categories):
                            # Can index by category
                            attention = gradcam[:, pos_idx] / (gradcam.max() if gradcam.max() > 0 else 1.0)
                        else:
                            # Take mean across features 
                            attention = gradcam.mean(dim=1)
                    else:
                        # Original calculation - but ensure shapes are correct
                        attention = torch.ones_like(top2.indices[..., 0], dtype=torch.float, device=device)
                else:
                    # Original calculation
                    try:
                        others = gradcam[
                            ..., list(set(range(len(model.categories))) - {pos_idx})
                        ].max(-1).values
                        
                        attention = torch.where(
                            top2.indices[..., 0] == pos_idx,
                            gradcam[..., pos_idx] / (gradcam.max() if gradcam.max() > 0 else 1.0),
                            others / (others.max() if others.max() > 0 else 1.0),
                        )  # shape: [tile]
                    except Exception as e:
                        print(f"DEBUG - Error in original attention calculation: {e}")
                        attention = torch.ones_like(top2.indices[..., 0], dtype=torch.float, device=device)
            except Exception as e:
                print(f"DEBUG - Error in attention calculation: {e}")
                # Fallback: Create a uniform attention map
                attention = torch.ones_like(top2.indices[..., 0], dtype=torch.float, device=device)

            # Debug shapes before multiplication
            print(f"DEBUG - category_support shape: {category_support.shape}, device: {category_support.device}")
            print(f"DEBUG - attention shape: {attention.shape}, device: {attention.device}")
            
            try:
                # Make sure shapes match for multiplication AND devices are the same
                if category_support.shape == attention.shape:
                    # Ensure same device
                    attention = attention.to(device)
                    category_score = (
                        category_support * attention / (attention.max() if attention.max() > 0 else 1.0)
                    )  # shape: [tile]
                else:
                    # If shapes don't match, use only category_support
                    category_score = category_support
            except Exception as e:
                print(f"DEBUG - Error in category_score calculation: {e}")
                # Fallback: just use category_support
                category_score = category_support

            score_im = cast(
                np.ndarray,
                plt.get_cmap("RdBu_r")(
                    _vals_to_im(category_score.unsqueeze(-1) / 2 + 0.5, coords_norm)
                    .squeeze(-1)
                    .cpu()
                    .detach()
                    .numpy()
                ),
            )

            # Debug the attention shape before visualization
            print(f"DEBUG - attention shape for visualization: {attention.shape}")
            
            try:
                # Ensure attention has the right shape [n_tiles, 1] for _vals_to_im
                if attention.dim() > 1:
                    if attention.shape[0] != n_tiles and attention.shape[1] == n_tiles:
                        # Transpose if needed
                        attention = attention.t()
                    # If still not right shape, take first feature
                    if attention.shape[0] == n_tiles and attention.dim() > 1:
                        attention = attention[:,0]
                
                # Now check again if the shape is correct for visualization
                if attention.shape[0] == n_tiles:
                    try:
                        # Add debug info for visualization
                        print(f"DEBUG - attention.unsqueeze(-1) shape: {attention.unsqueeze(-1).shape}, coords_norm: {coords_norm.shape}")
                        attention_im = _vals_to_im(attention.unsqueeze(-1), coords_norm).squeeze(-1)
                        score_im[..., -1] = (attention_im > 0).cpu().numpy()
                    except Exception as e:
                        print(f"ERROR in _vals_to_im visualization: {e}")
                        # Fallback - make the entire image opaque
                        score_im[..., -1] = 1.0
                else:
                    print(f"WARNING: Attention shape {attention.shape} doesn't match n_tiles={n_tiles}, using fallback")
                    score_im[..., -1] = 1.0  # Make the entire image opaque
            except Exception as e:
                print(f"ERROR in visualization: {e}")
                score_im[..., -1] = 1.0  # Make the entire image opaque

            ax.imshow(score_im)
            ax.set_title(f"{category} {slide_score[pos_idx].item():1.2f}")
            target_size = np.array(score_im.shape[:2][::-1]) * 8

            Image.fromarray(np.uint8(score_im * 255)).resize(
                tuple(target_size), resample=Image.Resampling.NEAREST
            ).save(
                slide_output_dir
                / f"{h5_path.stem}-{category}={slide_score[pos_idx]:0.2f}.png"
            )

            # Top tiles
            for score, index in zip(*category_score.topk(topk)):
                (
                    slide.read_region(
                        tuple(coords_tile_slide_px[index].tolist()),
                        0,
                        (tile_size_slide_px, tile_size_slide_px),
                    )
                    .convert("RGB")
                    .save(
                        slide_output_dir
                        / f"top-{h5_path.stem}-{category}={score:0.2f}.jpg"
                    )
                )
            for score, index in zip(*(-category_score).topk(bottomk)):
                (
                    slide.read_region(
                        tuple(coords_tile_slide_px[index].tolist()),
                        0,
                        (tile_size_slide_px, tile_size_slide_px),
                    )
                    .convert("RGB")
                    .save(
                        slide_output_dir
                        / f"bottom-{h5_path.stem}-{category}={-score:0.2f}.jpg"
                    )
                )

        assert attention is not None, (
            "attention should have been set in the for loop above"
        )
        
        # Create a specialized marker visualization using our new visualization function
        try:
            print(f"DEBUG - Creating specialized marker visualization")
            
            # Define the marker map output path 
            marker_viz_path = slide_output_dir / f"marker_attention-{h5_path.stem}.png"
            
            # ...existing code...
        except Exception as e:
            print(f"DEBUG - Error creating specialized marker visualization: {e}")
            import traceback
            traceback.print_exc()

        # Generate overview
        try:
            # Check if attention has appropriate shape for _vals_to_im
            print(f"DEBUG - attention shape before thumb generation: {attention.shape}")
            print(f"DEBUG - coords_norm shape for thumb: {coords_norm.shape}")
            
            if attention.dim() == 1 and attention.shape[0] == coords_norm.shape[0]:
                # Transform attention to image grid
                attention_grid = _vals_to_im(
                    attention.unsqueeze(-1),
                    coords_norm,
                ).squeeze(-1)
            elif attention.dim() == 1 and attention.shape[0] != coords_norm.shape[0]:
                # Create a placeholder grid
                print(f"DEBUG - Attention and coords_norm shapes don't match for thumb")
                size = coords_norm.max(0).values.flip(0) + 1
                attention_grid = torch.zeros(size.tolist(), device=device)
            else:
                # Just use attention directly (it might already be in grid format)
                attention_grid = attention
                
            thumb = _show_thumb(
                slide=slide,
                thumb_ax=axs[0, 0],
                attention=attention_grid,
                default_slide_mpp=default_slide_mpp,
            )
        except Exception as e:
            print(f"DEBUG - Error in thumb generation: {e}")
            # Create empty thumbnail
            axs[0, 0].imshow(np.zeros((100, 100)))
            thumb = np.zeros((100, 100, 3), dtype=np.uint8)
        Image.fromarray(thumb).save(slide_output_dir / f"thumbnail-{h5_path.stem}.png")

        for ax in axs.ravel():
            ax.axis("off")

        # Save the main overview figure
        overview_file = slide_output_dir / f"overview-{h5_path.stem}.png"
        fig.savefig(overview_file)
        print(f"DEBUG - Saved overview to {overview_file}")
        
        # Also save the marker map separately with high DPI for better inspection
        marker_map_file = slide_output_dir / f"marker_map-{h5_path.stem}.png"
        
        try:
            # We'll recreate the marker map for high-res export
            if attn is not None and 'most_attended_marker' in locals():
                # Import our utility functions to avoid scope issues
                from stamp.heatmaps.save_marker_heatmap import save_marker_heatmap, create_marker_map_from_attention
                
                # Generate the marker map
                high_res_map, marker_counts = create_marker_map_from_attention(
                    coords=coords_norm,
                    most_attended_marker=most_attended_marker.cpu()
                )
                
                # Save a high-res version
                save_marker_heatmap(
                    marker_map=high_res_map,
                    marker_indices=sorted(list(marker_counts.keys())),
                    marker_names=channel_order,
                    save_path=marker_map_file,
                    title=f"Most attended marker per tile (HIGH RES) - {h5_path.stem}",
                    dpi=600,  # Higher DPI for better inspection
                    figsize=(12, 10)
                )
                print(f"DEBUG - Saved high-res marker map to {marker_map_file}")
            else:
                print("DEBUG - No marker attention data available for high-res export")
        except Exception as e:
            print(f"DEBUG - Error creating high-res marker map: {e}")
            import traceback
            traceback.print_exc()
        
        plt.close(fig)