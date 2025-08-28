from pathlib import Path
from pydantic import BaseModel, ConfigDict

class AttentionHeatmapConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: Path
    feature_dir: Path
    wsi_dir: Path
    checkpoint_path: Path
    slide_paths: list[Path] | None = None
    device: str = "cuda"
    top_k_percent: float = 0.1  # Use top 10% patches
    channel_order: list[str]