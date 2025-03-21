import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from random_data import create_random_dataset

from stamp.modeling.crossval import categorical_crossval_


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:No positive samples in targets")
def test_crossval_integration(
    *,
    tmp_path: Path,
    n_patients: int = 800,
    max_slides_per_patient: int = 3,
    min_tiles_per_slide: int = 8,
    max_tiles_per_slide: int = 2**10,
    feat_dim: int = 25,
    n_categories: int = 3,
    use_alibi: bool = False,
    use_vary_precision_transform: bool = False,
) -> None:
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    clini_path, slide_path, feature_dir, categories = create_random_dataset(
        dir=tmp_path,
        n_categories=n_categories,
        n_patients=n_patients,
        max_slides_per_patient=max_slides_per_patient,
        min_tiles_per_slide=min_tiles_per_slide,
        max_tiles_per_slide=max_tiles_per_slide,
        feat_dim=feat_dim,
    )

    output_dir = tmp_path / "output"

    categorical_crossval_(
        clini_table=clini_path,
        slide_table=slide_path,
        feature_dir=feature_dir,
        output_dir=output_dir,
        patient_label="patient",
        ground_truth_label="ground-truth",
        filename_label="slide_path",
        categories=categories,
        # Dataset and -loader parameters
        bag_size=max_tiles_per_slide // 2,
        val_bag_size=None,
        num_workers=min(os.cpu_count() or 1, 7),
        # Training paramenters
        batch_size=8,
        max_epochs=2,
        patience=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        n_splits=2,
        # Experimental features
        use_vary_precision_transform=use_vary_precision_transform,
        use_alibi=use_alibi,
    )
