import tempfile
from pathlib import Path
import torch
import h5py
import pytest
from random_data import make_feature_file
from stamp.modeling.data import MultiplexBagDataset

@pytest.fixture
def multiplex_feature_files(tmp_path):
    # Simulate 3 markers, each with its own .h5 file for one sample
    markers = ['CD3', 'CD8', 'CD20']
    files = []
    for marker in markers:
        feats = torch.rand(10, 32)  # 10 tiles, 32 features
        coords = torch.rand(10, 2)
        file = make_feature_file(feats=feats, coords=coords)
        files.append(file)
    return files, markers

def test_multiplex_bag_dataset_shape(multiplex_feature_files):
    files, markers = multiplex_feature_files
    ds = MultiplexBagDataset(
        bags=[files],  # One sample with 3 marker files
        channel_order=markers,
        bag_size=8,
        ground_truths=torch.tensor([[1, 0, 0]], dtype=torch.bool),  # Dummy one-hot
        transform=None,
    )
    assert len(ds) == 1
    bag, coords, bag_size, target = ds[0]
    # bag: [markers, features, tiles] after permute in __getitem__
    assert bag.shape == (len(markers), 32, 8)
    assert coords.shape == (8, 2)
    assert bag_size == 8
    assert target.shape == (3,)

def test_multiplex_missing_marker(multiplex_feature_files):
    files, markers = multiplex_feature_files
    # Remove one marker file to simulate missing marker
    files = files[:-1]
    ds = MultiplexBagDataset(
        bags=[files],
        channel_order=markers,
        bag_size=8,
        ground_truths=torch.tensor([[1, 0, 0]], dtype=torch.bool),
        transform=None,
    )
    bag, coords, bag_size, target = ds[0]
    # The missing marker should be all zeros
    assert (bag[-1] == 0).all()

def test_multiplex_variable_bag_size(multiplex_feature_files):
    files, markers = multiplex_feature_files
    ds = MultiplexBagDataset(
        bags=[files],
        channel_order=markers,
        bag_size=None,  # Use all tiles
        ground_truths=torch.tensor([[1, 0, 0]], dtype=torch.bool),
        transform=None,
    )
    bag, coords, bag_size, target = ds[0]
    # Should use the minimum number of tiles across markers
    assert bag.shape[2] == min(10, 10, 10)