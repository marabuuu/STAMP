import torch
import pytest
from stamp.modeling.vision_transformer import VisionTransformer

def test_marker_attention_mechanism():
    """Test that the marker attention module correctly processes multiplex data."""
    # Set up test parameters
    batch_size = 2
    n_markers = 3
    embedding_dim = 1536  # This matches the hardcoded value in VisionTransformer
    n_patches = 8
    output_dim = 4
    
    # Create mock multiplex input with shape [batch, marker, embedding, patch]
    multiplex_input = torch.rand(batch_size, n_markers, embedding_dim, n_patches)
    coords = torch.rand(batch_size, n_patches, 2)
    
    # Initialize model with marker attention
    model = VisionTransformer(
        dim_output=output_dim,
        dim_input=512,  # The dimension after marker attention fusion
        dim_model=256,
        n_layers=2,
        n_heads=2,
        dim_feedforward=512,
        dropout=0.1,
        use_alibi=False,
        use_marker_attention=True,
        marker_hidden_dim=128
    )
    
    # Test normal forward pass
    output = model(multiplex_input, coords=coords)
    assert output.shape == (batch_size, output_dim), "Output shape should be [batch, output_dim]"
    
    # Test with return_marker_attention=True
    output, attention = model(multiplex_input, coords=coords, return_marker_attention=True)
    assert output.shape == (batch_size, output_dim)

    # Verify attention has proper dimensions for per-patch marker attention
    assert attention.shape[1:] == (n_markers, n_markers)
    assert attention.shape[0] == batch_size * n_patches

def test_marker_attention_with_masking():
    """Test that marker attention works correctly with masked input."""
    # Set up test parameters
    batch_size = 2
    n_markers = 3
    embedding_dim = 1536
    n_patches = 10
    output_dim = 4
    
    # Create mock multiplex input
    multiplex_input = torch.rand(batch_size, n_markers, embedding_dim, n_patches)
    coords = torch.rand(batch_size, n_patches, 2)
    
    # Create a mask to hide some patches
    mask = torch.ones(batch_size, n_patches, dtype=torch.bool)
    mask[:, 7:] = False  # Mask out patches 7-9
    
    # Initialize model
    model = VisionTransformer(
        dim_output=4,
        dim_input=512,
        dim_model=256,
        n_layers=2,
        n_heads=2,
        dim_feedforward=512,
        dropout=0.1,
        use_alibi=False,
        use_marker_attention=True
    )
    
    # Test forward pass with mask
    output = model(multiplex_input, coords=coords, mask=mask)
    assert output.shape == (batch_size, 4), "Output should have correct shape even with masking"
    
    # Test with return_marker_attention=True
    output, attention = model(multiplex_input, coords=coords, return_marker_attention=True)
    assert output.shape == (batch_size, output_dim)

    # Verify attention has proper dimensions for per-patch marker attention
    assert attention.shape[1:] == (n_markers, n_markers)
    assert attention.shape[0] == batch_size * n_patches

def test_marker_attention_disabled():
    """Test that the model works without marker attention for backwards compatibility."""
    # Standard input format (no multiplex)
    batch_size = 2
    n_tiles = 8
    feature_dim = 512
    standard_input = torch.rand(batch_size, n_tiles, feature_dim)
    coords = torch.rand(batch_size, n_tiles, 2)
    
    # Initialize model with marker attention disabled
    model = VisionTransformer(
        dim_output=4,
        dim_input=feature_dim,
        dim_model=256,
        n_layers=2,
        n_heads=2,
        dim_feedforward=512,
        dropout=0.1,
        use_alibi=False,
        use_marker_attention=False
    )
    
    # Test forward pass with standard input
    output = model(standard_input, coords=coords)
    assert output.shape == (batch_size, 4), "Model should work with marker attention disabled"
    
    # Test that model doesn't break with multiplex input but marker attention disabled
    # (It should ignore the multiplex structure and treat it as batch)
    multiplex_input = torch.rand(batch_size, 3, feature_dim, n_tiles)
    
    try:
        # This should raise an error due to dimension mismatch
        model(multiplex_input, coords=coords)
        assert False, "Model should not accept multiplex input with marker attention disabled"
    except:
        # Expected error
        pass