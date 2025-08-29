"""
In parts from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
"""

from collections.abc import Iterable
from typing import assert_never, cast, Any

import torch
from beartype import beartype
from einops import repeat
from jaxtyping import Bool, Float, jaxtyped
from torch import Tensor, nn

from stamp.modeling.alibi import MultiHeadALiBi
from fluoroformer.layers import MarkerAttention, PatchAttention


def feed_forward(
    dim: int,
    hidden_dim: int,
    dropout: float = 0.5,
) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout),
    )


class SelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        dropout: float,
        use_alibi: bool,
    ) -> None:
        super().__init__()
        self.heads = num_heads
        self.norm = nn.LayerNorm(dim)

        if use_alibi:
            self.mhsa = MultiHeadALiBi(
                embed_dim=dim,
                num_heads=num_heads,
            )
        else:
            self.mhsa = nn.MultiheadAttention(dim, num_heads, dropout, batch_first=True)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[Tensor, "batch sequence proj_feature"],
        *,
        coords: Float[Tensor, "batch sequence xy"],
        attn_mask: Bool[Tensor, "batch sequence sequence"] | None,
        # Help, my abstractions are leaking!
        alibi_mask: Bool[Tensor, "batch sequence sequence"] | None,
    ) -> Float[Tensor, "batch sequence proj_feature"]:
        """
        Args:
            attn_mask:
                Which of the features to ignore during self-attention.
                `attn_mask[b,q,k] == False` means that
                query `q` of batch `b` can attend to key `k`.
                If `attn_mask` is `None`, all tokens can attend to all others.
            alibi_mask:
                Which query-key pairs to apply ALiBi to.
                If this module was constructed using `use_alibi=False`,
                this has no effect.
        """
        x = self.norm(x)
        match self.mhsa:
            case nn.MultiheadAttention():
                attn_output, _ = self.mhsa(
                    x,
                    x,
                    x,
                    need_weights=False,
                    attn_mask=(
                        attn_mask.repeat(self.mhsa.num_heads, 1, 1)
                        if attn_mask is not None
                        else None
                    ),
                )
            case MultiHeadALiBi():
                attn_output = self.mhsa(
                    q=x,
                    k=x,
                    v=x,
                    coords_q=coords,
                    coords_k=coords,
                    attn_mask=attn_mask,
                    alibi_mask=alibi_mask,
                )
            case _ as unreachable:
                assert_never(unreachable)

        return attn_output


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dropout: float,
        use_alibi: bool,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        SelfAttention(
                            dim=dim,
                            num_heads=heads,
                            dropout=dropout,
                            use_alibi=use_alibi,
                        ),
                        feed_forward(
                            dim,
                            mlp_dim,
                        ),
                    ]
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(dim)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[Tensor, "batch sequence proj_feature"],
        *,
        coords: Float[Tensor, "batch sequence 2"],
        attn_mask: Bool[Tensor, "batch sequence sequence"] | None,
        alibi_mask: Bool[Tensor, "batch sequence sequence"] | None,
    ) -> Float[Tensor, "batch sequence proj_feature"]:
        for attn, ff in cast(Iterable[tuple[nn.Module, nn.Module]], self.layers):
            x_attn = attn(x, coords=coords, attn_mask=attn_mask, alibi_mask=alibi_mask)
            x = x_attn + x
            x = ff(x) + x

        x = self.norm(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim_output: int,
        dim_input: int,
        dim_model: int,
        n_layers: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        use_alibi: bool,
        use_marker_attention: bool = True,
        marker_hidden_dim: int = 256,
        total_steps: int = 1000,  
    ) -> None:
        super().__init__()
        self.total_steps = total_steps
        self.class_token = nn.Parameter(torch.randn(dim_model))

        # Add marker attention module only if needed
        self.use_marker_attention = use_marker_attention
        print(f"[DEBUG] Using MARKER ATTENTION:")
        if use_marker_attention:
            print(f"[DEBUG] Using MARKER ATTENTION with dim_input={dim_input}")
            # Make sure pre_projection uses the right input dimension (1536)
            self.pre_projection = nn.Linear(dim_input, 512)  # Hard-code the input dimension to match data
            self.marker_attention = MarkerAttention(
                embedding_dim=512,
                hidden_dim=marker_hidden_dim,
                num_heads=1,
                dropout=dropout
            )
            self.patch_attention = PatchAttention(
                input_dim=512,              # Use 'input_dim' instead of 'embedding_dim'
                hidden_dim=marker_hidden_dim,
                dropout=dropout
            )
    
            

        if use_marker_attention:
            self.pre_projection = nn.Linear(dim_input, 512)
            self.project_features = nn.Sequential(
                nn.Linear(512, dim_model, bias=True),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.project_features = nn.Sequential(
                nn.Linear(dim_input, dim_model, bias=True),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.transformer = Transformer(
            dim=dim_model,
            depth=n_layers,
            heads=n_heads,
            mlp_dim=dim_feedforward,
            dropout=dropout,
            use_alibi=use_alibi,
        )

        self.mlp_head = nn.Sequential(nn.Linear(dim_model, dim_output))

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        bags: Float[Tensor, "batch tile feature"] | Float[Tensor, "batch marker embedding patch"],
        *,
        coords: Float[Tensor, "batch tile 2"],
        mask: Bool[Tensor, "batch tile"] | None = None,
        return_marker_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Any] | tuple[Tensor, Any, Any]:
        # Handle multiplex data format if marker attention is enabled
        print(f"[DEBUG] use_marker_attention: {self.use_marker_attention}")
        print(f"[DEBUG] bags.shape at entry: {bags.shape}, bags.dim(): {bags.dim()}")
        print(f"[DEBUG] return_marker_attention: {return_marker_attention}")
        
        marker_attn = None
        patch_attn = None
        
        # Handle multiplex data (4D tensor)
        if self.use_marker_attention and bags.dim() == 4:
            # Input is [batch, marker, embedding, patch]
            batch_size, n_markers, embedding_dim, n_patches = bags.shape
            
            # Need to project from embedding_dim (1536) to 512 before marker attention
            # Reshape to apply linear projection
            bags_reshaped = bags.permute(0, 1, 3, 2).reshape(-1, embedding_dim)
            projected_bags = self.pre_projection(bags_reshaped)
            
            # Reshape back to [batch, marker, 512, patch]
            projected_bags = projected_bags.reshape(batch_size, n_markers, n_patches, 512).permute(0, 1, 3, 2)
            
            # Apply marker attention
            bags, marker_attn = self.marker_attention(projected_bags)

            print(f"[DEBUG] bags after marker_attention: {bags.shape}")
            # If bags is [batch, patch, embedding], coords should be [batch, patch, 2]
            if bags.shape[1] != coords.shape[1]:
                print(f"[WARNING] bags and coords sequence mismatch: {bags.shape[1]} vs {coords.shape[1]}")
                # Fix by slicing or pooling coords as needed
                coords = coords[:, :bags.shape[1], :]
            
        # Handle standard data (3D tensor)
        # Now process as standard input with shape [batch, tile, feature]
        print(f"[DEBUG] bags.shape before unpack: {bags.shape}")
        if bags.dim() == 3:
            batch_size, n_tiles, n_features = bags.shape
        elif bags.dim() == 2:
            # Likely [batch, feature] after pooling/attention
            batch_size, n_features = bags.shape
            n_tiles = 1
            bags = bags.unsqueeze(1)  # [batch, 1, feature]
            print(f"[DEBUG] bags reshaped to: {bags.shape}")
        else:
            raise ValueError(f"Unexpected bags shape: {bags.shape}")

        # Map input sequence to latent space
        # For standard (non-multiplex) features, bypass pre_projection if marker attention is enabled
        if self.use_marker_attention and n_features == 1536:
            # First handle the dimension mismatch by using pre_projection
            print(f"[DEBUG] Projecting standard features from {n_features} to 512")
            bags_reshaped = bags.reshape(-1, n_features)
            try:
                projected_bags = self.pre_projection(bags_reshaped)
                bags = projected_bags.reshape(batch_size, n_tiles, 512)
                print(f"[DEBUG] After projection: bags.shape={bags.shape}")
            except Exception as e:
                print(f"[ERROR] Projection failed: {e}")
                print(f"[DEBUG] bags_reshaped.shape={bags_reshaped.shape}, self.pre_projection.weight.shape={self.pre_projection.weight.shape}")
                # Fallback: use a new projection layer with correct dimensions
                temp_projection = nn.Linear(n_features, 512).to(bags.device)
                bags = temp_projection(bags_reshaped).reshape(batch_size, n_tiles, 512)
        
        bags = self.project_features(bags)
        
        # Prepend a class token to every bag
        cls_tokens = repeat(self.class_token, "d -> b 1 d", b=batch_size)
        bags = torch.cat([cls_tokens, bags], dim=1)
        coords = torch.cat(
            [torch.zeros(batch_size, 1, 2).type_as(coords), coords], dim=1
        )

        # --- FIX: Ensure bags and coords have matching sequence length ---
        if bags.shape[1] != coords.shape[1]:
            min_seq = min(bags.shape[1], coords.shape[1])
            print(f"[WARNING] Truncating bags, coords, and mask to min_seq={min_seq}")
            bags = bags[:, :min_seq]
            coords = coords[:, :min_seq]
            if mask is not None:
                mask = mask[:, :min_seq-1]  # -1 because mask does not include class token

        # The rest of the method stays exactly the same
        match mask:
            case None:
                bags = self.transformer(
                    bags, coords=coords, attn_mask=None, alibi_mask=None
                )

            case _:
                mask_with_class_token = torch.cat(
                    [torch.zeros(mask.shape[0], 1).type_as(mask), mask], dim=1
                )
                square_attn_mask = torch.einsum(
                    "bq,bk->bqk", mask_with_class_token, mask_with_class_token
                )
                # Don't allow other tiles to reference the class token
                square_attn_mask[:, 1:, 0] = True

                # Don't apply ALiBi to the query, as the coordinates don't make sense here
                alibi_mask = torch.zeros_like(square_attn_mask)
                alibi_mask[:, 0, :] = True
                alibi_mask[:, :, 0] = True

                # Truncate mask to match bags/coords sequence length
                seq_len = bags.shape[1]
                square_attn_mask = square_attn_mask[:, :seq_len, :seq_len]

                bags = self.transformer(
                    bags,
                    coords=coords,
                    attn_mask=square_attn_mask,
                    alibi_mask=alibi_mask,
                )

        # Only take class token
        bags = bags[:, 0]
        logits = self.mlp_head(bags)

        if return_marker_attention:
            return logits, marker_attn, patch_attn
        return logits
