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
from fluoroformer.layers import MarkerAttention

from stamp.modeling.alibi import MultiHeadALiBi


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
        alibi_mask: Bool[Tensor, "batch sequence sequence"],
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
        alibi_mask: Bool[Tensor, "batch sequence sequence"],
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
        hidden_dim: int,
        embedding_dim: int, 
    ) -> None:
        super().__init__()
        self.class_token = nn.Parameter(torch.randn(dim_model))

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

        self.marker_attention = MarkerAttention(embedding_dim, hidden_dim) # needed input: (batch, marker, embedding_dim, n_tiles)
        self.mlp_head = nn.Sequential(nn.Linear(dim_model, dim_output))

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        bags: Float[Tensor, "batch ... feature"],
        *,
        coords: Float[Tensor, "batch ... 2"],
        mask: Bool[Tensor, "batch ..."] | None,
    ) -> tuple[Tensor, Any]:
        print("BAGS SHAPE AT FORWARD ENTRY:", bags.shape)

        # --- Shape normalization ---
        if bags.dim() == 3:
            # Classic: (batch, tile, feature)
            batch_size, n_tiles, embedding_dim = bags.shape
            n_markers = 1
            bags = bags.unsqueeze(1)  # (batch, 1, tile, feature)
            coords = coords.unsqueeze(1)  # (batch, 1, tile, 2)
        elif bags.dim() == 4:
            # Multiplex: (batch, marker, tile, feature)
            batch_size, n_markers, n_tiles, embedding_dim = bags.shape
        else:
            raise ValueError(f"Unexpected bags shape: {bags.shape}")

        # Permute to (batch, marker, embedding_dim, n_tiles)
        bags = bags.permute(0, 1, 3, 2)  # (batch, marker, embedding_dim, n_tiles)
        coords = coords.permute(0, 1, 3, 2) if coords.dim() == 4 else coords  # (batch, marker, 2, n_tiles)

        # Flatten marker and tile for transformer input
        bags = bags.reshape(batch_size, n_markers * n_tiles, embedding_dim)  # (batch, sequence, embedding_dim)
        coords = coords.reshape(batch_size, n_markers * n_tiles, 2)  # (batch, sequence, 2)
        if mask is not None:
            mask = mask.reshape(batch_size, n_markers * n_tiles)  # (batch, sequence)

        # Project features to model dimension
        bags = self.project_features(bags)  # (batch, sequence, dim_model)

        # Prepend class token
        cls_tokens = repeat(self.class_token, "d -> b 1 d", b=batch_size)
        bags = torch.cat([cls_tokens, bags], dim=1)
        coords = torch.cat(
            [torch.zeros(batch_size, 1, 2).type_as(coords), coords], dim=1
        )

        # Prepare attention mask
        if mask is not None:
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

            bags = self.transformer(
                bags,
                coords=coords,
                attn_mask=square_attn_mask,
                alibi_mask=alibi_mask,
            )
        else:
            bags = self.transformer(bags, coords=coords, attn_mask=None, alibi_mask=None)

        # Remove the class token
        bags_wo_cls = bags[:, 1:, :]  # (batch, n_markers * n_tiles, embedding_dim)

        # Reshape to (batch, marker, n_tiles, embedding_dim)
        bags_reshaped = bags_wo_cls.view(batch_size, n_markers, n_tiles, embedding_dim)

        # Permute to (batch, marker, embedding_dim, n_tiles) for MarkerAttention
        bags_for_marker_attention = bags_reshaped.permute(0, 1, 3, 2)

        # MarkerAttention expects (batch, marker, embedding_dim, n_tiles)
        marker_attn_out = self.marker_attention(bags_for_marker_attention)

        # Final MLP head
        logits = self.mlp_head(marker_attn_out)

        return logits, marker_attn_out
