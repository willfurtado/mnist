import numpy as np
import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange


class ViT(nn.Module):
    """
    Vision transformer architecture definition
    """

    def __init__(
        self,
        patch_size: int = 7,
        dim: int = 64,
        depth: int = 6,
        heads: int = 8,
        mlp_dim: int = 128,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ):
        """
        Creates an instance of the `ViT` class

        Parameters:
            patch_size (int, optional): _description_. Defaults to 7.
            dim (int, optional): _description_. Defaults to 64.
            depth (int, optional): _description_. Defaults to 6.
            heads (int, optional): _description_. Defaults to 8.
            mlp_dim (int, optional): _description_. Defaults to 128.
            dim_head (int, optional): _description_. Defaults to 64.
            dropout (float, optional): _description_. Defaults to 0.0.
            emb_dropout (float, optional): _description_. Defaults to 0.0.
        """
        super().__init__()

        image_height, image_width = 28, 28
        num_classes = 10
        channels = 1

        patch_height = patch_size
        patch_width = patch_size

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.transformer = nn.Sequential(
            *nn.ModuleList(
                [ViTLayer(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)]
            )
        )

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the `ViT` model

        Parameters:
            img (torch.Tensor): _description_

        Returns:
            (torch.Tensor): _description_
        """
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.emb_dropout(x)

        x = self.transformer(x)

        x = x[:, 0]
        return self.mlp_head(x)


class ViTLayer(nn.Module):
    """
    Single transformer layer for ViT architecture
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        """
        Creates an instance of the `ViT` layer class

        Parameters:
            dim (int): _description_
            heads (int): _description_
            dim_head (int): _description_
            mlp_dim (int): _description_
            dropout (float, optional): _description_. Defaults to 0.0.
        """
        super().__init__()

        self.attn = nn.Sequential(
            nn.LayerNorm(dim),
            MultiHeadAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
            nn.Dropout(p=dropout),
            nn.Linear(heads * dim_head, dim, bias=False),
        )

        self.feedforward = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the `VitLayer` block

        Parameters:
            x (torch.Tensor): _description_

        Returns:
            (torch.Tensor): _description_
        """
        x = x + self.attn(x)
        x = x + self.feedforward(x)

        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-headed attention block
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        """
        Creates an instance of the `MultiHeadAttention` class

        Parameters:
            dim (int): _description_
            heads (int, optional): _description_. Defaults to 8.
            dim_head (int, optional): _description_. Defaults to 64.
            dropout (float, optional): _description_. Defaults to 0.0.
        """
        super().__init__()

        self.attention_heads = nn.ModuleList(
            [Attention(dim, dim_head, dropout=dropout) for _ in range(heads)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs a forward pass of the `MultiHeadAttention` block

        Parameters:
            x (torch.Tensor): _description_

        Returns:
            (torch.Tensor): _description_
        """
        return torch.cat([head.forward(x) for head in self.attention_heads], dim=-1)


class Attention(nn.Module):
    """
    Attention block used within ViT architecture

    A key component of the Vision Transformer is the self-attention mechanism. Described by
    the infamous paper entitled [Attention is All You Need ](https://arxiv.org/abs/1706.03762),
    we use scaled dot-product attention.
    """

    def __init__(self, input_dim: int, inner_dim: int, dropout: float = 0.0):
        """
        Creates an instance of the `Attention` class

        Parameters:
            input_dim (int): _description_
            inner_dim (int): _description_
            dropout (float, optional): _description_. Defaults to 0.0.
        """
        super().__init__()

        self.Q = nn.Linear(input_dim, inner_dim, bias=False)
        self.K = nn.Linear(input_dim, inner_dim, bias=False)
        self.V = nn.Linear(input_dim, inner_dim, bias=False)

        self.softmax = nn.Softmax(dim=1)
        self.scale = 1 / np.sqrt(inner_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the `Attention` block

        Parameters:
            x (torch.Tensor): _description_

        Returns:
            (torch.Tensor): _description_
        """
        Q_proj = self.Q(x)
        K_proj = self.K(x)
        V_proj = self.V(x)

        K_T = K_proj.transpose(1, 2)
        QK_T = torch.matmul(Q_proj, K_T) * self.scale
        attention = self.softmax(QK_T)
        self.attention_cache = attention
        out = torch.matmul(attention, V_proj)
        out = self.dropout(out)

        return out
