from torch import nn

class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim=768,
    num_heads=12, attn_dropout=0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(
            normalized_shape=embedding_dim) 
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(
            query=x, key=x, value=x, need_weights=False
        )
        return attn_output


class MLPBlock(nn.Module):
    def __init__(self, embedding_dim=128,
    mlp_size=512,
    mlp_dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(
            normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
            out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=mlp_dropout),
            nn.Linear(in_features=mlp_size,
            out_features=embedding_dim),
            nn.Dropout(p=mlp_dropout)
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim=128,
    num_heads=4, mlp_size=512, mlp_dropout=0.1, attn_dropout=0):
        super().__init__()
        self.msa_block = MultiheadSelfAttentionBlock(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout
        )
        self.mlp_block = MLPBlock(
            embedding_dim=embedding_dim,
            mlp_size=mlp_size,
            mlp_dropout=mlp_dropout)

    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x
        return x
