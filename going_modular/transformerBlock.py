from msa_mlp import TransformerEncoderBlock
import torch

class ViT(torch.nn.Module):
    def __init__(self,
    num_patches=28*28,
    num_transformer_layers=2,
    embedding_dim=128,
    mlp_size=512,
    num_heads=4,
    attn_dropout=0,
    mlp_dropout=0.1,
    embedding_dropout=0.1,
    num_classes=15,
    use_cls_token=True):
        super().__init__()
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.class_embedding = torch.nn.Parameter(data=torch.rand(1, 1, embedding_dim), requires_grad=True)
            self.position_embedding = torch.nn.Parameter(data=torch.rand(1, num_patches+1, embedding_dim), requires_grad=True)
        else:
            self.position_embedding = torch.nn.Parameter(data=torch.rand(1, num_patches, embedding_dim), requires_grad=True)

        self.embedding_dropout = torch.nn.Dropout(p=embedding_dropout)
        self.transformer_encoder = torch.nn.Sequential(*[TransformerEncoderBlock(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            mlp_size=mlp_size,
            mlp_dropout=mlp_dropout
        ) for _ in range(num_transformer_layers)])
        if self.use_cls_token:
            self.classifier = torch.nn.Sequential(
                torch.nn.LayerNorm(normalized_shape=embedding_dim),
                torch.nn.Linear(in_features=embedding_dim,
                out_features=num_classes)
            )
    def forward(self, x):
        batch_size = x.shape[0]
        if self.use_cls_token:
            class_token = self.class_embedding.expand(batch_size, -1, -1)
            x = torch.cat((class_token, x), dim=1)
        x = self.position_embedding + x
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        if self.use_cls_token:
            x = self.classifier(x[:, 0])
        return x
