from resnetBlock import ResNetBlock
from transformerBlock import ViT
from torch import nn
class HybridModel(nn.Module):
    def __init__(self, in_channels=3, hidden_units=64, 
    num_transformer_layers=2, embedding_dim=128, num_heads=4, mlp_size=512, num_classes=15):
        super().__init__()
        self.cnn = ResNetBlock()
        self.projection = nn.Linear(self.cnn.out_channels, embedding_dim)
        self.transformer = ViT(num_patches=196,
            num_transformer_layers=2,
            embedding_dim=128,
            mlp_size=512,
            num_heads=4,
            attn_dropout=0,
            mlp_dropout=0.1,
            embedding_dropout=0.1,
            num_classes=15)

    def forward(self, x):
        x = self.cnn(x)
        x = x.flatten(2)
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        x = self.transformer(x)    
        return x    
