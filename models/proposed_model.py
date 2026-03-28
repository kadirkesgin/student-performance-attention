import torch
import torch.nn as nn
import torch.optim as optim

class ProposedModel(nn.Module):
    """
    Enhanced Hybrid Attention Model for Tabular Educational Data.
    Features:
    - Feature-wise Linear Layers (Embedding-like)
    - Residual Multi-head Attention
    - Layer Normalization for stability
    """
    def __init__(self, input_dim, num_classes):
        super(ProposedModel, self).__init__()
        
        # Initial projection to high-dim space
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Attention Block with Residual and LayerNorm
        self.attn_layer = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(256)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )
        self.norm2 = nn.LayerNorm(256)
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, ablation_no_attn=False):
        # 1. Feature Extraction
        x = self.feature_extractor(x)
        
        if not ablation_no_attn:
            # 2. Self-Attention Block
            x_res = x.unsqueeze(1) # [Batch, 1, 256]
            attn_out, _ = self.attn_layer(x_res, x_res, x_res)
            attn_out = attn_out.squeeze(1)
            x = self.norm1(x + attn_out)
        
        # 3. FFN Block
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # 4. Classification
        return self.classifier(x)
