
import torch
import torch.nn as nn

class ECGTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_classes, num_heads=4, num_layers=2, dropout=0.1):
        super(ECGTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(model_dim, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.embedding(x)  # (batch, seq_len, model_dim)
        x = self.transformer(x)  # (batch, seq_len, model_dim)
        x = x.transpose(1, 2)  # for AdaptiveAvgPool1d: (batch, model_dim, seq_len)
        out = self.classifier(x)  # (batch, num_classes)
        return out
