import torch
import torch.nn as nn
import math

class ECGPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2) * -math.log(10000.0) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class ECGTransformer(nn.Module):
    def __init__(self, input_dim=1, seq_len=500, d_model=64, nhead=4, num_layers=2, num_classes=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = ECGPositionalEncoding(d_model, max_len=seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * seq_len, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)           # (B, seq_len, d_model)
        x = self.pos_encoding(x)        # (B, seq_len, d_model)
        x = self.encoder(x)             # (B, seq_len, d_model)
        return self.classifier(x)       # (B, num_classes)
