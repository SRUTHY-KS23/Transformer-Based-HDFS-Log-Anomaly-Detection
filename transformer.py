
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):

        x = x + self.pe[:, :x.size(1), :]

        return x


class TransformerModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model,
        nhead,
        dim_feedforward,
        num_layers,
        dropout,
        max_len
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=0
        )

        self.pos_encoder = PositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.attention = nn.Linear(d_model, 1)

        self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(d_model, 1)

        self.logit_scale = nn.Parameter(torch.tensor(1.0))


    def forward(self, x, return_attention=False):

        padding_mask = (x == 0)

       
        x = self.embedding(x)

       
        x = self.pos_encoder(x)

       
        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask
        )

        
        attn_weights = torch.softmax(
            self.attention(x),
            dim=1
        )

        context = torch.sum(
            attn_weights * x,
            dim=1
        )

        context = self.norm(context)

        context = self.dropout(context)

        out = self.fc(context).squeeze(-1)

        out = out * self.logit_scale

        if return_attention:
            return out, attn_weights

        return out