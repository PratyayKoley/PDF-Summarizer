import torch.nn as nn
import torch

class ChunkTransformer(nn.Module):
    def __init__(self, emb_dim=768, nhead=8, nlayers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 512, emb_dim))  # max 512 chunks

    def forward(self, chunk_embeddings):
        # chunk_embeddings: (batch=1, n_chunks, emb_dim)
        n = chunk_embeddings.size(1)
        x = chunk_embeddings + self.pos_embedding[:, :n, :].to(chunk_embeddings.device)
        x = x.permute(1,0,2)  # transformer expects (S, N, E) -> S=n_chunks, N=batch=1, E
        out = self.encoder(x)  # (n_chunks, batch, E)
        out = out.permute(1,0,2)
        return out  # (batch, n_chunks, E)