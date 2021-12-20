import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramModel(nn.Module):

    def __init__(self, num_nodes, dim):
        super(SkipGramModel, self).__init__()
        self.embedding_dim = dim
        self.u_embeddings = nn.Embedding(num_nodes, self.embedding_dim,
                                         sparse=True)

        self.v_embeddings = nn.Embedding(num_nodes, self.embedding_dim,
                                         sparse=True)

        initrange = 1.0 / self.embedding_dim
        nn.init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        nn.init.constant_(self.v_embeddings.weight.data, 0)

    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)

    def save_embedding(self, file_name):
        numpy.save(file_name, self.u_embeddings.weight.cpu().data.numpy())
