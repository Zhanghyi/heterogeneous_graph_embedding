import numpy
from tqdm import tqdm
import torch.optim as optim
import torch.sparse as sparse
from torch.utils.data import DataLoader
from model import SkipGramModel
from sampler import RandomWalkSampler
from dataset import NodeClassificationDataset
from evaluator import node_classification_with_LR
import os
import dgl


class Metapath2VecTrainer():
    def __init__(self, args):
        self.args = args
        self.dataset = NodeClassificationDataset()
        self.g = self.dataset.g
        self.model = SkipGramModel(self.g.num_nodes(), self.args.dim)
        self.mp2vec_sampler = None
        self.dataloader = None
        self.load_trained_embeddings = False
        self.device = self.args.device
        self.embeddings_file_path = 'dblp_' + self.args.meta_path_key + '_mp2vec_embeddings.npy'

    def preprocess(self):
        metapath = self.dataset.meta_paths_dict[self.args.meta_path_key]
        self.mp2vec_sampler = RandomWalkSampler(g=self.g.to('cpu'),
                                                metapath=metapath * self.args.rw_length,
                                                rw_walks=self.args.rw_walks,
                                                window_size=self.args.window_size,
                                                neg_size=self.args.neg_size)

        self.dataloader = DataLoader(self.mp2vec_sampler, batch_size=self.args.batch_size,
                                     shuffle=True, num_workers=self.args.num_workers,
                                     collate_fn=self.mp2vec_sampler.collate)

    def train(self):
        emb = self.load_embeddings()

        start_idx, end_idx = self.get_ntype_range(self.dataset.category)
        train_idx, test_idx = self.dataset.get_idx()
        node_classification_with_LR(emb[start_idx:end_idx], self.dataset.get_labels(), train_idx, test_idx)

    def load_embeddings(self):
        if not self.load_trained_embeddings or not os.path.exists(self.embeddings_file_path):
            self.train_embeddings()
        emb = numpy.load(self.embeddings_file_path)
        return emb

    def train_embeddings(self):
        self.preprocess()

        optimizer = optim.SparseAdam(list(self.model.parameters()), lr=self.args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

        for epoch in range(self.args.max_epoch):
            print('\n\n\nEpoch: ' + str(epoch + 1))
            running_loss = 0.0
            for i, sample_batched in enumerate(tqdm(self.dataloader)):
                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    scheduler.step()
                    optimizer.zero_grad()
                    loss = self.model.forward(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                    if i > 0 and i % 50 == 0:
                        print(' Loss: ' + str(running_loss))
        self.model.save_embedding(self.embeddings_file_path)

    def get_ntype_range(self, target_ntype):
        start_idx = 0
        for ntype in self.g.ntypes:
            if ntype == target_ntype:
                end_idx = start_idx + self.g.num_nodes(ntype)
                return start_idx, end_idx
            start_idx += self.g.num_nodes(ntype)


class HERecTrainer():
    def __init__(self, args):
        self.args = args
        self.dataset = NodeClassificationDataset()
        self.g = self.dataset.g
        self.model = SkipGramModel(self.g.num_nodes(), self.args.dim)
        self.mp2vec_sampler = None
        self.dataloader = None
        self.load_trained_embeddings = False
        self.device = self.args.device
        self.embeddings_file_path = 'dblp_' + self.args.meta_path_key + '_herec_embeddings.npy'

    def preprocess(self):
        self.metapath = self.dataset.meta_paths_dict[self.args.meta_path_key]
        for i, elem in enumerate(self.metapath):
            if i == 0:
                adj = self.g.adj(etype=elem)
            else:
                adj = sparse.mm(adj, self.g.adj(etype=elem))
        adj = adj.coalesce()

        g = dgl.graph(data=(adj.indices()[0], adj.indices()[1]))
        g.edata['rw_prob'] = adj.values()

        self.random_walk_sampler = RandomWalkSampler(g=g.to('cpu'),
                                                     rw_length=self.args.rw_length,
                                                     rw_walks=self.args.rw_walks,
                                                     window_size=self.args.window_size,
                                                     neg_size=self.args.neg_size, rw_prob='rw_prob')

        self.dataloader = DataLoader(self.random_walk_sampler, batch_size=self.args.batch_size,
                                     shuffle=True, num_workers=self.args.num_workers,
                                     collate_fn=self.random_walk_sampler.collate)

    def train(self):
        emb = self.load_embeddings()

        train_idx, test_idx = self.dataset.get_idx()
        node_classification_with_LR(emb, self.dataset.get_labels(), train_idx, test_idx)

    def load_embeddings(self):
        if not self.load_trained_embeddings or not os.path.exists(self.embeddings_file_path):
            self.train_embeddings()
        emb = numpy.load(self.embeddings_file_path)
        return emb

    def train_embeddings(self):
        self.preprocess()

        optimizer = optim.SparseAdam(list(self.model.parameters()), lr=self.args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

        for epoch in range(self.args.max_epoch):
            print('\n\n\nEpoch: ' + str(epoch + 1))
            running_loss = 0.0
            for i, sample_batched in enumerate(tqdm(self.dataloader)):

                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    scheduler.step()
                    optimizer.zero_grad()
                    loss = self.model.forward(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                    if i > 0 and i % 50 == 0:
                        print(' Loss: ' + str(running_loss))
        self.model.save_embedding(self.embeddings_file_path)
