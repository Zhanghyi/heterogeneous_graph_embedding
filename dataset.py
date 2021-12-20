import os
from dgl.data.utils import download, extract_archive
import torch as th
from dgl.data.utils import load_graphs


class NodeClassificationDataset():

    def __init__(self):
        self.g, self.category, self.num_classes = self.load_HIN()

    def load_HIN(self):
        dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
        path = os.path.join(dir, 'dblp.zip')
        g_path = os.path.join(dir, 'dblp/graph.bin')
        if os.path.exists(g_path):
            pass
        else:
            download('https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/dblp4MAGNN.zip', path=path)
            extract_archive(path, os.path.join(dir, 'dblp'))

        g = load_graphs(g_path)
        category = 'A'
        g = g[0][0].long()
        num_classes = 4
        self.meta_paths_dict = {
            'APVPA': [('A', 'A-P', 'P'), ('P', 'P-V', 'V'), ('V', 'V-P', 'P'), ('P', 'P-A', 'A')],
            'APA': [('A', 'A-P', 'P'), ('P', 'P-A', 'A')],
        }
        return g, category, num_classes

    def get_idx(self):
        train_mask = self.g.nodes[self.category].data.pop('train_mask').squeeze()
        test_mask = self.g.nodes[self.category].data.pop('test_mask').squeeze()
        train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
        test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
        return train_idx, test_idx

    def get_labels(self):
        return self.g.nodes[self.category].data.pop('labels').long()
