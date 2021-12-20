import configparser
import torch as th


class Config(object):
    def __init__(self, file_path, model, gpu):
        conf = configparser.ConfigParser()
        if gpu == -1:
            self.device = th.device('cpu')
        elif gpu >= 0:
            if th.cuda.is_available():
                self.device = th.device('cuda', int(gpu))
            else:
                print("cuda is not available, please set 'gpu' -1")
        conf.read(file_path)
        if model == 'Metapath2vec':
            self.lr = conf.getfloat("Metapath2vec", "learning_rate")
            self.max_epoch = conf.getint("Metapath2vec", "max_epoch")
            self.dim = conf.getint("Metapath2vec", "dim")
            self.batch_size = conf.getint("Metapath2vec", "batch_size")
            self.window_size = conf.getint("Metapath2vec", "window_size")
            self.num_workers = conf.getint("Metapath2vec", "num_workers")
            self.neg_size = conf.getint("Metapath2vec", "neg_size")
            self.rw_length = conf.getint("Metapath2vec", "rw_length")
            self.rw_walks = conf.getint("Metapath2vec", "rw_walks")
            self.meta_path_key = conf.get("Metapath2vec", "meta_path_key")

        elif model == 'HERec':
            self.lr = conf.getfloat("HERec", "learning_rate")
            self.max_epoch = conf.getint("HERec", "max_epoch")
            self.dim = conf.getint("HERec", "dim")
            self.batch_size = conf.getint("HERec", "batch_size")
            self.window_size = conf.getint("HERec", "window_size")
            self.num_workers = conf.getint("HERec", "num_workers")
            self.neg_size = conf.getint("HERec", "neg_size")
            self.rw_length = conf.getint("HERec", "rw_length")
            self.rw_walks = conf.getint("HERec", "rw_walks")
            self.meta_path_key = conf.get("HERec", "meta_path_key")
