from config import Config
from trainerflow import Metapath2VecTrainer, HERecTrainer
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Metapath2vec', type=str, help='name of models')
    parser.add_argument('--gpu', '-g', default='-1', type=int, help='-1 means cpu')
    args = parser.parse_args()

    config = Config(file_path=["./config.ini"], model=args.model, gpu=args.gpu)
    if args.model == 'Metapath2vec':
        trainerflow = Metapath2VecTrainer(args=config)
    elif args.model == 'HERec':
        trainerflow = HERecTrainer(args=config)

    trainerflow.train()
