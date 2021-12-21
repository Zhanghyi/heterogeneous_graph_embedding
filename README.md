# heterogeneous_graph_embedding

Implement two heterogeneous graph embedding models: Metapath2vec and HERec.

# Metapath2vec[KDD2017]

Paper: [**metapath2vec: Scalable Representation Learning for Heterogeneous Networks**](https://ericdongyx.github.io/metapath2vec/m2v.html)

Code from author: https://www.dropbox.com/s/w3wmo2ru9kpk39n/code_metapath2vec.zip

## How to run

```bash
python main.py --model Metapath2vec --gpu 0
```
If you do not have gpu, set -gpu -1.

## Performance

### Node Classification

| Dataset       |  Metapath | Macro-F1 | Micro-F1 |
| ------------- |--------   | -------- | -------- |
| dblp          |  APVPA    | 0.9256   | 0.9309   |


# HERec[TKDE2018]

Paper: [**Heterogeneous Information Network Embedding for Recommendation**](https://ieeexplore.ieee.org/abstract/document/8355676)

Code from author: https://github.com/librahu/HERec

## How to run

```bash
python main.py --model HERec --gpu 0
```

If you do not have gpu, set -gpu -1.

## Performance

### Node Classification

| Dataset       |  Metapath | Macro-F1 | Micro-F1 |
| ------------- |--------   | -------- | -------- |
| dblp          |  APVPA    | 0.9252   | 0.9303   |
