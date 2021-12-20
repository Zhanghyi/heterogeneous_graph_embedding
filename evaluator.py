from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression


def node_classification_with_LR(embeddings, labels, train_idx, test_idx):
    Y_train = labels[train_idx]
    Y_test = labels[test_idx]
    LR = LogisticRegression(max_iter=10000)
    X_train = embeddings[train_idx]
    X_test = embeddings[test_idx]
    LR.fit(X_train, Y_train)
    Y_pred = LR.predict(X_test)
    macro_f1 = f1_score(Y_test, Y_pred, average='macro')
    micro_f1 = f1_score(Y_test, Y_pred, average='micro')
    print('<node classification> macro_f1: {:.4f}, micro_f1: {:.4f}'.format(macro_f1, micro_f1))
