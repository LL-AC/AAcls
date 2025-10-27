import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error, accuracy_score, precision_score, \
    f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.multiclass import OneVsOneClassifier
import numpy as np
from .constant import PLOT_AXIS


def accuracy(predicets, targets):
    # average
    sw = compute_sample_weight(class_weight='balanced', y=targets)
    acc = accuracy_score(targets, predicets, sample_weight=sw)
    return round(acc, 4)


def f1_Score(predicets, targets):
    f1 = f1_score(targets, predicets, average='weighted')
    return round(f1, 4)


def AUC(predicets, targets):
    # ovr = OneVsOneClassifier(roc_auc_score)
    # #y_scores = ovr.fit(predicets, targets).decision_function(predicets)
    # aucs = ovr.fit(predicets, targets).score(predicets, targets, average=None)
    auc = roc_auc_score(targets, predicets, average='macro')
    return round(auc, 4)


def f1_score_multi(predicets, targets):
    f1 = f1_score(targets, predicets, average=None)
    f1 = [round(f, 4) for f in f1]
    return f1

    # return round(((predicets == targets).sum() / len(targets)).item(), 4)


def precision_multi(predicets, targets):
    precision = precision_score(targets, predicets, average=None)
    precision = [round(p, 4) for p in precision]
    return precision


def accuracy_multi(predicets, targets):
    # every single cls
    # TP+TN / TP+TN+FP+FN
    cls = len(PLOT_AXIS)
    bs = len(predicets)

    accuracys = []

    for i in range(cls):
        index = torch.tensor([i] * bs)
        # TP+TN+FP+FN = P+N = len(targets) or bs
        TP = (predicets == index) & (targets == index)
        TN = (predicets != index) & (targets != index)
        accuracys.append(round(((TP.sum() + TN.sum()) / bs).item(), 4))

    return accuracys


def specificity(predicets, targets):
    # tn / (tn+fp)

    zeros = torch.zeros(predicets.shape)
    tn = (zeros == predicets) & (zeros == targets)
    fp = (zeros != predicets) & (zeros == targets)

    return round((tn.sum() / (tn.sum() + fp.sum() + 1e-8)).item(), 4)


def specificity_multi(predicets, targets):
    # tn / (tn+fp)
    cls = len(PLOT_AXIS)
    bs = len(predicets)

    specificitys = []
    for i in range(cls):
        all = torch.tensor([i] * bs)
        # tn + fp = N  不是
        N = (all != targets)
        TN = (predicets != all) & N  # 预测不是 真的不是

        specificitys.append(round((TN.sum() / (N.sum() + 1e-8)).item(), 4))

    return specificitys


def sensitivity_multi(predicets, targets):
    # tp / (tp + fn)
    cls = len(PLOT_AXIS)
    bs = len(predicets)

    sensitivitys = []

    for i in range(cls):
        all = torch.tensor([i] * bs)
        # tp + fn = P 是
        P = (all == targets)
        TP = (predicets == all) & P  # 预测是 真的是

        sensitivitys.append(round((TP.sum() / (P.sum() + 1e-8)).item(), 4))

    return sensitivitys


def sensitivity(predicets, targets):
    # tp / (tp + fn)

    ones = torch.ones(predicets.shape)
    tp = (ones == predicets) & (ones == targets)
    fn = (ones != predicets) & (ones == targets)

    return round((tp.sum() / (tp.sum() + fn.sum() + 1e-8)).item(), 4)


def roc_auc(predicets, targets):
    output = F.softmax(predicets, dim=1)[:, 1].view(-1)
    return round(roc_auc_score(targets, output), 4)


def mae(predicets, targets):
    predicets = predicets.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    return round(mean_absolute_error(targets, predicets), 4)


def mse(predicets, targets):
    predicets = predicets.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    return round(mean_squared_error(targets, predicets), 4)


def rmse(predicets, targets):
    return round(np.sqrt(mse(targets, predicets)), 4)


def accuracy_top1(predicets, targets):
    predicets = predicets.cpu()
    targets = targets.cpu()
    targets = targets.argmax(dim=1)
    # predicets = predicets.argmax(dim=1)
    _, indices = torch.topk(predicets, 1)
    cnt = 0
    for label, i in zip(targets, indices):
        if label in i:
            cnt += 1
    return round(cnt / len(targets), 4)
    # return round(((predicets == targets).sum() / len(targets)).item(), 4)
