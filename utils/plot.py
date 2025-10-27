import matplotlib.pyplot as plt
from .metric import *
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from .constant import PLOT_AXIS, TABLE_CLOUMNS
import torch.nn.functional as F

def plot_roc(predicted, targets, output_name='ROC.png'):

    y_true = targets
    y_scores = F.softmax(predicted, dim=1)[:,1].view(-1)
    
    # 计算ROC曲线的值
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)
    
    # 绘制ROC曲线
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % metrics.auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(output_name)
    plt.close()

def plot_matrix(predicted, targets, output_name='matrix.png'):
    # bs, cls = predicted.shape

    # predicted = F.softmax(predicted, dim=1)
    # 示例真实标签和预测标签
    y_true = targets
    y_pred = predicted
    labels = PLOT_AXIS
    cls = len(PLOT_AXIS)
    print(cls, cls)

    cm = np.zeros((cls,cls))
    for i, j in zip(y_true, y_pred):
        cm[int(i)][int(j)] += 1
    
    plt.figure()
    plt.rcParams['font.family'] = ['Noto Sans CJK JP']
    # 生成混淆矩阵
    #cm = confusion_matrix(y_true, y_pred)
    
    # 使用Seaborn的heatmap来画混淆矩阵
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    print(output_name)
    plt.savefig(output_name)
    plt.close()


def plot_result(predicted, targets, output_name='result.png'):

    metric_fn=[sensitivity_multi, precision_multi, specificity_multi, f1_score_multi]
    # 创建数据
    data = [TABLE_CLOUMNS]
    data += [[fn.__name__.split('_')[0]] + fn(predicted, targets) for fn in metric_fn]
    # 创建数据
    acc = accuracy(predicted, targets)
    f1 = f1_Score(predicted, targets)


 
    # 绘制表格
    fig, ax = plt.subplots()
    ax.axis('off')
    table = ax.table(cellText=data, loc='center')
    
    # 设置表格样式
    plt.title(f'acc: {acc}, marco f1: {f1}')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)  # 调整表格大小
    
    # 保存为图片
    plt.savefig(output_name, bbox_inches='tight')
    plt.close()

    data.append(['acc', acc, 'f1', f1])

    df = pd.DataFrame(data[1:],columns=data[0])
    df.to_csv(output_name.replace('.png', '.csv'), index=False)