from sklearn.metrics import roc_auc_score
import torch
import numpy as np

def get_aucroc(y_true, output):
    if torch.min(y_true.data) == 1 or torch.max(y_true.data) == 0:
        aucroc = np.nan # return nan if there are only examples of one type in the batch, because AUCROC is not defined then. 
    else:
        y_true = y_true.cpu().detach().numpy().flatten().astype(bool)
        output = output.cpu().detach().numpy().flatten()
        aucroc = roc_auc_score(y_true,output)
    return aucroc