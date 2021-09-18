import numpy as np
import torch 
from sklearn.metrics import roc_auc_score


class Metric(object):
    def __init__(self, preds, labels, thresh=0.5):
        '''
        Args: 
            preds: (nd.array) in shape of (num_samples, ) with predicted classes (probabilities)
            labels: (nd.array) in shape of (num_samples, ) with labels
        '''
        
        self.preds = np.where(preds <= thresh, 0, 1).astype(np.float32)
        self.proba = preds
        self.labels = labels

    def confusion_matrix(self):
        tn = np.where((self.labels == 0) & (self.preds == 0))[0].size
        tp = np.where((self.labels == 1) & (self.preds == 1))[0].size
        fn = np.where((self.labels == 1) & (self.preds == 0))[0].size
        fp = np.where((self.labels== 0) & (self.preds == 1))[0].size

        return tn, tp, fn, fp
    
    def acc(self):
        tn, tp, fn, fp = self.confusion_matrix()
        acc = (tn + tp) / (tn + tp + fn + fp)
        
        return acc
    
    def sens_specs(self):
        tn, tp, fn, fp = self.confusion_matrix()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        return sensitivity, specificity
    
    def roc_curve(self):
        auc = roc_auc_score(self.labels, self.proba)
        return auc

def main():
    probabilities = torch.tensor([0.7, 0.5, 0.6, 0.2, 0.8], dtype = torch.float32)
    labels = torch.tensor([1, 0, 1, 1, 1], dtype = torch.float32)
    probabilities = probabilities.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    metric = Metric(probabilities, labels)
    print(f'acc  = {metric.acc()}')
    print(f'sensitivity,  specificity = {metric.sens_specs()}')
    print(f'AUC = {metric.roc_curve()}')

if __name__ == "__main__":
    main()