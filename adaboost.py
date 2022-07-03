import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.tree import DecisionTreeClassifier


class DtcAdaboost():

    def __init__(self,max_depth=3,min_samples_split=0.125,learning_rate=1,k=1/120,n_estimators=250,early_stop=False) -> None:
        self.md = max_depth
        self.mss = min_samples_split
        self.lr = learning_rate
        self.n = n_estimators
        self.k = k
        self.et = early_stop
        self.estimators = [DecisionTreeClassifier(max_depth=self.md,
                                                min_samples_split=self.mss,
                                                random_state=i) for i in range(self.n)]
        self.alphas = []
    
    def fit(self,X,y,rank,X_valid,y_valid):
        self.train_accuracy = []
        self.train_corr = []
        self.train_auc = []
        self.train_cm = []
        self.valid_accuracy = []
        self.X = X.copy(deep=True)
        self.y = y.copy(deep=True)
        self.rank = rank.copy(deep=True)
        self.y['weight'] = 1 / np.shape(y)[0]
        aggProbEst = np.zeros((np.shape(X)[0],2))
        aggProbEst_valid = np.zeros((np.shape(X_valid)[0],2))
        max_valid_accuracy = 0.5
        epoch = 0

        for i in range(self.n):
            dtc = self.estimators[i]
            dtc.fit(self.X,self.y.label,self.y.weight)
            probEst = dtc.predict_proba(X)
            probEst_valid = dtc.predict_proba(X_valid)
            classEst = [np.argmax(p) if np.argmax(p) > 0 else -1 for p in probEst]
            incorrect = classEst != self.y.label
            tp = (self.y.label == classEst)&(self.y.label == 1)

            error = np.average(incorrect,weights=self.y['weight'])
            if error >= 0.5:
                self.estimators.pop(-1)
                print("This weakclassifier is worser than random prediction")
                if len(self.estimators) == 0:
                    raise ValueError(
                        "BaseClassifier in AdaBoostClassifier "
                        "ensemble is worse than random, ensemble "
                        "can not be fit."
                        )
                continue

            alpha = self.lr * (0.5 * np.log((1.0 - error) / max(error, 1e-16)) + self.k * np.exp(self.y.loc[tp].weight.sum()))
            self.alphas.append(alpha)

            self.y.weight = np.exp(np.log(self.y.weight) + alpha * incorrect * (self.y.weight > 0))
            self.y.weight /= self.y.weight.sum()

            aggProbEst += alpha * probEst
            aggProbEst_valid += alpha * probEst_valid
            aggClassEst = [np.argmax(p) if np.argmax(p) > 0 else -1 for p in aggProbEst]
            aggClassEst_valid = [np.argmax(p) if np.argmax(p) > 0 else -1 for p in aggProbEst_valid]

            self.train_accuracy.append(accuracy_score(y_true=self.y.label,y_pred=aggClassEst))
            # self.train_auc.append(roc_auc_score(y_true=self.y.label,y_score=aggClassEst))
            # self.train_corr.append(np.corrcoef(self.rank,[ap[1]/(ap[0]+ap[1]) for ap in aggProbEst])[0,1])
            # self.train_cm.append(confusion_matrix(y_true=self.y.label,y_pred=aggClassEst))

            if not self.et:
                continue
            if i <= 40:
                continue
            temp_valid_accuracy = accuracy_score(y_true=y_valid.label,y_pred=aggClassEst_valid)
            if temp_valid_accuracy > max_valid_accuracy:
                max_valid_accuracy = temp_valid_accuracy
                epoch = 0
            else:
                epoch += 1
            if epoch == 25:
                self.n = i - 25
                for _ in range(25):
                    self.estimators.pop()
                    self.alphas.pop()
                break
            
    def predict_proba(self,X):
        m = np.shape(X)[0]
        aggProbEst = np.zeros((m,2))
        for i in range(self.n):
            dtc = self.estimators[i]
            alpha = self.alphas[i]
            probEst = dtc.predict_proba(X)
            aggProbEst += alpha * probEst                                            
        return [ap[1]/(ap[0]+ap[1]) for ap in aggProbEst]
    
    def predict(self,X):
        aggProbEst = self.predict_proba(X)
        return [np.argmax(p) if np.argmax(p) > 0 else -1 for p in aggProbEst]
    
    def score(self,X,y,method='accuracy'):
        if method == 'accuracy':
            return accuracy_score(y_true=y,y_pred=self.predict(X))
        if method == 'auc':
            return roc_auc_score(y_true=y,y_score=self.predict(X))

    # def valid(self,X,y,rank):
    #     valid_accuracy = []
    #     valid_corr = []
    #     valid_auc = []
    #     m = np.shape(X)[0]
    #     aggProbEst = np.zeros((m,2))
    #     for i in range(self.n):
    #         dtc = self.estimators[i]
    #         alpha = self.alphas[i]
    #         probEst = dtc.predict_proba(X)
    #         aggProbEst += alpha * probEst 
    #         aggClassEst = [np.argmax(a) if np.argmax(a) > 0 else -1 for a in aggProbEst]  
    #         valid_accuracy.append(accuracy_score(y_pred = aggClassEst,y_true = y.label))
    #         valid_corr.append(np.corrcoef(rank,[a[1]/(a[0]+a[1]) for a in aggProbEst])[0,1])
    #         valid_auc.append(roc_auc_score(y_score=aggClassEst,y_true=y.label))
    #     return valid_accuracy,valid_corr,valid_auc
    
    # def plot(self,X,y,rank,train=True):
    #     valid_accuracy,valid_corr,valid_auc = self.valid(X,y,rank)
    #     plt.figure(figsize=(18,5))
    #     ax1=plt.subplot(131)
    #     ax1.plot(valid_accuracy,label='valid')
    #     ax1.set_xlabel('numIt')
    #     ax1.set_ylabel('accuracy')
    #     ax1.set_title('Accuracy')
    #     ax2=plt.subplot(132)
    #     ax2.plot(valid_auc,label='valid')
    #     ax2.set_xlabel('numIt')
    #     ax2.set_ylabel('auc')
    #     ax2.set_title('AUC')
    #     ax3=plt.subplot(133)
    #     ax3.plot(valid_corr,label='valid')
    #     ax3.set_xlabel('numIt')
    #     ax3.set_ylabel('corr')
    #     ax3.set_title('Correlation')

    #     print('max Accuracy:{} with numIt {}'.format(np.max(valid_accuracy),np.argmax(valid_accuracy)+1))
    #     print('max AUC:{} with numIt {}'.format(np.max(valid_auc),np.argmax(valid_auc)+1))
    #     print('max Correlation:{} with numIt {}'.format(np.max(valid_corr),np.argmax(valid_corr)+1))

    #     if train:  
    #         ax1.plot(self.train_accuracy,label='train')
    #         ax2.plot(self.train_auc,label='train')
    #         ax3.plot(self.train_corr,label='train')
    #         ax1.legend()
    #         ax2.legend()
    #         ax3.legend() 
    #     plt.show()


    
        