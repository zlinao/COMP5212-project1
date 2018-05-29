#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 13:05:41 2018

@author: lin
"""
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,log_loss
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn import neural_network as nn

data1 = np.load("datasets/breast-cancer.npz")
data2 = np.load("datasets/diabetes.npz")
data3 = np.load("datasets/digit.npz")
data4 = np.load("datasets/iris.npz")
data5 = np.load("datasets/wine.npz")
class Config_nn(object):
    def __init__(self):
        pass
    def cancer(self):
        self.data = data1
        
        self.lr = 0.005
        self.lr_decay=0.001
    def diabetes(self):
        self.data = data2
        self.lr = 0.5
        self.lr_decay=0.01
    def digit(self):
        self.data = data3
        self.lr = 0.005
        self.lr_decay=0.001
    def iris(self):
        self.data = data4
        self.lr = 0.005
        self.lr_decay=0.001
    def wine(self):
        self.data = data5
        self.lr = 0.004
        self.lr_decay=0.0001
def model_selection(choice,config1):
    if choice ==1:
        config1.cancer()

    elif choice ==2:
        config1.diabetes()
    elif choice ==3:
        config1.digit()
    elif choice ==4:
        config1.iris()
    elif choice ==5:
        config1.wine()
        
    else:
        print("please choose the dataset number : 1-5")
    return config1

if __name__=='__main__':
    
    for choice in range(1,6):# datasets traverse
        print('')
        print("###########neural network for dataset",choice,"###########")
        print('')
        
        config1 = Config_nn()
        
        config = model_selection(choice,config1)
        #build the model
        nnh = nn.MLPClassifier(activation='logistic',random_state=2,alpha = 0.2,solver='sgd',max_iter=2000,learning_rate='invscaling',learning_rate_init=config.lr,power_t=config.lr_decay)
        
        parameters = {'hidden_layer_sizes':[1,2,3,4,5,6,7,8,9,10]}
        #grid search
        clf = GridSearchCV(nnh,parameters)
        clf.fit(config.data["train_X"],config.data["train_Y"])
        #print detail
        for i in range(10):
            print("hidden_layer_sizes:",i+1)
            print("average training time:",clf.cv_results_['mean_fit_time'][i])
            print("average validation accuracy",clf.cv_results_['mean_test_score'][i])
        print('')
        print("optimal hidden_layer_sizes:",clf.best_params_)
        print("best_accuracy:",clf.best_score_)
        print("training time for bset model:",clf.cv_results_['mean_fit_time'][clf.best_index_])
        #print training loss and test loss by using the best model
        y_train = clf.predict_proba(config.data["train_X"])
        print('average training loss for bset model:',log_loss(config.data["train_Y"], y_train)/len(y_train))
        y_pre = clf.predict(config.data["test_X"])
        y_test = clf.predict_proba(config.data["test_X"])
        print('average test loss for bset model:',log_loss(config.data["test_Y"], y_test)/len(y_test))
        print('test accuracy for best model:',accuracy_score(config.data["test_Y"],y_pre))
        print('confusion matix:')
        print(confusion_matrix(config.data["test_Y"],y_pre))
        print(classification_report(config.data["test_Y"],y_pre))
        
