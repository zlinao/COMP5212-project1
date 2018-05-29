#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 17:20:12 2018

@author: lin
"""
import time
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,log_loss
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import pickle
data1 = np.load("datasets/breast-cancer.npz")
data2 = np.load("datasets/diabetes.npz")
data3 = np.load("datasets/digit.npz")
data4 = np.load("datasets/iris.npz")
data5 = np.load("datasets/wine.npz")
class Config_svm(object):
    def __init__(self):
        pass
    def cancer(self):
        self.data = data1
        

    def diabetes(self):
        self.data = data2
        
    def digit(self):
        self.data = data3
      
    def iris(self):
        self.data = data4
       
    def wine(self):
        self.data = data5
        
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
    for choice in range(1,6):#dataset traverse
        print('')
        print("###########SVM for dataset",choice,"###########")
        print('')
    
        config1 = Config_svm()
        
        config = model_selection(choice,config1)
        #build the models
        ss1 = svm.SVC(kernel="linear",probability=True)
        ss2 = svm.SVC(kernel="rbf",probability=True)
        sss = [ss1,ss2]
        for ss in enumerate(sss):
            if ss[0] == 0:
                print('')
                print("#########linear kernel#######")
                print('')
                
                
                start = time.time()
                ss[1].fit(config.data["train_X"],config.data["train_Y"])
                print('average training time:',time.time()-start)
                y_train = ss[1].predict_proba(config.data["train_X"])
                
                print('average training loss :',log_loss(config.data["train_Y"], y_train)/len(y_train))
                y_pre = ss[1].predict(config.data["test_X"])
                y_test = ss[1].predict_proba(config.data["test_X"])
                print('average test loss :',log_loss(config.data["test_Y"], y_test)/len(y_test))
                print('test accuracy :',accuracy_score(config.data["test_Y"],y_pre))
                print("confusion metrix:")
                print(confusion_matrix(config.data["test_Y"],y_pre))
                print(classification_report(config.data["test_Y"],y_pre))
                
            else:
                print("")
                print("######### RBF kernel#######")
                print('')
                parameters = {'gamma':[1,0.1,0.01,0.001]}
                clf = GridSearchCV(ss[1],parameters,cv=5)
                clf.fit(config.data["train_X"],config.data["train_Y"])
                for i in range(4):
                    print("kernel parameter gamma:",i+1)
                    print("average training time:",clf.cv_results_['mean_fit_time'][i])
                    print("average validation accuracy",clf.cv_results_['mean_test_score'][i])
                print("optimal kernel parameter gamma:",clf.best_params_)
                print("best_accuracy:",clf.best_score_)
                print("training time for bset model:",clf.cv_results_['mean_fit_time'][clf.best_index_])
                y_train = clf.predict_proba(config.data["train_X"])
                print('average training loss for bset model:',log_loss(config.data["train_Y"], y_train)/len(y_train))
                y_pre = clf.predict(config.data["test_X"])
                y_test = clf.predict_proba(config.data["test_X"])
                print('average test loss for bset model:',log_loss(config.data["test_Y"], y_test)/len(y_test))
                print('test accuracy for best model:',accuracy_score(config.data["test_Y"],y_pre))
                print("confusion metrix:")
                print(confusion_matrix(config.data["test_Y"],y_pre))
                print(classification_report(config.data["test_Y"],y_pre))
            """
            import pickle
            s = pickle.dumps(clf)
            clf2 = pickle.loads(s)
            clf2.predict(X[0:1])
            """