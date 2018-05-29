#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 15:58:59 2018

@author: lin
"""
from until import *
import numpy as np
import time
import matplotlib.pyplot as plt
#from sklearn import linear_model
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
data1 = np.load("datasets/breast-cancer.npz")
data2 = np.load("datasets/diabetes.npz")
data3 = np.load("datasets/digit.npz")
data4 = np.load("datasets/iris.npz")
data5 = np.load("datasets/wine.npz")

class Linear_regression(object):
    def __init__(self,data,scale): 
        N, X = np.shape(data["train_X"])
        np.random.seed(13)
        self.w1 = scale*np.random.randn(X)
        self.b = 0
    def predict(self,x,y):
        return sigmoid(x.dot(self.w1)+self.b)

    def train(self,x,y,lr):
        a = sigmoid(x.dot(self.w1)+self.b)
        
        dw = x.T.dot(y-a)
        db = np.sum(y-a)
        model.w1 += dw*lr 
        model.b +=db*lr
        return a


class Config_linear(object):
    def __init__(self):
        pass
    def cancer(self):
        self.data = data1
        self.num_epoch = 10
        self.batch_size = 20
        self.learning_rate = 0.05
        self.lr_decay = 0.5
        self.scale = 1

    def diabetes(self):
        self.data = data2
        self.num_epoch = 15
        self.batch_size = 20
        self.learning_rate = 0.1
        self.lr_decay = 0.5
        self.scale = 2
    def digit(self):
        self.data = data3
        self.num_epoch = 10
        self.batch_size = 15
        self.learning_rate = 0.0005
        self.lr_decay = 0.5
        self.scale = 0.001
    def iris(self):
        self.data = data4
        self.num_epoch = 10
        self.batch_size = 10
        self.learning_rate = 0.05
        self.lr_decay = 0.5
        self.scale = 1
    def wine(self):
        self.data = data5
        self.num_epoch = 15
        self.batch_size = 10
        self.learning_rate = 0.00005
        self.lr_decay = 0.2
        self.scale = 0.0003

if __name__=='__main__':
 

    for choice in range(1,6):#dataset traverse
        print('')
        print("###########linear_regression for dataset",choice,"###########")
        print('')
        config1 = Config_linear()
        config = choose_dataset(choice, config1)
        #build the models
        model = Linear_regression(config.data,config.scale)
        start_time = time.time()
        loss_p = []
        acc_p = []
        #iterate training set
        for i in range(config.num_epoch):
            loss,acc = run_epoch(config.data,model,config.batch_size,config.lr_decay*config.learning_rate)
            print("epoch",i,':','loss:',loss,'accuracy:',acc)
            loss_p.append(loss)
            acc_p.append(acc)
        #plot the loss and acccuracy
        #plot_loss_acc(loss_p,acc_p,choice-1)
        time_cost = time.time()-start_time
        t_acc = accuracy(config.data["test_X"],config.data["test_Y"],model)
        y_pre = model.predict(config.data["test_X"],config.data["test_Y"])
        test_loss = -np.sum(config.data["test_Y"]*np.log(y_pre)+(1-config.data["test_Y"])*np.log(1-y_pre))
        test_loss /= np.size(config.data["test_Y"])
        print('test_loss:',test_loss)
        y_pre[y_pre>=0.5]=1
        y_pre[y_pre<0.5]=0
        
        print('confusion_matrix:')
        print(confusion_matrix(config.data["test_Y"],y_pre))
        print(classification_report(config.data["test_Y"],y_pre))
        print("Testset accuracy:",t_acc,"CPU time:",time_cost) 
    
         
         
         
         
         
         
         
         
         