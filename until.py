#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:29:52 2018

@author: lin
"""
import numpy as np
import matplotlib.pyplot as plt


def accuracy(x,y,model):
    a = model.predict(x,y)
    
    a[a>=0.5]=1
    a[a<0.5]=0

    return np.sum(a==y)/len(a)

data1 = np.load("datasets/breast-cancer.npz")
data2 = np.load("datasets/diabetes.npz")
data3 = np.load("datasets/digit.npz")
data4 = np.load("datasets/iris.npz")
data5 = np.load("datasets/wine.npz")



def run_epoch(data, model, batch_size,lr):
    epoch_size = (len(data["train_X"])//batch_size)+1
    
    loss_total=0
    for step in range(epoch_size):
        if step == epoch_size-1:
            input_data = data["train_X"][step*batch_size:,:]
            labels = data["train_Y"][step*batch_size:]
        else:
            input_data = data["train_X"][step*batch_size:(step+1)*batch_size,:]
            labels = data["train_Y"][step*batch_size:(step+1)*batch_size]
            
        a = model.train(input_data,labels,lr)

        loss = -np.sum(labels*np.log(a)+(1-labels)*np.log(1-a))

        loss_total += loss
    loss_avg = loss_total/len(data["train_X"])
    acc = accuracy(data["train_X"],data["train_Y"],model)
        #print("accuracy:",acc)
    return loss_avg ,acc
def plot_loss_acc(loss,acc,i):
    plt.figure(1+2*i)
    
    plt.plot(loss,label='loss per epoch')
    plt.title("dataset"+str(i+1)+" training loss")
    plt.legend()
    plt.xlabel('epoch_num')
    plt.figure(2+2*i)
    plt.plot(acc,color='orange',label='accuray per epoch')
    plt.title("dataset"+str(i+1)+" training accuracy")
    plt.legend()
    plt.xlabel('epoch_num')
def sigmoid(x):
    return 1/(1+np.exp(-x))


def choose_dataset(choice, config1):
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
