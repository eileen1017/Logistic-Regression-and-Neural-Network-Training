#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 16:19:38 2017

@author: mac
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

class NeuralNet:
    """
    This class implements a Neural Network Classifier.
    """
    
    def __init__(self, input_dim, output_dim,hidden_dim,rate):
        """
        Initializes the parameters of the neural network classifier to 
        random values.
        
        args:
            input_dim: Number of dimensions of the input data
            output_dim: Number of classes
        """
        self.r = rate
        self.theta1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.theta2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
        self.bias1 = np.zeros((1, hidden_dim))
        self.bias2 = np.zeros((1, output_dim))
        
    #--------------------------------------------------------------------------
    
    def compute_cost(self,X, y):
        """
        Computes the total cost on the dataset.
        
        args:
            X: Data array
            y: Labels corresponding to input data
        
        returns:
            cost: average cost per data sample
        """
        #TODO:
        n = len(X)
        z = np.dot(X,self.theta1) + self.bias1
        a = np.tanh(z)
        z2 = np.dot(a,self.theta2)+self.bias2
        exp_z2 = np.exp(z2)
        softmax_scores = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)
        for i in range(n):
            if y[i] == 0:
                one_hot_y = np.array([1,0])
            elif y[i] == 1:
                one_hot_y = np.array([0,1])
            CPS = -np.sum(one_hot_y*np.log(softmax_scores[i]))
            
        return np.mean(CPS)
        
    
    #--------------------------------------------------------------------------
 
    def predict(self,X):
        """
        Makes a prediction based on current model parameters.
        
        args:
            X: Data array
            
        returns:
            predictions: array of predicted labels
        """
        z = np.dot(X,self.theta1) + self.bias1
        a = np.tanh(z)
        z2 = np.dot(a,self.theta2)+self.bias2
        exp_z2 = np.exp(z2)
        softmax_scores = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)
        predictions = np.argmax(softmax_scores, axis = 1)
        return predictions
        
    #--------------------------------------------------------------------------
    
    def fit(self,X,y):
        """
        Learns model parameters to fit the data.
        """  

        n = len(X)
        for i in range(2000):
            gradient_bias1 = 0
            gradient_theta1 = 0
            gradient_theta2 = 0
            gradient_bias2 = 0
            

            #Forward propagation
            z = np.dot(X,self.theta1) + self.bias1
            a = np.tanh(z)
            z2 = np.dot(a,self.theta2)+self.bias2
            exp_z2 = np.exp(z2)
            softmax_scores = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)
            
            
            sigmoid_prime = 1-np.power(a,2)
            for j in range(n):                
                if y[j] == 0:
                    gt = np.array([1,0])
                elif y[j] == 1:
                    gt = np.array([0,1])
            beta2 = softmax_scores-gt
            gradient_theta2 = np.dot(a.T,beta2)
            gradient_bias2 = np.sum(beta2,axis=0,keepdims = True)
            beta1 = np.dot(beta2,(self.theta2).T) * sigmoid_prime
            gradient_theta1 = np.dot(X.T,beta1)
            gradient_bias1 = np.sum(beta1,axis=0)
            
            self.theta2 = self.theta2 - (gradient_theta2/n)*self.r
            self.bias2 = self.bias2 - (gradient_bias2/n)*self.r
            self.theta1 = self.theta1 - (gradient_theta1/n)*self.r
            self.bias1 = self.bias1 - (gradient_bias1/n)*self.r
           
#-------------------------------------------------------------------------
#--------------------------------------------------------------------------

def plot_decision_boundary(model, X, y):
    """
    Function to print the decision boundary given by model.
    
    args:
        model: model, whose parameters are used to plot the decision boundary.
        X: input data
        y: input labels
    """
    
    x1_array, x2_array = np.meshgrid(np.arange(-4, 4, 0.01), np.arange(-4, 4, 0.01))
    grid_coordinates = np.c_[x1_array.ravel(), x2_array.ravel()]
    Z = model.predict(grid_coordinates)
    Z = Z.reshape(x1_array.shape)
    plt.contourf(x1_array, x2_array, Z, cmap=plt.cm.bwr)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
    plt.show()


################################################################################    

X = np.genfromtxt('DATA/Linear/X.csv', delimiter=',')
y = np.genfromtxt('DATA/Linear/y.csv', delimiter=',')         
y.astype(int)


def hiddenLayers():
    
    layers = [3, 5, 7, 10]
    for i in layers:
        f = NeuralNet(2, 2, i, 0.001)
        f.fit(X,y)
        print("Number of hidden layers: ", i)
        plot_decision_boundary(f, X, y)

hiddenLayers()        
