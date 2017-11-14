"""
LogisticRegression.py

CS440/640: Lab-Week5

Lab goal: 1) Implement logistic regression classifier
"""

import numpy as np 
import matplotlib.pyplot as plt 

class LogisticRegression:
    """
    This class implements a Logistic Regression Classifier.
    """
    
    def __init__(self, input_dim, output_dim):
        """
        Initializes the parameters of the logistic regression classifer to 
        random values.
        
        args:
            input_dim: Number of dimensions of the input data
            output_dim: Number of classes
        """
        
        self.theta = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.bias = np.zeros((1, output_dim))
        
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
        z = np.dot(X,self.theta) + self.bias
        exp_z = np.exp(z)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
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
        z = np.dot(X,self.theta) + self.bias
        exp_z = np.exp(z)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        predictions = np.argmax(softmax_scores, axis = 1)
        return predictions
        
    #--------------------------------------------------------------------------
    
    def fit(self,X,y):
        """
        Learns model parameters to fit the data.
        """  
        n = len(X)
        for i in range(0,2000):
            gradient_bias = 0
            gradient_theta = 0
            for j in range(n):
                z = np.dot(X[j],self.theta) + self.bias
                exp_z = np.exp(z)
                softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
                for i in range(n):
                    if y[i] == 0:
                        ground_truth = np.array([1,0])
                    elif y[i] == 1:
                        ground_truth = np.array([0,1])
                abc = softmax_scores-ground_truth
                gradient_theta += np.dot(X[j].reshape(2,1),abc)
                gradient_bias += abc
            self.theta = self.theta - (gradient_theta/n)*0.001
            self.bias = self.bias - (gradient_bias/n)*0.001

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


#X = np.genfromtxt('DATA/Linear/X.csv', delimiter=',')
#y = np.genfromtxt('DATA/Linear/y.csv', delimiter=',')

X = np.genfromtxt('DATA/Linear/X.csv', delimiter=',')
y = np.genfromtxt('DATA/Linear/y.csv', delimiter=',')

plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.bwr)
plt.show()

f = LogisticRegression(2,2)
f.fit(X,y)
plot_decision_boundary(f,X,y)

