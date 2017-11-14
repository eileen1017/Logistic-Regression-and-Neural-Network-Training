"""
NeuralNet.py

CS440 - PA1

#7

"""
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
import pandas as pd


class NeuralNet:
    """
    This class implements a 3 layer neural network.
    """
    
    def __init__(self, input_dim, output_dim, hidden_dim):
        """
        Initializes the parameters of the logistic regression classifer to 
        random values.
        
        args:
            input_dim: Number of dimensions of the input data
            output_dim: Number of classes
        """
        
        self.theta1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.bias1 = np.zeros((1, hidden_dim))

        self.theta2 = np.random.randn(hidden, output_dim) / np.sqrt(hidden_dim)
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
        n = len(X)
        z = np.dot(X, self.theta1) + self.bias1
        a = np.tanh(z) 
        z2 = np.dot(a, self.theta2) + self.bias2
        exp_z = np.exp(z2)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        for i in range(n):
            if y[i] == 0:
                one_hot_y = np.array([1,0])
            elif y[i] == 1:
                one_hot_y = np.array([0, 1])
            CPS = -np.sum(one_hot_y * np.log(softmax_scores[i]))
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
        
        z = np.dot(X, self.theta1) + self.bias1
        a = np.tanh(z)        
        z2 = np.dot(a, self.theta2) + self.bias2
        exp_z = np.exp(z2)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        predictions = np.argmax(softmax_scores, axis = 1)

        return predictions
        
    #--------------------------------------------------------------------------
    
    def fit(self,X,y):
        
        """
        Learns model parameters to fit the data.
        """  

        for i in range(1000):
            
            gradient_theta1 = 0
            gradient_bias1 = 0
            gradient_theta2 = 0
            gradient_bias2 = 0
            n = len(X)
            
            for j in range(n):
                               
                len_j = len(X[j])
                
                z = np.dot(X[j], self.theta1) + self.bias1
                a = np.tanh(z)
                z2 = np.dot(a, self.theta2) + self.bias2
                exp_z = np.exp(z2)
               
                
                
                soft_output = exp_z / np.sum(exp_z, axis=1, keepdims=True)
                
                
                ground_truth = np.zeros(10, dtype = int)
                index = y[j].astype(int) 
                ground_truth[(np.array(index))] = 1                          
                beta2 = soft_output - ground_truth

                gradient_theta2 += np.dot(a.T, beta2)
                gradient_bias2 += np.dot(beta2.reshape(1,10), np.ones((10,10)))
                
                beta1 = np.dot(beta2, self.theta2.T) * (1-np.power(a,2)) 
                
                gradient_theta1 += np.dot(X[j].reshape(64, 1), beta1)
                gradient_bias1 += np.dot(beta2.reshape(1,10), np.ones((10,10)))
                
    
            self.theta1 = self.theta1 - 0.001 * gradient_theta1/n
            self.bias1 = self.bias1 - 0.001 * gradient_bias1/n
            self.theta2 = self.theta2 - 0.001 * gradient_theta2/n
            self.bias2 = self.bias2 - 0.001 * gradient_bias2/n
##--------------------------------------------------------------------------
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



X_train = np.genfromtxt('DATA/Digits/X_train.csv', delimiter = ',')
y_train = np.genfromtxt('DATA/Digits/y_train.csv', delimiter = ',')

X_test = np.genfromtxt('DATA/Digits/X_test.csv', delimiter = ',')
y_test = np.genfromtxt('DATA/Digits/y_test.csv', delimiter = ',')


g = NeuralNet (64, 10, 10)
g.fit(X_train, y_train)


y_actual = pd.Series(y_test.astype(int), name='Actual')
y_predicted = pd.Series(g.predict(X_test), name = 'Predicted')
confusion = pd.crosstab(y_actual, y_predicted, rownames=['Actual'], colnames=['Predicted'], margins=True)
print(confusion)


score = accuracy_score(y_test.astype(int), g.predict(X_test))
print(score)

