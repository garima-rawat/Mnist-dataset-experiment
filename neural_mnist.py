import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def sigmoidDer(X):
    return sigmoid(X) * (1 - sigmoid(X))

def softmax(X):
    f_X = np.exp(X) / np.sum(np.exp(X))
    return f_X

class Model:

    def __init__(self, hidden_layers=1):
        self.weights = [0,0]
        self.weights[0] = np.random.rand(16, 784) - 0.5
        self.weights[1] = np.random.rand(10, 16) - 0.5
        
        self.bias = [0,0]
        self.bias[0] = np.random.rand(16,1) - 0.5
        self.bias[1] = np.random.rand(10,1) - 0.5

        self.act = [0,0]
        self.act[0] ##input array
        self.act[1] = np.empty(shape=(16))


class NeuralNetwork:
    def __init__(self, epochs = 4, alpha = 0.1, batch_size = 100, hidden_layers = 1, neurons = 5):
        self.epochs = epochs
        self.alpha = alpha
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        self.neurons = neurons

        self.train_accuracy = [0]*(epochs+1)
        self.test_accuracy = [0]*(epochs+1)
        self.model = Model(hidden_layers)
        
    
    def readData(self):
        training_image = sys.argv[1]
        training_label = sys.argv[2]
        test_image = sys.argv[3]
        test_label = sys.argv[4]

        df = pd.read_csv(training_image, header=None)
        df['label'] = pd.read_csv(training_label, header=None)
        df = df.loc[0:9999]
        self.training_data = df

        df = pd.read_csv(test_image, header=None)
        df['label'] = pd.read_csv(test_label, header=None)
        self.test_data = df
    
    def accuracy(self, i):
        """
        input X and y of train and test data and outputs accuracy
        Note : the y value for each instance is a single predicted number instead of 10 length array storing probabilities
        """

        print("\n******************\nIn accuracy")

        train_X = self.training_data.iloc[:, 0:784]
        train_y = self.training_data['label'].to_numpy()
        n = len(train_X)
        gamma = self.feedForward(train_X, n)
        pred_y = np.argpartition(gamma, -2)[:, -1:].flatten()
        total = train_y.shape[0]
        self.train_accuracy[i] = (train_y == pred_y).sum()/total
        print("Train")
        print(pred_y)
        print(train_y)

        test_X = self.test_data.iloc[:, 0:784]
        test_y = self.test_data['label'].to_numpy()
        n = len(test_X)
        gamma = self.feedForward(test_X, n)
        pred_y = np.argpartition(gamma, -2)[:, -1:].flatten()
        total = test_y.shape[0]
        self.test_accuracy[i] = (test_y == pred_y).sum()/total
        print("Test")
        print(pred_y)
        print(test_y)

    def plot(self):
        # Define data values
        x = [k for k in range(self.epochs+1)]
        y = self.train_accuracy
        z = self.test_accuracy
        plt.rcParams["figure.autolayout"] = True

        line1, = plt.plot(y, label = "training")
        line2, = plt.plot(z, label = "test")
        leg = plt.legend(loc='upper right')
        plt.show()


    def feedForward(self, X, m):
        """
        Input : X is features in the input layer as pandas Df, m is no of samples i.e. batch size or leftover
        Algo : converts to numpy 2D Arr, 
        Outputs : probability of output layer
        for H samples, input is H X 764 and output is H X 10
        """
        ##print("\n\ninside feed forward")
       
        X = X.to_numpy()
        pred_y = np.empty(shape=(m, 10))
        
        #for data_idx in range(m):
            
            #print(X)s
            #print(X.loc[data_idx])
            ##input layer activation function
            
        self.model.act[0] = X.T/784
        
        #print("act 0")
        #print(self.model.act[0], self.model.act[0].shape)
        #for p in range(self.hidden_layers):
        Z1 = np.dot(self.model.weights[0], self.model.act[0]) + self.model.bias[0]
        self.model.act[1] = sigmoid(Z1)
        #print("act 1")
        #print(self.model.act[1], self.model.act[1].shape)

        ##last layer uses softmax for probabilty range
        #print("basic", self.model.weights[1].shape, self.model.act[1].shape, self.model.bias[1].shape)
        Z2 = np.dot(self.model.weights[1], self.model.act[1]) + self.model.bias[1]
        #print("beta")
        #print(beta, beta.shape)
        pred_y = softmax(Z2)
        pred_y = pred_y.T

        return pred_y, Z1, Z2
    
    def lossFunc(self, true_y, pred_y):
        """
        Takes as input numpy 2D matrix of prediction probabilities and 1D pandas DF of true_y values
        Convert true values to 2D matrix, 1 for true value and 0 for rest
        Matrix : for H samples H X 10
        Algo : Mean of MSE of each sample
        """

        #print("loss function")
        
        n = len(true_y)
        theta = np.empty(shape=(n, 10))
        theta.fill(0)

        true_y = true_y.to_numpy()
        for k in range(n):
            theta[k][true_y[k]] = 1

        print(pred_y.shape, theta.shape)
        
        return np.square(np.subtract(pred_y, theta)).mean(1).mean()

    def oneHot(self, true_y):

        n = len(true_y)
        theta = np.empty(shape=(n, 10))
        theta.fill(0)

        true_y = true_y.to_numpy()
        for k in range(n):
            theta[k][true_y[k]] = 1
        return theta




    
    def backPropagate(self, m, true_y, pred_y, Z1, Z2):
        """
        Takes as input a cost function value and adjusts the weights and biases of the model
        For each batch that went through the forward pass, we backpropagate using SGD(Stochastic Gradient Descent)
        """
        updated_weights = [0]*2
        updated_bias = [0]*2

        dZ2 = pred_y - self.oneHot(true_y)
        updated_weights[1] = 1 / m * np.dot(self.model.act[1], dZ2)
        updated_bias[1] = 1 / m * np.sum(dZ2)
        
        dZ1 = updated_weights[1].dot(dZ2) * sigmoidDer(Z1)
        updated_weights[0] = 1 / m * dZ1.dot(self.model.act[0])
        updated_bias[0] = 1 / m * np.sum(dZ1)
        
        self.updateParam(updated_weights, updated_bias)




    def updateParam(self, updated_weights, updated_bias):

        for i in range(2):
            self.model.weights[i] = self.model.weights[i] - self.alpha * updated_weights[i]
            self.model.bias[i] = self.model.bias[i] - self.alpha * updated_bias[i]

    def algo(self):
        """
        backbone of the algorithm
        For each epoch, shuffles data and divides in different batches
        For each batch, feedforward the training samples, calculate loss function and then backpropagate to change weights and biases
        After all batches of an epoch are computed, calculate their accuracy on training and test data
        """

        self.readData()
       
        for i in range (1, self.epochs+1):
            print("Epoch #", i)
            np.random.shuffle(self.training_data.values)
            n = len(self.training_data)
    
            for j in range(0, n, self.batch_size):
                print("Batch #", j)

                l = min(n, j+self.batch_size)
                m = l-j ##no of training samples in a batch
                train_X = self.training_data.iloc[j:l, 0:784]
                train_y = self.training_data.iloc[j:l, 784]

                print(train_X.shape, train_y.shape, m)
                pred_y, Z1, Z2 = self.feedForward(train_X, m)
                print(pred_y.shape, train_y.shape, Z1.shape, Z2.shape)
                #loss_value = self.lossFunc(train_y, pred_y)
                self.backPropagate(m, train_y, pred_y, Z1.T, Z2.T)

            self.accuracy(i)
        print(self.train_accuracy)
        self.plot()

if __name__ == "__main__":
    asd = NeuralNetwork()
    asd.algo()

        

