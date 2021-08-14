from math import sqrt
from math import pi
from math import exp
import scipy.io as scipy
import numpy as np

# Load the data from file
data = scipy.loadmat('fashion_mnist.mat')
trX, trY, tsX, tsY = data['trX'], np.reshape(
    data['trY'], -1), data['tsX'], np.reshape(data['tsY'], -1)

# This function transforms 784 features to 2 features
# X1 = Mean of the image
# X2 = Standard Deviation of the image


def transform_data(dataset):
    mean, std_dev = dataset.mean(axis=1), dataset.std(axis=1)
    return np.stack((mean, std_dev), axis=1)


# Transform the training and testing data from 784 to 2 features for each image
trX, tsX = transform_data(trX), transform_data(tsX)

# Gaussian Naive Bayes Class


class GaussianNB:

    # Initializes and populates the class-wise parameters from the training data
    def __init__(self, features, labels):
        self.label_params = self.cal_labelwise_parameters(features, labels)
        print(self.label_params)

    # Calculates the parameters of mean and standard deviation for each input parameter for each class
    def cal_labelwise_parameters(self, features, labels):
        d = dict()
        temp = np.unique(labels)

        # Separates input data based on their labels
        for i in range(temp.shape[0]):
            x_n = features[labels == i]
            d[i] = x_n

        s = dict()
        # Calculates parameter-wise mean and standard deviation for the class-wise separated data
        for l, data in d.items():
            s[l] = [data.mean(axis=0), data.std(axis=0), data.shape[0]]
        return s

    # Calculates the probability of x from given mean and std deviation assuming a normal distribution
    def cal_prob(self, mean, std_dev, x):
        e = exp(-((x-mean)**2/(2*std_dev**2)))
        return (1/(sqrt(2*pi)*std_dev))*e

    # Calculates labelwise probabilities for x from the parameters calculated earlier
    def cal_label_prob(self, parameters, x):
        count = sum([parameters[label][-1] for label in parameters])
        probabilities = dict()
        # From parameters computed earlier, it calculates probability for each label
        for label, label_params in parameters.items():
            probabilities[label] = np.log(parameters[label][-1]/count)
            for i in range(len(label_params)-1):
                probabilities[label] += np.log(self.cal_prob(
                    parameters[label][0][i], parameters[label][1][i], x[i]))
        return probabilities

    # Test the data based on learned parameters
    # Prints Confusion matrix and overall accuracy
    def test(self, tsX, tsY):
        self.confus_matrix = np.zeros((2, 2))

        for i in range(tsX.shape[0]):
            label_prob = self.cal_label_prob(self.label_params, tsX[i])
            predicted_label = 0
            max_ll = -1000
            for label, ll in label_prob.items():
                if ll > max_ll:
                    predicted_label = label
                    max_ll = ll
            self.confus_matrix[int(tsY[i])][int(predicted_label)] += 1

        print('Naive Bayes Confusion Matrix:')
        print(self.confus_matrix)
        correct_pred = sum([self.confus_matrix[i][i]
                            for i in range(self.confus_matrix.shape[0])])
        accuracy = correct_pred/tsY.size
        print('Naive Bayes Accuracy:')
        print(accuracy*100)

# Logistic Regression Class


class LogisticRegression:

    # Initializes weight co-efficients and bias as 0. Also assigns given learning_rate and iterations
    # Then calculates the weight co-efficients from given learning rate and iterations
    def __init__(self, features, labels, learning_rate=5, iterations=10000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = np.zeros(features.shape[1])
        self.bias = 0
        self.train(features, labels)

    # Calculates probability between 0-1 based on using weights, bias and sigmoid function
    def predict_labels(self, features):
        yhat = np.dot(self.weights, features.T) + self.bias
        return 1.0 / (1.0 + np.exp(-yhat))

    # Trains the model
    def train(self, features, labels):
        for i in range(self.iterations):
            loss = []
            y_pred = self.predict_labels(features)
            loss = (labels - y_pred)
            # Updates the weights and bias
            self.weights += self.learning_rate * \
                np.average(loss*features.T, axis=1)
            self.bias += self.learning_rate*np.average(loss)
            if i % 1000 == 0:
                print('loss:', np.average(loss))

    # Tests the model on test data using optimized weights, bias.
    # Probability >= 0.5 is assigned to class 1
    # Probability < 0.5 is assigned to class 0
    # Prints confusion matrix and overall accuracy
    def test(self, tsX, tsY):
        self.confus_matrix = np.zeros((2, 2))

        pred_labels = self.predict_labels(tsX)
        for i in range(pred_labels.size):
            pred_class = 1
            if pred_labels[i] < 0.5:
                pred_class = 0
            self.confus_matrix[int(tsY[i])][pred_class] += 1

        print('Logistic Regression Confusion Matrix:')
        print(self.confus_matrix)
        correct_pred = sum([self.confus_matrix[i][i]
                            for i in range(self.confus_matrix.shape[0])])
        accuracy = correct_pred/tsY.size
        print('Logistic Regression Accuracy:')
        print(accuracy*100)


nb = GaussianNB(trX, trY)
nb.test(tsX, tsY)
lg = LogisticRegression(trX, trY, 1, 15000)
lg.test(tsX, tsY)
