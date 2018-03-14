import numpy as np
import random
import matplotlib.pyplot as plt

class utils:
    def splitDataset(data, splitRatio):
        trainSet = []
        test = list(data)
        while len(trainSet) < int(len(data) * splitRatio):
            index = random.randrange(len(test))
            trainSet.append(test.pop(index))
        return [trainSet, test]

    def sigmoid(x):
        return np.exp(x)/(1+np.exp(x))

    def getTrainingData(filename):
        data = np.genfromtxt(filename, delimiter=',')
        return np.array(data)


class LogisticRegression:
    def __init__(self, learningrate, number_of_features):
        self.learningrate = learningrate
        self.weight_vector = np.zeros(number_of_features+1)


    def train(self, trainData, iterations = 10000):
        feature1 = np.asarray([float(item[0]) for item in trainData])
        feature2 = np.asarray([float(item[1]) for item in trainData])
        feature3 = np.asarray([float(item[2]) for item in trainData])
        feature4 = np.asarray([float(item[3]) for item in trainData])
        for j in range(iterations):
            delta = []
            for samples in trainData:
                s = np.asarray(samples[:4])
                predicted_y = utils.sigmoid(self.weight_vector[0] + s.T.dot(self.weight_vector[1:]))
                delta.append((samples[-1] - predicted_y)*self.learningrate)

            self.weight_vector[0] = self.weight_vector[0] + np.ones(len(feature1)).T.dot(delta)
            self.weight_vector[1] = self.weight_vector[1] + feature1.T.dot(delta)
            self.weight_vector[2] = self.weight_vector[2] + feature2.T.dot(delta)
            self.weight_vector[3] = self.weight_vector[3] + feature3.T.dot(delta)
            self.weight_vector[4] = self.weight_vector[4] + feature4.T.dot(delta)

    def test(self, testdata):
        wrong_prediction = 0
        for samples in testdata:
            probability = utils.sigmoid(self.weight_vector[0] + samples[:4].T.dot(self.weight_vector[1:]))
            label = 1 if probability>=0.5 else 0
            if(label!=samples[-1]):
                wrong_prediction = wrong_prediction + 1
        return 1-(wrong_prediction/len(testdata))



data = utils.getTrainingData(filename='data_banknote_authentication.csv')
splitRatio = [.01, .02, .05, .1, 0.2, 0.625, 1]
accuracy = []
data, testData = utils.splitDataset(data, 0.33)
for ratio in splitRatio:
    lr = LogisticRegression(learningrate=0.0005, number_of_features=len(data[0])-1)
    print("Learning Rate:\t", lr.learningrate)
    print("Initial Weights:\t", lr.weight_vector)
    trainData, useless = utils.splitDataset(data, splitRatio= ratio)
    print("Training Data\t", len(trainData), "\nTestData\t", len(testData))
    lr.train(trainData,iterations = 8000)
    print("Weights after running running the gradient ascent algorithm 8000 times:\n", lr.weight_vector)
    accuracy.append(lr.test(testData))
    print("Accuracy\t", accuracy[-1])


plt.plot(splitRatio, accuracy)
plt.ylabel('Accuracy')
plt.xlabel('Size of training set(as percentage)')
plt.show()