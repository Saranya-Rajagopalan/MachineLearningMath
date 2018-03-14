import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

class utils:
    def splitDataset(data, splitRatio, ls =True):
        trainSet = []
        if ls:
            test = list(data)
        else:
            test = list(data.values)
        while len(trainSet) < int(len(data) * splitRatio):
            index = random.randrange(len(test))
            trainSet.append(test.pop(index))
        return [trainSet, test]

    def gaussian(x, mean, sd):
        exponent = - (np.square(x - mean))/ 2 * np.square(sd)
        norm_constant = 1 / ((np.sqrt(2 * np.pi)) * sd)
        return norm_constant * np.exp(exponent)

    def getTrainingData(filename):
        data = pd.read_csv(filename, header=None)
        data.columns = [1, 2, 3, 4, "class"]
        return data


class GNB:
    def __init__(self, data):
        self.number_of_features = len(data[0])-1


    def train(self, trainData):
        label_0 = [d for d in trainData if d[-1] == 0]
        label_1 = [d for d in trainData if d[-1] == 1]
        self.prior0  = len(label_0) / float(len(trainData))
        self.mean = []
        self.sd = []
        for i in range(self.number_of_features):
            #For every feature, find the conditional mean given both y =0  and y =1
            self.mean.append([np.mean(label_0[i]), np.mean(label_1[i])])
            self.sd.append([np.std(label_0[i]), np.std(label_1[i])])


    def test(self, testData):
        error = 0
        for samples in testData:
            likelihood = [0, 0]
            for y in [0, 1]:
                likelihood[y] = 1
                for i in range(self.number_of_features):
                    x = samples[i]
                    likelihood[y] *= utils.gaussian(x, mean = self.mean[i][y], sd=self.sd[i][y])

            a = likelihood[0] * self.prior0
            b = likelihood[1] * (1 - self.prior0)

            posterior0 = a / float(a+b)
            label_predict = [0 if posterior0>=0.5 else 1]
            if label_predict != samples[-1]:
                error = error +1

        return 1-(error/len(testData))


    def generate_data(self, number_of_data):
        fake_data = []
        for i in range(number_of_data):
            x1 = utils.gaussian(i, mean= self.mean[0][1], sd = self.sd[0][1]) +np.random.normal(0, 0.01)
            x2 = utils.gaussian(i, mean= self.mean[1][1], sd = self.sd[1][1]) +np.random.normal(0, 0.01)
            x3 = utils.gaussian(i, mean= self.mean[2][1], sd = self.sd[2][1]) +np.random.normal(0, 0.01)
            x4 = utils.gaussian(i, mean= self.mean[3][1], sd = self.sd[3][1]) +np.random.normal(0, 0.01)
            fake_data.append([x1, x2, x3, x4])
        print (fake_data)
        return fake_data



data = utils.getTrainingData(filename='data_banknote_authentication.csv')
splitRatio = [.01, .02, .05, .1, 0.2, 0.625, 1]
accuracy = []
data, testData = utils.splitDataset(data, 0.33, ls =False)
for ratio in splitRatio:
    trainData, useless = utils.splitDataset(data, splitRatio= ratio)
    gnb = GNB(trainData)
    print("Training Data\t", len(trainData), "\nTestData\t", len(testData))
    gnb.train(trainData)
    print("Means of Gaussian distributions of feature vectors:\n", gnb.mean)
    accuracy.append(gnb.test(testData))
    print("Accuracy\t", accuracy[-1])


plt.plot(splitRatio, accuracy)
plt.ylabel('Accuracy')
plt.xlabel('Size of training set(as percentage)')
plt.show()


###################################################################
#Generating Fake data
###################################################################

fake_data = gnb.generate_data(400)







