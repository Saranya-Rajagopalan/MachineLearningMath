import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


class KNN():
    def __init__(self, k):
        self.k = k

    def predict(self, test_data, train_Data):
        errors = 0
        class_column = train_Data[0]
        y_test = np.empty(test_data.shape[0])
        for i in range(len(test_data)):
            test_sample = test_data.sample(i)
            distances = np.argsort([self.euclidean_distance(test_sample, train_data.sample(i)) for i in range(1, len(train_data))])
            cluster = distances[:self.k]
            k_nearest_neighbors = np.array([class_column.sample(vector) for vector in cluster])
            y_test[i] = np.bincount(k_nearest_neighbors.astype('int32')).argmax()
            if(y_test[i]!=test_data[0][0]):
                errors += 1
        error_rate = errors/len(test_data)
        return y_test, error_rate

    def euclidean_distance(self, x1, x2):
        distance = 0
        for i in range(len(x1)):
            distance += pow((x1[i] - x2[i+1]), 2)
        return math.sqrt(distance)

def getTrainingData(filename):
    data = pd.read_csv(filename, header=None)
    data.columns = list(range(0, 785))
    return data

filename_train   = "mnist_train.csv"
filename_test    = "mnist_test.csv"


k_values = [1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]

k_error = []

for k in k_values:
    train_data = getTrainingData(filename_train)
    test_data = getTrainingData(filename_test)
    knn = KNN(k)
    prediction, error = knn.predict(test_data, train_data)
    k_error.append(error)


plt.plot(k_values, k_error)
plt.axis([0, 100, 0, 0.25])
plt.xlabel('K value')
plt.ylabel('Test error')
plt.show()
