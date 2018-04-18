import pandas as pd
import math
import numpy as np



class K_means:
    def __init__(self, k, train_data, mean_vectors = None):
        self.k = k
        n, features = np.shape(train_data)
        self.data = train_data
        if mean_vectors!=None:
            self.centroids_list = mean_vectors
        else:
            random_vector_index = np.random.choice(range(n), self.k)
            self.centroids_list = [train_data[i] for i in random_vector_index]

    def classify(self, data_point):
        min = math.inf
        min_index = 0
        for i in range(len(self.centroids_list)):
            distance = self.euclidean_distance(self.centroids_list[i], data_point)
            if min > distance:
                min_index = i
                min = distance
        return min_index

    def cluster(self, train_data):
        clusters = [[] for _ in range(self.k)]
        for index, data_point in enumerate(train_data):
            centroid_vector = self.classify(data_point)
            clusters[centroid_vector].append(index)
        return clusters

    def recenter(self):
        for i, cluster in enumerate(self.cluster(self.data)):
            self.centroids_list[i] = np.mean(self.data[cluster], axis=0)

    def potential_function(self):
        distance = []
        for samples in self.data:
            distance.append(self.euclidean_distance(self.centroids_list[self.classify(samples)], samples))
        return sum(distance)

    def euclidean_distance(self, x1, x2):
        distance = 0
        for i in range(len(x1)):
            distance += pow((x1[i] - x2[i]), 2)
        return math.sqrt(distance)

    def fit(self):
        prev_mean = self.centroids_list
        self.recenter()
        keep_running = 1
        while(keep_running<1000):
            keep_running +=1
            self.recenter()

def getTrainingData(filename):
    data = pd.read_csv(filename, header=None)
    data.columns = list(range(0, 11))
    data = data.drop(columns=0)
    data = data.drop(columns=10)
    return data.as_matrix()


file_name = 'B:\\MyCodebase\\MachineLearningMath\\data\\bc.txt'
train_data = getTrainingData(file_name)


K = list(range(2,9))

for each_k in K:
    k = K_means(each_k, train_data)
    k.fit()
    print(k.potential_function())



