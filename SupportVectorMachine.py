"""
Support Vector Machine for handwritten digit classification

SVMs are great for small datasets

Created by  :   Saranya Rajagopalan
Date        :   17-02-2018
"""
import numpy as np
from matplotlib import pyplot as plt

x = np.array([
    [-2, 4, -1],
    [4, 1, -1],
    [1, 6, 1],
    [-2, 4, 1],
    [2, 4, 1],
])


y = np.array([-1, -1, -1, 1, 1])

for d, sample in enumerate(x):
    if d <=2:
        plt.scatter(sample[0], sample[1], s=120, marker= '_', linewidths=2)
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)



plt.plot([-2, 6],[6, 0.5])
plt.show()
