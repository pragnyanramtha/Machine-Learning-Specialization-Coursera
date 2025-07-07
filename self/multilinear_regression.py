import numpy as np  
import matplotlib.pyplot as plt
plt.style.use('dark_background')        



X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])


plt.scatter(X_train[:, 0], y_train, color='blue', marker='x', label='House Price vs Size')

m = X_train.shape[0]

w = np.zeros(m)
b = 0


def f_wb(x,w, b):
    """
    x : varible
    w : vector size m
    b : scalar
    """
    