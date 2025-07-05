import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')


house_price = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
house_yards = np.array([250, 300, 480, 430, 630, 730])

plt.scatter(house_price , house_yards, color='blue', marker='x' , label='House Price vs Yards')


w = 100
b = 100



# this is providing the current prediction of the model
def f_wb(x, w, b):
    m = house_yards.shape[0]
    y = np.zeros(m)
    for i in range(m):
        y[i] = w * x[i] + b 

    return y


# This is providing the cost of every point 
def j_wb(x,y,w,b):
    m = x.shape[0]
    f = f_wb(x, w, b)
    j = 0 

    for i in range(m):
        j = j + (f[i] - y[i]) ** 2

    j = j / (2 * m)
    return j


def dj_dw(x,y,w,b):
    m = x.shape[0]
    dj = 0
    f = f_wb(x, w, b)
    for i in range(m):
        dj = dj + (f[i] - y[i]) * x[i]

    return dj / m

def dj_db(x,y,w,b):
    m = x.shape[0]
    dj = 0
    f = f_wb(x, w, b)
    for i in range(m):
        dj = dj + (f[i] - y[i])
    return dj / m    


alpha = 0.01

def gradient_descent(x, y, w, b, alpha, max_iters=1000):
    for i in range(max_iters):
        w = w - alpha * dj_dw(x, y, w, b)
        b = b - alpha * dj_db(x, y, w, b)
        cost = j_wb(x, y, w, b)
        print(f"Cost: {cost}, w: {w}, b: {b}")
        
    
    a = [w,b]
    return a
    

a = gradient_descent(house_price, house_yards, w, b, alpha)
w = a[0]
b = a[1]
prediction_line = f_wb(house_price, w, b)
plt.plot(house_price , prediction_line, color='red', label='Prediction Line')
plt.show()