import numpy as np  
import matplotlib.pyplot as plt
plt.style.use('dark_background')        



X = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
Y = np.array([460, 232, 178])


plt.scatter(X[:, 0], Y, color='blue', marker='x', label='House Price vs Size')

m = X.shape[0]

w = np.array([0.1,1,1,1])
b = 20


"""w is a vector , x is the input vec , b is base, this is the prediction fuction"""
def fwb(x_vec, w, b):
    y = np.dot(w,x_vec)
    y = y + b
    return y    


def jwb(x, y, w, b): # ? returns cost 
    """this is the cost function"""
    m = X.shape[0]
    c = 0 
    for i in range(m):
        f = fwb(x[i], w, b)
        c += (f - y[i]) ** 2
    c = c / (2 * m)
    return c 

def dj_dw(x,y, w, b):
    """this is the dj/dw func """
    m = X.shape[0]
    c = 0 
    for i in range(m):
        f = fwb(x[i], w, b)
        c += (f - y[i])*x[i]
    c = c / m
    return c         


def dj_db(x,y, w, b):
    """this is the dj/db func """
    m = X.shape[0]
    c = 0 
    for i in range(m):
        f = fwb(x[i], w, b)
        c += (f - y[i])
    c = c / m
    return c 




def gradient(x,y,w,b,alpha=1 * (10 ** -7) ,iterations=10000):
    for i in range(100000):
        w = w - alpha * dj_dw(x,y,w,b)
        b = b - alpha * dj_db(x,y,w,b)
        cost = jwb(x,y,w,b)    
        if i % 50 == 0:
            print(f"iteration:{i} cost:{float(cost)}")
    return w,b


w,b = gradient(X,Y,w,b)

print(f"w:{w},b:{b}")

m = Y.shape[0]
for i in range(m):
    print(f"prediction: {np.dot(X[i], w) + b:0.2f}, target value: {Y[i]}")