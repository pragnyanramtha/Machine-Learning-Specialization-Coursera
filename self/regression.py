import pandas as pd 
import sklearn as sk
import numpy as np 
import matplotlib.pyplot as plt

df = pd.read_csv("/home/pik/Machine-Learning-Specialization-Coursera/C1 - Supervised Machine Learning - Regression and Classification/week3/C1W3A1/data/ex2data1.txt",sep=",")
df.columns = ["subA","subB",'bool']

suba = df.filter(like="subA").to_numpy()
subb = df.filter(like="subB").to_numpy()

sub = df.filter(like="sub").to_numpy()
p = df.filter(["bool"]).to_numpy()
m = sub.shape[1]

w = np.zeros(m)
b = 0

def g(x):
    return 1/(1+np.exp(-x))


def func(x,w,b):
    return g(np.dot(w,x) + b)


def loss(x,y,w,b):
    return -y * np.log(func(x,w,b)) - (1 - y) * np.log(1 - func(x,w,b))


def cost(x,y,w,b):
    m = x.shape[0]
    cost = 0 
    for i in range(m):
        cost += loss(x[i],y[i],w,b)
    cost = cost/m

    return cost 


def dj_dw(x,y, w, b):
    """this is the dj/dw func """
    m = x.shape[0]
    c = 0 
    for i in range(m):
        f = func(x[i], w, b)
        c += (f - y[i])*x[i]
    c = c / m
    return c         

def dj_db(x,y, w, b):
    """this is the dj/db func """
    m = x.shape[0]
    c = 0 
    for i in range(m):
        f = func(x[i], w, b)
        c += (f - y[i])
    c = c / m
    return c 

def gradient(x,y,w,b,alpha=(10 ** -2),iters=1000):
    for i in range(iters):
        w = w - alpha * dj_dw(x,y,w,b)
        b = b - alpha * dj_db(x,y,w,b)
        cos = cost(x,y,w,b)
        if i % 75 ==0 :
            print (f"iteration:{i}    cost:{cos}")
    return w,b


w,b = gradient(sub,p,w,b)
suba_flat = suba.flatten()
subb_flat = subb.flatten()
p_flat = p.flatten()

plt.figure(figsize=(8,6))
plt.scatter(suba_flat[p_flat == 1], subb_flat[p_flat == 1], c='g', marker='o', label='Passed')
plt.scatter(suba_flat[p_flat == 0], subb_flat[p_flat == 0], c='r', marker='x', label='Failed')

# Plot decision boundary
x1_vals = np.linspace(suba_flat.min(), suba_flat.max(), 100)
if w[1] != 0:
    x2_vals = -(w[0]/w[1]) * x1_vals - (b/w[1])
    plt.plot(x1_vals, x2_vals, 'b-', label='Decision Boundary')

plt.xlabel('Marks in Subject A')
plt.ylabel('Marks in Subject B')
plt.title('Student Results: Pass/Fail')
plt.legend()
plt.grid(True)
plt.show()