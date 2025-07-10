import pandas as pd 
import sklearn.linear_model as lm
import numpy as np 
import matplotlib.pyplot as plt

df = pd.read_csv("/home/pik/Machine-Learning-Specialization-Coursera/C1 - Supervised Machine Learning - Regression and Classification/week3/C1W3A1/data/ex2data1.txt",sep=",")
df.columns = ["subA","subB",'bool']

suba = df.filter(like="subA").to_numpy()
subb = df.filter(like="subB").to_numpy()

sub = df.filter(like="sub").to_numpy()

p = df.filter(["bool"]).to_numpy()

p = p.flatten()

model = lm.LogisticRegression()

model.fit(sub,p)

k = model.predict(sub[0].reshape(1, -1))
print(k[0], p[0])
f = model.score(sub,p)
print(f)
