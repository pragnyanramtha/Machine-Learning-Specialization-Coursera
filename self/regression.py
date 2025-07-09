import pandas as pd 
import sklearn as sk
import numpy as np 
import matplotlib.pyplot as plt

df = pd.read_csv("/home/pik/Machine-Learning-Specialization-Coursera/C1 - Supervised Machine Learning - Regression and Classification/week3/C1W3A1/data/ex2data1.txt",sep=",")
df.columns = ["subA","subB",'bool']

suba = df.filter(["subA"])

subb = df.filter("subB")



print(subb)