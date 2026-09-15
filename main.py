from sklearn.datasets import load_iris
import numpy as np

rng = np.random.default_rng()
iris =load_iris()
X = iris.data
y = iris.target

X=(X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))

w1 = rng.uniform(-5,5,( 8,4))
b1 = np.zeros((8,1))
w2 = rng.uniform(-5,5,(3,8))
b2 = np.zeros((3,1))

def cost_calculator(i,arr):
    expectation = np.zeros((3,1))
    expectation[y[i],0] = 1
    return np.sum(np.square(arr-expectation))
    
Cost = 0
t=(X.shape)[1]
for i in range(t):
    Z1 = X[i,:].T
    Z2 = w1@Z1 + b1
    logits = np.exp(1/(1+np.exp(-(w2@Z2 + b2))))
    sum = np.sum(logits)
    Z3 = logits/sum
    Cost = Cost+cost_calculator(i,Z3)








