from sklearn.datasets import load_iris
import numpy as np
import torch

rng = np.random.default_rng()
iris =load_iris()
X = iris.data
y = iris.target

X=(X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))

w1 = rng.uniform(-0.5,0.5,( 8,4))
b1 = np.zeros((8,1))
w2 = rng.uniform(-0.5,0.5,(3,8))
b2 = np.zeros((3,1))


def cost_calculator(arr,Y):
    return np.sum(-Y *np.log(arr+1e-10))

def softmax(arr):
    arr = arr - np.max(arr)
    exp_arr =np.exp(arr)
    return exp_arr/np.sum(exp_arr,axis =0)



Loss = 10
lr =0.01
epoch =1

t=(X.shape)[0]
while (epoch < 1000):
    Cost = 0
    for i in range(t):
        Y = np.zeros((3,1))
        Y[y[i],0] = 1
        inp = X[i,:].reshape(4,1)
        Z1 = w1@inp + b1
        A1 = 1/(1+np.exp(-(Z1)))
        Z2 = w2@A1 + b2
        A2 = softmax(Z2)
        Cost = Cost+cost_calculator(A2,Y)
    
        dZ2 = A2 - Y
        dw2 = dZ2 @ A1.T
        db2 = dZ2
        dA1 = w2.T @ dZ2
        dZ1 = dA1 * A1 * (1 - A1)
        dw1 = dZ1 @ inp.T
        db1 = dZ1

        w2 = w2- lr*dw2
        w1 =w1-lr*dw1
        b1=b1-lr*db1
        b2 = b2-lr*db2
    Loss = Cost/t
    print(f"Epoch:{epoch} Loss:{loss}",epoch,loss)
    epoch+=1
    


