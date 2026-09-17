Manual Neural Network on the Iris Dataset
1. Objective

The objective of this task was to train a small neural network on a real dataset and demonstrate that the loss decreases during training. Instead of using a high-level neural-network training library, I implemented the forward propagation, loss calculation, and backpropagation manually using NumPy.

I used the Iris dataset because it is a small classification dataset with three classes, making it suitable for implementing and debugging a neural network from scratch.

2. Dataset

The Iris dataset contains 150 samples belonging to three different classes of iris flowers:

Setosa
Versicolor
Virginica

Each sample has four numerical features:

Sepal length
Sepal width
Petal length
Petal width

The target variable has three possible classes, represented by the integers 0, 1, and 2.

I loaded the dataset using sklearn.datasets.load_iris() and stored the input features and target values in NumPy arrays.

iris = load_iris()
X = iris.data
y = iris.target
Feature scaling

Before training, I scaled every feature to the range 0 to 1 using min-max normalization:

[
X_{scaled} =
\frac{X-X_{min}}{X_{max}-X_{min}}
]

This was done separately for each feature. Scaling helps keep the inputs within a similar numerical range and makes optimization more stable.

X = (X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))
3. Network Architecture

I implemented a neural network with the following architecture:

4 input features
       ↓
8 hidden neurons
       ↓
3 output neurons

The input layer contains four values corresponding to the four Iris features.

The hidden layer contains eight neurons and uses the sigmoid activation function:

[
\sigma(z)=\frac{1}{1+e^{-z}}
]

The output layer contains three neurons, one for each Iris class. The outputs are converted into probabilities using the softmax function.

The weights were initialized randomly in the range ([-0.5,0.5]), while the biases were initialized to zero.

w1 = rng.uniform(-0.5,0.5,(8,4))
b1 = np.zeros((8,1))

w2 = rng.uniform(-0.5,0.5,(3,8))
b2 = np.zeros((3,1))
4. Forward Propagation

For a single training example, the input is represented as a (4\times1) column vector.

The first layer calculates:

[
Z_1=W_1X+b_1
]

The sigmoid activation is then applied:

[
A_1=\sigma(Z_1)
]

The hidden-layer output is passed to the output layer:

[
Z_2=W_2A_1+b_2
]

Finally, softmax converts the output logits into class probabilities:

[
A_{2,i}=
\frac{e^{Z_{2,i}}}
{\sum_j e^{Z_{2,j}}}
]

The predicted class is therefore the output neuron with the highest probability.

5. Numerical Stability of Softmax

During development, I initially used the exponential of the logits directly:

exp_arr = np.exp(arr)

This can become numerically unstable when the logits become large. Since the exponential function grows very quickly, very large logits can produce extremely large values and eventually cause overflow.

This contributed to numerical problems such as NaN values appearing during loss calculation.

I fixed this by subtracting the maximum logit before taking the exponential:

def softmax(arr):
    arr = arr - np.max(arr)
    exp_arr = np.exp(arr)
    return exp_arr / np.sum(exp_arr, axis=0)

Subtracting the same constant from every logit does not change the resulting softmax probabilities. It only changes the numerical values being passed into the exponential, making the calculation much safer.

6. Loss Function

I used categorical cross-entropy because this is a three-class classification problem.

The target was represented using one-hot encoding. For example, class 1 is represented as:

[
Y=
\begin{bmatrix}
0\
1\
0
\end{bmatrix}
]

The cross-entropy loss for one example is:

[
L=-\sum_iY_i\log(A_{2,i})
]

Since only the correct class has a value of 1 in the one-hot target, this is effectively the negative logarithm of the predicted probability of the correct class.

My implementation was:

def cost_calculator(arr, Y):
    return np.sum(-Y * np.log(arr + 1e-10))

The small value 1e-10 prevents taking the logarithm of zero.

7. Backpropagation

After calculating the loss, I manually calculated the gradients using the chain rule.

For softmax followed by cross-entropy, the gradient with respect to the output logits simplifies to:

[
dZ_2=A_2-Y
]

I therefore calculated:

dZ2 = A2 - Y

The gradient of the second-layer weights is:

[
dW_2=dZ_2A_1^T
]

and the bias gradient is:

[
db_2=dZ_2
]

The gradient is then propagated backwards into the hidden layer:

[
dA_1=W_2^TdZ_2
]

For the sigmoid activation,

[
\sigma'(Z_1)=A_1(1-A_1)
]

so:

[
dZ_1=dA_1\odot A_1(1-A_1)
]

where (\odot) represents element-wise multiplication.

The first-layer gradients are then:

[
dW_1=dZ_1X^T
]

and:

[
db_1=dZ_1
]

These gradients were used to update the parameters using gradient descent:

[
W=W-\alpha dW
]

where (\alpha) is the learning rate.

In my implementation, the learning rate was:

lr = 0.01
8. Gradient Mistakes and Debugging

A significant part of the implementation involved checking that the gradients were actually correct.




A painstaking  issue was numerical instability in the softmax calculation. Directly calculating np.exp(Z2) can cause very large values when the logits increase. This resulted in numerical problems that eventually affected the loss calculation.

I fixed this by subtracting the maximum value from the logits before calculating the exponential:

arr = arr - np.max(arr)

I also added a small constant to the softmax output inside the logarithm:

np.log(arr + 1e-10)

to avoid calculating log(0).

Numerical gradient checking

To independently verify my manually derived gradients, I used numerical gradient checking.

For an individual parameter (w), the numerical derivative can be approximated using:

[
\frac{\partial L}{\partial w}
\approx
\frac{L(w+\epsilon)-L(w-\epsilon)}
{2\epsilon}
]

where (\epsilon) is a very small number.

The numerical gradient can then be compared with the gradient calculated by backpropagation.

I used the relative error:

\frac{|g_{num}-g_{back}|}
{|g_{num}|+|g_{back}|+\epsilon}
]

A small relative error indicates that the manually calculated gradient agrees with the numerical approximation.

This provided an independent way to identify mistakes in matrix multiplication, transposes, activation derivatives, or the chain rule.

9. Training

The network was trained using 1000 epochs. During every epoch, all 150 Iris samples were processed individually.

For each sample, the program performed:

Input
 ↓
Forward propagation
 ↓
Softmax probabilities
 ↓
Cross-entropy loss
 ↓
Backpropagation
 ↓
Weight and bias updates

The loss was accumulated over all training examples and divided by the number of samples to obtain the average loss for the epoch.

Loss = Cost / t

I printed the loss after every epoch so that the decrease in loss could be observed during training.

The decreasing loss demonstrates that the network is learning to produce output probabilities that better match the target classes.

10. Results

The main result required for this task was to demonstrate that the training loss decreases.

The program prints the average loss after every epoch, allowing the training process to be observed directly. The final loss obtained after training was approximately:

0.05299

The loss decreased substantially during training, indicating that the neural network successfully learned a classification mapping from the four input features to the three Iris classes.

The gradient-checking procedure also provided evidence that the manually implemented backpropagation calculations were consistent with numerical differentiation.

11. Conclusion

I successfully implemented a small neural network for three-class Iris classification using NumPy. The implementation included feature normalization, random weight initialization, sigmoid activation, softmax output, categorical cross-entropy loss, and manually derived backpropagation.

The training loss decreased over the epochs, demonstrating that gradient descent was able to optimize the network parameters.

The debugging and numerical gradient-checking process was particularly useful for validating the backpropagation implementation. Issues involving the number of samples, numerical stability of softmax, logarithms near zero, matrix dimensions, and activation derivatives were identified and corrected.

This experiment demonstrated the complete training process of a neural network without relying on a high-level neural-network training framework.