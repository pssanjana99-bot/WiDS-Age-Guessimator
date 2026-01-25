import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(len(x_train), 784) / 255.0
x_test  = x_test.reshape(len(x_test), 784) / 255.0 #Turn each image into a 784-length vector and Normalize pixel values between 0 and 1
data = np.column_stack((y_train, x_train))#Put the label in front of each image, so every row now has [label | 784 pixels]
np.random.shuffle(data)#Mix all rows randomly so the model doesn’t see digits in order.
m, n = data.shape#m = number of images,n = number of values per image (785)

data_dev = data[:1000].T #Take first 1000 images for testing the model .T flips rows ↔ columns to match your math
Y_dev = data_dev[0]#Row 0 contains only the label values
X_dev = data_dev[1:]#Row 1 to 784 contains only the pixel values
Y_dev = Y_dev.astype(int)


data_train = data[1000:].T #Remaining training set
Y_train = data_train[0]
X_train = data_train[1:]
Y_train = Y_train.astype(int)
#Network structure = 784 inputs  →  10 hidden neurons  →  10 output neurons

def init_params():
    W1 = np.random.randn(10, 784)*0.01#Each hidden neuron needs one weight for every pixel:
    b1 = np.random.randn(10, 1)#One bias per hidden neuron.
    W2 = np.random.randn(10, 10)*0.01#Each output neuron connects to all 10 hidden neurons:
    b2 = np.random.randn(10, 1)#One bias per hidden neuron.
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z) 

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2,X):
    Z1=W1.dot(X)+b1
    A1=ReLU(Z1)
    Z2=W2.dot(A1)+b2
    A2=softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))#Y.size → how many images,Y.max()+1 → 10 (digits 0–9)- rows r images and coloumns r digits
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y=one_hot_Y.T
    return one_hot_Y
    #Ex Y = [3, 7, 1, 0]
    #Y.size = 4
    #np.arange(Y.size) becomes [0, 1, 2, 3]
    #one_hot_Y[ [0,1,2,3] , [3,7,1,0] ]
    #This means Set:
    '''
    (0,3) = 1
    (1,7) = 1
    (2,1) = 1
    (3,0) = 1
    So it becomes
    Img	    0	1	2	3	4	5	6	7	8	9
    0	    0	0	0	1	0	0	0	0	0	0
    1	    0	0	0	0	0	0	0	1	0	0
    2	    0	1	0	0	0	0	0	0	0	0
    3	    1	0	0	0	0	0	0	0	0	0
    and then we take the transpose
    '''

def deriv_ReLU(Z):
    return Z > 0

def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)

    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)#For each neuron, take the average of all its errors across all images (keep shape 10×1)

    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    # From the output probabilities of the network (10 × m),
    # pick the digit with the highest probability for each image
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    # Compare predicted digits with true digits and return
    # the fraction of correct predictions
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()#Creates random starting weights & biases.

    for i in range(iterations):#Loop that trains the network many times
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)#Push images through network → get predictions.
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)#Calculate how wrong the predictions were.
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)#Actually update weights to improve accuracy.

        if i % 10 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A2), Y))#Print how well the network is doing.

    return W1, b1, W2, b2#Return trained model.
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 1000, 0.1)

def make_predictions(X, W1, b1, W2, b2):
    # Pass input images through the trained network and get output probabilities
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    
    # Convert probabilities into predicted digits (0–9)
    predictions = get_predictions(A2)
    return predictions


def test_prediction(index, W1, b1, W2, b2):
    # Select one image from the training set (shape becomes 784 × 1)
    current_image = X_train[:, index, None]

    # Predict the digit for that image
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)

    # Get the real label for that image
    label = Y_train[index]

    # Print prediction and true label
    print("Prediction:", prediction)
    print("Label:", label)

test_prediction(5, W1, b1, W2, b2)








