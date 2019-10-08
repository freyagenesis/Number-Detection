import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.svm import SVC

def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    initialWeights = initialWeights.reshape((n_features + 1, 1))
    train_data_with_bias = np.hstack((np.ones((n_data, 1)), train_data))
    theta_n = sigmoid(np.dot(train_data_with_bias, initialWeights))

    Y_n = labeli
    
    error = (((-1)*(np.sum(np.multiply(Y_n, np.log(theta_n)) + np.multiply((1.0 - Y_n), np.log(1.0 - theta_n))))))
    
    error_grad = (theta_n - Y_n) * train_data_with_bias
    
    error_grad = np.sum(error_grad, axis = 0)
    
    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    label = sigmoid(np.dot(np.hstack((np.ones((data.shape[0], 1)), data)), W))
                    
    label = np.argmax(label, axis =1)
    
    label = label.reshape((data.shape[0], 1))
    
    return label

def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights_b: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))
    
    w = params.reshape((n_feature + 1, n_class))
    
    train_data_with_bias = np.hstack((np.ones((n_data, 1)), train_data))
    posterior = np.dot(train_data_with_bias, w)

    theta_sum = np.sum(np.exp(posterior), axis = 1).reshape(n_data, 1)
    final_theta = (np.exp(posterior)/theta_sum)
    
    error = (-1) * np.sum(np.sum(labeli * np.log(final_theta)))

    error_grad = (np.dot(np.transpose(train_data_with_bias), (final_theta - labeli))).flatten()

    return error, error_grad

def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    train_data_with_bias = np.hstack((np.ones((data.shape[0], 1)), data))
    
    post = np.dot(train_data_with_bias, W)
    theta_sum = np.sum(np.exp(post))

    posterior = (np.exp(post) / theta_sum)

    for i in range(posterior.shape[0]):
        label[i] = np.argmax(posterior[i])
    label = label.reshape(label.shape[0], 1)

    return label

"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}

for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))
    

# Find the accuracy on Training Dataset
predicted_label_train = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_train == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_valid = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_valid == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_test = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_test == test_label).astype(float))) + '%')

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')

train_indices = np.random.choice(train_data.shape[0], 10000)
train_data_svm = train_data[train_indices]
train_label_svm = train_label[train_indices]
X_train = train_data_svm
Y_train = train_label_svm

X_test = test_data
Y_test = test_label

X_valid = validation_data
Y_valid = validation_label

# Model: Linear
svm_model_linear = SVC(kernel = 'linear').fit(X_train, Y_train)
svm_predictions = svm_model_linear.predict(X_train)
accuracy_train_linear = svm_model_linear.score(X_train, Y_train)
svm_predictions = svm_model_linear.predict(X_test)
accuracy_test_linear = svm_model_linear.score(X_test, Y_test)
svm_predictions = svm_model_linear.predict(X_valid)
accuracy_valid_linear = svm_model_linear.score(X_valid, Y_valid)   

# Model: RBF with Gamma = 1

svm_model_radial_1 = SVC(kernel = 'rbf', gamma = 1).fit(X_train, Y_train)
svm_predictions = svm_model_radial_1.predict(X_train)
accuracy_train_radial_1 = svm_model_radial_1.score(X_train, Y_train)
svm_predictions = svm_model_radial_1.predict(X_test)
accuracy_test_radial_1 = svm_model_radial_1.score(X_test, Y_test)
svm_predictions = svm_model_radial_1.predict(X_valid)
accuracy_valid_radial_1 = svm_model_radial_1.score(X_valid, Y_valid)    

# Model: RBF with defaults

svm_model_radial_2 = SVC(kernel = 'rbf').fit(X_train, Y_train)
svm_predictions = svm_model_radial_2.predict(X_train)
accuracy_train_radial_2 = svm_model_radial_2.score(X_train, Y_train)
svm_predictions = svm_model_radial_2.predict(X_test)
accuracy_test_radial_2 = svm_model_radial_2.score(X_test, Y_test)
svm_predictions = svm_model_radial_2.predict(X_valid)
accuracy_valid_radial_2 = svm_model_radial_2.score(X_valid, Y_valid)    


n_groups = 3
acc_linear = [accuracy_train_linear, accuracy_test_linear, accuracy_valid_linear]
acc_radial_1 = [accuracy_train_radial_1, accuracy_test_radial_1, accuracy_valid_radial_1]
acc_radial_2 = [accuracy_train_radial_2, accuracy_test_radial_2, accuracy_valid_radial_2]

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.20
opacity = 0.8

rects1 = plt.bar(index, [acc_linear[0], acc_radial_1[0], acc_radial_2[0]], bar_width,
alpha=opacity,
color='b',
label='Train')

rects2 = plt.bar(index + bar_width, [acc_linear[1], acc_radial_1[1], acc_radial_2[1]], bar_width,
alpha=opacity,
color='g',
label='Test')

rects3 = plt.bar(index + 2*bar_width, [acc_linear[2], acc_radial_1[2], acc_radial_2[2]], bar_width,
alpha=opacity,
color='r',
label='Validation')

plt.xlabel('SVM Models')
plt.ylabel('Accuracies')
plt.title('SVM Models and accuracies')
plt.xticks(index + bar_width, ('Linear', 'RBF with Gamma=1', 'RBF with defaults'))
plt.legend()

plt.tight_layout()
plt.show()


# Model: RBF with C varying values

C = range(0,100,10)
C[0] = 1

accuracy_train_radial_c = []
accuracy_test_radial_c = []
accuracy_valid_radial_c = []

for c in C:
    svm_model_radial_c = SVC(C = c, kernel = 'rbf').fit(X_train, Y_train)
    svm_predictions = svm_model_radial_c.predict(X_train)
    accuracy_train_radial_c.append(svm_model_radial_c.score(X_train, Y_train))
    svm_predictions = svm_model_radial_c.predict(X_test)
    accuracy_test_radial_c.append(svm_model_radial_c.score(X_test, Y_test))
    svm_predictions = svm_model_radial_c.predict(X_valid)
    accuracy_valid_radial_c.append(svm_model_radial_c.score(X_valid, Y_valid))

plt.plot(C, accuracy_train_radial_c, 'k', label="Train")
plt.plot(C, accuracy_test_radial_c, 'bs', label="Test")
plt.plot(C, accuracy_valid_radial_c, 'r--', label="Validation")
plt.title('With Different C values')
plt.legend()
plt.show()

"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b_train = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b_train == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b_valid = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b_valid == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b_test = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b_test == test_label).astype(float))) + '%')


