import numpy as np
import math
from process_data import process_data

def initialize_parameters(layers):
  parameters = {}
  for i in range(len(layers) - 1):
    parameters["W" + str(i + 1)] = np.random.randn(layers[i + 1], layers[i]) * 0.01
    parameters["B" + str(i + 1)] = np.zeros((layers[i + 1], 1))
  
  return parameters

def forward_sigmoid(Z):
  return 1 / (1 + np.exp(-Z))

def forward_relu(Z):
  return np.maximum(Z, 0)

def forward_tanh(Z):
  return np.tanh(Z)

def forward_prop(X, layers, parameters, cache={}):
  A = X
  for i in range(len(layers) - 1):
    W = parameters["W" + str(i + 1)]
    B = parameters["B" + str(i + 1)]
    Z = np.dot(W, A) + B
    
    if i == len(layers) - 2:
      A = forward_sigmoid(Z)
    else:
      A = forward_relu(Z)

    cache["W" + str(i + 1)] = W
    cache["B" + str(i + 1)] = B
    cache["Z" + str(i + 1)] = Z
    cache["A" + str(i + 1)] = A
  
  return cache["A" + str(len(layers) - 1)]

def compute_cost(AL, Y, W, m, lambd):
  return (-1 / m) * np.sum(
    np.multiply(Y, np.log(AL)) + np.multiply((1 - Y),np.log(1 - AL))
  ) + lambd * np.sum(W**2) / (2*m)

def backward_prop(AL, Y, X, layers, cache, m, lambd, t, beta1=0.9, beta2=0.999, optimizer=None):
  grads = {}

  dZ = AL - Y
  dW = (1 / m) * np.dot(dZ, cache["A" + str(len(layers) - 2)].T)
  dB = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
  grads["dZ" + str(len(layers) - 1)] = dZ
  grads["dW" + str(len(layers) - 1)] = dW
  grads["dB" + str(len(layers) - 1)] = dB

  # adam op
  if optimizer == "adam":
    vdW = (1 - beta1) * dW
    vdB = (1 - beta1) * dB
    sdW = (1 - beta2) * (dW ** 2)
    sdB = (1 - beta2) * (dB ** 2)
    vdW_corrected = vdW / (1 - np.power(beta1, t))
    vdB_corrected = vdB / (1 - np.power(beta1, t))
    sdW_corrected = sdW / (1 - np.power(beta2, t))
    sdB_corrected = sdB / (1 - np.power(beta2, t))

    grads["vdW" + str(len(layers) - 1)] = vdW_corrected
    grads["vdB" + str(len(layers) - 1)] = vdB_corrected
    grads["sdW" + str(len(layers) - 1)] = sdW_corrected
    grads["sdB" + str(len(layers) - 1)] = sdB_corrected


  for i in range(len(layers) - 2, 0, -1):
    W = cache["W" + str(i + 1)]
    Z = cache["Z" + str(i)]
    if i > 1:
      A = cache["A" + str(i - 1)]
    else:
      A = X


    dZ = np.dot(W.T, dZ) * (Z > 0)
    dW = (1 / m) * np.dot(dZ, A.T)
    dB = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

    grads["dZ" + str(i)] = dZ
    grads["dW" + str(i)] = dW
    grads["dB" + str(i)] = dB
    
    # adam op
    if optimizer == "adam":
      vdW = (1 - beta1) * dW
      vdB = (1 - beta1) * dB
      sdW = (1 - beta2) * (dW ** 2)
      sdB = (1 - beta2) * (dB ** 2)
      vdW_corrected = vdW / (1 - np.power(beta1, t))
      vdB_corrected = vdB / (1 - np.power(beta1, t))
      sdW_corrected = sdW / (1 - np.power(beta2, t))
      sdB_corrected = sdB / (1 - np.power(beta2, t))

      grads["vdW" + str(i)] = vdW_corrected
      grads["vdB" + str(i)] = vdB_corrected
      grads["sdW" + str(i)] = sdW_corrected
      grads["sdB" + str(i)] = sdB_corrected

  return grads

def update_parameters(grads, parameters, layers, alpha, lambd, epsilon=10e-8, optimizer=None):
  for l in range(len(layers) - 1):
    if optimizer == 'adam':
      stepW = grads["vdW" + str(l + 1)] / (np.sqrt(grads["sdW" + str(l + 1)]) + epsilon)
      stepB = grads["vdB" + str(l + 1)] / (np.sqrt(grads["sdB" + str(l + 1)]) + epsilon)
    else:
      stepW = parameters["W" + str(l + 1)]
      stepB = parameters["B" + str(l + 1)]
    parameters["W" + str(l + 1)] -= alpha * grads["dW" + str(l + 1)] + (lambd / m) * stepW
    parameters["B" + str(l + 1)] -= alpha * grads["dB" + str(l + 1)] + (lambd / m) * stepB
  
  return parameters

def train_model(X, Y, layers, alpha = 0.01, lambd = 0, num_iterations = 10000, print_cost = True, optimizer=None):
  m = X.shape[0]
  parameters = initialize_parameters(layers)
  cache = {}

  for i in range(num_iterations):
    AL = forward_prop(X, layers, parameters, cache)
    cost = compute_cost(AL, Y, parameters["W" + str(len(layers) - 1)], m, lambd)
    grads = backward_prop(AL, Y, X, layers, cache, m, lambd, i + 1, optimizer=optimizer)

    parameters = update_parameters(grads, parameters, layers, alpha, lambd, optimizer=optimizer)

    if print_cost and i % 1000 == 0:
      print ("Cost after iteration %i: %f" %(i, cost))

  return parameters

# initialize hyperparameters
alpha = 0.01
lambd = 0.1

train, test = process_data('train.csv', 'test.csv')

train_arr = np.array(train)
train_X_orig = train_arr[:, 1:]
train_X = (train_X_orig - np.average(train_X_orig, axis=0)) / (np.std(train_X_orig, axis=0))
train_Y = train_arr[:, 0:1]
test_X = np.array(test)
m = train_X.shape[0]
split_pivot = math.floor(0.8*m)

train_X = train_X[0:split_pivot, :]
dev_X = train_X[split_pivot:,:]
train_Y = train_Y[0:split_pivot, :]
dev_Y = train_Y[split_pivot:, :]

# initialize NN architecture
layers = [train_X.shape[1], 5, 2, 1]

# print(train_X.shape, train_Y.shape)
# train model
parameters = train_model(train_X.T, train_Y.T, layers, alpha, lambd, num_iterations=10000)