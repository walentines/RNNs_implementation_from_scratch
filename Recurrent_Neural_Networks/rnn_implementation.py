import numpy as np
from utils import * 
import pandas as pd
import matplotlib.pyplot as plt
import math

train_dataset = pd.read_csv('dataset/Google_Stock_Price_Train.csv')
training_set = train_dataset.iloc[:, 1:2].values

scaled_training_set = (training_set - np.min(training_set)) / (np.max(training_set) - np.min(training_set))

X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(scaled_training_set[i - 60:i, 0])
    y_train.append(scaled_training_set[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (1, X_train.shape[0], X_train.shape[1]))
y_train = np.reshape(y_train, (1, y_train.shape[0], 1))

def initialize_adam(parameters):
    s = {}
    v = {}
    
    v['dWax'] = np.zeros((parameters['Wax'].shape[0], parameters['Wax'].shape[1]))
    v['dWaa'] = np.zeros((parameters['Waa'].shape[0], parameters['Waa'].shape[1]))
    v['dWya'] = np.zeros((parameters['Wya'].shape[0], parameters['Wya'].shape[1]))
    v['dba'] = np.zeros((parameters['ba'].shape[0], parameters['ba'].shape[1]))
    v['dby'] = np.zeros((parameters['by'].shape[0], parameters['by'].shape[1]))
    
    s['dWax'] = np.zeros((parameters['Wax'].shape[0], parameters['Wax'].shape[1]))
    s['dWaa'] = np.zeros((parameters['Waa'].shape[0], parameters['Waa'].shape[1]))
    s['dWya'] = np.zeros((parameters['Wya'].shape[0], parameters['Wya'].shape[1]))
    s['dba'] = np.zeros((parameters['ba'].shape[0], parameters['ba'].shape[1]))
    s['dby'] = np.zeros((parameters['by'].shape[0], parameters['by'].shape[1]))
             
    return v, s

def random_mini_batches(X, Y, mini_batch_size = 64):
    m = X.shape[1]
    mini_batches = []
    
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = X[:, k * mini_batch_size : (1 + k) * mini_batch_size, :]
        mini_batch_Y = Y[:, k * mini_batch_size : (1 + k) * mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Last case
    if m % mini_batch_size != 0:
        mini_batch_X = X[:, num_complete_minibatches * mini_batch_size:, :]
        mini_batch_Y = Y[:, num_complete_minibatches * mini_batch_size:, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches 

def initialize_parameters(n_x, n_y, n_a):
    parameters = {}
    parameters['Wax'] = np.random.randn(n_a, n_x) * 0.1
    parameters['Waa'] = np.random.randn(n_a, n_a) * 0.1
    parameters['Wya'] = np.random.randn(n_y, n_a) * 0.1
    parameters['ba'] = np.zeros((n_a, 1))
    parameters['by'] = np.zeros((n_y, 1))
    
    return parameters
    
def rnn_cell_forward(xt, a_prev, parameters):
    Wax = parameters['Wax']
    Waa = parameters['Waa']
    Wya = parameters['Wya']
    ba = parameters['ba']
    by = parameters['by']
    
    a_next = np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba)
    
    cache = (a_next, a_prev, xt, parameters)
    
    return a_next, cache

def rnn_forward(x, a0, parameters):
    caches = []
    
    Wya = parameters['Wya']
    by = parameters['by']
    
    n_x, m, T_x = x.shape
    n_y, n_a = Wya.shape
    
    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, 1))
    
    a_next = a0
    for t in range(T_x):
        xt = x[:, :, t]
        a_next, cache = rnn_cell_forward(xt, a_next, parameters)
        a[:, :, t] = a_next
        caches.append(cache)
    #print(Wya.shape, a_next.shape, by.shape)
    z_pred = np.dot(Wya, a_next) + by    
    y_pred = relu(z_pred)
    caches = (caches, x, y_pred, z_pred)
    
    return a, y_pred, caches

def compute_cost(y_pred, y_train):    
    cost = ((y_pred - y_train) ** 2).mean(axis = None)
    
    return cost

def backpropagation(X, AL, Y, parameters, caches):
    T_x = X.shape[2]
    (caches, x, y_pred, z_pred) = caches
    (a_next, a_prev, x1, parameters) = caches[T_x - 1]
        
    dA = 2 * (AL - Y).mean(axis = 1)
    
    dZ = relu_backward(dA, z_pred)
        
    dWya = np.dot(dZ, a_next.T)
    dby = np.sum(dZ, keepdims = True)
    
    dA = np.dot(parameters['Wya'].T, dZ)
    
    return dA, dWya, dby

def rnn_cell_backward(da_next, cache):
    (a_next, a_prev, xt, parameters) = cache
    
    Wax = parameters['Wax']
    Waa = parameters['Waa']
    ba = parameters['ba']
    
    dz = np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba)
    dxt = np.dot(Wax.T, da_next * (1 - dz ** 2))
    dWax = np.dot(da_next * (1 - dz ** 2), xt.T)
    
    da_prev = np.dot(Waa.T, da_next * (1 - dz ** 2))
    dWaa = np.dot(da_next * (1 - dz ** 2), a_prev.T)
    
    dba = np.sum(da_next * (1 - dz ** 2), axis = 1, keepdims = True)
    
    gradients = {'dxt': dxt,
                 'da_prev': da_prev,
                 'dWax': dWax,
                 'dWaa': dWaa,
                 'dba': dba}
    
    return gradients   

def rnn_backward(Y, da, dWya, dby, caches):
    (caches, x, y_pred, z_pred) = caches
    (a1, a0, x1, parameters) = caches[0]
    
    n_a, m = da.shape
    n_x, m = x1.shape
    T_x = x.shape[2]
    
    dx = np.zeros((n_x, m, T_x))
    dWax = np.zeros((n_a, n_x))
    dWaa = np.zeros((n_a, n_a))
    dba = np.zeros((n_a, 1))
    da0 = np.zeros((n_a, m))
    da_prevt = da0
    
    for t in reversed(range(T_x)):
        if t == T_x - 1:
            gradients = rnn_cell_backward(da + da_prevt, caches[t])
        else:
            gradients = rnn_cell_backward(da_prevt, caches[t])
        dxt, da_prevt, dWaxt, dWaat, dbat = gradients['dxt'], gradients['da_prev'], gradients['dWax'], gradients['dWaa'], gradients['dba']
        dx[:, :, t] = dxt
        dWax += dWaxt
        dWaa += dWaat
        dba += dbat
    
    dWax /= m
    dWaa /= m
    dba /= m
    dWya /= m
    dby /= m
    
    da0 = da_prevt
    
    gradients = {'dx': dx,
                 'da0': da0,
                 'dWax': dWax,
                 'dWaa': dWaa,
                 'dWya': dWya,
                 'dba': dba,
                 'dby': dby}
    
    return gradients

def update_parameters(parameters, gradients, learning_rate):
    parameters['Wax'] = parameters['Wax'] - np.dot(learning_rate, gradients['dWax'])
    parameters['Waa'] = parameters['Waa'] - np.dot(learning_rate, gradients['dWaa'])
    parameters['Wya'] = parameters['Wya'] - np.dot(learning_rate, gradients['dWya'])
    parameters['ba'] = parameters['ba'] - np.dot(learning_rate, gradients['dba'])
    parameters['by'] = parameters['by'] - np.dot(learning_rate, gradients['dby'])
    
    return parameters

def adam(parameters, gradients, learning_rate, v, s, t, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
    v_corrected = {}
    s_corrected = {}

    v['dWax'] = beta1 * v['dWax'] + (1 - beta1) * gradients['dWax']
    v['dWaa'] = beta1 * v['dWaa'] + (1 - beta1) * gradients['dWaa']
    v['dWya'] = beta1 * v['dWya'] + (1 - beta1) * gradients['dWya']
    v['dba'] = beta1 * v['dba'] + (1 - beta1) * gradients['dba']
    v['dby'] = beta1 * v['dby'] + (1 - beta1) * gradients['dby']

    v_corrected['dWax'] = v['dWax'] / (1 - beta1 ** t)
    v_corrected['dWaa'] = v['dWaa'] / (1 - beta1 ** t)
    v_corrected['dWya'] = v['dWya'] / (1 - beta1 ** t)
    v_corrected['dba'] = v['dba'] / (1 - beta1 ** t)
    v_corrected['dby'] = v['dby'] / (1 - beta1 ** t)
    
    s['dWax'] = beta2 * s['dWax'] + (1 - beta2) * (gradients['dWax'] ** 2)
    s['dWaa'] = beta2 * s['dWaa'] + (1 - beta2) * (gradients['dWaa'] ** 2)
    s['dWya'] = beta2 * s['dWya'] + (1 - beta2) * (gradients['dWya'] ** 2)
    s['dba'] = beta2 * s['dba'] + (1 - beta2) * (gradients['dba'] ** 2)
    s['dby'] = beta2 * s['dby'] + (1 - beta2) * (gradients['dby'] ** 2)
    
    s_corrected['dWax'] = s['dWax'] / (1 - beta2 ** t)
    s_corrected['dWaa'] = s['dWaa'] / (1 - beta2 ** t)
    s_corrected['dWya'] = s['dWya'] / (1 - beta2 ** t)
    s_corrected['dba'] = s['dba'] / (1 - beta2 ** t)
    s_corrected['dby'] = s['dby'] / (1 - beta2 ** t)

    parameters['Wax'] = parameters['Wax'] - np.dot(learning_rate, v_corrected['dWax'] / (np.sqrt(s_corrected['dWax']) + epsilon))
    parameters['Waa'] = parameters['Waa'] - np.dot(learning_rate, v_corrected['dWaa'] / (np.sqrt(s_corrected['dWaa']) + epsilon))
    parameters['Wya'] = parameters['Wya'] - np.dot(learning_rate, v_corrected['dWya'] / (np.sqrt(s_corrected['dWya']) + epsilon))
    parameters['ba'] = parameters['ba'] - np.dot(learning_rate, v_corrected['dba'] / (np.sqrt(s_corrected['dba']) + epsilon))
    parameters['by'] = parameters['by'] - np.dot(learning_rate, v_corrected['dby'] / (np.sqrt(s_corrected['dby']) + epsilon))
    
    return parameters, v, s

def model(X, Y, dims, learning_rate, num_epochs):
    n_x, n_y, n_a = dims
    parameters = initialize_parameters(n_x, n_y, n_a)
    v, s = initialize_adam(parameters)
    minibatch_size = 64
    m = minibatch_size
    a0 = np.zeros((n_a, m))

    
    for epoch in range(1, num_epochs + 1):
        epoch_cost = 0.
        num_minibatches = m // minibatch_size
        minibatches = random_mini_batches(X, Y)
        minibatches = minibatches[:-1]
        t = 0
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            a, y_pred, caches = rnn_forward(minibatch_X, a0, parameters)
            cost = compute_cost(y_pred, minibatch_Y)
            da, dWya, dby = backpropagation(minibatch_X, y_pred, minibatch_Y, parameters,  caches)
            gradients = rnn_backward(minibatch_Y, da, dWya, dby, caches)
            t += 1
            parameters, v, s = adam(parameters, gradients, learning_rate, v, s, t)
            epoch_cost += cost
        print('Cost is: ' + str(epoch_cost / num_minibatches) + ' after iteration ' + str(epoch))
    
    return parameters

n_a = 1
dims = [X_train.shape[0], y_train.shape[0], n_a]
parameters = model(X_train, y_train, dims, learning_rate = 0.001, num_epochs = 1000)
print(parameters['Wya'].shape)
a0 = np.zeros((n_a, X_train.shape[1]))

_, predictions, _ = rnn_forward(X_train, a0, parameters)    

unscaled_predictions = predictions * (np.max(training_set) - np.min(training_set)) + np.min(training_set)
unscaled_predictions = unscaled_predictions.T

plt.plot(training_set, color = 'red', label = 'Real Google Stock Price')
plt.plot(unscaled_predictions, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.show()

test_dataset = pd.read_csv('dataset/Google_Stock_Price_Test.csv')
real_stock_prices = test_dataset.iloc[:, 1:2].values

dataset_total = pd.concat((train_dataset['Open'], test_dataset['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(test_dataset) - 60:].values
inputs = inputs.reshape(-1, +1)
scaled_inputs = (inputs - np.min(inputs)) / (np.max(inputs) - np.min(inputs))
X_test = []
for i in range(60, 80):
    X_test.append(scaled_inputs[i - 60: i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (1, X_test.shape[0], X_test.shape[1]))
a0 = np.zeros((n_a, X_test.shape[1]))
_, test_predictions, _ = rnn_forward(X_test, a0, parameters)
unscaled_test_predictions = test_predictions * (np.max(inputs) - np.min(inputs)) + np.min(inputs)
unscaled_test_predictions = unscaled_test_predictions.T

plt.plot(real_stock_prices, color = 'red', label = 'Real Google Stock Price')
plt.plot(unscaled_test_predictions, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()