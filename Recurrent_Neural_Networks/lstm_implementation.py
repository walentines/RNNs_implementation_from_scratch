import numpy as np
import utils
import pandas as pd
import matplotlib.pyplot as plt
import math

train_dataset = pd.read_csv('dataset/Google_Stock_Price_Train.csv')
training_set = train_dataset.iloc[:, 1:2].values
training_set_scaled = (training_set - np.min(training_set)) / (np.max(training_set) - np.min(training_set))

X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i - 60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (1, X_train.shape[0], X_train.shape[1]))
y_train = np.reshape(y_train, (1, y_train.shape[0], 1))

def initialize_parameters(n_x, n_y, n_a):
    Wf = np.random.randn(n_a, n_a + n_x) * 0.1
    bf = np.zeros((n_a, 1))
    Wi = np.random.randn(n_a, n_a + n_x) * 0.1
    bi = np.zeros((n_a, 1))
    Wc = np.random.randn(n_a, n_a + n_x) * 0.1
    bc = np.zeros((n_a, 1))
    Wo = np.random.randn(n_a, n_a + n_x) * 0.1
    bo = np.zeros((n_a, 1))
    Wy = np.random.randn(n_y, n_a) * 0.1
    by = np.zeros((n_y, 1))
    
    parameters = {'Wf': Wf,
                  'bf': bf,
                  'Wi': Wi,
                  'bi': bi,
                  'Wc': Wc,
                  'bc': bc,
                  'Wo': Wo,
                  'bo': bo,
                  'Wy': Wy,
                  'by': by}
    
    return parameters

def initialize_adam(parameters):
    v = {}
    s = {}
    
    v['dWf'] = np.zeros((parameters['Wf'].shape[0], parameters['Wf'].shape[1]))
    v['dbf'] = np.zeros((parameters['bf'].shape[0], parameters['bf'].shape[1]))
    v['dWi'] = np.zeros((parameters['Wi'].shape[0], parameters['Wi'].shape[1]))
    v['dbi'] = np.zeros((parameters['bi'].shape[0], parameters['bi'].shape[1]))
    v['dWc'] = np.zeros((parameters['Wc'].shape[0], parameters['Wc'].shape[1]))
    v['dbc'] = np.zeros((parameters['bc'].shape[0], parameters['bc'].shape[1]))
    v['dWo'] = np.zeros((parameters['Wo'].shape[0], parameters['Wo'].shape[1]))
    v['dbo'] = np.zeros((parameters['bo'].shape[0], parameters['bo'].shape[1]))
    v['dWy'] = np.zeros((parameters['Wy'].shape[0], parameters['Wy'].shape[1]))
    v['dby'] = np.zeros((parameters['by'].shape[0], parameters['by'].shape[1]))
    
    s['dWf'] = np.zeros((parameters['Wf'].shape[0], parameters['Wf'].shape[1]))
    s['dbf'] = np.zeros((parameters['bf'].shape[0], parameters['bf'].shape[1]))
    s['dWi'] = np.zeros((parameters['Wi'].shape[0], parameters['Wi'].shape[1]))
    s['dbi'] = np.zeros((parameters['bi'].shape[0], parameters['bi'].shape[1]))
    s['dWc'] = np.zeros((parameters['Wc'].shape[0], parameters['Wc'].shape[1]))
    s['dbc'] = np.zeros((parameters['bc'].shape[0], parameters['bc'].shape[1]))
    s['dWo'] = np.zeros((parameters['Wo'].shape[0], parameters['Wo'].shape[1]))
    s['dbo'] = np.zeros((parameters['bo'].shape[0], parameters['bo'].shape[1]))
    s['dWy'] = np.zeros((parameters['Wy'].shape[0], parameters['Wy'].shape[1]))
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

def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    Wf = parameters['Wf']
    bf = parameters['bf']
    Wi = parameters['Wi']
    bi = parameters['bi']
    Wc = parameters['Wc']
    bc = parameters['bc']
    Wo = parameters['Wo']
    bo = parameters['bo']
    Wy = parameters['Wy']
    
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    concat = np.concatenate((a_prev, xt), axis = 0)
    
    ft = utils.sigmoid(np.dot(Wf, concat) + bf)
    it = utils.sigmoid(np.dot(Wi, concat) + bi)
    cct = utils.sigmoid(np.dot(Wc, concat) + bc)
    c_next = ft * c_prev + it * cct
    ot = utils.sigmoid(np.dot(Wo, concat) + bo)
    a_next = ot * np.tanh(c_next)
        
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)
    
    return a_next, c_next, cache

def lstm_forward(x, a0, parameters):
    caches = []
    
    Wy = parameters['Wy']
    by = parameters['by']
    
    n_x, m, T_x = x.shape
    n_y, n_a = Wy.shape
    
    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))
    
    a_next = a0
    c_next = np.zeros((n_a, m))
    for t in range(T_x):
        xt = x[:, :, t]
        a_next, c_next, cache = lstm_cell_forward(xt, a_next, c_next, parameters)
        a[:, :, t] = a_next
        c[:, :, t] = c_next
        caches.append(cache)
    z = np.dot(Wy, a_next) + by
    y = utils.relu(z)
    caches = (caches, x, z)
    
    return a, y, c, caches

def compute_cost(y_train, y_pred):
    cost = ((y_pred - y_train) ** 2).mean(axis = None)
    
    return cost

def backpropagation(X, y_pred, y_train, parameters, caches):
    T_x = X.shape[2]
    (caches, x, z) = caches
    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = caches[T_x - 1]
    
    da = 2 * (y_pred - y_train).mean(axis = 1)
    dZ = utils.relu_backward(da, z)
    
    dWy = np.dot(dZ, a_next.T)
    dby = np.sum(dZ, keepdims = True)
    
    da = np.dot(parameters['Wy'].T, dZ)
    
    return da, dWy, dby

def lstm_cell_backward(da_next, dc_next, cache):
    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache
    
    n_x, m = xt.shape
    n_a, m = a_next.shape
    
    dot = da_next * np.tanh(c_next) * ot * (1 - ot)
    dcct = (dc_next * it + ot * (1 - np.tanh(c_next) ** 2) * it * da_next) * (1 - cct ** 2)
    dit = (dc_next * cct + ot * (1 - np.tanh(c_next) ** 2) * cct * da_next) * it * (1 - it)
    dft = (dc_next * c_prev + ot * (1 - np.tanh(c_next) ** 2) * c_prev * da_next) * ft * (1 - ft)
    
    dWf = np.dot(dft, np.concatenate((a_prev, xt), axis = 0).T)
    dWi = np.dot(dit, np.concatenate((a_prev, xt), axis = 0).T)
    dWc = np.dot(dcct, np.concatenate((a_prev, xt), axis = 0).T)
    dWo = np.dot(dot, np.concatenate((a_prev, xt), axis = 0).T)
    dbf = np.sum(dft, axis = 1, keepdims = True)
    dbi = np.sum(dit, axis = 1, keepdims = True)
    dbc = np.sum(dcct, axis = 1, keepdims = True)
    dbo = np.sum(dot, axis = 1, keepdims = True)
    
    da_prev = np.dot(parameters['Wf'][:, :n_a].T, dft) + np.dot(parameters['Wi'][:, :n_a].T, dit) + np.dot(parameters['Wc'][:, :n_a].T, dcct) + np.dot(parameters['Wo'][:, :n_a].T, dot)
    dc_prev = dc_next * ft + ot * (1 - np.tanh(c_next) ** 2) * ft * da_next
    dxt = np.dot(parameters['Wf'][:, n_a:].T, dft) + np.dot(parameters['Wi'][:, n_a:].T, dit) + np.dot(parameters['Wc'][:, n_a:].T, dcct) + np.dot(parameters['Wo'][:, n_a:].T, dot)
    
    gradients = {'dxt': dxt,
                 'da_prevt': da_prev,
                 'dc_prevt': dc_prev,
                 'dWf': dWf,
                 'dbf': dbf,
                 'dWi': dWi,
                 'dbi': dbi,
                 'dWc': dWc,
                 'dbc': dbc,
                 'dWo': dWo,
                 'dbo': dbo}
    
    return gradients

def lstm_backward(da, caches):
    (caches, x, z) = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]
    
    n_a, m = da.shape
    n_x, m = x1.shape
    T_x = x.shape[2]
    
    dx = np.zeros((n_x, m, T_x))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))
    dc_prevt = np.zeros((n_a, m))
    dWf = np.zeros((n_a, n_a + n_x))
    dWi = np.zeros((n_a, n_a + n_x))
    dWc = np.zeros((n_a, n_a + n_x))
    dWo = np.zeros((n_a, n_a + n_x))
    dbf = np.zeros((n_a, 1))
    dbi = np.zeros((n_a, 1))
    dbc = np.zeros((n_a, 1))
    dbo = np.zeros((n_a, 1))
    
    for t in reversed(range(T_x)):
        if t == T_x - 1:
            gradients = lstm_cell_backward(da + da_prevt, dc_prevt, caches[t])
        else:
            gradients = lstm_cell_backward(da_prevt, dc_prevt, caches[t])
        da_prevt = gradients['da_prevt']
        dc_prevt = gradients['dc_prevt']
        dx[:, :, t] = gradients['dxt']
        dWf += gradients['dWf']
        dWi += gradients['dWi']
        dWc += gradients['dWc']
        dWo += gradients['dWo']
        dbf += gradients['dbf']
        dbi += gradients['dbi']
        dbc += gradients['dbc']
        dbo += gradients['dbo']
    
    dWf /= m
    dWi /= m
    dWc /= m
    dWo /= m
    dbf /= m
    dbi /= m
    dbc /= m
    dbo /= m
    
    da0 = da_prevt
    
    gradients = {'dx': dx,
                 'da0': da0,
                 'dWf': dWf,
                 'dbf': dbf,
                 'dWi': dWi,
                 'dbi': dbi,
                 'dWc': dWc,
                 'dbc': dbc,
                 'dWo': dWo,
                 'dbo': dbo}
    
    return gradients

def adam(parameters, gradients, learning_rate, v, s, t, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
    v_corrected = {}
    s_corrected = {}
    
    v['dWf'] = beta1 * v['dWf'] + (1 - beta1) * gradients['dWf']
    v['dbf'] = beta1 * v['dbf'] + (1 - beta1) * gradients['dbf']
    v['dWi'] = beta1 * v['dWi'] + (1 - beta1) * gradients['dWi']
    v['dbi'] = beta1 * v['dbi'] + (1 - beta1) * gradients['dbi']
    v['dWc'] = beta1 * v['dWc'] + (1 - beta1) * gradients['dWc']
    v['dbc'] = beta1 * v['dbc'] + (1 - beta1) * gradients['dbc']
    v['dWo'] = beta1 * v['dWo'] + (1 - beta1) * gradients['dWo']
    v['dbo'] = beta1 * v['dbo'] + (1 - beta1) * gradients['dbo']
    
    v_corrected['dWf'] = v['dWf'] / (1 - beta1 ** t)
    v_corrected['dbf'] = v['dbf'] / (1 - beta1 ** t)
    v_corrected['dWi'] = v['dWi'] / (1 - beta1 ** t)
    v_corrected['dbi'] = v['dbi'] / (1 - beta1 ** t)
    v_corrected['dWc'] = v['dWc'] / (1 - beta1 ** t)
    v_corrected['dbc'] = v['dbc'] / (1 - beta1 ** t)
    v_corrected['dWo'] = v['dWo'] / (1 - beta1 ** t)
    v_corrected['dbo'] = v['dbo'] / (1 - beta1 ** t)
    
    s['dWf'] = beta2 * s['dWf'] + (1 - beta2) * (gradients['dWf'] ** 2)
    s['dbf'] = beta2 * s['dbf'] + (1 - beta2) * (gradients['dbf'] ** 2)
    s['dWi'] = beta2 * s['dWi'] + (1 - beta2) * (gradients['dWi'] ** 2)
    s['dbi'] = beta2 * s['dbi'] + (1 - beta2) * (gradients['dbi'] ** 2)
    s['dWc'] = beta2 * s['dWc'] + (1 - beta2) * (gradients['dWc'] ** 2)
    s['dbc'] = beta2 * s['dbc'] + (1 - beta2) * (gradients['dbc'] ** 2)
    s['dWo'] = beta2 * s['dWo'] + (1 - beta2) * (gradients['dWo'] ** 2)
    s['dbo'] = beta2 * s['dbo'] + (1 - beta2) * (gradients['dbo'] ** 2)
    
    s_corrected['dWf'] = s['dWf'] / (1 - beta2 ** t)
    s_corrected['dbf'] = s['dbf'] / (1 - beta2 ** t)
    s_corrected['dWi'] = s['dWi'] / (1 - beta2 ** t)
    s_corrected['dbi'] = s['dbi'] / (1 - beta2 ** t)
    s_corrected['dWc'] = s['dWc'] / (1 - beta2 ** t)
    s_corrected['dbc'] = s['dbc'] / (1 - beta2 ** t)
    s_corrected['dWo'] = s['dWo'] / (1 - beta2 ** t)
    s_corrected['dbo'] = s['dbo'] / (1 - beta2 ** t)
    
    parameters['Wf'] = parameters['Wf'] - np.dot(learning_rate, v_corrected['dWf'] / (np.sqrt(s_corrected['dWf']) + epsilon))
    parameters['bf'] = parameters['bf'] - np.dot(learning_rate, v_corrected['dbf'] / (np.sqrt(s_corrected['dbf']) + epsilon))
    parameters['Wi'] = parameters['Wi'] - np.dot(learning_rate, v_corrected['dWi'] / (np.sqrt(s_corrected['dWi']) + epsilon))
    parameters['bi'] = parameters['bi'] - np.dot(learning_rate, v_corrected['dbi'] / (np.sqrt(s_corrected['dbi']) + epsilon))
    parameters['Wc'] = parameters['Wc'] - np.dot(learning_rate, v_corrected['dWc'] / (np.sqrt(s_corrected['dWc']) + epsilon))
    parameters['bc'] = parameters['bc'] - np.dot(learning_rate, v_corrected['dbc'] / (np.sqrt(s_corrected['dbc']) + epsilon))
    parameters['Wo'] = parameters['Wo'] - np.dot(learning_rate, v_corrected['dWo'] / (np.sqrt(s_corrected['dWo']) + epsilon))
    parameters['bo'] = parameters['bo'] - np.dot(learning_rate, v_corrected['dbo'] / (np.sqrt(s_corrected['dbo']) + epsilon))
    
    return parameters, v, s

def model(X, Y, dims, learning_rate, num_epochs):
    m = 64
    n_x, n_y, n_a = dims
    parameters = initialize_parameters(n_x, n_y, n_a)
    v, s = initialize_adam(parameters)
    a0 = np.zeros((n_a, m))
    batch_size = 64
    
    for epoch in range(1, num_epochs + 1):
        epoch_cost = 0.
        minibatches = random_mini_batches(X, Y)
        minibatches = minibatches[:-1]
        num_minibatches = math.floor(m / batch_size)
        t = 0
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            a, y, c, caches = lstm_forward(minibatch_X, a0, parameters)
            cost = compute_cost(minibatch_Y, y)
            da, dWy, dby = backpropagation(minibatch_X, y, minibatch_Y, parameters, caches)
            gradients = lstm_backward(da, caches)
            t += 1
            parameters, v, s = adam(parameters, gradients, learning_rate, v, s, t)
            epoch_cost += cost
        print('Cost after epoch: %i %f' % (epoch, epoch_cost / num_minibatches))
    
    return parameters

n_a = 1
m = X_train.shape[1]
dims = [X_train.shape[0], y_train.shape[0], n_a]
parameters = model(X_train, y_train, dims, learning_rate = 0.01, num_epochs = 5000)

a0 = np.zeros((n_a, m))
_, predictions, _, _ = lstm_forward(X_train, a0, parameters)

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
_, test_predictions, _, _ = lstm_forward(X_test, a0, parameters)
unscaled_test_predictions = test_predictions * (np.max(inputs) - np.min(inputs)) + np.min(inputs)
unscaled_test_predictions = unscaled_test_predictions.T

plt.plot(real_stock_prices, color = 'red', label = 'Real Google Stock Price')
plt.plot(unscaled_test_predictions, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()