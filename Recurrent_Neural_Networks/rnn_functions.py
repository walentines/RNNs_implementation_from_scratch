import numpy as np
from utils import *

# RNN
# ------------------------------------------------------------------------------------------------

def rnn_cell_forward(xt, a_prev, parameters):
    Wax = parameters['Wax']
    Waa = parameters['Waa']
    Wya = parameters['Wya']
    ba = parameters['ba']
    by = parameters['by']
    
    a_next = np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba)
    yt_pred = softmax(np.dot(Wya, a_next) + by)
    
    cache = (a_next, a_prev, xt, parameters)
    
    return a_next, yt_pred, cache

def rnn_forward(x, a0, parameters):
    caches = []
    
    n_x, m, T_x = x.shape
    n_y, n_a = parameters['Wya'].shape
    
    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))
    
    a_next = a0
    for t in range(T_x):
        xt = x[:, :, t]
        a_next, yt_pred, cache = rnn_cell_forward(xt, a_next, parameters)
        a[:, :, t] = a_next
        y_pred[:, :, t] = yt_pred
        caches.append(cache)
        
    caches = (caches, x)
    
    return a, y_pred, caches

def rnn_cell_backward(da_next, cache):
    (a_next, a_prev, xt, parameters) = cache
    
    Wax = parameters['Wax']
    Waa = parameters['Waa']
    Wya = parameters['Wya']
    ba = parameters['ba']
    by = parameters['by']
    
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

def rnn_backward(da, caches):
    (caches, x) = caches
    (a1, a0, x1, parameters) = caches[0]
    
    n_a, m, T_x = da.shape
    n_x, m = x1.shape
    
    dx = np.zeros((n_x, m, T_x))
    dWax = np.zeros((n_a, n_x))
    dWaa = np.zeros((n_a, n_a))
    dba = np.zeros((n_a, 1))
    da0 = np.zeros((n_a, m))
    da_prevt = da0
    
    for t in reversed(range(T_x)):
        gradients = rnn_cell_backward(da[:, :, t] + da_prevt, caches[t])
        dxt, da_pervt, dWaxt, dWaat, dbat = gradients['dxt'], gradients['da_prev'], gradients['dWax'], gradients['dWaa'], gradients['dba']
        dx[:, :, t] = dxt
        dWax += dWaxt
        dWaa += dWaat
        dba += dbat
        
    da0 = da_prevt
    
    gradients = {'dx': dx,
                 'da0': da0,
                 'dWax': dWax,
                 'dWaa': dWaa,
                 'dba': dba}
    
    return gradients

# ----------------------------------------------------------------------------------------------


# LSTM
# ----------------------------------------------------------------------------------------------

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
    by = parameters['by']
    
    n_x, m = xt.shape
    n_y, n_a = Wy.shape
    
    concat = np.concatenate((a_prev, xt), axis = 0)
    
    ft = sigmoid(np.dot(Wf, concat) + bf)
    it = sigmoid(np.dot(Wi, concat) + bi)
    cct = sigmoid(np.dot(Wc, concat) + bc)
    c_next = ft * c_prev + it * cct
    ot = sigmoid(np.dot(Wo, concat) + bo)
    a_next = ot * np.tanh(c_next)
    
    yt_pred = softmax(np.dot(Wy, a_next) + by)
    
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)
    
    return a_next, c_next, yt_pred, cache

def lstm_forward(x, a0, parameters):
    caches = []
    
    Wy = parameters['Wy']
    
    n_x, m, T_x = x.shape
    n_y, n_a = Wy.shape
    
    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))
    
    a_next = a0
    c_next = np.zeros((n_a, m))
    for t in range(T_x):
        xt = x[:, :, t]
        a_next, c_next, yt, cache = lstm_cell_forward(xt, a_next, c_next, parameters)
        a[:, :, t] = a_next
        c[:, :, t] = c_next
        y[:, :, t] = yt
        caches.append(cache)
        
    caches = (caches, x)
    
    return a, y, c, caches

def lstm_cell_backward(da_next, dc_next, cache):
    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache
    
    n_x, m = xt.shape
    n_a, m = a_next.shape
    
    dot = da_next * np.tanh(c_next) * ot * (1 - ot)
    dcct = (dc_next * it + ot * (1 - np.tanh(c_next) ** 2) * it * da_next) * (1 - cct * 2)
    dit = (dc_next * cct + ot * (1 - np.tanh(c_next) ** 2) * cct * da_next) * it * (1 - it)
    dft = (dc_next * c_prev + ot * (1 - np.tamh(c_next) ** 2) * c_prev * da_next) * ft * (1 - ft)
    
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
                 'da_prev': da_prev,
                 'dc_prev': dc_prev,
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
    (caches, x) = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]
    
    n_a, m, T_x = da.shape
    n_x, m = x1.shape
    
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
        gradients = lstm_cell_backward(da[:, :, t] + da_prevt, dc_prevt, caches[t])
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