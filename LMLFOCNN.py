import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import random
import time

def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

def initialize_parameters(layers):
    new_layers = []
    for i, layer in enumerate(layers):
        mode = layer['mode']
        if mode == 'pool':
            new_layers.append(layer)
            continue
        elif mode == 'fc':
            n_now = layer['n_now']
            n_prev = layer['n_prev']
            layer['W']=(np.random.rand(n_now, n_prev) - 0.5) * 0.2
            layer['b']=(np.random.rand(n_now,1) - 0.5) * 0.2
            layer['W_k']=[np.zeros_like(layer['W'])]
            layer['b_k']=[np.zeros_like(layer['b'])]
            layer['dW']=np.zeros_like(layer['W'])
            layer['db']=np.zeros_like(layer['b'])
            layer['IO_dW_k-1']=(np.random.rand(n_now, n_prev) - 0.5) * 0.2
            layer['IO_db_k-1']=(np.random.rand(n_now,1) - 0.5) * 0.2
        elif mode == 'conv':
            f = layer['f']
            n_C = layer['n_C']
            n_C_prev = layer['n_C_prev']
            layer['W']=(np.random.rand(f, f, n_C_prev, n_C) - 0.5) * 0.2
            layer['b']=(np.random.rand(1, 1, 1, n_C) - 0.5) * 0.2
            layer['W_k']=[np.zeros_like(layer['W'])]
            layer['b_k']=[np.zeros_like(layer['b'])]
            layer['dW']=np.zeros_like(layer['W'])
            layer['db']=np.zeros_like(layer['b'])
            layer['IO_dW_k-1']=(np.random.rand(f, f, n_C_prev, n_C) - 0.5) * 0.2
            layer['IO_db_k-1']=(np.random.rand(1, 1, 1, n_C) - 0.5) * 0.2
        else:
            print('Wrong layer in [{}]'.format(i))
        new_layers.append(layer)
            
    return new_layers

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ

def relu(Z):
    A = np.maximum(0,Z)
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z < 0] = 0
    return dZ

def softmax(Z):
    n, m = Z.shape
    A = np.exp(Z)
    A_sum = np.sum(A, axis = 0)
    A_sum = A_sum.reshape(-1, m)
    A = A / A_sum
    cache = Z
    return A, cache

def softmax_backward(A, Y):
    m = A.shape[1]
    dZ = (A - Y) / np.float(m)
    return dZ

def linear_activation_forward(A_prev, layer, activation='relu'):
    W = layer['W']
    b = layer['b']
    if activation=='sigmoid':
        Z, linear_cache=np.dot(W, A_prev)+b, (A_prev, W, b)
        A, activation_cache=sigmoid(Z)
    elif activation=='relu':
        Z, linear_cache=np.dot(W, A_prev)+b, (A_prev, W, b)
        A, activation_cache=relu(Z)
    else:
        Z = np.dot(W, A_prev)+b
        A = Z
    return A, Z

def IO_linear_activation_backward(dA, layer, activation):
    A_prev = layer['A_prev']
    W = layer['W']
    b = layer['b']
    Z = layer['Z']
    if activation=='relu':
        dZ=relu_backward(dA, Z)
    elif activation=='sigmoid':
        dZ=sigmoid_backward(dA, Z)
    else:
        dZ = dA 
    n, m = dA.shape
    dA_prev=np.dot(W.T, dZ)
    dW = np.dot(dZ, A_prev.T)
    db = np.sum(dZ, axis = 1).reshape(n,1)
    
    return dA_prev, dW, db

def FO_linear_activation_backward(dA, layer, activation, delta=1e-2):
    A_prev = layer['A_prev']
    W = layer['W']
    b = layer['b']
    c_W = layer['W_k'][-1]
    c_b = layer['b_k'][-1]
    Z = layer['Z']
    alpha = layer['alpha']
    if activation=='relu':
        dZ=relu_backward(dA, Z)
    elif activation=='sigmoid':
        dZ=sigmoid_backward(dA, Z)
    else:
        dZ = dA
    n, m = dA.shape
    dA_prev = np.dot(W.T, dZ)
    IO_dW = np.dot(dZ, A_prev.T)
    IO_db = np.sum(dZ, axis = 1).reshape(n,1)
    dW = layer['IO_dW_k-1'] * np.power(np.abs(W-c_W)+delta, 1-alpha) / math.gamma(2-alpha)  
    db = layer['IO_db_k-1'] * np.power(np.abs(b-c_b)+delta, 1-alpha) / math.gamma(2-alpha)

    return dA_prev, dW, db, IO_dW, IO_db

def zero_pad(X, pad, value = 0):
    X_pad = np.pad(X, ((0, 0),(pad, pad),(pad, pad),(0, 0)), 'constant', constant_values=value)
    return X_pad

def conv_forward(A_prev, layer):
    W = layer['W']
    b = layer['b']
    stride = layer['s']
    pad = layer['p']
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    n_H = 1 + int((n_H_prev + 2 * pad - f) / stride)
    n_W = 1 + int((n_W_prev + 2 * pad - f) / stride)
    Z = np.zeros((m, n_H, n_W, n_C))
    if pad > 0:
        A_prev_pad = zero_pad(A_prev, pad)
    else:
        A_prev_pad = A_prev
    
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    Z[i, h, w, c] = np.sum(np.multiply(a_slice_prev, W[:, :, :, c])) + b[0, 0, 0, c]
    return Z

def IO_conv_backward(dZ, layer):
    A_prev = layer['A_prev']
    W = layer['W']
    b = layer['b']
    Z = layer['Z']
    alpha = layer['alpha']
    stride = layer['s']
    pad = layer['p']
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    (m, n_H, n_W, n_C) = dZ.shape
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))                           
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))
    if pad > 0:
        A_prev_pad = zero_pad(A_prev, pad)
        dA_prev_pad = zero_pad(dA_prev, pad)
    else:
        A_prev_pad = A_prev
        dA_prev_pad = np.copy(dA_prev)
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    a_slice = A_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :]
                    dA_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += dZ[i, h, w, c] * a_slice
                    db[:,:,:,c] += dZ[i, h, w, c]
        if pad == 0:
            dA_prev[i, :, :, :] = dA_prev_pad[i, :, :, :]
        else:
            dA_prev[i, :, :, :] = dA_prev_pad[i, pad:-pad, pad:-pad, :]
    return dA_prev, dW, db

def FO_conv_backward(dZ, layer, delta=1e-2):
    A_prev = layer['A_prev']
    W = layer['W']
    b = layer['b']
    c_W = layer['W_k'][-1]
    c_b = layer['b_k'][-1]
    Z = layer['Z']
    alpha = layer['alpha']
    stride = layer['s']
    pad = layer['p']
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    (m, n_H, n_W, n_C) = dZ.shape
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))                           
    IO_dW = np.zeros((f, f, n_C_prev, n_C))
    IO_db = np.zeros((1, 1, 1, n_C))
    if pad > 0:
        A_prev_pad = zero_pad(A_prev, pad)
        dA_prev_pad = zero_pad(dA_prev, pad)
    else:
        A_prev_pad = A_prev
        dA_prev_pad = np.copy(dA_prev)

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    a_slice = A_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :]
                    dA_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    IO_dW[:,:,:,c] += dZ[i, h, w, c] * a_slice
                    IO_db[:,:,:,c] += dZ[i, h, w, c]
        if pad == 0:
            dA_prev[i, :, :, :] = dA_prev_pad[i, :, :, :]
        else:
            dA_prev[i, :, :, :] = dA_prev_pad[i, pad:-pad, pad:-pad, :]
    dW = layer['IO_dW_k-1'] * np.power(np.abs(W-c_W)+delta, 1-alpha) / math.gamma(2-alpha)
    db = layer['IO_db_k-1'] * np.power(np.abs(b-c_b)+delta, 1-alpha) / math.gamma(2-alpha)
    return dA_prev, dW, db, IO_dW, IO_db

def pool_forward(A_prev, layer, mode = "max"):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = layer["f"]
    stride = layer["s"]
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    A = np.zeros((m, n_H, n_W, n_C))              
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range (n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)

    return A

def distribute_value(dz, shape):
    (n_H, n_W) = shape
    average = dz / (n_H * n_W)
    a = np.ones(shape) * average 
    return a

def create_mask_from_window(x):
    mask = (x == np.max(x))
    return mask

def pool_backward(dA, layer, mode = "max"):
    A_prev = layer['A_prev']
    stride = layer['s']
    f = layer['f']
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    dA_prev = np.zeros_like(A_prev)
    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    if mode == "max":
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        mask = create_mask_from_window(a_prev_slice)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask * dA[i, h, w, c]
                        
                    elif mode == "average":
                        da = dA[i, h, w, c]
                        shape = (f, f)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)

    return dA_prev

def forward_propogation(X, layers):
    m = X.shape[0]
    layers[0]['A_prev'] = X
    Z = conv_forward(X, layers[0])
    layers[0]['Z'] = Z
    A, _ = relu(Z)
    layers[1]['A_prev'] = A
    A = pool_forward(A, layers[1], mode = "average")
    layers[2]['A_prev'] = A
    Z = conv_forward(A, layers[2])
    layers[2]['Z'] = Z
    A, _ = relu(Z)
    layers[3]['A_prev'] = A
    A = pool_forward(A, layers[3], mode = "average")
    layers[4]['A_prev'] = A
    Z = conv_forward(A, layers[4])
    layers[4]['Z'] = Z
    A, _ = relu(Z)
    layers[5]['A_prev'] = (A.reshape(m,-1)).T
    A, Z = linear_activation_forward((A.reshape(m,-1)).T, layers[5], activation='relu')
    layers[5]['Z'] = Z
    layers[6]['A_prev'] = A
    _, Z = linear_activation_forward(A, layers[6], activation='none')
    layers[6]['Z'] = Z
    AL, _ = softmax(Z)

    return AL, layers

def compute_cost(AL, Y):
    n, m = Y.shape
    cost = - np.sum(np.log(AL) * Y) / m
    cost=np.squeeze(cost)

    return cost

def IO_backward_propogation(AL, Y, layers):
    m = Y.shape[1]
    dZ = softmax_backward(AL, Y)
    dA_prev, dW, db = IO_linear_activation_backward(dZ, layers[6], 'none')
    layers[6]['dW'] = dW
    layers[6]['db'] = db
    dA_prev, dW, db = IO_linear_activation_backward(dA_prev, layers[5], 'relu')
    layers[5]['dW'] = dW
    layers[5]['db'] = db
    dA = (dA_prev.T).reshape(m,1,1,layers[4]['n_C'])
    dZ = relu_backward(dA, layers[4]['Z'])
    dA_prev, dW, db = IO_conv_backward(dZ, layers[4])
    layers[4]['dW'] = dW
    layers[4]['db'] = db
    dA_prev = pool_backward(dA_prev, layers[3], mode = "average")

    dZ = relu_backward(dA_prev, layers[2]['Z'])
    dA_prev, dW, db = IO_conv_backward(dZ, layers[2])
    layers[2]['dW'] = dW
    layers[2]['db'] = db
    dA_prev = pool_backward(dA_prev, layers[1], mode = "average")
    dZ = relu_backward(dA_prev, layers[0]['Z'])
    dA_prev, dW, db = IO_conv_backward(dZ, layers[0])
    layers[0]['dW'] = dW
    layers[0]['db'] = db
    return layers

def FO_backward_propogation(AL, Y, layers):
    m = Y.shape[1]
    dZ = softmax_backward(AL, Y)
    dA_prev, dW, db, IO_dW, IO_db = FO_linear_activation_backward(dZ, layers[6], 'none')
    layers[6]['dW'] = dW
    layers[6]['db'] = db
    layers[6]['IO_dW_k-1'] = IO_dW
    layers[6]['IO_db_k-1'] = IO_db
    dA_prev, dW, db, IO_dW, IO_db = FO_linear_activation_backward(dA_prev, layers[5], 'relu')
    layers[5]['dW'] = dW
    layers[5]['db'] = db
    layers[5]['IO_dW_k-1'] = IO_dW
    layers[5]['IO_db_k-1'] = IO_db
    dA = (dA_prev.T).reshape(m,1,1,layers[4]['n_C'])
    dZ = relu_backward(dA, layers[4]['Z'])
    dA_prev, dW, db, IO_dW, IO_db = FO_conv_backward(dZ, layers[4])
    layers[4]['dW'] = dW
    layers[4]['db'] = db
    layers[4]['IO_dW_k-1'] = IO_dW
    layers[4]['IO_db_k-1'] = IO_db
    dA_prev = pool_backward(dA_prev, layers[3], mode = "average")
    dZ = relu_backward(dA_prev, layers[2]['Z'])
    dA_prev, dW, db, IO_dW, IO_db = FO_conv_backward(dZ, layers[2])
    layers[2]['dW'] = dW
    layers[2]['db'] = db
    layers[2]['IO_dW_k-1'] = IO_dW
    layers[2]['IO_db_k-1'] = IO_db
    dA_prev = pool_backward(dA_prev, layers[1], mode = "average")
    dZ = relu_backward(dA_prev, layers[0]['Z'])
    dA_prev, dW, db, IO_dW, IO_db = FO_conv_backward(dZ, layers[0])
    layers[0]['dW'] = dW
    layers[0]['db'] = db
    layers[0]['IO_dW_k-1'] = IO_dW
    layers[0]['IO_db_k-1'] = IO_db
    return layers

def IO_update_parameters(layers, learning_rate):
    num_layer = len(layers)
    for i in range(num_layer):
        mode = layers[i]['mode']
        if mode == 'pool':
            continue
        elif (mode == 'fc' or mode == 'conv'):
            layers[i]['W'] = layers[i]['W'] - learning_rate*layers[i]['dW']
            layers[i]['b'] = layers[i]['b'] - learning_rate*layers[i]['db']
        else:
            print('Wrong layer mode in [{}]'.format(i))
    return layers

def FO_update_parameters(layers, learning_rate):
    num_layer = len(layers)
    for i in range(num_layer):
        mode = layers[i]['mode']
        if mode == 'pool':
            continue
        elif (mode == 'fc' or mode == 'conv'):
            layers[i]['W_k'][0] = layers[i]['W']
            layers[i]['b_k'][0] = layers[i]['b']
            layers[i]['W'] = layers[i]['W'] - learning_rate*layers[i]['dW']
            layers[i]['b'] = layers[i]['b'] - learning_rate*layers[i]['db']
        else:
            print('Wrong layer mode in [{}]'.format(i))
    return layers

def predict(X_test, Y_test, layers):
    m = X_test.shape[0]
    n = Y_test.shape[1]
    pred = np.zeros((n,m))
    pred_count = np.zeros((n,m)) - 1
    AL, _ = forward_propogation(X_test, layers)
    max_index = np.argmax(AL, axis = 0)
    pred[max_index, list(range(m))] = 1
    pred_count[max_index, list(range(m))] = 1
    accuracy = np.float(np.sum(pred_count == Y_test.T)) / m
    return pred, accuracy

def compute_accuracy(AL, Y):
    n, m = Y.shape
    pred_count = np.zeros((n,m)) - 1
    max_index = np.argmax(AL, axis = 0)
    pred_count[max_index, list(range(m))] = 1
    accuracy = np.float(np.sum(pred_count == Y)) / m
    return accuracy

def IO_train_mini_batch(X_train, Y_train, X_test, Y_test, layers, num_exp=0, batch_size=10, num_epoch=1, learning_rate=0.01):
    num_sample=X_train.shape[0]
    num_iteration = num_sample // batch_size
    index = list(range(num_sample))
    accuracy_train_list = []
    accuracy_test_list = []
    for epoch in range(num_epoch):
        losses = []
        accuracies = []
        random.seed(num_exp*10+epoch)
        random.shuffle(index) # random sampling every epoch
        for iteration in range(num_iteration):
            batch_start = iteration * batch_size
            batch_end = (iteration + 1) * batch_size
            if batch_end > num_sample:
                batch_end = num_sample
            X_train_batch = X_train[index[batch_start:batch_end]]
            Y_train_batch = Y_train[index[batch_start:batch_end]]
            AL, layers = forward_propogation(X_train_batch, layers)
            loss = compute_cost(AL, Y_train_batch.T)
            accuracy = compute_accuracy(AL, Y_train_batch.T)
            layers = IO_backward_propogation(AL, Y_train_batch.T, layers)
            layers = IO_update_parameters(layers, learning_rate)
            losses.append(loss)
            accuracies.append(accuracy)
            if (iteration+1) % 600 == 0:
                np.save('data/FO_layers_{}_{}.npy'.format(epoch, iteration+1), layers)

        _, accuracy_test = predict(X_test, Y_test, layers)
        pred_train, _ = forward_propogation(X_train[:10000], layers)
        loss_train = compute_cost(pred_train, Y_train[:10000].T)
        accuracy_train = compute_accuracy(pred_train, Y_train[:10000].T)
        accuracy_train_list.append(accuracy_train)
        accuracy_test_list.append(accuracy_test)
        print('Epoch [{}] train_loss = {} train_accuracy = {}'.format(epoch, loss_train, accuracy_train))
        print('Epoch [{}] test_accuracy = {}'.format(epoch, accuracy_test))
    return layers, accuracy_train_list, accuracy_test_list

def FO_train_mini_batch(X_train, Y_train, X_test, Y_test, layers, num_exp=0, batch_size=10, num_epoch=1, learning_rate=0.01):
    num_sample=X_train.shape[0]
    num_iteration = num_sample // batch_size
    index = list(range(num_sample))
    accuracy_train_list = []
    accuracy_test_list = []
    for epoch in range(num_epoch):
        losses = []
        accuracies = []
        random.seed(num_exp*10+epoch)
        random.shuffle(index)
        for iteration in range(num_iteration):
            batch_start = iteration * batch_size
            batch_end = (iteration + 1) * batch_size
            if batch_end > num_sample:
                batch_end = num_sample
            X_train_batch = X_train[index[batch_start:batch_end]]
            Y_train_batch = Y_train[index[batch_start:batch_end]]
            AL, layers = forward_propogation(X_train_batch, layers)
            loss = compute_cost(AL, Y_train_batch.T)
            accuracy = compute_accuracy(AL, Y_train_batch.T)
            layers = FO_backward_propogation(AL, Y_train_batch.T, layers)
            layers = FO_update_parameters(layers, learning_rate)
            losses.append(loss)
            accuracies.append(accuracy)
            if (iteration+1) % 600 == 0:
                np.save('data/FO_layers_{}_{}.npy'.format(epoch, iteration+1), layers)
        _, accuracy_test = predict(X_test, Y_test, layers)
        pred_train, _ = forward_propogation(X_train[:10000], layers)
        loss_train = compute_cost(pred_train, Y_train[:10000].T)
        accuracy_train = compute_accuracy(pred_train, Y_train[:10000].T)
        accuracy_train_list.append(accuracy_train)
        accuracy_test_list.append(accuracy_test)
        print('Epoch [{}] train_loss = {} train_accuracy = {}'.format(epoch, loss_train, accuracy_train))
        print('Epoch [{}] test_accuracy = {}'.format(epoch, accuracy_test))
    
    return layers, accuracy_train_list, accuracy_test_list

def change_order(layers, order):
    layers[0]['alpha'] = order
    layers[2]['alpha'] = order
    layers[4]['alpha'] = order
    layers[5]['alpha'] = order
    layers[6]['alpha'] = order

    return layers

X_train, Y_train = load_mnist('data', 'train')
X_test, Y_test = load_mnist('data', 'test')
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)
X_train = X_train / 255.0
X_test = X_test / 255.0
(num_train,) = Y_train.shape
Y = np.zeros((num_train, 10))
for i in range(num_train):
    Y[i, Y_train[i]] = 1
Y_train = Y
(num_test,) = Y_test.shape
Y = np.zeros((num_test, 10))
for i in range(num_test):
    Y[i, Y_test[i]] = 1
Y_test = Y
layer1={}
layer1['mode'] = 'conv'
layer1['f'] = 5
layer1['n_C_prev'] = 1
layer1['n_C'] = 6
layer1['p'] = 2
layer1['s'] = 1
layer1['alpha'] = 1.0
layer2={}
layer2['mode'] = 'pool'
layer2['f'] = 2
layer2['s'] = 2
layer3={}
layer3['mode'] = 'conv'
layer3['f'] = 5
layer3['n_C_prev'] = 6
layer3['n_C'] = 16
layer3['p'] = 0
layer3['s'] = 1
layer3['alpha'] = 1.0
layer4={}
layer4['mode'] = 'pool'
layer4['f'] = 2
layer4['s'] = 2
layer5={}
layer5['mode'] = 'conv'
layer5['f'] = 5
layer5['n_C_prev'] = 16
layer5['n_C'] = 120
layer5['p'] = 0
layer5['s'] = 1
layer5['alpha'] = 1.0
layer6={}
layer6['mode'] = 'fc'
layer6['n_now'] = 84
layer6['n_prev'] = 120
layer6['alpha'] = 1.0
layer7={}
layer7['mode'] = 'fc'
layer7['n_now'] = 10
layer7['n_prev'] = 84
layer7['alpha'] = 1.0
construct_layers = [layer1, layer2, layer3, layer4, layer5, layer6, layer7]
num_experiments = 1
for index in range(num_experiments):
    initial_layers_path = 'data/initial_layers_{}.npy'.format(index+1)
    if os.path.exists(initial_layers_path):
        initial_layers = np.load(initial_layers_path)
    else:
        initial_layers = initialize_parameters(construct_layers)
        np.save(initial_layers_path, initial_layers)
    IO_initial_layers = copy.deepcopy(initial_layers)
    FO19_initial_layers = copy.deepcopy(initial_layers)
    FO19_initial_layers = change_order(FO19_initial_layers, 1.9)
    FO18_initial_layers = copy.deepcopy(initial_layers)
    FO18_initial_layers = change_order(FO18_initial_layers, 1.8)
    FO17_initial_layers = copy.deepcopy(initial_layers)
    FO17_initial_layers = change_order(FO17_initial_layers, 1.7)
    FO16_initial_layers = copy.deepcopy(initial_layers)
    FO16_initial_layers = change_order(FO16_initial_layers, 1.6)
    FO15_initial_layers = copy.deepcopy(initial_layers)
    FO15_initial_layers = change_order(FO15_initial_layers, 1.5)
    FO14_initial_layers = copy.deepcopy(initial_layers)
    FO14_initial_layers = change_order(FO14_initial_layers, 1.4)
    FO13_initial_layers = copy.deepcopy(initial_layers)
    FO13_initial_layers = change_order(FO13_initial_layers, 1.3)
    FO12_initial_layers = copy.deepcopy(initial_layers)
    FO12_initial_layers = change_order(FO12_initial_layers, 1.2)
    FO11_initial_layers = copy.deepcopy(initial_layers)
    FO11_initial_layers = change_order(FO11_initial_layers, 1.1)
    FO9_initial_layers = copy.deepcopy(initial_layers)
    FO9_initial_layers = change_order(FO9_initial_layers, 0.9)
    FO8_initial_layers = copy.deepcopy(initial_layers)
    FO8_initial_layers = change_order(FO8_initial_layers, 0.8)
    FO7_initial_layers = copy.deepcopy(initial_layers)
    FO7_initial_layers = change_order(FO7_initial_layers, 0.7)
    FO6_initial_layers = copy.deepcopy(initial_layers)
    FO6_initial_layers = change_order(FO6_initial_layers, 0.6)
    FO5_initial_layers = copy.deepcopy(initial_layers)
    FO5_initial_layers = change_order(FO5_initial_layers, 0.5)
    FO4_initial_layers = copy.deepcopy(initial_layers)
    FO4_initial_layers = change_order(FO4_initial_layers, 0.4)
    FO3_initial_layers = copy.deepcopy(initial_layers)
    FO3_initial_layers = change_order(FO3_initial_layers, 0.3)
    FO2_initial_layers = copy.deepcopy(initial_layers)
    FO2_initial_layers = change_order(FO2_initial_layers, 0.2)
    FO1_initial_layers = copy.deepcopy(initial_layers)
    FO1_initial_layers = change_order(FO1_initial_layers, 0.1)
    layers, train_acc, test_acc = IO_train_mini_batch(X_train, Y_train, X_test, Y_test, IO_initial_layers,
                                    num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO19_initial_layers,
                                    num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO18_initial_layers,
                                    num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO17_initial_layers,
                                    num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO16_initial_layers,
                                    num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO15_initial_layers,
                                    num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO14_initial_layers,
                                    num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO13_initial_layers,
                                    num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO12_initial_layers,
                                    num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO11_initial_layers,
                                    num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO9_initial_layers,
                                    num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO8_initial_layers,
                                    num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO7_initial_layers,
                                    num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO6_initial_layers,
                                    num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO5_initial_layers,
                                    num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO4_initial_layers,
                                    num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO3_initial_layers,
                                    num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO2_initial_layers,
                                    num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO1_initial_layers,
                                    num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)