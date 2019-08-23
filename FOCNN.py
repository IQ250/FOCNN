import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import random
import logging
import time
import pdb

# Creating log file
def create_logger(output_path, cfg_name):
    log_file = '{}_{}.log'.format(cfg_name, time.strftime('%Y-%m-%d-%H-%M'))
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(output_path, log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    return logger

# Loading and precessing data
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

# ----------------------------------------- Parts of Model ------------------------------------------
def initialize_parameters(layers):
    """
    Initialize parameters according to different types of layers
    
    Argument:
    layers -- list, the length denotes the depth of networks, every element is a dictionary which contains the 
              mode of layer and shape of weight and bias 
    
    Returns:
    new_layers -- list, every element corresponds to original layer and its intialized parameter
    """
    new_layers = []
    for i, layer in enumerate(layers):
        mode = layer['mode'] # 'fc', 'conv', 'pool'
        if mode == 'pool':
            new_layers.append(layer)
            continue
        elif mode == 'fc':
            n_now = layer['n_now']
            n_prev = layer['n_prev']
            # layer['W']=np.random.randn(n_now, n_prev)*0.01 # normal distribution (mu=0, sigma=0.01)
            layer['W']=(np.random.rand(n_now, n_prev) - 0.5) * 0.2 # random sample in [-0.1, 0.1] 
            layer['b']=(np.random.rand(n_now,1) - 0.5) * 0.2
            layer['W_k']=[np.zeros_like(layer['W'])]
            layer['b_k']=[np.zeros_like(layer['b'])]
            layer['dW']=np.zeros_like(layer['W'])
            layer['db']=np.zeros_like(layer['b'])
            layer['IO_dW_k-1']=(np.random.rand(n_now, n_prev) - 0.5) * 0.2 # integer order gradient of last step
            layer['IO_db_k-1']=(np.random.rand(n_now,1) - 0.5) * 0.2
        elif mode == 'conv':
            f = layer['f']
            n_C = layer['n_C']
            n_C_prev = layer['n_C_prev']
            # layer['W']=np.random.randn(f, f, n_C_prev, n_C)*0.01 # normal distribution (mu=0, sigma=0.01)
            layer['W']=(np.random.rand(f, f, n_C_prev, n_C) - 0.5) * 0.2 # random sample in [-0.1, 0.1] 
            layer['b']=(np.random.rand(1, 1, 1, n_C) - 0.5) * 0.2
            layer['W_k']=[np.zeros_like(layer['W'])]
            layer['b_k']=[np.zeros_like(layer['b'])]
            layer['dW']=np.zeros_like(layer['W'])
            layer['db']=np.zeros_like(layer['b'])
            layer['IO_dW_k-1']=(np.random.rand(f, f, n_C_prev, n_C) - 0.5) * 0.2 # integer order gradient of last step
            layer['IO_db_k-1']=(np.random.rand(1, 1, 1, n_C) - 0.5) * 0.2
        else:
            print('Wrong layer in [{}]'.format(i))
        new_layers.append(layer)
            
    return new_layers

def sigmoid(Z):
    # Sigmoid activation function
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

def sigmoid_backward(dA, cache):
    # Backpropogation of sigmoid activation function
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ

def relu(Z):
    # Relu activation function
    A = np.maximum(0,Z)
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    # Backpropogation of Relu activation function 
    Z = cache
    
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
     
    # dZ[Z <= 0] = 0 # when z <= 0, you should set dz to 0 as well.
    dZ[Z < 0] = 0
    return dZ

def softmax(Z):
    # Softmax activation function
    n, m = Z.shape
    A = np.exp(Z)
    A_sum = np.sum(A, axis = 0)
    A_sum = A_sum.reshape(-1, m)
    A = A / A_sum
    cache = Z
    return A, cache

def softmax_backward(A, Y):
    # Backpropogation of softmax activation function
    # loss = - ln a[j] (y[j] = 1, j = {0, ..., n}) 
    m = A.shape[1]
    # dZ = A - Y
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
    # Backward propagatIon module - linear activation backward
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
    # Backward propagatIon module - linear activation backward
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
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
    as illustrated in Figure 1.
    
    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """
    X_pad = np.pad(X, ((0, 0),(pad, pad),(pad, pad),(0, 0)), 'constant', constant_values=value)
    
    return X_pad

def conv_forward(A_prev, layer):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    layer -- a dictionary contains weights, bias, hyperparameters and shape of data
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    """
    # Retrieve information from layer
    W = layer['W']
    b = layer['b']
    stride = layer['s']
    pad = layer['p']
    
    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape
    
    # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor
    n_H = 1 + int((n_H_prev + 2 * pad - f) / stride)
    n_W = 1 + int((n_W_prev + 2 * pad - f) / stride)
    
    # Initialize the output volume Z with zeros
    Z = np.zeros((m, n_H, n_W, n_C))
    
    # Create A_prev_pad by padding A_prev
    if pad > 0:
        A_prev_pad = zero_pad(A_prev, pad)
    else:
        A_prev_pad = A_prev
    
    for i in range(m):                                 # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i]                     # Select ith training example's padded activation
        for h in range(n_H):                           # loop over vertical axis of the output volume
            for w in range(n_W):                       # loop over horizontal axis of the output volume
                for c in range(n_C):                   # loop over channels (= #filters) of the output volume
                    
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    
                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell)
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                  
                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron
                    # Z[i, h, w, c] = np.sum(np.multiply(a_slice_prev, W[:, :, :, c]) + b[:, :, :, c])
                    Z[i, h, w, c] = np.sum(np.multiply(a_slice_prev, W[:, :, :, c])) + b[0, 0, 0, c]

    return Z

def IO_conv_backward(dZ, layer):
    """
    Implement the backward propagation for a convolution function
    
    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    layer -- a dictionary contains weights, bias, hyperparameters and shape of data
    
    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    """
    # Retrieve informations from layer
    A_prev = layer['A_prev']
    W = layer['W']
    b = layer['b']
    Z = layer['Z']
    alpha = layer['alpha']
    stride = layer['s']
    pad = layer['p']
    
    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape
    
    # Retrieve dimensions from dZ's shape
    (m, n_H, n_W, n_C) = dZ.shape
    
    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))                           
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    # Pad A_prev and dA_prev
    if pad > 0:
        A_prev_pad = zero_pad(A_prev, pad)
        dA_prev_pad = zero_pad(dA_prev, pad)
    else:
        A_prev_pad = A_prev
        dA_prev_pad = np.copy(dA_prev)

    for i in range(m):                         # loop over the training examples
        for h in range(n_H):                   # loop over vertical axis of the output volume
            for w in range(n_W):               # loop over horizontal axis of the output volume
                for c in range(n_C):           # loop over the channels of the output volume
                    
                    # Find the corners of the current "slice"
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    
                    # Use the corners to define the slice from a_prev_pad
                    a_slice = A_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :]

                    # Update gradients for the window and the filter's parameters using the code formulas
                    dA_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += dZ[i, h, w, c] * a_slice
                    db[:,:,:,c] += dZ[i, h, w, c]
                    
                    
        # Set the ith training example's dA_prev to the unpaded da_prev_pad
        if pad == 0:
            dA_prev[i, :, :, :] = dA_prev_pad[i, :, :, :]
        else:
            dA_prev[i, :, :, :] = dA_prev_pad[i, pad:-pad, pad:-pad, :]
    
    return dA_prev, dW, db

def FO_conv_backward(dZ, layer, delta=1e-2):
    """
    Implement the backward propagation for a convolution function
    
    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    layer -- a dictionary contains weights, bias, hyperparameters and shape of data
    
    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    """
    # Retrieve informations from layer
    A_prev = layer['A_prev']
    W = layer['W']
    b = layer['b']
    c_W = layer['W_k'][-1]
    c_b = layer['b_k'][-1]
    Z = layer['Z']
    alpha = layer['alpha']
    stride = layer['s']
    pad = layer['p']
    
    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape
    
    # Retrieve dimensions from dZ's shape
    (m, n_H, n_W, n_C) = dZ.shape
    
    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))                           
    IO_dW = np.zeros((f, f, n_C_prev, n_C))
    IO_db = np.zeros((1, 1, 1, n_C))

    # Pad A_prev and dA_prev
    if pad > 0:
        A_prev_pad = zero_pad(A_prev, pad)
        dA_prev_pad = zero_pad(dA_prev, pad)
    else:
        A_prev_pad = A_prev
        dA_prev_pad = np.copy(dA_prev)

    for i in range(m):                         # loop over the training examples
        for h in range(n_H):                   # loop over vertical axis of the output volume
            for w in range(n_W):               # loop over horizontal axis of the output volume
                for c in range(n_C):           # loop over the channels of the output volume
                    
                    # Find the corners of the current "slice"
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    
                    # Use the corners to define the slice from a_prev_pad
                    a_slice = A_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :]

                    # Update gradients for the window and the filter's parameters using the code formulas
                    dA_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    IO_dW[:,:,:,c] += dZ[i, h, w, c] * a_slice
                    IO_db[:,:,:,c] += dZ[i, h, w, c]
                    
                    
        # Set the ith training example's dA_prev to the unpaded da_prev_pad
        if pad == 0:
            dA_prev[i, :, :, :] = dA_prev_pad[i, :, :, :]
        else:
            dA_prev[i, :, :, :] = dA_prev_pad[i, pad:-pad, pad:-pad, :]
    
    # Fractional order gradient
    dW = layer['IO_dW_k-1'] * np.power(np.abs(W-c_W)+delta, 1-alpha) / math.gamma(2-alpha)
    db = layer['IO_db_k-1'] * np.power(np.abs(b-c_b)+delta, 1-alpha) / math.gamma(2-alpha)

    return dA_prev, dW, db, IO_dW, IO_db

def pool_forward(A_prev, layer, mode = "max"):
    """
    Implements the forward pass of the pooling layer
    
    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    """
    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve hyperparameters from "hparameters"
    f = layer["f"]
    stride = layer["s"]
    
    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    
    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))              

    for i in range(m):                           # loop over the training examples
        for h in range(n_H):                     # loop on the vertical axis of the output volume
            for w in range(n_W):                 # loop on the horizontal axis of the output volume
                for c in range (n_C):            # loop over the channels of the output volume
                    
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    
                    # Use the corners to define the current slice on the ith training example of A_prev, channel c.
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                    
                    # Compute the pooling operation on the slice. Use an if statment to differentiate the modes.
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)

    return A

def distribute_value(dz, shape):
    """
    Distributes the input value in the matrix of dimension shape
    
    Arguments:
    dz -- input scalar
    shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
    
    Returns:
    a -- Array of size (n_H, n_W) for which we distributed the value of dz
    """
    # Retrieve dimensions from shape
    (n_H, n_W) = shape
    # Compute the value to distribute on the matrix
    average = dz / (n_H * n_W)
    # Create a matrix where every entry is the "average" value
    a = np.ones(shape) * average
    
    return a

def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.
    Arguments:
    x -- Array of shape (f, f)
    Returns:
    mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    """
    mask = (x == np.max(x))
    
    return mask

def pool_backward(dA, layer, mode = "max"):
    """
    Implements the backward pass of the pooling layer
    
    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """
    # Retrieve information from layer
    A_prev = layer['A_prev']
    stride = layer['s']
    f = layer['f']
    
    # Retrieve dimensions from A_prev's shape and dA's shape
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    
    # Initialize dA_prev with zeros (≈1 line)
    dA_prev = np.zeros_like(A_prev)
    
    for i in range(m):                         # loop over the training examples
        # select training example from A_prev
        a_prev = A_prev[i]
        for h in range(n_H):                   # loop on the vertical axis
            for w in range(n_W):               # loop on the horizontal axis
                for c in range(n_C):           # loop over the channels (depth)
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    
                    # Compute the backward propagation in both modes.
                    if mode == "max":
                        # Use the corners and "c" to define the current slice from a_prev
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        # Create the mask from a_prev_slice
                        mask = create_mask_from_window(a_prev_slice)
                        # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask * dA[i, h, w, c]
                        
                    elif mode == "average":
                        # Get the value a from dA
                        da = dA[i, h, w, c]
                        # Define the shape of the filter as fxf
                        shape = (f, f)
                        # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da.
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)

    return dA_prev

def forward_propogation(X, layers):
    m = X.shape[0]
    # -1- convolution layer
    layers[0]['A_prev'] = X
    Z = conv_forward(X, layers[0])
    layers[0]['Z'] = Z
    A, _ = relu(Z)
    
    # -2- average pooling layer
    layers[1]['A_prev'] = A
    A = pool_forward(A, layers[1], mode = "average")
    
    # -3- convolution layer
    layers[2]['A_prev'] = A
    Z = conv_forward(A, layers[2])
    layers[2]['Z'] = Z
    A, _ = relu(Z)
    
    # -4- average pooling layer
    layers[3]['A_prev'] = A
    A = pool_forward(A, layers[3], mode = "average")
    
    # -5- convolution layer
    layers[4]['A_prev'] = A
    Z = conv_forward(A, layers[4])
    layers[4]['Z'] = Z
    A, _ = relu(Z)
    
    # -6- fully connected layer
    layers[5]['A_prev'] = (A.reshape(m,-1)).T # flatten
    A, Z = linear_activation_forward((A.reshape(m,-1)).T, layers[5], activation='relu')
    layers[5]['Z'] = Z
    
    # -7- fully connected layer
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
    # -7- fully connected layer
    dZ = softmax_backward(AL, Y)
    dA_prev, dW, db = IO_linear_activation_backward(dZ, layers[6], 'none')
    layers[6]['dW'] = dW
    layers[6]['db'] = db
    
    # -6- fully connected layer
    dA_prev, dW, db = IO_linear_activation_backward(dA_prev, layers[5], 'relu')
    layers[5]['dW'] = dW
    layers[5]['db'] = db
    
    # -5- convolution layer
    dA = (dA_prev.T).reshape(m,1,1,layers[4]['n_C']) # flatten backward
    dZ = relu_backward(dA, layers[4]['Z']) # relu_backward(dA, layers[4]['Z'])
    dA_prev, dW, db = IO_conv_backward(dZ, layers[4])
    layers[4]['dW'] = dW
    layers[4]['db'] = db
    
    # -4- average pooling layer
    dA_prev = pool_backward(dA_prev, layers[3], mode = "average")
    
    # -3- convolution layer
    dZ = relu_backward(dA_prev, layers[2]['Z']) # relu_backward(dA_prev, layers[2]['Z'])
    dA_prev, dW, db = IO_conv_backward(dZ, layers[2])
    layers[2]['dW'] = dW
    layers[2]['db'] = db
    
    # -2- average pooling layer
    dA_prev = pool_backward(dA_prev, layers[1], mode = "average")
    
    # -1- convolution layer
    dZ = relu_backward(dA_prev, layers[0]['Z']) # relu_backward(dA_prev, layers[0]['Z'])
    dA_prev, dW, db = IO_conv_backward(dZ, layers[0])
    layers[0]['dW'] = dW
    layers[0]['db'] = db
    
    return layers

def FO_backward_propogation(AL, Y, layers):
    m = Y.shape[1]
    # -7- fully connected layer
    dZ = softmax_backward(AL, Y)
    dA_prev, dW, db, IO_dW, IO_db = FO_linear_activation_backward(dZ, layers[6], 'none')
    layers[6]['dW'] = dW
    layers[6]['db'] = db
    layers[6]['IO_dW_k-1'] = IO_dW
    layers[6]['IO_db_k-1'] = IO_db
    
    # -6- fully connected layer
    dA_prev, dW, db, IO_dW, IO_db = FO_linear_activation_backward(dA_prev, layers[5], 'relu')
    layers[5]['dW'] = dW
    layers[5]['db'] = db
    layers[5]['IO_dW_k-1'] = IO_dW
    layers[5]['IO_db_k-1'] = IO_db
    
    # -5- convolution layer
    dA = (dA_prev.T).reshape(m,1,1,layers[4]['n_C']) # flatten backward
    dZ = relu_backward(dA, layers[4]['Z'])
    dA_prev, dW, db, IO_dW, IO_db = FO_conv_backward(dZ, layers[4])
    layers[4]['dW'] = dW
    layers[4]['db'] = db
    layers[4]['IO_dW_k-1'] = IO_dW
    layers[4]['IO_db_k-1'] = IO_db
    
    # -4- average pooling layer
    dA_prev = pool_backward(dA_prev, layers[3], mode = "average")
    
    # -3- convolution layer
    dZ = relu_backward(dA_prev, layers[2]['Z'])
    dA_prev, dW, db, IO_dW, IO_db = FO_conv_backward(dZ, layers[2])
    layers[2]['dW'] = dW
    layers[2]['db'] = db
    layers[2]['IO_dW_k-1'] = IO_dW
    layers[2]['IO_db_k-1'] = IO_db
    
    # -2- average pooling layer
    dA_prev = pool_backward(dA_prev, layers[1], mode = "average")
    
    # -1- convolution layer
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
        mode = layers[i]['mode'] # 'fc', 'conv', 'pool'
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
        mode = layers[i]['mode'] # 'fc', 'conv', 'pool'
        if mode == 'pool':
            continue
        elif (mode == 'fc' or mode == 'conv'):
#             memory_length = len(layers[i]['W_k'])
#             for j in range(memory_length-1):
#                 layers[i]['W_k'][memory_length-1-j] = layers[i]['W_k'][memory_length-2-j]
#                 layers[i]['b_k'][memory_length-1-j] = layers[i]['b_k'][memory_length-2-j]
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
    pred_count = np.zeros((n,m)) - 1 # for counting accurate predictions 
    
    # Forward propagation
    AL, _ = forward_propogation(X_test, layers)

    # convert prediction to 0/1 form
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

def IO_train_mini_batch(X_train, Y_train, X_test, Y_test, layers, logger, num_exp=0, batch_size=10, num_epoch=1, learning_rate=0.01):
    logger.info('------------ Integer order CNN with mini batch ------------')
    logger.info('Initial weights: FC [-0.1, 0.1], CONV [-0.1, 0.1]')
    logger.info('Initial bias: FC [-0.1, 0.1], CONV [-0.1, 0.1]')
    logger.info('Batch size: {}'.format(batch_size))
    logger.info('Learning rate: {}'.format(learning_rate))
    
    # number of iteration
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
                logger.info('Epoch [{}] Iteration [{}]: loss = {} accuracy = {}'.format(epoch, iteration+1, loss, accuracy))
                print('Epoch [{}] Iteration [{}]: loss = {} accuracy = {}'.format(epoch, iteration+1, loss, accuracy))
                np.save('data/FO_layers_{}_{}.npy'.format(epoch, iteration+1), layers)

        _, accuracy_test = predict(X_test, Y_test, layers)
        pred_train, _ = forward_propogation(X_train[:10000], layers)
        loss_train = compute_cost(pred_train, Y_train[:10000].T)
        accuracy_train = compute_accuracy(pred_train, Y_train[:10000].T)
        accuracy_train_list.append(accuracy_train)
        accuracy_test_list.append(accuracy_test)
        print('Epoch [{}] average_loss = {} average_accuracy = {}'.format(epoch, np.mean(losses), np.mean(accuracies)))
        logger.info('Epoch [{}] average_loss = {} average_accuracy = {}'.format(epoch, np.mean(losses), np.mean(accuracies)))
        print('Epoch [{}] train_loss = {} train_accuracy = {}'.format(epoch, loss_train, accuracy_train))
        logger.info('Epoch [{}] train_loss = {} train_accuracy = {}'.format(epoch, loss_train, accuracy_train))
        print('Epoch [{}] test_accuracy = {}'.format(epoch, accuracy_test))
        logger.info('Epoch [{}] test_accuracy = {}'.format(epoch, accuracy_test))
    
    return layers, accuracy_train_list, accuracy_test_list

def FO_train_mini_batch(X_train, Y_train, X_test, Y_test, layers, logger, num_exp=0, batch_size=10, num_epoch=1, learning_rate=0.01):
    logger.info('------------ Fractional order CNN with mini batch ------------')
    logger.info('Initial weights: FC [-0.1, 0.1], CONV [-0.1, 0.1]')
    logger.info('Absolute initial weights: FC 0, CONV 0')
    logger.info('Initial bias: FC [-0.1, 0.1], CONV [-0.1, 0.1]')
    logger.info('Absolute initial bias: FC 0, CONV 0')
    logger.info('Alpha: {}'.format(layers[0]['alpha']))
    logger.info('Batch size: {}'.format(batch_size))
    logger.info('Learning rate: {}'.format(learning_rate))
    
    # number of iteration
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
            layers = FO_backward_propogation(AL, Y_train_batch.T, layers)
            # pdb.set_trace()
            layers = FO_update_parameters(layers, learning_rate)
            losses.append(loss)
            accuracies.append(accuracy)
            if (iteration+1) % 600 == 0:
                logger.info('Epoch [{}] Iteration [{}]: loss = {} accuracy = {}'.format(epoch, iteration+1, loss, accuracy))
                print('Epoch [{}] Iteration [{}]: loss = {} accuracy = {}'.format(epoch, iteration+1, loss, accuracy))
                np.save('data/FO_layers_{}_{}.npy'.format(epoch, iteration+1), layers)

        _, accuracy_test = predict(X_test, Y_test, layers)
        pred_train, _ = forward_propogation(X_train[:10000], layers)
        loss_train = compute_cost(pred_train, Y_train[:10000].T)
        accuracy_train = compute_accuracy(pred_train, Y_train[:10000].T)
        accuracy_train_list.append(accuracy_train)
        accuracy_test_list.append(accuracy_test)
        print('Epoch [{}] average_loss = {} average_accuracy = {}'.format(epoch, np.mean(losses), np.mean(accuracies)))
        logger.info('Epoch [{}] average_loss = {} average_accuracy = {}'.format(epoch, np.mean(losses), np.mean(accuracies)))
        print('Epoch [{}] train_loss = {} train_accuracy = {}'.format(epoch, loss_train, accuracy_train))
        logger.info('Epoch [{}] train_loss = {} train_accuracy = {}'.format(epoch, loss_train, accuracy_train))
        print('Epoch [{}] test_accuracy = {}'.format(epoch, accuracy_test))
        logger.info('Epoch [{}] test_accuracy = {}'.format(epoch, accuracy_test))
    
    return layers, accuracy_train_list, accuracy_test_list

def change_order(layers, order):
    layers[0]['alpha'] = order
    layers[2]['alpha'] = order
    layers[4]['alpha'] = order
    layers[5]['alpha'] = order
    layers[6]['alpha'] = order

    return layers

# -------------------------------------------------------------------------------------------------------------------------

# Create log file
logger = create_logger('output', 'Comparison_train_log')

# Load dataset and reshape image set as (m, n_H, n_W, n_C)
X_train, Y_train = load_mnist('data', 'train')
X_test, Y_test = load_mnist('data', 'test')
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

# Normalization for images
# X_train = (X_train / 255.0 - 0.5) * 2
# X_test = (X_test / 255.0 - 0.5) * 2
X_train = X_train / 255.0
X_test = X_test / 255.0

# Transform the label into one-hot form
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

# Construct model
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
    print('------------------------------------- Experiment {} --------------------------------------'.format(index+1))
    logger.info('------------------------------------- Experiment {} -------------------------------------'.format(index+1))

    initial_layers_path = 'data/initial_layers_{}.npy'.format(index+1)
    if os.path.exists(initial_layers_path):
        initial_layers = np.load(initial_layers_path)
        print('Load initial parameters from {}'.format(initial_layers_path))
        logger.info('Load initial parameters from {}'.format(initial_layers_path))
    else:
        initial_layers = initialize_parameters(construct_layers)
        np.save(initial_layers_path, initial_layers)
        print('Initialize layers and save as {}'.format(initial_layers_path))
        logger.info('Initialize layers and save as {}'.format(initial_layers_path))
    
    # nitialize paraters with alpha = 1 
    IO_initial_layers = copy.deepcopy(initial_layers)
    
    # initialize paraters with alpha = 1.9
    FO19_initial_layers = copy.deepcopy(initial_layers)
    FO19_initial_layers = change_order(FO19_initial_layers, 1.9)

    # initialize paraters with alpha = 1.8
    FO18_initial_layers = copy.deepcopy(initial_layers)
    FO18_initial_layers = change_order(FO18_initial_layers, 1.8)

    # initialize paraters with alpha = 1.7
    FO17_initial_layers = copy.deepcopy(initial_layers)
    FO17_initial_layers = change_order(FO17_initial_layers, 1.7)

    # initialize paraters with alpha = 1.6
    FO16_initial_layers = copy.deepcopy(initial_layers)
    FO16_initial_layers = change_order(FO16_initial_layers, 1.6)

    # initialize paraters with alpha = 1.5
    FO15_initial_layers = copy.deepcopy(initial_layers)
    FO15_initial_layers = change_order(FO15_initial_layers, 1.5)

    # initialize paraters with alpha = 1.4
    FO14_initial_layers = copy.deepcopy(initial_layers)
    FO14_initial_layers = change_order(FO14_initial_layers, 1.4)

    # initialize paraters with alpha = 1.3
    FO13_initial_layers = copy.deepcopy(initial_layers)
    FO13_initial_layers = change_order(FO13_initial_layers, 1.3)

    # initialize paraters with alpha = 1.2
    FO12_initial_layers = copy.deepcopy(initial_layers)
    FO12_initial_layers = change_order(FO12_initial_layers, 1.2)

    # initialize paraters with alpha = 1.1
    FO11_initial_layers = copy.deepcopy(initial_layers)
    FO11_initial_layers = change_order(FO11_initial_layers, 1.1)

    # initialize paraters with alpha = 0.9
    FO9_initial_layers = copy.deepcopy(initial_layers)
    FO9_initial_layers = change_order(FO9_initial_layers, 0.9)

    # initialize paraters with alpha = 0.8
    FO8_initial_layers = copy.deepcopy(initial_layers)
    FO8_initial_layers = change_order(FO8_initial_layers, 0.8)

    # initialize paraters with alpha = 0.7
    FO7_initial_layers = copy.deepcopy(initial_layers)
    FO7_initial_layers = change_order(FO7_initial_layers, 0.7)

    # initialize paraters with alpha = 0.6
    FO6_initial_layers = copy.deepcopy(initial_layers)
    FO6_initial_layers = change_order(FO6_initial_layers, 0.6)

    # initialize paraters with alpha = 0.5
    FO5_initial_layers = copy.deepcopy(initial_layers)
    FO5_initial_layers = change_order(FO5_initial_layers, 0.5)

    # initialize paraters with alpha = 0.4
    FO4_initial_layers = copy.deepcopy(initial_layers)
    FO4_initial_layers = change_order(FO4_initial_layers, 0.4)

    # initialize paraters with alpha = 0.3
    FO3_initial_layers = copy.deepcopy(initial_layers)
    FO3_initial_layers = change_order(FO3_initial_layers, 0.3)

    # initialize paraters with alpha = 0.2
    FO2_initial_layers = copy.deepcopy(initial_layers)
    FO2_initial_layers = change_order(FO2_initial_layers, 0.2)

    # initialize paraters with alpha = 
    FO1_initial_layers = copy.deepcopy(initial_layers)
    FO1_initial_layers = change_order(FO1_initial_layers, 0.1)

    print('----------------------------------------------------------------------------------------')
    logger.info('----------------------------------------------------------------------------------------')
    print('- alpha = 1.0 -')
    logger.info('- alpha = 1.0 -')
    # IOBPNN
    print('Integer order back propagation neural networks')
    layers, train_acc, test_acc = IO_train_mini_batch(X_train, Y_train, X_test, Y_test, IO_initial_layers,
                                    logger, num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    print('\n')
    logger.info('\n')


    print('----------------------------------------------------------------------------------------')
    logger.info('----------------------------------------------------------------------------------------')
    print('- alpha = 1.9 -')
    logger.info('- alpha = 1.9 -')
    # FOBPNN
    print('Fractional order back propagation neural networks')
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO19_initial_layers,
                                    logger, num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    print('\n')
    logger.info('\n')

    print('-----------------------------------------------------------------------------------------')
    logger.info('-----------------------------------------------------------------------------------------')
    print('- alpha = 1.8 -')
    logger.info('- alpha = 1.8 -')
    # FOBPNN
    print('Fractional order back propagation neural networks')
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO18_initial_layers,
                                    logger, num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    print('\n')
    logger.info('\n')

    print('-----------------------------------------------------------------------------------------')
    logger.info('-----------------------------------------------------------------------------------------')
    print('- alpha = 1.7 -')
    logger.info('- alpha = 1.7 -')
    # FOBPNN
    print('Fractional order back propagation neural networks')
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO17_initial_layers,
                                    logger, num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    print('\n')
    logger.info('\n')

    print('-----------------------------------------------------------------------------------------')
    logger.info('-----------------------------------------------------------------------------------------')
    print('- alpha = 1.6 -')
    logger.info('- alpha = 1.6 -')
    # FOBPNN
    print('Fractional order back propagation neural networks')
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO16_initial_layers,
                                    logger, num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    print('\n')
    logger.info('\n')

    print('-----------------------------------------------------------------------------------------')
    logger.info('-----------------------------------------------------------------------------------------')
    print('- alpha = 1.5 -')
    logger.info('- alpha = 1.5 -')
    # FOBPNN
    print('Fractional order back propagation neural networks')
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO15_initial_layers,
                                    logger, num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    print('\n')
    logger.info('\n')

    print('-----------------------------------------------------------------------------------------')
    logger.info('-----------------------------------------------------------------------------------------')
    print('- alpha = 1.4 -')
    logger.info('- alpha = 1.4 -')
    # FOBPNN
    print('Fractional order back propagation neural networks')
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO14_initial_layers,
                                    logger, num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    print('\n')
    logger.info('\n')

    print('-----------------------------------------------------------------------------------------')
    logger.info('-----------------------------------------------------------------------------------------')
    print('- alpha = 1.3 -')
    logger.info('- alpha = 1.3 -')
    # FOBPNN
    print('Fractional order back propagation neural networks')
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO13_initial_layers,
                                    logger, num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    print('\n')
    logger.info('\n')

    print('-----------------------------------------------------------------------------------------')
    logger.info('-----------------------------------------------------------------------------------------')
    print('- alpha = 1.2 -')
    logger.info('- alpha = 1.2 -')
    # FOBPNN
    print('Fractional order back propagation neural networks')
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO12_initial_layers,
                                    logger, num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    print('\n')
    logger.info('\n')

    print('-----------------------------------------------------------------------------------------')
    logger.info('-----------------------------------------------------------------------------------------')
    print('- alpha = 1.1 -')
    logger.info('- alpha = 1.1 -')
    # FOBPNN
    print('Fractional order back propagation neural networks')
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO11_initial_layers,
                                    logger, num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    print('\n')
    logger.info('\n')

    print('-----------------------------------------------------------------------------------------')
    logger.info('-----------------------------------------------------------------------------------------')
    print('- alpha = 0.9 -')
    logger.info('- alpha = 0.9 -')
    # FOBPNN
    print('Fractional order back propagation neural networks')
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO9_initial_layers,
                                    logger, num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    print('\n')
    logger.info('\n')

    print('-----------------------------------------------------------------------------------------')
    logger.info('-----------------------------------------------------------------------------------------')
    print('- alpha = 0.8 -')
    logger.info('- alpha = 0.8 -')
    # FOBPNN
    print('Fractional order back propagation neural networks')
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO8_initial_layers,
                                    logger, num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    print('\n')
    logger.info('\n')

    print('-----------------------------------------------------------------------------------------')
    logger.info('-----------------------------------------------------------------------------------------')
    print('- alpha = 0.7 -')
    logger.info('- alpha = 0.7 -')
    # FOBPNN
    print('Fractional order back propagation neural networks')
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO7_initial_layers,
                                    logger, num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    print('\n')
    logger.info('\n')

    print('-----------------------------------------------------------------------------------------')
    logger.info('-----------------------------------------------------------------------------------------')
    print('- alpha = 0.6 -')
    logger.info('- alpha = 0.6 -')
    # FOBPNN
    print('Fractional order back propagation neural networks')
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO6_initial_layers,
                                    logger, num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    print('\n')
    logger.info('\n')

    print('-----------------------------------------------------------------------------------------')
    logger.info('-----------------------------------------------------------------------------------------')
    print('- alpha = 0.5 -')
    logger.info('- alpha = 0.5 -')
    # FOBPNN
    print('Fractional order back propagation neural networks')
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO5_initial_layers,
                                    logger, num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    print('\n')
    logger.info('\n')

    print('-----------------------------------------------------------------------------------------')
    logger.info('-----------------------------------------------------------------------------------------')
    print('- alpha = 0.4 -')
    logger.info('- alpha = 0.4 -')
    # FOBPNN
    print('Fractional order back propagation neural networks')
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO4_initial_layers,
                                    logger, num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    print('\n')
    logger.info('\n')

    print('-----------------------------------------------------------------------------------------')
    logger.info('-----------------------------------------------------------------------------------------')
    print('- alpha = 0.3 -')
    logger.info('- alpha = 0.3 -')
    # FOBPNN
    print('Fractional order back propagation neural networks')
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO3_initial_layers,
                                    logger, num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    print('\n')
    logger.info('\n')

    print('-----------------------------------------------------------------------------------------')
    logger.info('-----------------------------------------------------------------------------------------')
    print('- alpha = 0.2 -')
    logger.info('- alpha = 0.2 -')
    # FOBPNN
    print('Fractional order back propagation neural networks')
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO2_initial_layers,
                                    logger, num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    print('\n')
    logger.info('\n')

    print('-----------------------------------------------------------------------------------------')
    logger.info('-----------------------------------------------------------------------------------------')
    print('- alpha = 0.1 -')
    logger.info('- alpha = 0.1 -')
    # FOBPNN
    print('Fractional order back propagation neural networks')
    layers, train_acc, test_acc = FO_train_mini_batch(X_train, Y_train, X_test, Y_test, FO1_initial_layers,
                                    logger, num_exp=index, batch_size=10, num_epoch=1, learning_rate=0.1)
    print('\n')
    logger.info('\n')