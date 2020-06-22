#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 12:47:55 2020

@author: mrpaolo
"""

import numpy as np
import torch
import torch.nn as nn
np.random.seed(2)

# ================================================== DATA =========================================================


input_layer = np.array([[1, 2, 3 ,2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8] ])

weight_1 = np.array([[0.2, 0.8, -0.5, 1.0],
          [0.5, -0.91, 0.26, -0.5],
          [-0.26, -0.27, 0.17, 0.87]])

weight_2 = 0.1 * np.random.randn(4, 3)

weight_3 = 0.2 * np.random.randn(4, 3)

# ============================================== Test and example functions ===================================================

# TODO: batches should be randomly sampled random_sample=True
# ??? With or without repetition?? 
def create_batch(input_data, batch_size=1, random_sample=False):
    '''
        Function that demonstrates how the batches may be created for simple ANN
        
        batch_size: size of every batch of the input data  
        
        Whereas in Keras batch_size does completely opposite to this function
        in Keras this parameter means more of a batch_num --> number of batches
        to which data should be divided to
    '''
    
    if batch_size < 1:
        batch_size = 1
    
    if not np.any(input_data):
        print("input_data cannot be empty")
        return 0
    
    input_size =  len(input_data)
    
    resid = int(input_size % int(batch_size))
        
    batch_num = ((input_size - resid) / batch_size) + (np.ceil(resid / batch_size))
    batch_num = int(batch_num)
    print(batch_num)
    
    global index
    index = 0
    output = []
    # for the batch_num - 1 we divide input data into batches
    for batch in range(batch_num):
        start = index
        index = batch_size * (batch + 1)

        if index <= input_size:
            output.append(list(input_data[start:index]))
        # for residuals we fill with zeros

        else:
            output.append(list(input_data[start:input_size]))
            for _ in range(start, index):
                output[batch].append(0)
    
    return np.array(output)

inputs_real = [1, 2, 3 ,2.5, 2.0, 5.0, -1.0, 2.0, -1.5, 2.7, 3.3, -0.8 ]
weights_real = [0.2, 0.8, -0.5, 1.0, 0.5, -0.91, 0.26, -0.5, -0.26, -0.27, 0.17, 0.87]

input_505 = np.random.randn(505)

batches = create_batch(input_data=input_505, batch_size=256)

# sigmoid functions for output layer mocking purposes
sigmoid = lambda x: 1/(1 + np.exp(-x))

class LayerSimpleRNN():
    def __init__(self, n_inputs, n_neurons):
        # Random weights less then 1
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.recurrent_weights = 0.10 * np.random.randn(n_neurons, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + \
            np.dot(self.recurrent_weights, self.recurrent_weights) + self.biases



# ============================================ GENERAL CALCULATIONS ===============================================

# one row is one feature, one bias --> one row
# size of bias depends on the size of weights layer specifically rows shape[1] a.k.a features
bias_1 = np.random.randn(weight_1.T.shape[1]) 
bias_2 = np.random.randn(weight_2.T.shape[1])

output_1 = np.dot(input_layer, weight_1.T) + bias_1
output_2 = np.dot(output_1, weight_2.T) + bias_2

final_output = sigmoid(output_2)
print(final_output)

# ================================================ NO ACTIVATION ======================================================

'''
Without activations funcions NN is a just a combination of linear transformations and 
all hidden layers can be .dot producted onto one layer
'''
input_layer = torch.from_numpy(input_layer)
weight_1 = torch.from_numpy(weight_1)
weight_2 = torch.from_numpy(weight_2)
weight_3 = torch.from_numpy(weight_3)

# Calculate the first and second hidden layer
hidden_1 = torch.matmul(input_layer, weight_1)
hidden_2 = torch.matmul(hidden_1, weight_2)

# Calculate the output
print(torch.matmul(hidden_2, weight_3)) # tensor([[0.2655, 0.1311, 3.8221, 3.0032]])

# Calculate weight_composed_1 and weight
weight_composed_1 = torch.matmul(weight_1, weight_2)
weight = torch.matmul(weight_composed_1, weight_3)

# Multiply input_layer with weight
print(torch.matmul(input_layer, weight)) # tensor([[0.2655, 0.1311, 3.8221, 3.0032]])

# RESULTS ARE THE SAME

# =================================== Constructing NN OOP Style ==============================================

X = np.array([[1, 2, 3 ,2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8] ])


weights = [[0.2, 0.8, -0.5, 1.0],
          [0.5, -0.91, 0.26, -0.5],
          [-0.26, -0.27, 0.17, 0.87]]



# Dense layers --> type of layers that applies weights to all nodes from previous layer.
class LayerDense():
    def __init__(self, n_inputs, n_neurons):
        # Random weights less then 1
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        print("Weights shape: ",self.weights.shape, ", Biases shape: ", self.biases.shape)
        
    def forward(self, inputs):
        inputs = np.array(inputs)
        print("Inputs shape: ", inputs.shape)
        self.output = np.dot(inputs, self.weights) + self.biases


layer1 = LayerDense(n_inputs=4, n_neurons=5)
layer2 = LayerDense(n_inputs=5, n_neurons=2) # the output of the previous layer should be input to this one

layer1.forward(X)
print(layer1.output)

layer2.forward(layer1.output)
print(layer2.output)

# ========================================== NN LOSS ====================================================

def MSE(prediction, goal_prediction):
    '''
        Mean squared error prediction
    '''
    delta = prediction - goal_prediction
    loss = np.mean((delta ** 2))
    return delta, loss
    

def softmax_cross_entropy(logits, ground_truth):
    '''
        logits --> np.array(): predictions from the NN 
        
        ground_truth --> np.array(): probabilities for correct classes 1 for correct, 0 for incorrect 
        
        order and shape of the inputs shoould correspond to each other
        
        returns: one number which if loss value for correct class
    '''
    
    assert logits.shape == ground_truth.shape, \
    ("Shapes of logits and ground truth should correspond to each other")
    exps = np.array([np.exp(score) for score in logits])
    sum_exps = np.sum(exps)
    probs = np.round(exps/sum_exps,2)
    
    #adj_probs = ((probs * ground_truth) / ground_truth)
    #correct_class = adj_probs.max()
    #ground_truth = ground_truth.max()
    #delta = ground_truth - correct_class    
    
    # we will get negative delta for correct class which will force weigths for correct clas to be increased
    # and for incorrect classes to decrease
    delta = probs - ground_truth # JUST LIKE THAT
    
    correct_class = (probs * ground_truth).max()
    # Cross-enthropy loss
    loss = -np.log(correct_class)
    
    return delta, loss 

softmax_cross_entropy([3.2,5.1,-1.7], [1,0,0])    
# cat car frog example
softmax_cross_entropy([-1.2,0.12,4.8], [0,0,1])


# ============================================= NN Training ==================================================

def hot_cold_train(inputs, weights, ground_truth, loss_func, epochs = 10, step = 0.001):
    '''
        HOT/COLD Method
        
        Simple NN training function, online learning, no activation
    '''
    
    assert inputs.shape == ground_truth.shape, \
    ("Shapes of inputs and ground truth should correspond to each other")
    
    for epoch in range(epochs):
        
        prediction = inputs .dot(weights)
        error = loss_func(prediction, ground_truth)
        
        print("Epoch: " + str(epoch), " Error: " + str(error) + " Prediction: " + str(prediction))
        
        up_weights = weights + step
        up_prediction = inputs.dot(up_weights)
        up_error = loss_func(up_prediction, ground_truth)
    
        down_weights = weights - step
        down_prediction = inputs.dot(down_weights)
        down_error = loss_func(down_prediction, ground_truth)        
        
        if(down_error < up_error):
            weights = down_weights
    
        if(down_error > up_error):
            weights =  up_weights

def simple_gradient_descent(inputs, weights, ground_truth, loss_func, epochs = 10, alpha = 0.01):
    '''
        Simple gradient Descent
        alpha --> learning rate    
    '''
    
    assert inputs.shape == ground_truth.shape, \
    ("Shapes of inputs and ground truth should correspond to each other")
    
    # Derivative of ReLu function converts negative numbers to 0 positive to 1
    # thus on the backward propagation only those deltas will influence weights 
    # that are non-negative
    def relu_deriv(layer_output):
        return layer_output >=0
    
    for epoch in range(epochs):
        
        prediction = inputs.dot(weights)
        delta, error = loss_func(prediction, ground_truth)
        
        weight_delta = delta.dot(inputs.T) * relu_deriv(inputs)
        
        # alpha is scalar;
        weights = weights - (weight_delta*alpha)
    
        print("Epoch: " + str(epoch), " Error: " + str(error) + " Prediction: " + str(prediction))
        
        
        
        
        


