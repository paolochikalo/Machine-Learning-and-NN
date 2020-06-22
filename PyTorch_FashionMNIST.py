#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:22:19 2020

@author: mrpaolo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim


import numpy as np
import matplotlib.pyplot as plt


# ========================================== DOWNLOAD DATA ==========================================

train_set = torchvision.datasets.FashionMNIST(
        root='./__DATA/FashionMNIST',
        train=True,
        download=True, # data will be downloaded if it doesn't exist on disk
        transform = transforms.Compose([
                transforms.ToTensor()
                ])
        )


# ========================================= SINGLE IMAGE TESTING ===========================================
len(train_set)
print(train_set.targets) # 9 --> ankle boot..., 0 --> t-shirt
print(len(train_set.targets))

print(train_set.train_labels.bincount())#frequency distribution of the values inside the tensor

rand_ind = np.random.randint(low=0, high=60000, size =1)[0]

sample = train_set[rand_ind]
len(sample)

image, label = sample # sequence unpacking
image.shape

image.squeeze() # -- > removes 3rd dimention 1 which in our case is channel 

plt.imshow(image.squeeze(), cmap='gray')
print(label)

# ========================================== CREATING NETWORK ================================================

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


class Network(nn.Module):
    
    def __init__(self):
        super(Network, self).__init__()
        # kernel_size here refers to the width/height of the filter mask
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        # this layer will contain 6x12==72 separate 5x5 tensors and 12 feature maps on output
        # 6 out_channels from the previous layer should be imagined as 6 color channels (depth) within 1 image
        # number of samples and data structure will stay the same
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        # in_features --> 
        # 4*4 --> shrinked from 28*28
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
    
    
    # We don't call the forward method directly
    def forward(self, tensor):
        # Operations that does not involve operating with weights are invoked from functional module
        
        # (1) input layer added for brevity
        self.t = tensor
        
        # (2) 1st hidden Conv layer --> performs 3 operations
        self.t = self.conv1(self.t)
        self.t = F.relu(self.t)
        self.t = F.max_pool2d(self.t, kernel_size=2, stride=2)
        
        # (3) 2nd hiddden Conv Layer
        self.t = self.conv2(self.t)
        self.t = F.relu(self.t)
        self.t = F.max_pool2d(self.t, kernel_size=2, stride=2)
        
        # (4) 1st hidden linear layer
        
        # reshaping to flat form after convolutional layer
        # 12 --> number of feature maps from 2nd CNN layer
        # 4*4 --> width*height, size of each image after 2 layers of convolving
        self.t = self.t.reshape(-1, 12*4*4)
        self.t = self.fc1(self.t)
        self.t = F.relu(self.t)
        
        # (5) second hidden layer
        self.t = self.fc2(self.t)
        self.t = F.relu(self.t)
        
        # (6) output layer
        self.output = self.out(self.t)
        self.output = F.softmax(self.t, dim=1)
        
        return self.output
    

my_cnn = Network()

# -------------------------------------------- NETWORK ARCHITECTURE --------------------------------------------
print(my_cnn)
print(my_cnn.out)

# returns 6 tensors (out_channels=6) of size 5x5 (kernel_size=5) inited with small random values
print(my_cnn.conv1.weight)
print(my_cnn.conv1.weight.shape) # torch.Size([6, 1, 5, 5]) --> RANK 4

# Reshaping example
# Returns torch.Size([1, 150])
print(my_cnn.conv1.weight.reshape(-1, 6*5*5))
print(my_cnn.conv1.weight.shape)

# Following adds additional dimention at position .unsqueeze(index)
print(my_cnn.conv1.weight.unsqueeze(1).shape)# torch.Size([6, 1, 1, 5, 5])

# ??? Difference between reshape and view
print(my_cnn.conv1.weight.view(-1, 6*5*5))
print(my_cnn.conv1.weight.shape)


print(my_cnn.conv2.weight)
print(my_cnn.conv2.weight.shape) # torch.Size([12, 6, 5, 5]) --> RANK 4

print(my_cnn.conv1.weight[0].shape) # torch.Size([1, 5, 5]) --> single filter
print(my_cnn.conv2.weight[0].shape) # torch.Size([6, 5, 5])


print(my_cnn.fc1.weight.shape) # torch.Size([120, 192]) (out_features==120, in_features=12*4*4==192)

print(next(iter(my_cnn.parameters())))
print(next(iter(my_cnn.named_parameters())))

# ============================= LOADING AND DISPLAYING ONE BATCH OF DATA  =================================

train_set.root # './__DATA/FashionMNIST'
train_loader  = torch.utils.data.DataLoader(train_set, batch_size=100)


len(train_loader)
images, labels = next(iter(train_loader))

print(images.shape)# torch.Size([100, 1, 28, 28]) 100 images in each batch as we specified in batch_size

images.squeeze().shape # 1 removed from images data again

grid = torchvision.utils.make_grid(images, nrow=10)

plt.figure(figsize=(15,15))

grid.shape #  torch.Size([3, 32, 302])
np.transpose(grid,(1,2,0)).shape # torch.Size([32, 302, 3])
plt.imshow(np.transpose(grid,(1,2,0)))

# ===================================== SINGLE BATCH NETWORK TRAINING LOOP ==================================

# my_cnn.parameters() contains all of the weights and biases if the network
optimizer = optim.Adam(my_cnn.parameters(), lr=0.01)

# ------------------ LOOP STARTS HERE ---------------------
preds = my_cnn(images)

# loss tensor contains computational graph itself and has all of the information needed for backpropagation 
loss = F.cross_entropy(preds, labels)

# interpretation of the loss depends on the loss functiopn we are dealing with
loss.item() # 4.0940327644348145 ... 4.0527143478393555 ......

print(my_cnn.conv1.weight.grad) # None --> No gradients on the first pass
loss.backward()

# returns tensor of shape torch.Size([6, 1, 5, 5]) filled with small gradient changes to the weights
print(my_cnn.conv1.weight.grad)

get_num_correct(preds,labels)# 0 of the first batchepoch 11 on the second 15 on the 6th etc. 

# Actual updating of the weights
optimizer.step() # step in the direction of the loss function minimum

# GOTO 172 and repeat

# ------------------ LOOP ENDS HERE ---------------------

# ======================================== MY_CNN NETWORK TRAINING LOOP ====================================
optimizer = optim.Adam(my_cnn.parameters(), lr=0.0001)

total_loss = 0
total_correct = 0
epochs = 1 

len(train_loader)

for epoch in range(epochs):

    for batch in train_loader:
        images, labels = batch
        
        preds = my_cnn(images)
        
        loss = F.cross_entropy(preds, labels)
        
        # Needed due to the way PyTorch calculates grads
        # on each iteration PyTorch accumulates summarizes grad  with the ones from previous steps
        # for our purposes gradients (derivatives) should be calculated from scratch on each iteration
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
        total_loss += loss.item()
        total_correct += get_num_correct(preds,labels)
        
        print("Epoch: ", epoch, " Total correct: ", total_correct, " Loss: ",  np.round(loss.item(), 4), 
              " Total loss: ", np.round(total_loss, 4))
    
print("Epochs: ", epoch + 1, " Total correct: ", total_correct, " Total loss: ", total_loss)
