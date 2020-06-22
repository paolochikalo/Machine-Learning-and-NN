#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 19:57:55 2020

@author: mrpaolo
"""

import numpy as np
import pandas as pd

from utility_functions import get_vocabulary, get_max_len

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, SimpleRNN, TimeDistributed, LSTM

'''

The idea is to generate lucky comany name from the list of S&P 500 Using RNN/LSTM

'''


lucky_comps = pd.read_csv("__DATA/SP_500.csv")

print(lucky_comps.head())
print(len(lucky_comps))

# ==================================== CLEANING UP NAMES ============================================

lucky_comps['Clean Name'] =  lucky_comps['Name'].str.replace(r'.', '')
lucky_comps['Clean Name'] =  lucky_comps['Clean Name'].str.replace(r',', '')
lucky_comps['Clean Name'] =  lucky_comps['Clean Name'].str.replace(r'*', '')
lucky_comps['Clean Name'] =  lucky_comps['Clean Name'].str.replace(r'\'', '')
lucky_comps['Clean Name'] =  lucky_comps['Clean Name'].str.replace(r'&', '')
lucky_comps['Clean Name'] =  lucky_comps['Clean Name'].str.replace(r'!', '')
lucky_comps['Clean Name'] =  lucky_comps['Clean Name'].str.replace(r'[Cc]orp\n', '')
lucky_comps['Clean Name'] =  lucky_comps['Clean Name'].str.replace(r'[Cc]orp$', '')
lucky_comps['Clean Name'] =  lucky_comps['Clean Name'].str.replace(r'[Cc]orporation\n', '')
lucky_comps['Clean Name'] =  lucky_comps['Clean Name'].str.replace(r'[Cc]orporation$', '')
lucky_comps['Clean Name'] =  lucky_comps['Clean Name'].str.replace(r'[Ii]nc', '')
lucky_comps['Clean Name'] =  lucky_comps['Clean Name'].str.replace(r'(\([A-z]*\))', '')

lucky_comps['Clean Name'] =  lucky_comps['Clean Name'].str.lower()

print(lucky_comps.head())

lucky_comps.iloc[492]
lucky_comps['Clean Name'].to_csv("__DATA/SP_500_Names.csv", index=False, header=True, sep=',')


# ==================================== NN Preparation ============================================

lucky_comps = pd.read_csv("__DATA/SP_500_Names.csv")
print(lucky_comps.head())
print(len(lucky_comps))

vocabulary = sorted(get_vocabulary(lucky_comps['Clean Name']))

print(len(vocabulary)) # was 62 become 33

longest_name = get_max_len(lucky_comps['Clean Name']) 
print(longest_name) # 38

# Create the mapping of the vocabulary chars to integers
char_to_idx = { char : idx for idx, char in enumerate(vocabulary) }

# Create the mapping of the integers to vocabulary chars
idx_to_char = { idx : char for idx, char in enumerate(vocabulary)  }

# Print the dictionaries
print(char_to_idx)
print(idx_to_char)

# Initialize the input vector
input_data = np.zeros((len(lucky_comps['Clean Name']), longest_name+1, len(vocabulary)), dtype='float32')

# Initialize the target vector
target_data = np.zeros((len(lucky_comps['Clean Name']), longest_name+1, len(vocabulary)), dtype='float32')

# Iterate for each name in the dataset
for n_idx, name in enumerate(lucky_comps['Clean Name']):
  # Iterate over each character and convert it to a one-hot encoded vector
  for c_idx, char in enumerate(name):
    input_data[n_idx, c_idx, char_to_idx[char]] = 1

# Iterate for each name in the dataset
for n_idx, name in enumerate(lucky_comps['Clean Name']):
  # Iterate over each character and convert it to a one-hot encoded vector
  for c_idx, char in enumerate(name):
    target_data[n_idx, c_idx, char_to_idx[char]] = 1


# ===================================== PREDICTION USING SIMPLE RNN =========================================

# Create a Sequential model
model = Sequential()

# Network architecture has 50 simple RNN nodes in the first layer followed by a dense layer

# Increasing number of weights (neurons in each layer from 50 to 75 increases total model parameters twice)

# Formula for simple RNN: recurrent_weights + input_weights + biases or
# num_units* num_units + num_features* num_units + biases --> (num_features + num_units)* num_units + biases
# in our case: 80*80 + len(vocabulary)*80 + 80


# return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
model.add(SimpleRNN(80, input_shape=(longest_name+1, len(vocabulary)), return_sequences=True))# Try with False

# example of more layer
model.add(SimpleRNN(30, return_sequences=True))

# One more hidden layer
# Adding third layer doesn't increase the quality of the model
#model.add(SimpleRNN(50, return_sequences=True))

# Add a TimeDistributed Dense layer of size same as the vocabulary
# This wrapper allows to apply a layer to every temporal slice of an input.
model.add(TimeDistributed(Dense(len(vocabulary), activation='softmax')))

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer='adam')

# Print the model summary
model.summary()


# (505, 39, 33) --> (number of samples in the dataset, longest name + 1, vocabulary size)
print(input_data.shape)
print(target_data.shape)

# batch_size = 1 --> online learning, updating weights after passing through whole dataset --> much slower learning 
# Despite adding two more hidden layer time of training had not changed significantly
model.fit(input_data, target_data, batch_size=256, epochs=20)


# Function to generate company names
def generate_company_names(n):
    
    # Repeat for each name to be generated
    for i in range(0,n):

        # Flag to indicate when to stop generating characters
        stop=False

	# Number of characters generated so far
        counter=1

	# Define a zero vector to contain the output sequence
        output_seq = np.zeros((1, longest_name+1, len(vocabulary)))

        # Initialize the first character of output sequence as the start token
        output_seq[0, 0, char_to_idx['\t']] = 1

	# Variable to contain the name
        name = ''

        # Repeat until the end token is generated or we get the maximum no of characters
        while stop == False and counter < 10:

            # Get probabilities for the next character in sequence
            probs = model.predict_proba(output_seq, verbose=0)[:,counter-1,:]
            
            
            # Sample the vocabulary according to the probability distribution
            c = np.random.choice(sorted(list(vocabulary)), replace=False, p=probs.reshape(len(vocabulary)))
            
            if c=='\n':
                # Stop if end token is encountered, else append to existing sequence
                stop=True
            else:
                # Append this character to the name generated so far
                name = name + c

                # Append this character to existing sequence for prediction of next characters
                output_seq[0,counter , char_to_idx[c]] = 1.
                
                # Increment the number of characters generated
                counter=counter+1

        # Output generated sequence or name
        print(name)

generate_company_names(10)


# ========================================= Prediction using LSTM =============================================


# Create Sequential model 
model = Sequential()

# Add an LSTM layer of 128 units
model.add(LSTM(128, input_shape=(longest_name, len(vocabulary))))

# Add a Dense output layer
model.add(Dense(len(vocabulary), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Fit the model
model.fit(x, y, batch_size=64, epochs=1, validation_split=0.2)

def generate_text(sentence, n):
    """
    Function to generate text
    Inputs: seed sentence and number of characters to be generated.
    Output: returns nothing but prints the generated sequence.
    """
    
    # Initialize the generated sequence with the seed sentence
    generated = ''
    generated += sentence
    
    # Iterate for each character to be generated
    for i in range(n):
      
        # Create input vector from the input sentence
        x_pred = np.zeros((1, maxlen, len(vocabulary)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_to_idx[char]] = 1.

        # Get probability distribution for the next character
        preds = model.predict(x_pred, verbose=0)[0]
        
        # Get the index with maximum probability
        next_index = np.argmax(preds)
        next_char = idx_to_char[next_index]

        # Append the new character to the input sentence for next iteration
        sentence = sentence[1:] + next_char

        # Append the new character to the text generated so far
        generated += next_char
    
    # Print the generated text
    print(generated)






