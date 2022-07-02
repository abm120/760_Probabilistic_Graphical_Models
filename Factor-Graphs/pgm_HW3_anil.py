""" 
ECEN 760 PGM Project - Homework 3
Author: @AnilBMurthy
Software used: Python Canopy Environment
"""
import numpy as np
import string
import random

u = '1101001100' # information bits
x = ''           # encoded bits
print('\nThe given information bits are: \t')
print(u)
"""
Approach #1: 
#Encoding acc to FSM description
state = 0;
for i in range(len(u)):
    if(u[i]=='1'):
        if(state == 0):
            output = '111'
            state = 1
        elif(state == 1):
            output = '010'
            state = 3
        elif(state == 2):
            output ='100'
            state = 1
        elif(state == 3):
            output = '001'
            state = 3
        else:
            print('Invalid state !!')
    elif(u[i] == '0'):
        if(state == 0):
            output = '000'
            state = 0
        elif(state == 1):
            output = '101'
            state = 2
        elif(state == 2):
            output ='011'
            state = 0
        elif(state == 3):
            output = '110'
            state = 2
        else:
            print('Invalid state !!')
    else:
        print('Invalid information bit !!')
    x = x + output
    
print('\n The encoded bits are: \t')
print(x)

# BSC(p) channel Modelling

p = 2 #BSC(p) - Probability of bit error in the Binary Symmetric Channel

flip_positions = [2,8,12,17,21,28]  # bit positions which are flipped and erraneous due to noisy channel - Trying Particular bits - White Box Testing
y = x
for i in range(len(x)):
    if(i in flip_positions):
        if(x[i] == '1'):
            y = y[:i] + '0' + x[i+1:]
        elif(x[i] == '0'):
            y = y[:i] + '1' + x[i+1:]
    else:
        continue

print('The received vector (noisy version of x) is:\t')
print(y)
"""
def encode(input_message):
    input_message = map(int,input_message)
    state =(0,0)
    output = []
    for itr in input_message:
        output.append(output_map[state][itr])
        state = state_map[state][itr]
    print 'The encoded bits are : ' ,output
    return output
    
def bit_flipping(codeword):
    codeword = [i[j] for i in codeword for j in range(len(i))]
    sample = random.sample(range(len(codeword)),6)
    for item in sample:
        if codeword[item] == 1:
            codeword[item] = 0
        else:
            codeword[item] = 1
    print 'The received vector (noisy version of x) is: ',codeword
    print 'Flip positions :',sample
    return codeword

def gamma(output_channel,p):
    gamma_values = []
    a = []
    probable_y = ['000','001','010','011','100','101','110','111']
    for bit in probable_y:
        bit = map(int,str(bit))
        a.append(bit)
    sep = lambda output_channel, size: [output_channel[i:i+size] for i in range(0, len(output_channel), size)]
    sets = sep(output_channel,3)
    for item in sets:
        for sample in a:
            prob = 1
            for itr1,itr2 in zip(item,sample):
                if itr1 == itr2:
                    prob = prob * (1-p)
                else:
                    prob = prob * p
            gamma_values.append(prob)
    gamma_values = np.reshape(np.array(gamma_values),(len(sets),len(probable_y)))
    return gamma_values
                                    

state_map = {(0, 0): ((0,0), (1,0)),
    (1, 0): ((0,1), (1,1)),
    (0, 1): ((0,0), (1,0)),
    (1, 1): ((0,1), (1,1))}

output_map = {(0, 0): ((0,0,0), (1,1,1)),
    (1, 0): ((1,0,1), (0,1,0)),
    (0, 1): ((0,1,1), (1,0,0)),
    (1, 1): ((1,1,0), (0,0,1))}

state_space = [(0,0),(1,0),(0,1),(1,1)]

output_space = [(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)]

v=encode(u)

alpha = np.zeros((11, 4))
beta = np.zeros((11, 4))
output_y = bit_flipping(v)
gamma = gamma(output_y,0.2)
y = np.array(output_y)
y=y.reshape((10,3))

alpha[0] = [1, 0, 0, 0]

for i in range(1,alpha.shape[0]):
    row_state = np.zeros(4)
    for j in range(alpha.shape[1]):
        message = 0
        next_state = state_space[j]
        for itr in state_space:
            if(next_state in state_map[itr]):
                u = state_map[itr].index(next_state)
                x = output_map[itr][u]
                alpha_previous = alpha[i-1][state_space.index(itr)]
                gamma_previous = gamma[i-1][output_space.index((x))]
                message = message + alpha_previous*gamma_previous
        row_state[j] = message
    alpha[i] = row_state
    

beta[10] = [1, 0, 0, 0]

for i in range(beta.shape[0]-2,-1,-1):
    row_state = np.zeros(4)
    for j in range(beta.shape[1]):
        message = 0
        previous_state = state_space[j]
        for itr in state_space:
            if(itr in state_map[previous_state]):
                u = state_map[previous_state].index(itr)
                x = output_map[previous_state][u]
                beta_next = beta[i+1][state_space.index(itr)]
                gamma_current = gamma[i][output_space.index((x))]
                message = message + beta_next*gamma_current
        row_state[j] = message
    beta[i] = row_state
    
U = np.zeros((10,2))
for i in range(10):
    u_0 = 0
    u_1 = 0
    message = 0
    for previous_state in state_space:
        for current_state in state_space:
            if(current_state in state_map[previous_state]):
                u = state_map[previous_state].index(current_state)
                x = output_map[previous_state][u]
                beta_current = beta[i+1][state_space.index(current_state)]
                alpha_previous = alpha[i][state_space.index(previous_state)]
                gamma_current = gamma[i][output_space.index((x))]
                message = beta_current*alpha_previous*gamma_current
                if (u == 0):
                    u_0 = u_0 + message
                else:
                    u_1 = u_1 + message
    U[i][0] = u_0
    U[i][1] = u_1

print("Probabilities of Ui's: \n")
for u in U:
    print (u/np.sum(u))[0] ,  (u/np.sum(u))[1]
    
print("Most likely sent codeword:\nU=")
message = []
for u in U:
    if((u/np.sum(u))[0]>=0.5):
        message.append(0)
    else:
        message.append(1)
print message


X = np.zeros((10, 8))
message = 0
for i in range(10):
    row_state = np.zeros(8)
    for previous_state in state_space:
        for current_state in state_space:
            if(current_state in state_map[previous_state]):
                u = state_map[previous_state].index(current_state)
                x = output_map[previous_state][u]
                beta_current = beta[i+1][state_space.index(current_state)]
                alpha_previous = alpha[i][state_space.index(previous_state)]
                message = beta_current*alpha_previous
                row_state[output_space.index((x))] = message
    X[i] = row_state                                                                                                                                                    
print("Probabilities of Xi's: \n")
print(X);

"""
Testing:
from utils import (
    product, argmax, element_wise_product, matrix_multiplication,
    vector_to_diagonal, vector_add, scalar_vector_product, inverse_matrix,
    weighted_sample_with_replacement, isclose, probability, normalize
)
from logic import extend

import random
from collections import defaultdict
from functools import reduce

class HiddenMarkovModel:
    #A Hidden markov model which takes Transition model and Sensor model as input
    ## transition model = Transition probability matrix 
    ## Sensor Model = Observation matrix or emission probability matrix
    def __init__(self, transition_model, sensor_model, prior=[0.5, 0.5]):
        self.transition_model = transition_model
        self.sensor_model = sensor_model
        self.prior = prior

    def sensor_dist(self, ev):
        if ev is True:
            return self.sensor_model[0]
        else:
            return self.sensor_model[1]


def forward(HMM, fv, ev):
    prediction = vector_add(scalar_vector_product(fv[0], HMM.transition_model[0]),
                            scalar_vector_product(fv[1], HMM.transition_model[1]))
    sensor_dist = HMM.sensor_dist(ev)

    return normalize(element_wise_product(sensor_dist, prediction))


def backward(HMM, b, ev):
    sensor_dist = HMM.sensor_dist(ev)
    prediction = element_wise_product(sensor_dist, b)

    return normalize(vector_add(scalar_vector_product(prediction[0], HMM.transition_model[0]),
                                scalar_vector_product(prediction[1], HMM.transition_model[1])))


def forward_backward(HMM, ev, prior):

    #Forward-Backward algorithm for smoothing. Computes posterior probabilities
    #of a sequence of states given a sequence of observations.
    t = len(ev)
    ev.insert(0, None)  

    fv = [[0.0, 0.0] for i in range(len(ev))]
    b = [1.0, 1.0]
    bv = [b]    
    sv = [[0, 0] for i in range(len(ev))]

    fv[0] = prior

    for i in range(1, t + 1):
        fv[i] = forward(HMM, fv[i - 1], ev[i])
    for i in range(t, -1, -1):
        sv[i - 1] = normalize(element_wise_product(fv[i], b))
        b = backward(HMM, b, ev[i])
        bv.append(b)

    sv = sv[::-1]

    return sv
"""

        
        

        
        
        



