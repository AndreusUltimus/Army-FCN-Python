# ------------------------------------------------------------
#   FCN Python
#   Fully Connected Neural Network (FCN) Python Implementation
#   Author: Andrew Wilson
#   
#   This code is the product of a U.S. Government employee
#   within the scope of their employment.
#   
#   Under 17 U.S.C. 105, the United States Government does not
#   hold any copyright in works produced by U.S. Government 
#   employees within the scope of their employment.
#
#   Distribution Statement A: 
#      For Public Release per AMRDEC PAO (20190023)
# ------------------------------------------------------------

import scipy.special as spy
from sklearn.utils import check_random_state
import numpy as np


def sigmoid( z ):
    return spy.expit(z)
def sigmoid_slope( z ):
    return spy.expit(z)*(1-spy.expit(z))
    
def sigmoid_bip( z ):
    return sigmoid(z)*2.-1.
def sigmoid_bip_slope( z ):
    return sigmoid_slope(z)*2.

def tanh( z ):
    return np.tanh(z)
def tanh_slope( z ):
    return 1.-np.tanh(z)**2

def tanh_mon( z ):
    return np.tanh(z)/2.+.5
def tanh_mon_slope( z ):
    return tanh_slope(z)/2.
    
def gaussian( z ):
    return np.exp( - z*z )
def gaussian_slope( z ):
    return -2.*z*np.exp( -z*z )
    
def relu( z ):
    return np.maximum( 0., z )
def relu_slope( z ):
    return (np.sign(z)+1)/2.
    # if z > 0:
        # return 1.
    # else: # z < 0
        # return 0.
    
Activations = {
    'linear' : ( lambda x:x, lambda x:1. ),
    'sigmoid' : (sigmoid,sigmoid_slope),
    'sigmoid_bip' : (sigmoid_bip,sigmoid_bip_slope),
    'tanh' : (np.tanh,tanh_slope),
    'tanh_mon' : (tanh_mon,tanh_mon_slope),
    'gauss' : (gaussian,gaussian_slope),
    'relu' : (relu,relu_slope),
    'sin' : (np.sin,np.cos),
    }

valid_activations_for_random = [
    # 'sigmoid',
    # 'sigmoid_bip',
    'tanh',
    'tanh_mon',
    'gauss',
    'relu',
    ]

def get_activation( name, random_state ):
    if name=='random':
        return random_state.choice(valid_activations_for_random)
    else:
        return name
    
MU_FACTOR = 10.
MAX_MU = 1e7
MIN_MU = 1e-10
    
# MU_FACTOR = 2
# MAX_MU = 1e3
# MIN_MU = 1e-3


class Neuron:
    def __init__( self, N_in = 1, activation_type = 'tanh_mon',epsilon=1.,random_state=None):
        self.activation_name = activation_type
        self.activation, self.activation_slope = Activations[activation_type]
        self.N_in = N_in
        # rand_func = random.rand # Standard normal
        random_state = check_random_state( random_state )
        rand_func = random_state.random_sample # Uniform [0,1)
        self.weights = (rand_func(self.N_in)*2. - 1.)*epsilon
        self.in_range = range(self.N_in)
        
class StaticNeuron:
    def __init__( self, weights, activation_type='tanh_mon' ):
        self.activation_name = activation_type
        self.activation, self.activation_slope = Activations[activation_type]
        self.N_in = len(weights)
        self.weights = weights
        self.in_range = range(self.N_in)
