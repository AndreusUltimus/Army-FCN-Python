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

from neurons import *
import numpy as np

    
def fcn_feedforward( network, inputs ):
    
    # A valid FCN network has a particular structure; check it.
    for i_n in range(len(network)-1):
        assert network[i_n+1].N_in == network[i_n].N_in + 1

    X = inputs
    N_p = X.shape[0]
    N_i = X.shape[1]

    # The number of weights of the first neuron should equal the
    # number of input variables, plus one for the bias term.
    assert len(network[0].weights) == N_i+1
    
    # This is restricted to networks with one output.
    N_o = 1
    N_n = len(network)
    Y = np.zeros( [1+N_i+N_n] )
    Y[0] = 1. 
        
    y = np.zeros( [N_p,N_o] )

    for i_p in range(N_p):
        Y[1:1+N_i] = X[i_p,:]  # Inputs for this pattern
        for i_n in range(N_n):
            netval = np.dot( Y[:network[i_n].N_in], network[i_n].weights )
            Y[1+N_i+i_n] = network[i_n].activation( netval )
            # print( i_n, netval, Y[1+N_i+i_n], network[i_n].activation )
        y[i_p,:] = Y[-N_o:]
            
    return y
    

    
if __name__ == "__main__":
    a_network = [
        StaticNeuron( [.2,.3], 'tanh_mon' ),
        StaticNeuron( [-.1,.15,.4], 'tanh_mon' ),
        StaticNeuron( [.05,.04,.03,.5], 'tanh_mon' ),
        StaticNeuron( [.6,.8,.04,-.8,.6], 'tanh_mon' ),
        StaticNeuron( [-.2,0.,.2,0.,.15,.7], 'tanh_mon' ),
        ]

    inputs = np.arange( -4,1,.2 )
    inputs = inputs[:,np.newaxis ]
    
    outputs = fcn_feedforward( a_network, inputs )
    