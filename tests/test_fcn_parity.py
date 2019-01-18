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

import sys
sys.path.append('..')

import matplotlib as mpl
mpl.use('Agg')

import os
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pylab as pl
import time

#import fcn_sgd as fcn
from fcn import fcn2 as fcn

sep = ' '
    
def y_true( X ):
    y_true = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        y_true[i] = (sum(x for x in X[i,:] if x>0)%2-.5)*2
    return y_true
    
def parity_data( n_parity, mono=False ):    
    X = []
    y = []
    for line in open( '%i_bit_parity.txt' % n_parity ):
        data = [float(word) for word in line.split(sep)]
        X.append( data[:-1] )
        y.append( data[-1] )
    
    X = np.array(X)
    y = np.array(y)
    

    X = X.astype( np.float )
    y = y.astype( np.float )

    # y_t = y_true( X )
    # print( "y errors: ", abs(y-y_t) )

    y = y[:,np.newaxis]
    if (n_parity in [6]):
        y = -1.*y
        
    if mono:
        X = (X+1)/2
        y = (y + 1)/2
    
    XY = np.concatenate( [X,y], axis=1 )
    np.random.shuffle(XY)
    return XY[:,:-1],XY[:,-1:]
    

N_samples = 100
def parity_curve(estimator_class, params, verbose=0):
    parity_range = range(2,8)
    
    successes = []
    for n_parity in parity_range:
        if len(successes)>0 and successes[-1] == 0:
            successes.append(0.)
        else:
            tic=time.time()
            X,y = parity_data( n_parity )
            nnets = []
            for i_sample in range(N_samples):
                random_state = n_parity*N_samples + i_sample
                nnet = estimator_class(random_state=random_state,**params)
                nnet.fit(X,y)
                nnets.append(nnet)
            successes.append( sum( [nnet.converged_ for nnet in nnets]) / float(N_samples) )
            if verbose>0:
                toc=time.time()
                print( "     %2i parity: %8f (time %3i sec)" % (n_parity,successes[-1],toc-tic) )
    return parity_range, successes
    
if __name__ == "__main__":

    # for i in range(10):
        # random.seed(i)
        # X,y = parity_data( 2, mono=True )
        # nnet = fcn.FCNForward(
            # tol=1e-9,
            # n_extra_neurons=1,
            # max_iter=101,
            # out_activation='tanh_mon',
            # hidden_activation='tanh_mon',
            # verbose=2,
            # )
        # nnet.fit(X,y)
        # if nnet.converged_:
            # print( "Got one!" )
            # for c in [n.weights for n in nnet.neurons_]: print( c )
 
 
    fig,axis = pl.subplots(1,1,figsize=(6,3))
    for n_neurons in range(1,11):
        print( "%3i neuron networks" % n_neurons )
        prange, psuccess = parity_curve( 
            estimator_class = fcn.FCNForward,
            params = dict(
                n_extra_neurons = n_neurons-1, 
                max_iter = 201, 
                tol=1e-2, 
                verbose=0, 
                # mu0=1e-3,
                reg_param = 1e-5, 
                out_activation='tanh',
                hidden_activation='relu',
                # dropout=False, dropout_frac = .02
                ),
            verbose=1
            )
        axis.plot( prange, psuccess, label="%i neurons" % n_neurons )
    axis.legend(frameon=False)
    axis.set_xlabel( 'N-parity problem' )
    axis.set_ylabel( 'Fraction of successfully trained networks' )
    
    # fig.savefig( 'N-parity curve.png', dpi=600, bbox_inches='tight' )
 
    # for n_parity in range(2,8):
        
        # print( " %i parity problem:" % n_parity )
        # # X = np.array([
            # # [-1,-1,-1],
            # # [-1,-1, 1],
            # # [-1, 1,-1],
            # # [-1, 1, 1],
            # # [ 1,-1,-1],
            # # [ 1,-1, 1],
            # # [ 1, 1,-1],
            # # [ 1, 1, 1],
            # # ])
        # # y=np.array([-1,1,1,-1,1,-1,-1,1])

        # # X = (X+1.)/2.
        # # y = (y+1.)/2.
        # X,y = parity_data( n_parity )

        
        # # for dropout in [True,False]:
            # # print( "Dropout = ", repr(dropout) )
        # for i_seed in range(10):
            # random.seed(i_seed)
            # print( "    Attampt ",i_seed )
            # my_net = fcn.FCNForward( n_extra_neurons = n_parity-1, max_iter = 500, 
                                        # tol=1e-7, verbose=1, mu0=1e-3,
                                        # #reg_param = 1e-5, 
                                        # out_activation='tanh',
                                        # hidden_activation='tanh',
                                        # dropout=False, dropout_frac = .02
                                        # )
            # my_net.fit(X,y)
        
        # # random.seed(n_parity)
        # # my_net = fcn.FCNForward( n_extra_neurons = n_parity-1, max_iter = 400, 
                                    # # tol=1e-7, verbose=1, mu0=1e-3,
                                    # # reg_param = 1e-4, 
                                    # # activation='tanh',
                                    # # )
        # # my_net.fit(X,y)

        # print( "" )