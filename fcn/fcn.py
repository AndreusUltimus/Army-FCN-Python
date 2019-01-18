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

import numpy as np
import sklearn
import sklearn.metrics
from sklearn.utils import check_random_state

from .neurons import *


class FCNForward( sklearn.base.BaseEstimator ):
    def __init__( self, 
            n_extra_neurons=0, 
            hidden_activation='random',
            out_activation='tanh',
            mu0=1e1,
            max_iter=1000,
            tol=1e-4,
            verbose=0,
            epsilon=.3,
            reg_param=0.,
            class_weight = None,
            dropout=False,
            dropout_frac=0.1,
            random_state=None,
            ):
        self.n_extra_neurons = n_extra_neurons
        self.mu0 = mu0
        self.max_iter = max_iter
        self.hidden_activation = hidden_activation
        self.out_activation = out_activation
        self.tol=tol
        self.verbose=verbose
        self.epsilon=epsilon
        self.reg_param = reg_param
        self.class_weight = class_weight
        if self.class_weight == 'auto':
            print( "'auto' class weight depricated; use 'balanced'.")
            self.class_weight = 'balanced'
        self.dropout = dropout
        self.dropout_frac = dropout_frac
        self.random = check_random_state( random_state )
        
    def predict(self,X):
        
        N_p = X.shape[0]
        N_i = X.shape[1]
        N_o = self.n_outputs_
        neurons = self.neurons_
        N_n = len(neurons)
        Y = np.zeros( [1+N_i+N_n] )
        Y[0] = 1.
        
        y = np.zeros( [N_p,N_o] )

        for i_p in range(N_p):
            Y[1:1+N_i] = X[i_p,:]  # Inputs for this pattern
            for i_n in range(N_n):
                netval = np.dot( Y[:neurons[i_n].N_in], neurons[i_n].weights )
                Y[1+N_i+i_n] = neurons[i_n].activation( netval )
                # print( i_n, netval, Y[1+N_i+i_n], neurons[i_n].activation)
            y[i_p,:] = Y[-N_o:]
            # print( y)
            # print( Y)
            
        
        
        
        # for i_p in range(N_p):
            # Y[1:1+N_i] = X[i_p,:]  # Inputs for this pattern
            # for i_n in range(N_n):
                # neuron = neurons[i_n]
                # netval = np.dot( Y[:neuron.N_in], neuron.weights )
                # Y[1+N_i+i_n] = neuron.activation( netval )
                
            # y[i_p,:] = Y[-N_o:]
        # print( 'a', y)
        return y
        
    def fit( self, X, y ):
        if len(y.shape)==1:
            y = y[:,np.newaxis]
        assert y.shape[1] == 1, "Only one output variable supported for now, due to class weights scheme."
        assert y.shape[1] == 1, "Only one output variable supported for now, n_o assumed to be 1."
        
        N_p = X.shape[0]
        # TODO: Check for single-variable input/output
        N_i = X.shape[1]
        N_o = y.shape[1]
        N_n = N_o + self.n_extra_neurons

        classes = list(set(list(y[:,0])))
        classes.sort()
        N_classes = len(classes)

        if self.class_weight == None:
            cweights = np.ones(y.shape)
        else:
            if self.class_weight == 'balanced':
                class_weight = {}
                for i_class in range(N_classes):
                    class_count = sum( y==classes[i_class] )
                    class_weight[ classes[i_class] ] = 1./class_count
                normalizer = max( class_weight.values() )
                for k in class_weight.keys():
                    class_weight[k] /= normalizer
            else:
                class_weight = self.class_weight
                for c in classes:
                    assert (c in class_weight.keys())
                for k in class_weight.keys():
                    assert c in classes
            cweights = np.ones(y.shape)
            for i_p in range(N_p):
                cweights[i_p] = class_weight[ y[i_p,0] ]
        # print( cweights)
        
        # First, the hidden neurons
        neurons = [
            Neuron(N_in = N_i + 1 + n,activation_type=get_activation(self.hidden_activation,self.random),epsilon=self.epsilon)
            for n in range(self.n_extra_neurons)
            ]
        # Second, the output neurons
        neurons.extend( [ 
            Neuron(N_in = N_i + 1 + n,activation_type=self.out_activation,epsilon=self.epsilon)
            for n in range(self.n_extra_neurons,N_n)
            ])
        
        self.neurons_ = neurons
        self.n_outputs_ = N_o

        N_weights = sum( [neuron.N_in for neuron in neurons] )
        
        H = np.zeros( [N_weights,N_weights] )
        g = np.zeros( [N_weights] )
        
        # Y is Activation values of bias, inputs, hidden neurons, and output neurons, in that order.
        Y = np.zeros( [1+N_i+N_n] )
        Y[0] = 1.
        
        delta_w = np.zeros([N_n,N_n])
        # J = np.zeros([N_weights,N_o*N_p])
        # e = np.zeros([N_o*N_p])
        J = np.zeros([N_weights,N_o])
        e = np.zeros([N_o])
        
        errs = np.zeros([self.max_iter])
        mu = self.mu0
        E_new = 0.
        
        theEye = np.eye(N_weights)
        
        # times = [0.]*12
        # import time
            
        self.converged_ = False
        
        pattern_range = range(N_p)
        neuron_range = range(N_n)
        output_range = range(N_o)
        
        for i_iter in range(self.max_iter):
            E_total = 0.
            H[:,:] = 0.
            g[:] = 0.
            
            if self.dropout:
                # TODO: probabilities (e.g., p(0) = .2, p(1) = .8)
                # TODO: ? set up a chooser object for performance?
                mask = self.random.choice(
                    [0,1],N_weights,
                    p = [self.dropout_frac,1-self.dropout_frac]
                    )
                mask[0] = 1
                # print( "Keeping %i of %i weights" % (np.sum(mask),len(mask)))
            
            # tic = time.time(); i_t=0
            for i_p in pattern_range:
                # tic = time.time(); i_t=0

                # The upper triangle of the delta_w matrix is weights of the 
                # neuron-neuron connections (no input weights, no bias weights).
                for i_n in range(1,N_n):
                    # This *column* are the weights connecting earlier neurons to this neuron 
                    weights = neurons[i_n].weights
                    if self.dropout:
                        # print( weights)
                        # print( mask)
                        weights = weights * mask[:neurons[i_n].N_in]
                        # print( weights)
                        # assert False
                    delta_w[:i_n,i_n] = weights[N_i+1:]
                    
                #toc=time.time(); times[i_t] += (toc-tic); tic=toc; i_t += 1; # 0
                
                # The network values.
                # Y[0] = 1.  # Bias.
                Y[1:1+N_i] = X[i_p,:]  # Inputs for this pattern
                #Y[1+N_i:] = np.nan  # Neuron outputs to be computed
                #toc=time.time(); times[i_t] += (toc-tic); tic=toc;i_t += 1; # 1
                
                for i_n in neuron_range:
                    neuron = neurons[i_n]
                    weights = neuron.weights
                    if self.dropout:
                        weights = weights * mask[:neuron.N_in]
                    netval = weights.dot( Y[:neuron.N_in] )
                    Y[1+N_i+i_n] = neuron.activation( netval )
                    # print( i_n, netval, Y[1+N_i+i_n])
                    s = neuron.activation_slope( netval )
                    delta_w[i_n,i_n] = s
                    for j_n in range(i_n):
                        #=================================================
                        # Begin choice.
                        #-------------------------------------------------
                        # accum = 0.
                        # for i_loop in range( j_n,i_n ):
                            # accum += delta_w[i_loop,i_n]*delta_w[i_loop,j_n]
                        # delta_w[i_n,j_n] = s*accum
                        #-------------------------------------------------
                        delta_w[i_n,j_n] = s*sum([
                            delta_w[i_loop,i_n]*delta_w[i_loop,j_n]
                            for i_loop in range( j_n,i_n )
                            ])
                        #-------------------------------------------------
                        #delta_w[i_n,j_n] = s*np.dot( delta_w[j_n:i_n,i_n], delta_w[j_n:i_n,j_n] )
                        #-------------------------------------------------
                        # End choice.
                        #=================================================
                #toc=time.time(); times[i_t] += (toc-tic); tic=toc;i_t += 1; # 2
                    
                i_w = -1
                delta_w_out = delta_w[-N_o:,:]
                for i_n in neuron_range:
                    for i_in in neurons[i_n].in_range:
                        i_w += 1
                        #=================================================
                        # Begin choice.
                        #-------------------------------------------------
                        # TODO: Abuses only one output
                        J[i_w,0] = Y[i_in]*delta_w_out[0,i_n]
                        #-------------------------------------------------
                        # # J[i_w,:] = Y[i_in]*delta_w_out[:,i_n]
                        #-------------------------------------------------
                        # for i_o in output_range:
                            # # import pdb; pdb.set_trace()
                            # # J[i_w,i_p*N_o + i_o] = Y[i_in]*delta_w_out[i_o,i_n]
                            # J[i_w,i_o] = Y[i_in]*delta_w_out[i_o,i_n]
                        #-------------------------------------------------
                        # End choice.
                        #=================================================
                #toc=time.time(); times[i_t] += (toc-tic); tic=toc;i_t += 1; # 3

                # My J is the traditional J.T
                H += J.dot(J.T)
                
                #=================================================
                # Begin choice.
                #-------------------------------------------------
                #e = Y[-N_o:] - y[i_p,:] 
                #-------------------------------------------------
                # TODO: Abuses only one output.
                # e = Y[-1] - y[i_p,0]
                #-------------------------------------------------
                # e_weighted = e[:]
                #-------------------------------------------------
                # for i_o in range(N_o):
                    # e_weighted *= cweights[ y[i_p,i_o] ]
                # TODO: Abuses only one output.
                #-------------------------------------------------
                e_weighted = ( Y[-1] - y[i_p,0])*cweights[i_p]
                #-------------------------------------------------
                # End choice.
                #=================================================

                #toc=time.time(); times[i_t] += (toc-tic); tic=toc;i_t += 1; # 4

                #=================================================
                # Begin choice.
                #-------------------------------------------------
                # My J is the traditional J.T
                g += J.dot(e_weighted)
                #-------------------------------------------------
                # i_w = -1
                # for i_n in range(N_n):
                    # neuron = neurons[i_n]
                    # for i_in in range(neuron.N_in):
                        # i_w += 1
                        # #g[i_w] += Y[i_in] * np.dot( delta_w_out[:,i_n], e )
                        # for i_o in range(N_o):
                            # g[i_w] += Y[i_in] * delta_w_out[i_o,i_n] * e[i_o]
                #-------------------------------------------------
                # End choice.
                #=================================================
                
                #=================================================
                # Begin choice.
                #-------------------------------------------------
                # TODO: Abuses only one output
                E_total += e_weighted[0]*e_weighted[0]
                #-------------------------------------------------
                # for i_o in output_range:
                    # E_total += e_weighted[i_o]**2
                #-------------------------------------------------
                #E_total += e.dot(e)
                #-------------------------------------------------
                # End choice.
                #=================================================
            # -------------------------------------------------------------------------
                #toc=time.time(); times[i_t] += (toc-tic); tic=toc;i_t += 1; # 5

            # tic = time.time(); i_t=5

            for neuron in neurons:
                neuron.old_weights = neuron.weights.copy()

            # mu = self.mu0
            for m in range(5):
                matr = (H + mu*theEye)
                matr = np.linalg.inv( matr )
                # print( matr.shape, H.shape, matr.min(), matr.max())
                matr = matr.dot( g )
                if self.reg_param > 0.:
                    i_w = -1
                    for i_n in neuron_range:
                        neuron = neurons[i_n]
                        for i_in in neuron.in_range:
                            if i_in > 0:
                                toAdd = self.reg_param/mu * neuron.old_weights[i_n]
                                if self.dropout:
                                    toAdd *= mask[i_w]
                                matr[i_w] += toAdd
                            i_w += 1
                        #matr[i_w+1:i_w+1+neuron.N_in-1] += self.reg_param * neuron.weights[1:]
                        #i_w += neuron.N_in
                # print( matr.shape, H.shape, matr.min(), matr.max())
                
                i_w = -1
                for i_n in neuron_range:
                    neuron = neurons[i_n]
                    for i_in in range(neuron.N_in):
                        i_w += 1
                        # if (not self.dropout) or (mask[i_w]):
                        neuron.weights[i_in] = neuron.old_weights[i_in] - matr[i_w]
                        

                E_new = 0.
                for i_p in pattern_range:
                    Y[1:1+N_i] = X[i_p,:]  # Inputs for this pattern
                    for i_n in range(N_n):
                        weights = neurons[i_n].weights
                        if self.dropout:
                            weights = weights * mask[:neurons[i_n].N_in]
                        netval = np.dot( Y[:neurons[i_n].N_in], weights )
                        
                        Y[1+N_i+i_n] = neurons[i_n].activation( netval )
                    e_mini = Y[-N_o:] - y[i_p,:] 
                    # E_new += e.dot(e)
                    for i_o in range(N_o):
                        E_new += (e_mini[i_o]*cweights[i_p])**2
            

                if E_new <= E_total:
                    mu /= MU_FACTOR
                    mu = max(MIN_MU,mu)
                    break
                elif mu >= MAX_MU:
                    break
                else:
                    mu *= MU_FACTOR
                    mu = min(MAX_MU,mu)
            #toc=time.time(); times[i_t] += (toc-tic); tic=toc; i_t += 1; # 5
   
            if (i_iter%20==0) and (self.verbose>=2):
                print( "%6i %e %6.1f %6.1f H %6.1f +/- %.2e" % (i_iter, mu, E_new, E_total, H.mean(), H.std()))
                
            
                    
            errs[i_iter] = E_total
            N_err_steps=10
            
            if E_total <= self.tol and E_new <= E_total:
                if (self.verbose>=1):
                    print( "Tolerance reached.",)
                    print( "%6i %e %8f %8f" % (i_iter, mu, E_new, E_total))
                self.converged_ = True
                break
            elif i_iter>N_err_steps:
                old_errs = errs[ i_iter-N_err_steps  : i_iter   ]
                new_errs = errs[ i_iter-N_err_steps+1: i_iter+1 ]
                if max(abs(old_errs-new_errs)) < self.tol*1e-3:
                    if (self.verbose>=1):
                        print( "Error not changing.",)
                        print( "%6i %e %8f %8f" % (i_iter, mu, E_new, E_total))
                    self.converged_ = False
                    break
        
        if E_total > self.tol:
            if (self.verbose>=1):
                print( "Tolerance NOT reached.",)
                print( "%6i %e %8f %8f" % (i_iter, mu, E_new, E_total))
            
        self.neurons_ = neurons
        self.n_outputs_ = N_o
        self.errors_ = errs[:i_iter+1]
        
        # for i_t in range(len(times)):
            # print( " %3i bin, time: %8.2f sec" % (i_t,times[i_t]))
            
        return self


def np_inv( *args, **kwargs ):
    return np.linalg.inv( *args, **kwargs )
            
class FCNForward_Classify( FCNForward ):
    def __init__(self, 
            n_extra_neurons=0, 
            hidden_activation='random',
            out_activation='tanh',
            mu0=1e1,
            max_iter=1000,
            tol=1e-4,
            verbose=0,
            epsilon=.3,
            reg_param=0.,
            class_weight = None,
            dropout=False,
            dropout_frac=0.1,
            adjust_threshold=False,
            random_state=None,
            ):
        FCNForward.__init__(self,
            n_extra_neurons, 
            hidden_activation,
            out_activation,
            mu0,
            max_iter,
            tol,
            verbose,
            epsilon,
            reg_param,
            class_weight,
            dropout,
            dropout_frac,
            random_state,
            )
        self.adjust_threshold = adjust_threshold
        
    def fit(self,X,y):
        # print( 'fit')
        FCNForward.fit(self,X,y)
        if self.adjust_threshold:
            scores = self.decision_function(X)
            fpr, tpr, threshes = sklearn.metrics.roc_curve(y, scores, pos_label=1)
            f1s = [(sklearn.metrics.f1_score(y, scores>thresh),thresh) for thresh in threshes]
            self.thresh = max(f1s)[1]
            if self.thresh > 0:
                self.thresh *= .99
            elif self.thresh < 0:
                self.thresh *= 1.01
            # for s,t in f1s: print( "      f1 %8f thresh %8f" % (s,t))
            # print( max(f1s), self.thresh)
        else:
            self.thresh = (y.max()-y.min())/2.
        return self
            
        
    def score(self,X,y):
        # print( 'score')
        y_p = self.predict(X)
        return sklearn.metrics.accuracy_score(y,y_p)
        #return (1.-abs(y_p-y)).mean()
    
    def predict( self, X ):
        # print( 'predict')
        # print( "here, thresh:", self.thresh)
        y = FCNForward.predict(self, X )
        # print( y.min(), y.max())
        return (y>self.thresh).astype(np.int)
    def decision_function( self, X ):
        # print( 'decision')
        return FCNForward.predict(self, X )
    
    
    
class FCNBestInClass_Classify( sklearn.base.BaseEstimator ):
    def __init__( self, n_networks=5, score='f1', 
            n_extra_neurons=0, 
            hidden_activation='gauss',
            out_activation='tanh',
            mu0=1e1,
            max_iter=1000,
            tol=1e-4,
            verbose=0,
            epsilon=.3,
            reg_param=0.,
            class_weight = None,
            dropout=False,
            dropout_frac=0.1,
            adjust_threshold=False):
        self.n_networks = n_networks
        self.score = score

        self.n_extra_neurons = n_extra_neurons
        self.mu0 = mu0
        self.max_iter = max_iter
        self.hidden_activation = hidden_activation
        self.out_activation = out_activation
        self.tol=tol
        self.verbose=verbose
        self.epsilon=epsilon
        self.reg_param = reg_param
        self.class_weight = class_weight
        self.dropout = dropout
        self.dropout_frac = dropout_frac
        self.adjust_threshold = adjust_threshold
    
    def fit( self, X, y ):
        #print( 'M fit', X.shape)
        self.networks_ = [
            FCNForward_Classify( 
                n_extra_neurons = self.n_extra_neurons,
                mu0 = self.mu0,
                max_iter = self.max_iter,
                hidden_activation = self.hidden_activation,
                out_activation = self.out_activation,
                tol = self.tol,
                verbose = self.verbose,
                epsilon = self.epsilon,
                reg_param = self.reg_param,
                class_weight = self.class_weight,
                dropout = self.dropout,
                dropout_frac = self.dropout_frac,
                adjust_threshold = self.adjust_threshold,
                )
            for i in range( self.n_networks )
            ]
        self.best_score_ = 0.
        self.best_net_ = self.networks_[0]
        
        if self.score == 'f1':
            scorer = sklearn.metrics.f1_score
        elif self.score == 'accuracy':
            scorer = sklearn.metrics.accuracy_score
        else:
            raise NotImplementedError( "Only F1 or accuracy scoring is supported right now." )
        
        for net in self.networks_:
            net.fit( X, y )
            y_p = net.predict( X )
            score = scorer( y, y_p )
            # print( "scoring nets...",score, self.best_score_)
            if score > self.best_score_:
                self.best_score_ = score
                self.best_net_ = net
        return self
    def predict( self, X ):
        #print( 'M predict', X.shape)
        return self.best_net_.predict( X )
    def score(self,X,y):
        print( 'M score')
        y_p = self.predict(X)
        return sklearn.metrics.accuracy_score(y,y_p)
        #return (1.-abs(y_p-y)).mean()
    def decision_function( self, X ):
        #print( 'M decision_function')
        return self.best_net_.decision_function( X )
