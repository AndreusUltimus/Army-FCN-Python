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
import pylab as pl
from fcn import fcn


def surface( X, Y ):

    Z = np.zeros(X.shape)-1.
    conditional1 = np.logical_and( (X>3),(Y>3) )
    Z[conditional1] = 1.
    
    conditional2 = np.logical_and( (X<-3), (Y<-3) )
    Z[conditional2] = 0.
    return Z
    

def plot_extra( npts, fignum, my_net ):
    [X2,Y2] = np.meshgrid(
        np.linspace(-5,5,npts),
        np.linspace(-5,5,npts)
        )
        
    X = X2.reshape([-1])
    Y = Y2.reshape([-1])

    Z2 = surface(X2,Y2)
    Z = Z2.reshape([-1])
    
    XY = np.concatenate( 
        [X[:,np.newaxis], Y[:,np.newaxis]],
        axis=1,
        )
    Zp = my_net.predict( XY )
    Zp2 = Zp.reshape( [npts,npts] )
    
    pl.figure(fignum)


    pl.subplot(1,2,1)
    pl.contourf( X2,Y2,Z2,20 )
    pl.subplot(1,2,2)
    pl.contourf( X2,Y2,Zp2,20 )

    pl.colorbar()

    
    
npts = 31
  
[X2,Y2] = np.meshgrid(
    np.linspace(-5,5,npts),
    np.linspace(-5,5,npts)
    )
    
X = X2.reshape([-1])
Y = Y2.reshape([-1])

Z2 = surface(X2,Y2)
Z = Z2.reshape([-1])


XYZ = np.concatenate( 
    [X[:,np.newaxis], Y[:,np.newaxis], Z[:,np.newaxis]],
    axis=1,
    )
np.random.shuffle( XYZ )
XY = XYZ[:,:2]
Z = XYZ[:,2:]
    

for i_net in range(5):
    my_net = fcn.FCNForward( 
        n_extra_neurons = 9, 
        max_iter = 201, 
        tol=1e-3, 
        verbose=2, 
        # mu0=1e2,
        # reg_param = 1e-5, 
        hidden_activation='relu',
        out_activation='tanh',
        # epsilon = .1,,
        random_state=i_net,
        )

    my_net.fit(XY,Z)
    # Zp = my_net.predict( XY )
    # Zp2 = Zp.reshape( [npts,npts] )

    # print [n.activation_name for n in my_net.neurons_]




    # pl.figure(i_net)


    # pl.subplot(1,2,1)
    # pl.contourf( X2,Y2,Z2,20 )
    # pl.subplot(1,2,2)
    # pl.contourf( X2,Y2,Zp2,20 )

    # pl.colorbar()

    plot_extra( npts*2, i_net, my_net )
