'''
This is the optimizer module for Shift Invariant Sparse Coding with \
non-negative alpha. This module uses theano for easier GPU/Parallel processing.
-------------------------------------------------------------------------------
    Coded by Md. Iftekhar Tanveer (itanveer@cs.rochester.edu)
    Rochester Human-Computer Interaction (ROCHCI)
    University of Rochester
-------------------------------------------------------------------------------
'''
import fileio as fio
import matplotlib.pyplot as pp

import numpy as np
import theano as th
import theano.tensor as T
from theano import function
from theano.tensor.signal.conv import conv2d


# Plot the original alpha, psi and X
def dispOriginal(alpha,psi):
    pp.figure('Original Alpha and Psi')
    pp.clf()
    _,D = np.shape(alpha)
    for d in xrange(D):
        pp.subplot(D,2,2*d+1)
        pp.plot(psi[:,:,d])
        pp.title('psi')
        pp.subplot(D,2,2*d+2)
        pp.plot(alpha[:,d])
        pp.title('alpha')
        pp.draw()
        pp.pause(1)
  

def main():
    # Get alpha, psi and put in shared variable
    # psi is M x K x D (axis-1: x, y and z components of AEB)
    # alpha is N x D (axis-0:time/frame, axis-1: various AEB)
    #alpha_np,psi_np = fio.toyExample_medium_3d_multicomp()
    alpha_np = [[0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,0]]
    psi_np = [[[1,2,3],[4,5,6]],[[3,2,1],[6,5,4]]]
#    alpha = th.shared(value=alpha_np,name='alpha',borrow=True)
#    psi = th.shared(value=psi_np,name='psi',borrow=True)
    alpha = T.ftensor3('alpha')
    psi = T.fmatrix('psi')

    # Build a convolution expression
    convexpr = conv2d(psi_,alpha_,border_mode='full')
    consig,upd = th.scan(fn=convexpr,sequences=[psi,alpha])
    lossf = function([psi,alpha],outputs=convexpr)
    
    
    print lossf(psi_np,alpha_np)
    


if __name__=='__main__':
    main()

