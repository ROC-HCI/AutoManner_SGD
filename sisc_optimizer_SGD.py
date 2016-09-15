'''
This is the SGD optimizer module for Shift Invariant Sparse Coding.
It enforces l0 norm using projection and alternating stochastic gradient method
Besides the optimizer module, it also contains several helper functions.
These functions are mainly required for debuging and plotting the results.
-------------------------------------------------------------------------------
    Coded by Md. Iftekhar Tanveer (itanveer@cs.rochester.edu)
    Rochester Human-Computer Interaction (ROCHCI)
    University of Rochester
-------------------------------------------------------------------------------
'''
from functools import partial
from multiprocessing import Pool
from itertools import izip
import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as pp
import math
import time
import random
from heapq import nlargest

############################## Helper Functions ###############################
# Display the gradients
def dispGrads(gralpha,grpsi):
    _,D = np.shape(gralpha)    
    for d in xrange(D):    
        pp.figure('Plot of Gradients for component # '+'{:0}'.format(d))
        pp.clf()
        # Plot Gradient wrt psi
        pp.subplot(211)    
        pp.plot(grpsi[:,:,d])
        pp.title('Gradf wrt psi')
        # Plot Gradient wrt alpha 
        pp.subplot(212)        
        pp.plot(gralpha[:,d])
        pp.title('Gradf wrt alpha')
        pp.draw()
        pp.pause(1)
# Plots alpha, psi, original and reconstructed data
def dispPlots(alpha,psi,X,figureName,p):
    _,D = np.shape(alpha)
    for d in xrange(D):
        pp.figure(figureName + ' for component # '+'{:0}'.format(d))
        pp.clf()    
        pp.subplot(511)
        pp.plot(X)
        yrange = pp.ylim()        
        pp.title('Original Data')        
        pp.subplot(512)
        L = recon(alpha,psi,p)
        pp.plot(L)
        pp.ylim(yrange)
        pp.title('Reconstructed Data')    
        pp.subplot(513)
        pp.plot(L - X)
        pp.ylim(yrange)
        pp.title('Reconstructed - Original')        
        pp.subplot(514)
        pp.plot(psi[:,:,d])
        pp.title('psi')
        pp.subplot(515)
        pp.stem(alpha[:,d])
        pp.title('alpha')     
        pp.draw()
        pp.pause(0.3)
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
# Find the next power of 2
def nextpow2(i):
    # do not use numpy here, math is much faster for single values
    buf = math.ceil(math.log(i) / math.log(2))
    return int(math.pow(2, buf))
################### Functions for calculating objectives ######################
# Actual value of the objective function
def calcObjf(X,alpha,psi,p):
    N = np.size(alpha,axis=0)
    L = recon(alpha,psi,p)
    return 0.5*np.sum((X-L)**2.)/N
# Logarithm of the objective function
def logcost(X,alpha,psi,p):
    return math.log(calcObjf(X,alpha,psi,p))
########################### Projection Functions ##############################
# Project psi in a set {Norm(psi) < c}
def projectPsi(psi,c):
    M,K,D = np.shape(psi)
    for d in xrange(D):
        psiNorm = np.linalg.norm(psi[:,:,d])
        if psiNorm == 0.:
            psi[:,:,d]*=0.
        else:
            psi[:,:,d] = min(c,psiNorm)*(psi[:,:,d]/psiNorm)
    return psi
# Project to enforce positivity and l0 norm
def proj_l0(alpha,max_l0):
    alpha = np.nan_to_num(alpha)
    alpha[alpha<0]=0. # Project alpha into {x:x>=0}
    N,D = np.shape(alpha)
    for d in xrange(D):
        temp = np.zeros_like(alpha[:,d])
        idx = nlargest(max_l0,enumerate(alpha[:,d]),key = lambda x:x[1])
        idx = [x[0] for x in idx]
        temp[idx] = alpha[idx,d]
        alpha[:,d] = temp
    return alpha
######################## Functions for Data Reconstruction ####################
# This function performs convolution (*) of alpha and psi
# alpha is N x D (axis-0:time/frame, axis-1: various pattern)
# alpha represents sparse coefficient for various pattern's
# psi is M x K x D (axis-1: x, y and z components of pattern)
# psi represents all the patterns (D number of them)
# OUTPUT: returns an N x k x d tensor which is alpha*psi over axis-0
# Note: The method uses multiple processes for faster operation
def __myconvolve(in2,in1,mode):
    return sg.fftconvolve(in1,in2,mode)
def convAlphaPsi(alpha,psi,p):
    szAlpha = np.shape(alpha)
    szPsi = np.shape(psi)
    assert len(szAlpha) == 2 and len(szPsi) == 3 and szAlpha[1] == szPsi[2]
    convRes = np.zeros((szAlpha[0],szPsi[1],szAlpha[1]))
    for d in xrange(szAlpha[1]):
        partconvolve = partial(__myconvolve,in1=alpha[:,d],mode='same')
        convRes[:,:,d] = np.array(p.map(partconvolve,psi[:,:,d].T,1)).T
    return convRes
# Reconstruct the data from components
# alpha is N x D, psi is M x K X D
# OUTPUT: sum of alpha*psi
def recon(alpha,psi,p):
    szAlpha = np.shape(alpha)
    szPsi = np.shape(psi)
    assert len(szAlpha) == 2 and len(szPsi) == 3 and szAlpha[1] == szPsi[2]
    convRes = convAlphaPsi(alpha,psi,p)
    return np.sum(convRes,axis=2)
####################### Exact calculation of Gradient #########################
# Manually Checked with sample data -- Working as indended
# Grad of P wrt alpha is sum((X(t)-L(t))psi(t-t'))
# Returns an NxD tensor representing gradient of P with respect to psi
# Note: The method uses multiple processes for faster operation
def __myconvolve1(parArgs):
    return sg.fftconvolve(parArgs[0],parArgs[1],'full')
def calcGrad_alpha(alpha,psi,X,p):
    N,D = np.shape(alpha)
    M,K,_ = np.shape(psi)
    gradP_alpha = np.zeros((N,D,K))
    L = recon(alpha,psi,p)
    lxDiff = (L-X).T
    for d in xrange(D):
        parArgs = izip(lxDiff,psi[::-1,:,d].T)
        gradP_alpha[:,d,:] = np.array(p.map(__myconvolve1,parArgs,1))\
        [:,(M+N)/2-N/2:(M+N)/2+N/2].T
    gradP_alpha = np.sum(gradP_alpha,axis=2) # Sum over axis for k's
    return gradP_alpha/float(N)
# Manually Checked with sample data -- Working as indended
# Grad of P wrt psi is sum((X(t)-L(t))alpha(t-t'))
# Returns an MxKxD tensor representing gradient of P with respect to psi
# Note: The method uses multiple processes for faster operation    
def calcGrad_psi(alpha,psi,X,p):
    N,D = np.shape(alpha)
    M,K,_ = np.shape(psi)
    gradP_psi = np.zeros((M,K,D))
    L = recon(alpha,psi,p)
    lxDiff = (L - X).T
    for d in xrange(D):
        partconvolve = partial(sg.fftconvolve,in2=alpha[::-1,d],mode='full')
        gradP_psi[:,:,d] = np.array(p.map(partconvolve,lxDiff))\
        [:,(N-M/2):(N+M/2)].T
    return gradP_psi/float(N)


################## Stochastic Gradient Descent Procedure #######################
# Shift Invariant Sparse Coding using Stochastic Gradient Descent Algorithm.
# It assumes the data (X) is given in batch mode using X_array.
# X is N x K data matrix
# M is a scalar integer representing time/frame length for pattern     
# D represents how many pattern we want to capture
# The last two parameters (psi_orig and alpha_orig) are for debug purposes only

# Optimization through Stochastic Gradient Descent
def optimize_SGD(X_array,M,D,beta,iter_thresh=5,\
    dispObj=False,dispGrad=False,dispIteration=False,\
    totWorker=4,psi_orig=[], alpha_orig=[]):

    # Assign workers for parallel processing
    workers = Pool(processes=totWorker)

    # Initialization
    # alpha is N x D (axis-0:time/frame, axis-1: various pattern)
    # alpha represents sparse coefficient for various patterns
    # psi is M x K x D (axis-1: x, y and z components of pattern)
    # psi represents all the patterns (D number of them)
    # M is a scalar integer representing time/frame length for pattern    
    totaldata = len(X_array)
    for X in X_array:
        N,K = np.shape(X)
        # M and N must be nonzero and even
        assert M!=0 and N!=0 and M % 2 == 0 and N % 2 == 0
    
    # Initialize alpha, psi, and cost
    alpha_array = [np.zeros((len(X),D)) for X in X_array]
    psi = projectPsi(np.random.randn(M,K,D),1.0)
    cost_array = [0 for i in xrange(totaldata)]
    
    # Iterate over all data points
    iter = 0
    itStartTime = time.time()
    while iter < iter_thresh:
        # Debug 
        # if iter>79:
        #     dispIteration=True
        # Shuffle the array for each calculation
        shIdx = np.random.permutation(totaldata)
        # For one datapoint at a time
        for i in shIdx:
            X = X_array[i]
            alpha = alpha_array[i]

            # Initialize over datapoints
            N,K = np.shape(X)
            sigPerSampl = np.linalg.norm(X)/N # Signal per sample
            initLR_alpha = float(M/2.)
            initLR_psi = float(M/2.)
            print str(iter),'data #'+str(i),
         
            # Update alpha
            # ============
            gralpha = calcGrad_alpha(alpha,psi,X,workers)
            prevcost = logcost(X,alpha,psi,workers)
            gamma_alpha = initLR_alpha
            # BoldDriver: Try to use a large learning rate without diverging
            for trialno in xrange(int(math.ceil(math.log(initLR_alpha,2)))+5):
                # Project to l0norm<=N/M/beta space
                newAlpha = proj_l0(alpha - gamma_alpha*gralpha,int(beta))
                if logcost(X,newAlpha,psi,workers) <= prevcost:
                    break
                else:
                    gamma_alpha = gamma_alpha/2.0
            alpha = newAlpha.copy()
            # Save the updated alpha
            alpha_array[i] = alpha 

            # Update psi
            # ==========
            # Calculate gradient of P with respect to psi
            grpsi = calcGrad_psi(alpha,psi,X,workers)
            prevcost = logcost(X,alpha,psi,workers)
            gamma_psi = initLR_psi
            # BoldDriver: Try to use a large learning rate without diverging
            for trialno in xrange(int(math.ceil(math.log(initLR_psi,2.0)))+5):
                newPsi = projectPsi(psi - gamma_psi*grpsi,1.0)
                if logcost(X,alpha,newPsi,workers) <= prevcost:
                    break
                else:
                    gamma_psi = gamma_psi/2.0
            psi = newPsi.copy()
            print 'LR_a/p','{:.2f}'.format(gamma_alpha),'{:.2f}'.format(gamma_psi),
            
            # ======================== Display ============================
            # Display graphs and print status
            if dispGrad: 
                # Display Gradiants
                dispGrads(gralpha,grpsi)
            if dispIteration:
                # Display alpha, psi, X and L
                dispPlots(alpha,psi,X,'Iteration Data',workers)
            # Display Log Objective
            cost_array[i] = logcost(X,alpha,psi,workers)            
            if dispObj:
                pp.figure('Log likelihood plot')
                pp.scatter(iter,cost_array[i],c = 'b')
                pp.title('Likelihood Plot')
                pp.draw()
                pp.pause(1)
            # Print iteration status.
            SNR = sigPerSampl/math.exp(cost_array[i])
            print 'N',str(N),'K',str(K),'M',str(M),'D',str(D),'Beta',beta,\
                'logObj','{:.2f}'.format(cost_array[i]),\
                'SNR','{:.2f}'.format(SNR),\
                'AvgL0', '{:.2f}'.format(\
                    np.mean([np.count_nonzero(item)/D for item in alpha_array])),\
                'Elapsed Time','{:.2f}'.format(time.time() - itStartTime)

            # ===================== End Display ==========================
            # Count the iteration
            iter += 1
    L0 = [np.count_nonzero(alpha) for alpha in alpha_array]
    return alpha_array,psi,cost_array,cost_array,L0,SNR
