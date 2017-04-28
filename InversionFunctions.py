# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 09:22:16 2017
Inversion Routine Main File
@author: John Vogel
Call the whole function in chuncks based on location
"""
import numpy as np
"""
CONDX:
Calculate the conductivity in each direction as well as the contact resistance
"""


def cond(T, Data, location):
    DataDim = Data.shape    # Find numbers of rows and columns
    NValid = DataDim[0]     # Number of valid experiments (start with all)
    if(DataDim[1] < 5):     # If you don't have enough columns than error
        print('**** ERROR: not enough columns in "Data" ****')
        return
    for ind in range(0, DataDim[0]):     # For all rows
        V1 = Data[ind, 0]                # Store V1
        V2 = Data[ind, 1]                # Store V2
        V3 = Data[ind, 2]                # Store V3
        I1 = Data[ind, 3]                # Store I1
        E1 = Data[ind, 4]          # Store Experiment Type
    condx = 0
    condy = 0
    r0 = 0
    rc = 0
    return [condx, condy, r0, rc]

"""
INVERT:
Function to actually calculate the optimization
"""


def invertIsotropic():

    return


"""
INITIALIZATION:
Initialize the data for inversion and precompute some of the intensive stuff
(Timed at 1.17 seconds average)
"""
def precompute():
    # ---------------------------------------------------------------------
    # Probe Dimensions and Layout
    # ---------------------------------------------------------------------
    LL = 250    # Line Length in um
    LW = 20     # Line Width in um
    LC = 40     # Center to Center distance um
    
    p1 = np.array([((LC-LW)/2), ((LC-LW)/2)+LC, ((LC-LW)/2)+2*LC])  # Pos of inside
    p2 = np.array([p1[0]+LW, p1[1]+LW, p1[2]+LW])   # Pos of outside
    
    # ---------------------------------------------------------------------
    # Quadrature (numerical integration) over lambda which is the
    # eigenvalue in the x direction (across the lines)
    # ---------------------------------------------------------------------
    N = 60      # Number of bins in integration by parts over lambda
    lnlmax = 5               # Integration upper limit
    lnlmin = -12             # Integration lower limit
    Ninf = 500               # number of terms in quadrature (INFTY)
    lnlmaxinf = 1            # Integration upper limit (INFTY)
    lnlmininf = -18          # Integration lower limit (INFTY)
    hx = (lnlmax-lnlmin)/N   # delta ln(lambda)
    exphx = np.exp(hx)
    lmbda = 2/LL*np.exp(lnlmin-hx)
    bigH = np.zeros((6, N))
    bigHH = np.zeros((3, N))
    lam2 = np.zeros((N, 1))
    for ind in range(0, N):
        lmbda = lmbda * exphx
        lam2[ind, 0] = lmbda**2
        Ht = np.zeros((3, 1))
        Ho = np.zeros((3, 1))
        for ind2 in range(0, 3):    # Loop across the 3 probe lines
            Ht[ind2, 0] = (np.cos(lmbda*p2[ind2]) -
                           np.cos(lmbda*p1[ind2])/(lmbda*LW))
            Ho[ind2, 0] = (np.sin(lmbda*p2[ind2]) -
                           np.sin(lmbda*p1[ind2])/(lmbda*LW))
        Hconst = hx * lmbda/np.pi
        bigH[:, ind] = np.array([Ht[0, 0]*Ht[1, 0]*Hconst*4,
                                 Ho[0, 0]*Ho[1, 0]*Hconst,
                                 Ht[0, 0]*Ht[2, 0]*Hconst*4,
                                 Ho[0, 0]*Ho[2, 0]*Hconst,
                                 Ht[1, 0]*Ht[2, 0]*Hconst*4,
                                 Ho[1, 0]*Ho[2, 0]*Hconst])
        bigHH[:, ind] = np.array([(Ho[0, 0]**2-Ht[0, 0]**2)*Hconst,
                                  (Ho[1, 0]**2-Ht[1, 0]**2)*Hconst,
                                  (Ho[2, 0]**2-Ht[2, 0]**2)*Hconst])
    
    # ---------------------------------------------------------------------
    # Quadrature (numerical integration) over mu which is the
    # eigenvalue in the z direction (along axis of lines)
    # ---------------------------------------------------------------------
    M = 1240                     # Number of bins in integration by parts
    lambdaLbig = 5.216480381023  # Cutoff
    bigJ = np.zeros((M, 1))
    mu2 = np.zeros((M, 1))
    muL2 = np.zeros((M, 1))
    lnmLmax = 25                 # Integration +- limits
    hz = 2*lnmLmax/M             # delta ln(mu)
    muL = np.pi*np.exp(-hz*(M+2)/2)
    exphz = np.exp(hz)
    for ind in range(1, M):
        # Calculate bigJ and eigvalue mu
        muL = muL * exphz
        mu = muL * 2 / LL
        muL2[ind, 0] = muL**2        # store mu*L^2 values
        mu2[ind, 0] = mu**2          # store mu^2 values
        sinmuL = np.sin(muL)/muL     # average of e-function cos(mu*z)
        bigJ[ind, 0] = sinmuL**2 * hz * muL * 2 / np.pi
    
    # ---------------------------------------------------------------------
    # Shape factors for thickness = infinity
    # ---------------------------------------------------------------------
    STinf = np.zeros((6, 1))
    STinfL = np.zeros((3, 1))
    hx = (lnlmaxinf - lnlmininf) / Ninf
    lmbda = np.exp(lnlmininf-hx)
    exphx = np.exp(hx)
    for ind in range(0, Ninf):
        lmbda = lmbda * exphx
        lambdaL = lmbda * LL / 2
        if lambdaL < lambdaLbig:
            JY = 0
            lambdaL2 = lambdaL**2
            for ind2 in range(0, M):
                JY = JY + bigJ[ind2, 0] / np.sqrt(muL2[ind2, 0] + lambdaL2)
            JY = JY * lambdaL
        else:
            JY = 1 - (1 / (np.pi * lambdaL))
        for ind2 in range(0, 3):
            Ht[ind2, 0] = (np.cos(lmbda*p2[ind2]) -
                           np.cos(lmbda*p1[ind2])/(lmbda*LW))
            Ho[ind2, 0] = (np.sin(lmbda*p2[ind2]) -
                           np.sin(lmbda*p1[ind2])/(lmbda*LW))
        Hconst = JY * hx / np.pi
        STinf[:, 0] = np.array([STinf[0, 0] + Ht[0, 0]*Ht[1, 0]*Hconst*4,
                                STinf[1, 0] + Ho[0, 0]*Ho[1, 0]*Hconst,
                                STinf[2, 0] + Ht[0, 0]*Ht[2, 0]*Hconst*4,
                                STinf[3, 0] + Ho[0, 0]*Ho[2, 0]*Hconst,
                                STinf[4, 0] + Ht[1, 0]*Ht[2, 0]*Hconst*4,
                                STinf[5, 0] + Ho[1, 0]*Ho[2, 0]*Hconst])
        STinfL[:, 0] = np.array([STinfL[0, 0] + (Ho[0, 0]**2-Ht[0, 0]**2)*Hconst,
                                 STinfL[1, 0] + (Ho[1, 0]**2-Ht[1, 0]**2)*Hconst,
                                 STinfL[2, 0] + (Ho[2, 0]**2-Ht[2, 0]**2)*Hconst])

    E2 = np.array([2, 2, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2])  # Conversion E1 to E2

    return
