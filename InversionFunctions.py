"""
Created on Thu Apr 27 09:22:16 2017
Inversion Routine Functions
@author: John Vogel
This file has all the functions called in the main file and is the meat of
the actual data inversion
"""
import numpy as np
import time


"""
Get Shape Factor
Function to get the shape factors for a given R0 and thickness
"""


def getS(R0, T, N, M, bigJ, bigH, bigHH, mu2, lam2, STinf, STinfL):
    Shape = []
    dSdaT = []
    dSdR0 = []
    ShapeL = []
    dSLdaT = []
    dSLdR0 = []
    for Etype in range(0, 6):
        Shape.append(0)
        dSdaT.append(0)
        dSdR0.append(0)
    for Etype in range(0, 3):
        ShapeL.append(0)
        dSLdaT.append(0)
        dSLdR0.append(0)
    for ind in range(0, N):     # Outer Integral over lambda
        JY = 0
        dJYdT = 0
        dJYdR = 0
        for ind2 in range(0, M):    # Inner Integral over mu
            gam = np.sqrt(mu2[ind2] + lam2[ind])
            egT = np.exp(-2 * gam * T)
            RgT = R0 * gam * T
            RgT1 = 1 / (RgT + 1)
            RgTe = (1 - RgT) * RgT1 * egT
            RgTe1 = 1 / (RgTe + 1)
            deriv = 4 * egT * RgT1 * RgTe1 * RgTe1
            bigY = -2 * RgTe * RgTe1 / gam
            dYdT = deriv * (R0 * RgT1 + 1 - RgT)
            dYdR = deriv * T * RgT1
            JY = JY + bigJ[ind2] * bigY
            dJYdT = dJYdT + bigJ[ind2] * dYdT
            dJYdR = dJYdR + bigJ[ind2] * dYdR
        for Etype in range(0, 6):
            Shape.append(Shape[Etype] + bigH[Etype, ind] * JY)
            dSdaT.append(dSdaT[Etype] + bigH[Etype, ind] * dJYdT)
            dSdR0.append(dSdR0[Etype] + bigH[Etype, ind] * dJYdR)
        for Etype in range(0, 3):
            ShapeL.append(ShapeL[Etype] + bigHH[Etype, ind] * JY)
            dSLdaT.append(dSLdaT[Etype] + bigHH[Etype, ind] * dJYdT)
            dSLdR0.append(dSLdR0[Etype] + bigHH[Etype, ind] * dJYdR)
    for Etype in range(0, 6):       # Departure shape factor T = inf value
        Shape[Etype] = Shape[Etype] + STinf[Etype]
    for Etype in range(0, 3):
        ShapeL[Etype] = ShapeL[Etype] + STinfL[Etype]

    return [Shape, dSdaT, dSdR0, ShapeL, dSLdaT, dSLdR0]


"""
INVERT:
Function to actually calculate the optimization
"""


def invertIsotropic(T, rho, rhoL, rcert, rcertL, E1, N, M, bigJ, bigH, bigHH,
                    mu2, lam2, STinf, STinfL):
    b2 = 0.0001         # Weight for R0 -> infinity
    b2 = 0.05           # Weight for R0 -> 0
    Nexp = len(rho)     # Number of experiments
    rcert2 = 0.1        # Scale certainty of mixed experiments
    Nexp2 = int(Nexp + Nexp / 2)  # 1.5x number of independent experiments
    Neqn = Nexp2 + 1    # number of equations to solve

    et = []     # E1 values for experiments
    et2 = []    # E2 values for experiments

    # Convert E1 values to base values (assume Rt is infty in SF models)
    E2 = np.array([2, 2, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2])  # Conversion E1 to E2
    for ind in range(0, Nexp):
        et.append(np.mod(E1[ind]-1, 6)+1)
        et2.append(E2[E1[ind]-1])
    for ind in range(Nexp, Nexp2):
        ii = (ind-Nexp) * 2
        et.append(np.mod(E1[ii]-1, 6)+1)
        et2.append(E2[E1[ii]-1])

    convrg = 0                  # Initially assume exper don't converge FALSE
    JT = np.zeros((2, Neqn))    # Initialize Jacobian transpose matrix
    R0 = 5                      # Initial Guess values
    alpha = 1

    [Shape, dSdaT, dSdR0, ShapeL, dSLdaT, dSLdR0] = getS(R0, alpha * T, N, M,
                                                         bigJ, bigH, bigHH,
                                                         mu2, lam2, STinf,
                                                         STinfL)
    sigy = 0
    for ind in range(0, Nexp):
        sigy = sigy + Shape[int(et[ind])] / (rho[ind] * Nexp)

    # Convert sigy, and R0 to independent vectors using log scale, this
    # prevents them from going negative and allows the guess values to change
    # quickly, covering a lot of ground
    v = []
    v.append(np.log(sigy))
    v.append(np.log(R0))

    # Convergence tuning parameters
    maxit = 100         # Maximum number of iterations
    step = 0.5          # Initial step relaxation parameter
    stepfinal = 1.2     # Final relax parameter (> 1 speed, < 1 stability)
    tol = []
    tol.append(0.0001)  # Convergence Tolerance for ln(sigy)
    tol.append(0.0001)  # Convergence Tolerance for ln(R0)
    convrg = 0          # Initial value for convergence flag is FALSE
    f = []              # Function values for solving

    for itr in range(0, maxit):     # Start convergence optimization
        errtot = 0                  # Total number of errors
        for ind in range(0, Nexp):
            asrr = alpha * sigy * rho[ind] * rcert[ind]
            f.append(Shape[int(et[ind])] * rcert[ind] - asrr)
            errtot = errtot + f[ind]**2
            JT[0, ind] = -asrr * np.sqrt(1 + sigy * sigy) / sigy
            JT[1, ind] = np.sqrt(1 + R0 * R0) * dSdR0[int(et[ind])] * rcert[ind]
        for ind in range(Nexp, Nexp2):  # indices of orthogonal experiments
            ii = (ind - Nexp) * 2
            asrr = alpha * sigy * rhoL[ii] * rcertL[ii] * rcert2
            f.append(ShapeL[int(et2[ind])] * rcertL[ii] * rcert2 - asrr)
            errtot = errtot + f[ind]**2
            JT[0, ind] = -asrr * np.sqrt(1 + sigy * sigy) / sigy
            JT[1, ind] = np.sqrt(1 + R0 * R0) * dSLdR0[int(et2[ind])] * rcertL[ii] * rcert2
        f.append(b2 * np.exp(-b2 * R0))
        temp = np.array(([0], [-b2 * np.sqrt(1 + R0**2) * f[Nexp2]]))
        JT = np.hstack((JT, temp))
        errtot = errtot + f[Nexp2]**2

        # Multiply J Transpose by J
        JTf = [0, 0]
        JTJ = np.zeros((2, 2))
        for ind in range(0, Neqn):
            JT1 = JT[0, ind]
            JT2 = JT[1, ind]
            JTJ[0, 0] = JTJ[0, 0] + JT1**2
            JTJ[1, 0] = JTJ[1, 0] + JT2 * JT1
            JTJ[1, 1] = JTJ[1, 1] + JT2**2
            JTf[0] = JTf[0] + JT1 * f[ind]
            JTf[1] = JTf[1] + JT2 * f[ind]
        JTJ[0, 1] = JTJ[1, 0]
        detJTJ = JTJ[0, 0] * JTJ[1, 1] - JTf[1] * JTJ[0, 1]
        dv = []
        if(detJTJ == 0):
            # Matrix is singular
            dv.append(-JTf[0] * JTJ[1, 1] + JTf[1] * JTJ[0, 1])
            dv.append(JTf[0] * JTJ[1, 0] - JTf[1] * JTJ[0, 0])
            break
        else:
            dv.append((-JTf[0] * JTJ[1, 1] + JTf[1] * JTJ[0, 1])/detJTJ)
            dv.append((JTf[0] * JTJ[1, 0] - JTf[1] * JTJ[0, 0])/detJTJ)
        # Calculate next step in v by newton method
        v[0] = v[0] + step * dv[0]
        v[1] = v[1] + step * dv[1]

        # Convert back to needed parameters
        sigy = np.exp(v[0])
        R0 = np.exp(v[1])

        if np.abs(dv[0]) < tol[0] and np.abs(dv[1]) < tol[1]:
            convrg = 1
            break

        # Recompute shape factors with new R0
#        [Shape, dSdaT, dSdR0, ShapeL, dSLdaT, dSLdR0] = getS(R0, alpha * T, N, M,
#                                                             bigJ, bigH, bigHH,
#                                                             mu2, lam2, STinf,
#                                                             STinfL)
        step = stepfinal - 0.8 * (stepfinal - step)  # Increment step
    return [R0, sigy, alpha, convrg, itr]


"""
CONDX:
Calculate the conductivity in each direction as well as the contact resistance
"""


def cond(T, Data, location, LL, N, M, bigJ, bigH, bigHH, mu2, lam2,
         STinf, STinfL):
    startcond = time.time()
    Uconvert = 0.0000001    # Conversion from ohm-um to kOhm-cm
    DataDim = Data.shape    # Find numbers of rows and columns
    NValid = DataDim[0]     # Number of valid experiments (start with all)
    Nunique = 0             # Always start with 0 unique experiments
    breakflag = 0           # Initially don't break from any loops
    rho = []
    rhoL = []
    rcert = []
    rcertL = []
    E1 = []
    warning = ''
    if(DataDim[1] < 5):     # If you don't have enough columns than error
        print('**** ERROR: not enough columns in "Data" ****')
        return
    for ind in range(0, DataDim[0]):     # For all rows
        V1 = Data[ind, 0]                # Store V1
        V2 = Data[ind, 1]                # Store V2
        V3 = Data[ind, 2]                # Store V3
        I1 = Data[ind, 3]                # Store I1
        E1.append(Data[ind, 4])          # Store Experiment Type

        # Check for errors with the experiments
        if E1[ind] > 12:            # If experiment is not real
            E1[ind] = 0
        elif I1 <= 0:               # If current is negative
            E1[ind] = 0
        elif V1 < 0:                # If voltage is negative
            E1[ind] = 0
        elif V2 >= V1 or V2 <= 0:   # If V2 is more than power or less GND
            E1[ind] = 0
        elif V3 >= V1 or V3 <= 0:   # if V2 is more than power or less GND
            E1[ind] = 0
        elif np.mod(E1[ind], 2) == 1 and V2 <= V3:  # V1 > V2 > V3 > GND
            E1[ind] = 0

        if E1[ind] == 0:            # If the experiment failed
            NValid = NValid - 1     # One less experiment is valid
            if NValid == 0:         # If they are all invalid
                breakflag = 1       # Don't Finish inversion
        elif np.mod(E1[ind], 2):    # Tangential Experiment
            deltaV = V2 - V3
            Vleft = 2 * V1 - (V2 + V3)
            Vright = V2 + V3
            rho.append(deltaV * LL * Uconvert / I1)
            rhoL.append(Vleft * Vright / V1 * LL * Uconvert / I1)
            rcert.append(10 * deltaV)
            rcertL.append(10 * Vleft * Vright / V1)
        else:                       # Orthogonal Experiment
            deltaV = 0.5 * (V2 + V3)
            rho.append(deltaV * LL * Uconvert / I1)
            rhoL.append(V1 * LL * Uconvert / I1)
            rcert.append(10 * deltaV)
            rcertL.append(10 * V1)
            end = len(rhoL)-1
            if(end > 0):
                rhoL[end] = rhoL[end] - 0.25 * rhoL[end-1]
                if((E1[end] - E1[end-1]) == 1):
                    rcert[end] = np.sqrt(rcert[end] * rcert[end - 1])
                else:
                    rcert[end] = 0
        if(breakflag == 0):
            Nunique = Nunique + 1         # Add another experiment to the list
            for ind2 in range(0, ind-1):  # Check if experiment already run
                if E1[ind2] == E1[ind]:     # If it has
                    Nunique = Nunique - 1   # then less are unique
                    break                   # Only needs to be true once
    # HERE PASS IN DATA TO INVERT
    # T, rho, rhoL, rcert, rcertL, E1, R0, sigy, alpha, convrg, iter
    if(breakflag == 0):
        [R0, sigy, alpha, convrg, itr] = invertIsotropic(T, rho, rhoL, rcert,
                                                          rcertL, E1, N, M, bigJ,
                                                          bigH, bigHH, mu2, lam2,
                                                          STinf, STinfL)
        if convrg:
            warning = ["conv ", itr]
        else:
            warning = ['no conv ', itr]
        condx = sigy
        condy = sigy * alpha**2
        r0 = R0
        rc = R0 * T / (sigy + 0.0000000001) / 10
        endcond = time.time()
        print('Inversion finished, Location ', location + 1,
              ' (', round(endcond-startcond, 5), 'seconds elapsed )',
              len(rho), '/', DataDim[0], 'Points', warning)
    elif(breakflag == 1):
        condx = -1
        condy = -1
        r0 = -1
        rc = -1
        endcond = time.time()
        print('Inversion Failed, Location ', location + 1,
              ' (', round(endcond-startcond, 5),
              'seconds elapsed )')
    return [condx, condy, r0, rc]


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

    p1 = np.array([((LC-LW)/2), ((LC-LW)/2)+LC, ((LC-LW)/2)+2*LC])  # PosInside
    p2 = np.array([p1[0]+LW, p1[1]+LW, p1[2]+LW])   # PosOutside

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

    return [LL, N, M, bigJ, bigH, bigHH, mu2, lam2, STinf, STinfL]
