import numpy as np
import sys
import scipy.stats as ss
from math import factorial
from scipy import sparse
from scipy.sparse.linalg import splu
import scipy as scp
import scipy.stats as ss
from functools import partial
import scipy.special as scps
from scipy import signal
from scipy.integrate import quad
import time

class VG_process():
    """
    Class for the Variance Gamma process:
    r = risk free constant rate
    Using the representation of Brownian subordination, the parameters are: 
        theta = drift of the Brownian motion
        sigma = standard deviation of the Brownian motion
        kappa = variance of the of the Gamma process 
    """
    def __init__(self, r=0.1, sigmaJ=0.2, theta=-0.1, kappa=0.1):
        self.r = r
        self.c = self.r
        self.theta = theta
        self.kappa = kappa
        if (sigmaJ<0) :
            raise ValueError("sigma must be positive")
        else:
            self.sigma = sigmaJ
            
        # moments
        self.mean = self.c + self.theta
        self.var = self.sigma**2 + self.theta**2 * self.kappa 
        self.skew = (2 * self.theta**3 * self.kappa**2 + 3*self.sigma**2 * self.theta * self.kappa) / (self.var**(1.5)) 
        self.kurt = ( 3*self.sigma**4 * self.kappa +12*self.sigma**2 * self.theta**2 \
                     * self.kappa**2 + 6*self.theta**4 * self.kappa**3 ) / (self.var**2)

    def exp_RV(self, S0, T, N):
        w = -np.log(1 - self.theta * self.kappa - self.kappa/2 * self.sigma**2 ) /self.kappa    # coefficient w
        rho = 1 / self.kappa
        G = ss.gamma(rho * T).rvs(N) / rho     # The gamma RV
        Norm = ss.norm.rvs(0,1,N)              # The normal RV  
        VG = self.theta * G + self.sigma * np.sqrt(G) * Norm     # VG process at final time G
        S_T = S0 * np.exp( (self.r-w)*T + VG )                 # Martingale exponential VG       
        return S_T.reshape((N,1))
    
    def path(self, T=1, N=10000, paths=1):
        """
        Creates Variance Gamma paths    
        N = number of time points (time steps are N-1)
        paths = number of generated paths
        """
        dt = T/(N-1)          # time interval        
        X0 = np.zeros((paths,1))
        G = ss.gamma( dt/self.kappa, scale=self.kappa).rvs( size=(paths,N-1) )     # The gamma RV
        Norm = ss.norm.rvs(loc=0, scale=1, size=(paths,N-1))                       # The normal RV  
        increments = self.c*dt + self.theta * G + self.sigma * np.sqrt(G) * Norm
        X = np.concatenate((X0,increments), axis=1).cumsum(1)
        return X

class VG_pricer():
    """
    Closed Formula.
    Monte Carlo.
    Finite-difference PIDE: Explicit-implicit scheme, with Brownian approximation         
    
        0 = dV/dt + (r -(1/2)sig^2 -w) dV/dx + (1/2)sig^2 d^V/dx^2
                 + \int[ V(x+y) nu(dy) ] -(r+lam)V  
    """
    def __init__(self, Option_info, Process_info ):
        """
        Process_info:  of type VG_process. It contains the interest rate r and the VG parameters (sigma, theta, kappa) 
    
        Option_info:  of type Option_param. It contains (S0,K,T) i.e. current price, strike, maturity in years
        """
        self.r = Process_info.r           # interest rate
        self.sigma = Process_info.sigma      # VG parameter
        self.theta = Process_info.theta       # VG parameter
        self.kappa = Process_info.kappa       # VG parameter
        self.exp_RV = Process_info.exp_RV     # function to generate exponential VG Random Variables
        self.w = -np.log(1 - self.theta * self.kappa - self.kappa/2 * self.sigma**2 ) /self.kappa    # coefficient w
        
        self.S0 = Option_info.S0          # current price
        self.K = Option_info.K            # strike
        self.T = Option_info.T            # maturity in years
        
        self.price = 0
        self.S_vec = None
        self.price_vec = None
        self.mesh = None
        self.exercise = Option_info.exercise
        self.payoff = Option_info.payoff
        
        
        
    def payoff_f(self, S):
        if self.payoff == "call":
            Payoff = np.maximum( S - self.K, 0 )
        elif self.payoff == "put":    
            Payoff = np.maximum( self.K - S, 0 )  
        return Payoff
    
 
    
    def closed_formula(self):
        """ 
        VG closed formula.  Put is obtained by put/call parity.
        """
        
        def Psy(a,b,g):
            f = lambda u: ss.norm.cdf(a/np.sqrt(u) + b*np.sqrt(u)) * u**(g-1) * np.exp(-u) / scps.gamma(g)
            result = quad( f, 0, np.inf )
            return result[0]
        
        # Ugly parameters
        xi = - self.theta / self.sigma**2
        s = self.sigma / np.sqrt( 1+ ((self.theta/self.sigma)**2) * (self.kappa/2) )
        alpha = xi * s
    
        c1 = self.kappa/2 * (alpha + s)**2 
        c2 = self.kappa/2 * alpha**2
        d = 1/s * ( np.log(self.S0/self.K) + self.r*self.T + self.T/self.kappa * np.log( (1-c1)/(1-c2) )  )

        # Closed formula 
        call = self.S0 * Psy( d * np.sqrt((1-c1)/self.kappa) , (alpha+s) * np.sqrt(self.kappa/(1-c1)) , self.T/self.kappa ) - \
            self.K * np.exp(-self.r*self.T) * Psy( d * np.sqrt((1-c2)/self.kappa) , \
                            (alpha) * np.sqrt(self.kappa/(1-c2)) , self.T/self.kappa )
        
        if self.payoff == "call":
            return call
        elif self.payoff == "put":
            return call - self.S0 + self.K * np.exp(-self.r * self.T)
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")
           

    def PIDE_price(self, steps, Time=False):
        """
        steps = tuple with number of space steps and time steps
        payoff = "call" or "put"
        exercise = "European" or "American"
        Time = Boolean. Execution time.
        """
        t_init = time.time()
        
        Nspace = steps[0]   
        Ntime = steps[1]
        
        S_max = 6*float(self.K)                
        S_min = float(self.K)/6
        x_max = np.log(S_max)
        x_min = np.log(S_min)
        
        dev_X = np.sqrt(self.sigma**2 + self.theta**2 * self.kappa)     # std dev VG process
        
        dx = (x_max - x_min)/(Nspace-1)
        extraP = int(np.floor(5*dev_X/dx))            # extra points beyond the B.C.
        x = np.linspace(x_min-extraP*dx, x_max+extraP*dx, Nspace + 2*extraP)   # space discretization
        t, dt = np.linspace(0, self.T, Ntime, retstep=True)       # time discretization
        
        Payoff = self.payoff_f(np.exp(x))
        offset = np.zeros(Nspace-2)
        V = np.zeros((Nspace + 2*extraP, Ntime))       # grid initialization
        
        if self.payoff == "call":
            V[:,-1] = Payoff                   # terminal conditions 
            V[-extraP-1:,:] = np.exp(x[-extraP-1:]).reshape(extraP+1,1) * np.ones((extraP+1,Ntime)) - \
                 self.K * np.exp(-self.r* t[::-1] ) * np.ones((extraP+1,Ntime))  # boundary condition
            V[:extraP+1,:] = 0
        else:    
            V[:,-1] = Payoff
            V[-extraP-1:,:] = 0
            V[:extraP+1,:] = self.K * np.exp(-self.r* t[::-1] ) * np.ones((extraP+1,Ntime))
        
        
        A = self.theta/(self.sigma**2)
        B = np.sqrt( self.theta**2 + 2*self.sigma**2/self.kappa ) / self.sigma**2
        levy_m = lambda y: np.exp( A*y - B*np.abs(y) ) / (self.kappa*np.abs(y))   # Levy measure VG        

        eps = 1.5*dx    # the cutoff near 0
        lam = quad(levy_m,-(extraP+1.5)*dx,-eps)[0] + quad(levy_m,eps,(extraP+1.5)*dx)[0] # approximated intensity

        int_w = lambda y: (np.exp(y)-1) * levy_m(y)
        int_s = lambda y: np.abs(y) * np.exp( A*y - B*np.abs(y) ) / self.kappa  # avoid division by zero

        w = quad(int_w, -(extraP+1.5)*dx, -eps)[0] + quad(int_w, eps, (extraP+1.5)*dx)[0]   # is the approx of omega

        sig2 = quad(int_s,-eps,eps)[0]         # the small jumps variance
        
        
        dxx = dx * dx
        a = ( (dt/2) * ( (self.r - w - 0.5*sig2)/dx - sig2/dxx ) )
        b = ( 1 + dt * ( sig2/dxx + self.r + lam) )
        c = (-(dt/2) * ( (self.r - w - 0.5*sig2)/dx + sig2/dxx ) )
        D = sparse.diags([a, b, c], [-1, 0, 1], shape=(Nspace-2, Nspace-2)).tocsc()
        DD = splu(D)
       
        nu = np.zeros(2*extraP+3)        # Lévy measure vector
        x_med = extraP+1                 # middle point in nu vector
        x_nu = np.linspace(-(extraP+1+0.5)*dx, (extraP+1+0.5)*dx, 2*(extraP+2) )    # integration domain
        for i in range(len(nu)):
            if (i==x_med) or (i==x_med-1) or (i==x_med+1):
                continue
            nu[i] = quad(levy_m, x_nu[i], x_nu[i+1])[0]


        if self.exercise=="European":        
            # Backward iteration
            for i in range(Ntime-2,-1,-1):
                offset[0] = a * V[extraP,i]
                offset[-1] = c * V[-1-extraP,i]
                V_jump = V[extraP+1 : -extraP-1, i+1] + dt * signal.convolve(V[:,i+1],nu[::-1],mode="valid",method="auto")
                V[extraP+1 : -extraP-1, i] = DD.solve( V_jump - offset ) 
        elif self.exercise=="American":
            for i in range(Ntime-2,-1,-1):
                offset[0] = a * V[extraP,i]
                offset[-1] = c * V[-1-extraP,i]
                V_jump = V[extraP+1 : -extraP-1, i+1] + dt * signal.convolve(V[:,i+1],nu[::-1],mode="valid",method="auto")
                V[extraP+1 : -extraP-1, i] = np.maximum( DD.solve( V_jump - offset ), Payoff[extraP+1 : -extraP-1] )
                
        X0 = np.log(self.S0)                            # current log-price
        self.S_vec = np.exp(x[extraP+1 : -extraP-1])        # vector of S
        self.price = np.interp(X0, x, V[:,0])
        self.price_vec = V[extraP+1 : -extraP-1,0]
        self.mesh = V[extraP+1 : -extraP-1, :]
        
        if (Time == True):
            elapsed = time.time()-t_init
            return self.price, elapsed
        else:
            return self.price

class Option_param():
    """
    Option class wants the option parameters:
    S0 = current stock price
    K = Strike price
    T = time to maturity
    v0 = (optional) spot variance 
    exercise = European or American
    """
    def __init__(self, S0=15, K=15, T=1, v0=0.04, payoff="call", exercise="European"):
        self.S0 = S0
        self.v0 = v0
        self.K = K
        self.T = T
        
        if (exercise=="European" or exercise=="American"):
            self.exercise = exercise
        else: 
            raise ValueError("invalid type. Set 'European' or 'American'")
        
        if (payoff=="call" or payoff=="put"):
            self.payoff = payoff
        else: 
            raise ValueError("invalid type. Set 'call' or 'put'")