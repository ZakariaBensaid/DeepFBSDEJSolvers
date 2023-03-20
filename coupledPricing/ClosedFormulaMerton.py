import numpy as np
import sys
import scipy.stats as ss
from math import factorial


class BS_pricer():
    """
    Closed Formula.
    Monte Carlo.
    Finite-difference Black-Scholes PDE:
     df/dt + r df/dx + 1/2 sigma^2 d^f/dx^2 -rf = 0
    """
    def __init__(self, Option_info, Process_info ):
        """
        Process_info:  of type Diffusion_process. It contains (r,mu, sig) i.e.  interest rate, drift coefficient, diffusion coefficient
    
        Option_info:  of type Option_param. It contains (S0,K,T) i.e. current price, strike, maturity in years
        """
        self.r = Process_info.r           # interest rate
        self.sig = Process_info.sig       # diffusion coefficient
        self.S0 = Option_info.S0          # current price
        self.K = Option_info.K            # strike
        self.T = Option_info.T            # maturity in years
        self.exp_RV = Process_info.exp_RV # function to generate solution of GBM
        
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
        
    
    @staticmethod
    def BlackScholes(payoff='call', S0=100., K=100., T=1., r=0.1, sigma=0.2 ):
        """ Black Scholes closed formula:
            payoff: call or put.
            S0: float.    initial stock/index level.
            K: float strike price.
            T: float maturity (in year fractions).  
            r: float constant risk-free short rate.
            sigma: volatility factor in diffusion term. """
   
        d1 = (np.log(S0/K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S0/K) + (r - sigma**2 / 2) * T) / (sigma * np.sqrt(T))

        if payoff=="call":
            return S0 * ss.norm.cdf( d1 ) - K * np.exp(-r * T) * ss.norm.cdf( d2 )
        elif payoff=="put":
            return K * np.exp(-r * T) * ss.norm.cdf( -d2 ) - S0 * ss.norm.cdf( -d1 )
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")
    
    
    @staticmethod        
    def vega(sigma, S0, K, T, r):
        """ BS vega: derivative of the price with respect to the volatility """
        d1 = (np.log(S0/K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        return S0 * np.sqrt(T) * ss.norm.pdf(d1)
    
    
    def closed_formula(self):
        """ 
        Black Scholes closed formula:
        """
        d1 = (np.log(self.S0/self.K) + (self.r + self.sig**2 / 2) * self.T) / (self.sig * np.sqrt(self.T))
        d2 = (np.log(self.S0/self.K) + (self.r - self.sig**2 / 2) * self.T) / (self.sig * np.sqrt(self.T))

        if self.payoff=="call":
            return self.S0 * ss.norm.cdf( d1 ) - self.K * np.exp(-self.r * self.T) * ss.norm.cdf( d2 )
        elif self.payoff=="put":
            return self.K * np.exp(-self.r * self.T) * ss.norm.cdf( -d2 ) - self.S0 * ss.norm.cdf( -d1 )
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

class Merton_process():
    """
    Class for the Merton process:
    r = risk free constant rate
    sig = constant diffusion coefficient
    lam = jump activity
    muJ = jump mean
    sigJ = jump standard deviation
    """
    def __init__(self, r=0.1, sig=0.2, lam = 0.8, muJ = 0, sigJ = 0.5):
        self.r = r
        self.lam = lam
        self.muJ = muJ
        if (sig<0 or sigJ<0):
            raise ValueError("sig and sigJ must be positive")
        else:
            self.sig = sig
            self.sigJ = sigJ
        
        # moments
        self.var = self.sig**2 + self.lam * self.sigJ**2 + self.lam * self.muJ**2
        self.skew = self.lam * (3* self.sigJ**2 * self.muJ + self.muJ**3) / self.var**(1.5)
        self.kurt = self.lam * (3* self.sigJ**3 + 6 * self.sigJ**2 * self.muJ**2 + self.muJ**4) / self.var**2
     
    def exp_RV(self, S0, T, N):
        m = self.lam * (np.exp(self.muJ + (self.sigJ**2)/2) -1)    # coefficient m
        W = ss.norm.rvs(0, 1, N)              # The normal RV vector  
        P = ss.poisson.rvs(self.lam*T, size=N)    # Poisson random vector (number of jumps)
        Jumps = np.asarray([ss.norm.rvs(self.muJ, self.sigJ, ind).sum() for ind in P ]) # Jumps vector
        S_T = S0 * np.exp( (self.r - 0.5*self.sig**2 -m )*T + np.sqrt(T)*self.sig*W + Jumps )     # Martingale exponential Merton
        return S_T.reshape((N,1))

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

class Merton_pricer():
    """
    Closed Formula.
    Monte Carlo.
    Finite-difference PIDE: Explicit-implicit scheme         
    
        0 = dV/dt + (r -(1/2)sig^2 -m) dV/dx + (1/2)sig^2 d^V/dx^2
                 + \int[ V(x+y) nu(dy) ] -(r+lam)V  
    """
    def __init__(self, Option_info, Process_info ):
        """
        Process_info:  of type Merton_process. It contains (r, sig, lam, muJ, sigJ) i.e.  
        interest rate, diffusion coefficient, jump activity and jump distribution parameters
    
        Option_info:  of type Option_param. It contains (S0,K,T) i.e. current price, strike, maturity in years
        """
        self.r = Process_info.r           # interest rate
        self.sig = Process_info.sig       # diffusion coefficient
        self.lam = Process_info.lam       # jump activity
        self.muJ = Process_info.muJ       # jump mean
        self.sigJ = Process_info.sigJ     # jump std
        self.exp_RV = Process_info.exp_RV # function to generate exponential Merton Random Variables
        
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
        Merton closed formula.
        """

        m = self.lam * (np.exp(self.muJ + (self.sigJ**2)/2) -1)    # coefficient m
        lam2 = self.lam * np.exp(self.muJ + (self.sigJ**2)/2)

        tot=0
        for i in range(18):
            tot += ( np.exp(-lam2*self.T) * (lam2*self.T)**i / factorial(i) ) \
            * BS_pricer.BlackScholes(self.payoff, self.S0, self.K, self.T, self.r-m+i*(self.muJ+0.5*self.sigJ**2)/self.T, 
                                np.sqrt(self.sig**2 + (i*self.sigJ**2)/self.T) )  
        return tot