import numpy as np
import tensorflow as tf
from scipy.integrate import quad




class ModelCoupledFBSDE:
    def __init__(self, T , QAver,  R0 ,  jumpFactor , A, K, pi, p0, p1, f0, f1, theta,C, S0 , h1, h2, sig0, sig, alphaTarget, jumpModel, coeffEqui):
        self.T = T
        self.QAver= QAver
        self.R0= R0
        self.jumpFactor= jumpFactor
        self.A =A
        self.K =K
        self.pi = pi
        self.p0 = p0
        self.p1 = p1
        self.f0 = f0
        self.f1 = f1
        self.theta = theta
        self.N = len(QAver)-1
        self.dt =self.T/self.N
        self.C = C
        self.S0 = S0
        self.h1=h1
        self.h2 = h2
        self.sig0 = sig0
        self.sig = sig
        self.alphaTarget = alphaTarget
        self.jumpModel = jumpModel
        self.coeffEqui = coeffEqui
        

    # initialize
    def init(self, batchSize):
        self.batchSize = batchSize
        self.hQ = self.QAver[0]*tf.ones([batchSize])
        self.Q = self.QAver[0]*tf.ones([batchSize])
        self.expHQ = tf.ones([batchSize])
        self.expQ = tf.ones([batchSize])
        self.R =  self.R0*tf.ones([batchSize])
        self.hS = self.S0*tf.ones([batchSize])
        self.S = self.S0*tf.ones([batchSize])
        self.iStep =0


    # generate jumps dN at current date And compensator
    def dN(self):
        # rate of jumps
        if self.jumpModel == 'stochastic':
            self.lam = self.jumpFactor*(self.hQ)**2
        else:
            self.lam = self.jumpFactor*tf.ones([self.batchSize])
        # number of jump in dt
        return tf.random.poisson( [1], self.lam*self.dt)[0] , self.lam*self.dt
       

    # one steps
    def oneStepFrom(self, dW0, dW, dN , hY, Y):
        # update S
        self.hS= self.hS + self.calpha_hat(hY)*self.dt
        self.S= self.S + self.calpha(hY, Y)*self.dt
        # R evolution
        self.R = self.R +self.dt - tf.where(dN>0, self.R, 0)
        # update exp terms
        self.expHQ = self.expHQ*tf.exp(-0.5*self.sig0*self.sig0*self.dt + self.sig0*dW0)
        self.expQ = self.expQ*tf.exp(-0.5*(self.sig0*self.sig0 + self.sig*self.sig)*self.dt + self.sig0*dW0 + self.sig*dW)
        # update hQ, Q
        self.hQ = self.QAver[self.iStep+1]*self.expHQ
        self.Q = self.QAver[self.iStep+1]*self.expQ
        # update step
        self.iStep +=1
            
    # compute control  hAlpha  taking uncertainties at the current state
    #Compute stochastic alphaTarget
    def calphaTarget(self):
        if self.jumpModel == 'stochastic':
            return self.alphaTarget*(self.hQ)
        return self.alphaTarget*tf.ones([self.batchSize])

    # hY  Y value for BSDE
    def calpha_hat(self, hY):
        kTheta = self.A+(1-self.pi)*self.coeffEqui*self.p1+self.K+self.coeffEqui*self.f1*tf.where(self.R <= self.theta,1.,0.)
        return -(1/kTheta)*(self.p0 + self.pi*self.p1*self.hQ+ ((1-self.pi)*self.coeffEqui*self.p1 + self.K)*self.hQ + hY + \
                            (self.f0+self.coeffEqui*self.f1*(self.hQ - self.QAver[self.iStep]- self.calphaTarget()))*tf.where(self.R <= self.theta,1.,0.))

    def calpha(self, hY, Y):
      return -(1/(self.A + self.K))*(self.K*self.Q + self.p0 + self.pi*self.p1*self.hQ+ (1-self.pi)*self.coeffEqui*self.p1*(self.hQ +self.calpha_hat(hY)) + Y + \
          (self.f0+self.coeffEqui*self.f1*(self.hQ - self.QAver[self.iStep] + self.calpha_hat(hY) - self.calphaTarget()))*tf.where(self.R <= self.theta,1.,0.))

    # driver as function of Y
    def f(self, U):
        return U*self.C


    # terminal condition
    def g(self, X):
        return self.h1 + self.h2*X


    # get all states
    def getProjectedStates( self ):
        return self.iStep*self.dt, self.hQ, self.hS, self.R

    # get  states 
    def getAllStates( self ):
        return  self.iStep*self.dt, self.Q, self.S, self.hQ, self.hS, self.R


#Pricing Models
############################################################################################################################
# Variance Gamma Model
class VGmodel:
    def __init__(self, T, N, r, theta, kappa, sigmaJ, sigma, K, x0, epsilon):
        self.T = T                                                                         #Maturity
        self.r = r                                                                         #Interest rate
        self.sig = sigma                                                                   #volatility of diffusion part
        self.sigJ = sigmaJ                                                                 #volatility of small jumps
        self.theta = theta                                                                 #the drift of the BM approximating small jumps
        self.kappa = kappa                                                                 #variance of the Gamma process
        self.K = K                                                                         #srike
        self.N = N                                                                         #steps number
        self.dt = T/N                                                                      #steps length
        self.x0 = x0                                                                       #spot price
        self.eps = epsilon                                                                 #epsilon    
        self.correction = -tf.math.log(1 - theta * kappa - kappa/2 * sigmaJ**2 ) /kappa    #correction of the drift of the jump part
        #Levy measure
        self.mu = lambda z: np.exp((self.theta*z)/self.sigJ**2)*np.exp(-(np.sqrt(2/self.kappa + self.theta**2/self.sigJ**2))*abs(z)/(self.sigJ))/(self.kappa*abs(z))
        self.mu_sig =  lambda z: abs(z)*np.exp((self.theta*z)/self.sigJ**2)*np.exp(-(np.sqrt(2/self.kappa + self.theta**2/self.sigJ**2))*abs(z)/(self.sigJ))/(self.kappa)
        self.lam = quad(self.mu, self.eps, np.inf) + quad(self.mu, -np.inf, -self.eps)  
        self.lam = self.lam[0]                                        
        self.sigEps = quad(self.mu_sig, -self.eps, self.eps)[0]                             #compute truncated volatility 
        self.maxJumps = np.amax(np.random.poisson(self.lam*self.dt, 10**5))+1               #maximum number of Jumps

    #initialize
    def init(self, batchSize):
        self.iStep = 0                                                                     
        self.batchSize = batchSize
        self.X = self.x0*tf.ones([batchSize])

    
    #Go to next step
    def oneStepFrom(self, dW, gaussJ):
        self.X = self.x0*tf.exp((self.r - self.correction - 0.5*(self.sig + self.sigEps)*(self.sig + self.sigEps))*self.iStep*self.dt + (self.sig + self.sigEps)*dW + gaussJ)
        self.iStep += 1
    
    #jumps
    def jumps(self):
      dN = tf.random.poisson([self.batchSize], self.lam*self.dt, dtype = tf.float32)
      bindN = tf.sequence_mask(dN, self.maxJumps)
      gammaSim = tf.random.gamma([self.batchSize, self.maxJumps], self.iStep*self.dt/self.kappa, 1)*self.kappa
      gaussSim = tf.random.normal([self.batchSize, self.maxJumps], 0, 1)
      simulations =  tf.math.multiply(self.theta, gammaSim) + self.sigJ*tf.sqrt(gammaSim)*gaussSim
      listJumps = tf.where(bindN, simulations, tf.zeros([self.batchSize, self.maxJumps]))
      unstackedList = tf.unstack(listJumps, axis = 1)
      gaussJ = tf.reduce_sum(listJumps, axis = 1)
      return dN, unstackedList, gaussJ

    #Driver
    def f(self, Y):
        return -self.r*Y

    #Payoff 
    def g(self, X):
        return tf.maximum(X-self.K, 0)

    #Get states variables to inject in the nn
    def getStates(self, Y, dN, listJumps, gaussJ):
        return self.iStep*self.dt*tf.ones([self.batchSize]), self.X, self.g(self.X), Y*tf.ones([self.batchSize]), (self.iStep>0)*dN, *[(self.iStep>0)*x for x in listJumps], (self.iStep>0)*gaussJ

# Merton jump model
############################################################################################################################
class MertonJumpModel:
    def __init__(self, T, N, r, muJ, sigmaJ, sigma, lam, K, x0, maxJumps):
        self.T = T                      #Maturity
        self.r = r                      #Interest rate
        self.sig = sigma                #sigma BM asset 
        self.muJ = muJ                  #jump mean
        self.sigJ = sigmaJ              #jump std
        self.lam = lam                  #jump intensity
        self.K = K                      #srike
        self.N = N                      #steps number
        self.dt = T/N                   #steps length
        self.x0 = x0                    #current price asset
        self.maxJumps = maxJumps        #estimation of the maximal number of jumps

    #initialize
    def init(self, batchSize):
        self.iStep = 0                  #first step
        self.batchSize = batchSize
        self.expX = tf.ones([batchSize])
        self.X = self.x0*tf.ones([batchSize])
        

    
    #Go to next step
    def oneStepFrom(self, dW, gaussJ):
        self.expX = self.expX*tf.exp((self.r - 0.5*self.sig*self.sig - self.lam*(tf.exp(self.muJ  + self.sigJ*self.sigJ*0.5) - 1))*self.dt + self.sig*dW + gaussJ) 
        self.X = self.x0*self.expX
        self.iStep += 1

    #jumps
    def jumps(self):
      dN = tf.random.poisson([self.batchSize], self.lam*self.dt, dtype = tf.float32)
      bindN = tf.sequence_mask(dN, self.maxJumps)
      listJumps = tf.where(bindN,
                            tf.random.normal([self.batchSize, self.maxJumps], self.muJ, self.sigJ),
                            tf.zeros([self.batchSize, self.maxJumps]))
      unstackedList = tf.unstack(listJumps, axis = 1)
      gaussJ = tf.reduce_sum(listJumps, axis = 1)
      return dN, unstackedList, gaussJ
  

    #Driver
    def f(self, Y):
        return -self.r*Y

    #Payoff 
    def g(self, X):
        return tf.maximum(X-self.K, 0)

    #Get states variables to inject in the nn
    def getStates(self, dN, listJumps, gaussJ):
        return self.iStep*self.dt*tf.ones([self.batchSize]), self.X, self.g(self.X), (self.iStep>0)*dN, *[(self.iStep>0)*x for x in listJumps], (self.iStep>0)*gaussJ