import numpy as np
import tensorflow as tf

class MertonJumpModel:
    def __init__(self, T, N, r, muJ, sigmaJ, sigma, lam, K, x0, maxJumps, func):
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
        self.func = func                #functor
        self.dist = tfd.Normal(loc=0., scale=1.)

    #initialize
    def init(self, batchSize):
        self.iStep = 0                  #first step
        self.batchSize = batchSize
        self.expX = tf.ones([batchSize])
        self.Xbar = self.x0*tf.ones([batchSize])
        self.X = self.x0*tf.ones([batchSize])
        self.F = 0.
    
    #Analytic approach
    #BS closed formula
    def BS(self, rBS, sigBS):
        shape = rbs.shape[0]
        X = tf.stack([self.Xbar]*shape, axis = 1)
        d1 = (tf.math.log(X/self.K) + (rBS + sigBS**2/2)*(self.T- self.iStep*self.dt))/(sigbs*tf.sqrt(self.T- self.iStep*self.dt))
        d2 = (tf.math.log(X/self.K) + (rBS - sigBS**2/2)*(self.T- self.iStep*self.dt))/(sigbs*tf.sqrt(self.T- self.iStep*self.dt))
        return X*self.dist.cdf(d1) - self.K*tf.exp(-rBS*(self.T- self.iStep*self.dt))*self.dist.cdf(d2) 

    #Merton closed formula
    def A(self):
        I = tf.range(100, dtype = tf.float32)
        rBS = self.r - self.lam*(tf.exp(self.muJ  + self.sigJ*self.sigJ*0.5) - 1) + I*(self.muJ + 0.5*self.sigJ*self.sigJ)/(self.T - self.iStep*self.dt)
        sigBS = tf.sqrt(self.sig**2 + I*(self.sigJ**2)/(self.T - self.iStep*self.dt))
        BSincrements = self.BS(rBS, sigBS)
        lam2 = self.lam*tf.exp(self.muJ + 0.5*self.sigJ**2)
        coefficients = tf.exp(-lam2*(self.T - self.iStep*self.dt))*((lam2*(self.T - self.iStep*self.dt))**I)/tf.exp(tf.math.lgamma(I + 1))
        return tf.reduce_sum(coefficients*BSincrements, axis = 1)

  
    #Go to next step
    def oneStepFrom(self, dW, gaussJ, Y):
        self.expX = self.expX*tf.exp((self.r - 0.5*self.sig*self.sig - self.lam*(tf.exp(self.muJ  + self.sigJ*self.sigJ*0.5) - 1))*self.dt + self.sig*dW + gaussJ) 
        self.F += self.func(Y - self.A())*self.dt
        self.Xbar = self.x0*self.expX
        self.X = self.Xbar + self.F
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
    def getStates(self, Y, dN, listJumps, gaussJ):
        return self.iStep*self.dt*tf.ones([self.batchSize]), self.X, self.g(self.X), Y*tf.ones([self.batchSize]), (self.iStep>0)*dN, *[(self.iStep>0)*x for x in listJumps], (self.iStep>0)*gaussJ