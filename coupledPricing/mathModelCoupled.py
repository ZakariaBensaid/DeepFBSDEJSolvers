import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

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
        self.batchSize = batchSize
        return self.x0 *tf.ones([batchSize])
    
    #Analytic approach
    #BS closed formula
    def BS(self,iStep ,X , rbs, sigbs):
        shape = rbs.shape[0]
        d1 = (tf.math.log(X/self.K) + (rbs + sigbs**2/2)*(self.T- iStep*self.dt))/(sigbs*tf.sqrt(self.T- iStep*self.dt))
        d2 = (tf.math.log(X/self.K) + (rbs - sigbs**2/2)*(self.T- iStep*self.dt))/(sigbs*tf.sqrt(self.T- iStep*self.dt))
        return X*self.dist.cdf(d1) - self.K*tf.exp(-rbs*(self.T- iStep*self.dt))*self.dist.cdf(d2) 

    #Merton closed formula
    def A(self, iStep, X):
        if iStep < self.N :
          I = tf.range(20, dtype = tf.float32)
          rBS = tf.tile(tf.expand_dims(self.r - self.lam*(tf.exp(self.muJ  + self.sigJ*self.sigJ*0.5) - 1) + I*(self.muJ + 0.5*self.sigJ*self.sigJ)/(self.T - iStep*self.dt), axis=0),[tf.shape(X)[0],1])
          sigBS = tf.tile(tf.expand_dims(tf.sqrt(self.sig**2 + I*(self.sigJ**2)/(self.T - iStep*self.dt)), axis=0),[tf.shape(X)[0],1])
          BSincrements = self.BS(iStep,tf.tile(tf.expand_dims(X, axis=-1),[1,20]) , rBS, sigBS)
          lam2 = self.lam*tf.exp(self.muJ + 0.5*self.sigJ**2)
          coefficients = tf.tile(tf.expand_dims(tf.exp(-lam2*(self.T - iStep*self.dt))*((lam2*(self.T - iStep*self.dt))**I)/tf.exp(tf.math.lgamma(I + 1)), axis=0),[tf.shape(X)[0],1])
          return tf.reduce_sum(coefficients*BSincrements, axis = 1)
        return self.g(X)

  
    #Go to next step
    def oneStepFrom(self, iStep, X, dW, gaussJ, Y):
        return  X*tf.exp((self.r - 0.5*self.sig*self.sig - self.lam*(tf.exp(self.muJ  + self.sigJ*self.sigJ*0.5) - 1))*self.dt + self.sig*dW + gaussJ) + self.func(Y - self.A(iStep,X))*self.dt

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