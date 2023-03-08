import numpy as np
import tensorflow as tf

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
        self.XJsum = self.x0*tf.ones([batchSize, self.maxJumps])
        

    
    #Go to next step
    def oneStepFrom(self, dW, gaussJ, listJumps):
        self.expX = self.expX*tf.exp((self.r - 0.5*self.sig*self.sig - self.lam*(tf.exp(self.muJ  + self.sigJ*self.sigJ*0.5) - 1))*self.dt + self.sig*dW + gaussJ) 
        self.X = self.x0*self.expX
        self.XJsum = tf.stack([self.X]*self.maxJumps, axis = 1) + tf.math.cumsum(tf.stack(listJumps, axis = 1), axis = 1)
        self.iStep += 1

    #jumps
    def jumps(self):
      lam = self.lam*tf.ones([self.batchSize])
      dN = tf.random.poisson([1], lam*self.dt, dtype = tf.float32)[0]
      bindN = tf.sequence_mask(dN, self.maxJumps, dtype = tf.float32)
      listJumps = tf.where(bindN > 0,
                            tf.random.normal([self.batchSize, self.maxJumps], self.muJ, self.sigJ),
                            tf.zeros([self.batchSize, self.maxJumps]))
      unstackedList = tf.unstack(listJumps, axis = 1)
      gaussJ = tf.reduce_sum(listJumps, axis = 1)
      return dN, unstackedList, gaussJ
  

    #Driver
    def f(self, Y):
        return -tf.multiply(self.r,Y)

    #Payoff 
    def g(self, X):
        return tf.maximum(X-self.K, 0)

    #Get states variables to inject in the nn
    def getStates(self, Y, dN, listJumps, gaussJ):
        return self.iStep*self.dt*tf.ones([self.batchSize]), self.X, self.g(self.X), Y*tf.ones([self.batchSize]), (self.iStep>0)*dN, *[(self.iStep>0)*x for x in listJumps], (self.iStep>0)*gaussJ

    #Get states for U
    def getStatesU(self, l):
      return self.iStep*self.dt*tf.ones([self.batchSize]), self.XJsum[:,l], self.g(self.XJsum[:,l])

    #Get states for Z
    def getStatesZ(self):
      return self.iStep*self.dt*tf.ones([self.batchSize]), self.X, self.g(self.X)