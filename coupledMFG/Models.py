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


