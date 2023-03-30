import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


class MFGSolutionsFixedTrajectory:
    def __init__(self, mathModel, kerasModel, method,  dW0_arr, dW_arr, dN, savefig = 'OFF'):
        self.mathModel = mathModel
        self.kerasModel = kerasModel
        self.savefig = savefig
        self.method = method
        self.dW0_arr = dW0_arr
        self.dW_arr = dW_arr
        self.dN = dN
        #time vector
        self.t = np.arange(self.mathModel.N+1)
        #step
        self.dt = self.mathModel.dt
        #theta
        self.theta = self.mathModel.theta
    

    def simulateAllProcesses(self, nbSimulations):
        if nbSimulations > tf.shape(self.dN)[0]:
          return 'Shape error, please change the number of simulations.'
        #Nb Simulations
        dW0_arr = self.dW0_arr[:nbSimulations]
        dW_arr = self.dW_arr[:nbSimulations]
        dN_arr = self.dN[:nbSimulations]        
        #Init
        self.mathModel.init(nbSimulations)        
        self.R = np.zeros((nbSimulations, self.mathModel.N+1))
        self.hQ = np.zeros((nbSimulations, self.mathModel.N+1))
        self.meanhQ = np.zeros(self.mathModel.N+1)
        self.Q = np.zeros((nbSimulations, self.mathModel.N+1))
        self.lam = np.zeros((nbSimulations, self.mathModel.N+1))
        self.hS = np.zeros((nbSimulations, self.mathModel.N+1))
        self.S = np.zeros((nbSimulations, self.mathModel.N+1))
        self.alpha_hat = np.zeros((nbSimulations, self.mathModel.N+1))
        self.alpha = np.zeros((nbSimulations, self.mathModel.N+1))
        if self.method in ['Global']:
            tensorhY = self.kerasModel.model_hat.Y0_hat
            tensorY = self.kerasModel.model.Y0
            for iStep in range(self.mathModel.N+1):
                self.Q[:,iStep], self.S[:,iStep], self.hQ[:,iStep], self.hS[:,iStep], self.R[:,iStep] \
                    = [x.numpy() for x in self.mathModel.getAllStates()[1:]]
                self.meanhQ[iStep] = self.mathModel.meanhQ
                # get back adjoint variables
                hZ0, hGam = self.kerasModel.model_hat(self.mathModel.getProjectedStates())
                Z0, Gam, Z = self.kerasModel.model(self.mathModel.getAllStates())
                # jump and compensation
                dN , compens  = dN_arr[:,iStep], self.mathModel.dN()[1]
                self.lam[:,iStep] = self.mathModel.lam
                # Noise increment
                dW0 = dW0_arr[:,iStep]
                dW = dW_arr[:,iStep]
                #BSDEs
                tensorhYNext = tensorhY - self.mathModel.dt* self.mathModel.f(self.mathModel.hS)\
                        + hZ0* dW0 + hGam*(dN-compens)
                tensorYNext = tensorY - self.mathModel.dt* self.mathModel.f(self.mathModel.S) \
                    + Z0* dW0 + Gam*(dN-compens) + Z*dW
                #Compute alpha
                self.alpha_hat[:,iStep] = self.mathModel.calpha_hat(tensorhY).numpy()
                self.alpha[:,iStep] = self.mathModel.calpha(tensorhY, tensorY).numpy()
                #go to next step
                if not iStep == self.mathModel.N:
                    self.mathModel.oneStepFrom(dW0, dW, dN , tensorhY, tensorY)
                #next values
                tensorhY, tensorY =  tensorhYNext, tensorYNext
        else :
            tensorhY, = self.kerasModel.model_hat(self.mathModel.getProjectedStates())
            tensorY, = self.kerasModel.model(self.mathModel.getAllStates())
            for iStep in range(self.mathModel.N+1):
                self.Q[:,iStep], self.S[:,iStep], self.hQ[:,iStep], self.hS[:,iStep], self.R[:,iStep] \
                    = [x.numpy() for x in self.mathModel.getAllStates()[1:]]
                self.meanhQ[iStep] = self.mathModel.meanhQ
                #Compute alpha
                self.alpha_hat[:,iStep] = self.mathModel.calpha_hat(tensorhY).numpy()
                self.alpha[:,iStep] = self.mathModel.calpha(tensorhY, tensorY).numpy()
                #go to next step
                if not iStep == self.mathModel.N:
                    # jump and compensation
                    dN , compens  = dN_arr[:,iStep], self.mathModel.dN()[1]
                    self.lam[:,iStep] = self.mathModel.lam
                    # Noise increment
                    dW0 = dW0_arr[:,iStep]
                    dW = dW_arr[:,iStep]
                    self.mathModel.oneStepFrom(dW0, dW, dN , tensorhY, tensorY)
                    # get back adjoint variables
                    tensorhY, = self.kerasModel.model_hat(self.mathModel.getProjectedStates())
                    tensorY, = self.kerasModel.model(self.mathModel.getAllStates())

    def computeTarget(self, nbSimulations):
        if self.mathModel.jumpModel == 'stochastic':
            self.alphaTg = self.mathModel.alphaTarget*(np.concatenate([[self.meanhQ]*nbSimulations], axis = 1))
        else :
            self.alphaTg = self.mathModel.alphaTarget*np.ones((nbSimulations, self.mathModel.N+1))

    def price(self, pi, alpha):
        return self.mathModel.p0 + pi*self.mathModel.p1*self.hQ + (1-pi)*self.mathModel.p1*(self.hQ + alpha) 
    
    def objectiveFunction(self):
        #Compute the increment of the integral
        increment = self.mathModel.A*0.5*self.alpha**2 + self.mathModel.C*0.5*self.S**2 + self.mathModel.K*0.5*(self.Q + self.alpha)**2 + \
            (self.Q + self.alpha)*(self.mathModel.p0 + self.mathModel.p1*self.mathModel.pi*self.hQ + self.mathModel.p1*(1-self.mathModel.pi)*(self.hQ + self.alpha_hat)) \
                + (self.R < self.mathModel.theta)*(self.Q - self.meanhQ + self.alpha - self.alphaTg)*(self.mathModel.f0 + self.mathModel.f1*(self.hQ - \
                    self.mathModel.meanhQ + self.alpha_hat - self.alphaTg))
        #Compute the integral
        cost_integral = np.sum(increment*self.mathModel.dt, axis = 1) + self.mathModel.h1*self.S[:,-1] + self.mathModel.h2*0.5*self.S[:,-1]**2
        return np.mean(cost_integral), np.std(cost_integral)