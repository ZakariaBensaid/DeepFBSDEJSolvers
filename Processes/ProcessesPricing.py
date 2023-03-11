import numpy as np
import tensorflow as tf

class Processes:
    def __init__(self, mathModel, kerasModel, method):
        self.mathModel = mathModel
        self.kerasModel = kerasModel
        self.method = method
        #time vector
        self.t = np.arange(self.mathModel.N+1)
        #step
        self.dt = self.mathModel.dt
    

    def simulateAllProcesses(self, nbSimulations):   
        #Init
        self.mathModel.init(nbSimulations)        
        self.Y = np.zeros((nbSimulations, self.mathModel.N+1))
        self.Z = np.zeros((nbSimulations, self.mathModel.N+1))
        self.Gam = np.zeros((nbSimulations, self.mathModel.N+1))
        self.Compens = np.zeros((nbSimulations, self.mathModel.N+1))
        self.X = np.zeros((nbSimulations, self.mathModel.N+1))
        if self.method in ['Global']:
            self.Y[:,0] = self.kerasModel.Y0 + np.zeros(nbSimulations)
            for iStep in range(self.mathModel.N+1):
                #Simulations
                gaussian = tf.random.normal([nbSimulations])
                dW =  np.sqrt(self.mathModel.dt)*gaussian 
                # jump and compensation
                dN, listJumps, gaussJ  = self.mathModel.jumps()
                # variable states
                self.X[:,iStep] = self.mathModel.X.numpy()
                # get back adjoint variables
                self.Z[:,iStep], self.Gam[:,iStep], self.Compens[:,iStep] = [x.numpy() for x in self.kerasModel(self.mathModel.getStates(dN, listJumps, gaussJ))]
                #go to next step
                if not iStep == self.mathModel.N:
                    #BSDEs
                    self.Y[:,iStep + 1] = self.Y[:,iStep] - self.mathModel.dt* self.mathModel.f(self.Y[:,iStep])\
                        + self.Z[:,iStep]*dW.numpy() + self.Gam[:,iStep] - self.Compens[:,iStep]*self.mathModel.dt
                    self.mathModel.oneStepFrom(dW, gaussJ)