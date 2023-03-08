import numpy as np
import tensorflow as tf
import time
from tensorflow.keras import  optimizers


class SolverBase:
    # mathModel          Math model
    # modelKeras         Keras model 
    # lRate              Learning rate
    def __init__(self, mathModel,   modelKeras, lRate):
        # to store les different networks
        self.mathModel = mathModel
        self.modelKeras= modelKeras
        self.lRate= lRate

    
class SolverMultiStepFBSDE(SolverBase):
    def __init__(self, mathModel, modelKeras, lRate):
        super().__init__(mathModel,  modelKeras, lRate)

    def train(self,  batchSize,  batchSizeVal, num_epoch, num_epochExt):

        @tf.function
        def optimizeBSDE( nbSimul):
            # initialize
            self.mathModel.init(nbSimul)
            dN, listJumps, gaussJ  = self.mathModel.jumps()
            # Target
            listOfForward = []
            for istep in range(self.mathModel.N):
                # Common and individual noises
                gaussian = tf.random.normal([nbSimul])
                dW =  np.sqrt(self.mathModel.dt)*gaussian 
                # Adjoint variables 
                Z = self.modelKeras.modelZ(self.mathModel.getStatesZ())
                YList = tf.stack([self.modelKeras.model(self.mathModel.getStatesU(l)) for l in range(self.mathModel.maxJumps)], axis = 1)
                # target
                toAdd = - self.mathModel.dt*self.mathModel.f(YList[:,0]) + Z*dW + tf.reduce_sum(YList, axis = 1) - tf.reduce_sum(YList[:,:-1], axis = 1)\
                 - tf.reduce_mean(tf.reduce_sum(YList, axis = 1) - tf.reduce_sum(YList[:,:-1] , axis = 1))
                #update list and error
                listOfForward.append(YList[:,0])
                #forward
                for i in range(len(listOfForward)):
                    listOfForward[i] = listOfForward[i] + toAdd
                # next t step
                self.mathModel.oneStepFrom(dW, gaussJ, listJumps)
                # jump and compensation
                dN, listJumps, gaussJ  = self.mathModel.jumps()
            #Final Y
            Yfinal = self.mathModel.g(self.mathModel.X)
            listOfForward = tf.stack(listOfForward, axis=0)
            return tf.reduce_sum(tf.reduce_mean(tf.reduce_mean(tf.square(listOfForward - tf.tile(tf.expand_dims(Yfinal,axis=0), [self.mathModel.N,1])), axis=-1), axis=-1))
                
        # train to optimize control
        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= optimizeBSDE( nbSimul)
            CoupledVariables = self.modelKeras.model.trainable_variables + self.modelKeras.modelZ.trainable_variables
            gradients= tape.gradient(objFunc,  CoupledVariables)
            optimizer.apply_gradients(zip(gradients,  CoupledVariables))
            return objFunc

        optimizer= optimizers.Adam(learning_rate = self.lRate)

        self.listY0 = []
        self.lossList = []
        for iout in range(num_epochExt):
            start_time = time.time()
            for epoch in range(num_epoch):
                # un pas de gradient stochastique
                trainOpt(batchSize, optimizer)
            end_time = time.time()
            rtime = end_time-start_time 
            objError = optimizeBSDE(batchSizeVal)
            self.mathModel.init(1)
            dN, listJumps, gaussJ  = self.mathModel.jumps()
            Y0 = self.modelKeras.model(self.mathModel.getStatesU(0))[0][0]
            print(" Error",objError.numpy(),  " took %5.3f s" % rtime, "Y0 sofar ",Y0.numpy(), 'epoch', iout)
            self.listY0.append(Y0.numpy())
            self.lossList.append(objError)   
        return self.listY0 
                
        
class SolverSumLocalFBSDE(SolverBase):
    def __init__(self, mathModel,   modelKeras, lRate):
        super().__init__(mathModel,   modelKeras, lRate)

    def train(self,  batchSize,  batchSizeVal, num_epoch, num_epochExt):

        @tf.function
        def optimizeBSDE( nbSimul):
            # initialize
            self.mathModel.init(nbSimul)
            dN, listJumps, gaussJ  = self.mathModel.jumps()
            #error
            error = 0
            #init val
            ZPrev = self.modelKeras.modelZ(self.mathModel.getStatesZ())
            YListPrev = tf.stack([self.modelKeras.model(self.mathModel.getStatesU(l)) for l in range(self.mathModel.maxJumps)], axis = 1)
            for istep in range(self.mathModel.N):
                # increment
                gaussian = tf.random.normal([nbSimul])
                dW =  np.sqrt(self.mathModel.dt)*gaussian 
                # target
                toAdd = self.mathModel.dt*self.mathModel.f(YListPrev[:,0]) - ZPrev*dW + tf.reduce_sum(YListPrev, axis = 1) - tf.reduce_sum(YListPrev[:,:-1] , axis = 1)\
                 - tf.reduce_mean(tf.reduce_sum(YListPrev, axis = 1) - tf.reduce_sum(YListPrev[:,:-1] , axis = 1))
                # next step
                self.mathModel.oneStepFrom(dW, gaussJ, listJumps)
                # jump and compensation
                dN, listJumps, gaussJ  = self.mathModel.jumps()
                if (istep == (self.mathModel.N-1)):
                    YListNext = tf.stack([self.mathModel.g(self.mathModel.X)]*self.mathModel.maxJumps, axis =1)
                else:
                  ZNext = self.modelKeras.modelZ(self.mathModel.getStatesZ())
                  YListNext = tf.stack([self.modelKeras.model(self.mathModel.getStatesU(l)) for l in range(self.mathModel.maxJumps)], axis = 1)
                error = error + tf.reduce_mean(tf.square(YListNext[:,0] - YListPrev[:,0] + toAdd))
                YListPrev = YListNext
            return error
                
        # train to optimize control
        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= optimizeBSDE( nbSimul)
            CoupledVariables = self.modelKeras.model.trainable_variables + self.modelKeras.modelZ.trainable_variables
            gradients= tape.gradient(objFunc,  CoupledVariables)
            optimizer.apply_gradients(zip(gradients,  CoupledVariables))
            return objFunc

        optimizer= optimizers.Adam(learning_rate = self.lRate)

        self.listY0 = []
        self.lossList = []
        for iout in range(num_epochExt):
            start_time = time.time()
            for epoch in range(num_epoch):
                # un pas de gradient stochastique
                trainOpt(batchSize, optimizer)
            end_time = time.time()
            rtime = end_time-start_time 
            objError = optimizeBSDE(batchSizeVal)
            self.mathModel.init(1)
            dN, listJumps, gaussJ  = self.mathModel.jumps()
            Y0 = self.modelKeras.model(self.mathModel.getStatesU(0))[0][0]
            print(" Error",objError.numpy(),  " took %5.3f s" % rtime, "Y0 sofar ",Y0.numpy(), 'epoch', iout)
            self.listY0.append(Y0.numpy())
            self.lossList.append(objError)   
        return self.listY0 