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

class SolverGlobalFBSDE(SolverBase):
    def __init__(self, mathModel, modelKeras, lRate):
        super().__init__(mathModel, modelKeras, lRate)

    def train(self,  batchSize,  batchSizeVal, num_epoch, num_epochExt):

        @tf.function
        def optimizeBSDE( nbSimul):
            # initialize
            self.mathModel.init(nbSimul)
            # get back initial value
            Y = self.modelKeras.model.Y0
            for istep in range(self.mathModel.N):
                # increment
                gaussian = tf.random.normal([nbSimul])
                dW =  np.sqrt(self.mathModel.dt)*gaussian 
                # jump and compensation
                dN, listJumps, gaussJ  = self.mathModel.jumps()
                # get back Z,U
                Z = self.modelKeras.modelZ(self.mathModel.getStatesZ())
                YList = tf.stack([self.modelKeras.model(self.mathModel.getStatesU(l)) for l in range(self.mathModel.maxJumps + 1)], axis = 1) 
                # target
                Y = Y - self.mathModel.dt* self.mathModel.f(Y) + Z*dW + tf.reduce_sum(YList[:,1:] - YList[:,:-1], axis = 1)\
                 - tf.reduce_mean(tf.reduce_sum(YList[:,1:] - YList[:,:-1], axis = 1))
                # next t step
                self.mathModel.oneStepFrom(dW, gaussJ, listJumps)
            return  tf.reduce_mean(tf.square(Y- self.mathModel.g(self.mathModel.X)))

        # train to optimize control
        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc = optimizeBSDE( nbSimul)
            CoupledVariables = self.modelKeras.model.trainable_variables + self.modelKeras.modelZ.trainable_variables
            gradients= tape.gradient(objFunc,  CoupledVariables)
            optimizer.apply_gradients(zip(gradients,  CoupledVariables))
            return objFunc

        optimizer= optimizers.Adam(learning_rate = self.lRate)

        self.listY0 = []
        self.lossList = []
        self.duration = 0
        for iout in range(num_epochExt):
            start_time = time.time()
            for epoch in range(num_epoch):
                # un pas de gradient stochastique
                trainOpt(batchSize, optimizer)
            end_time = time.time()
            rtime = end_time-start_time
            self.duration += rtime
            objError = optimizeBSDE(batchSizeVal)
            Y0 = self.modelKeras.model.Y0
            print(" Error",objError.numpy(),  " elapsed time %5.3f s" % self.duration, "Y0 sofar ",Y0.numpy(), 'epoch', iout)
            self.listY0.append(Y0.numpy())
            self.lossList.append(objError)   
        return self.listY0 

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
                YList = tf.stack([self.modelKeras.model(self.mathModel.getStatesU(l)) for l in range(self.mathModel.maxJumps + 1)], axis = 1)
                # target
                toAdd = - self.mathModel.dt*self.mathModel.f(YList[:,0]) + Z*dW + tf.reduce_sum(YList[:,1:] - YList[:,:-1], axis = 1)\
                 - tf.reduce_mean(tf.reduce_sum(YList[:,1:] - YList[:,:-1], axis = 1))
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
            YListPrev = tf.stack([self.modelKeras.model(self.mathModel.getStatesU(l)) for l in range(self.mathModel.maxJumps + 1)], axis = 1)
            for istep in range(self.mathModel.N):
                # increment
                gaussian = tf.random.normal([nbSimul])
                dW =  np.sqrt(self.mathModel.dt)*gaussian 
                # target
                toAdd = self.mathModel.dt*self.mathModel.f(YListPrev[:,0]) - ZPrev*dW - tf.reduce_sum(YListPrev[:,1:] - YListPrev[:,:-1] , axis = 1)\
                 + tf.reduce_mean(tf.reduce_sum(YListPrev[:,1:] - YListPrev[:,:-1] , axis = 1))
                # next step
                self.mathModel.oneStepFrom(dW, gaussJ, listJumps)
                # jump and compensation
                dN, listJumps, gaussJ  = self.mathModel.jumps()
                if (istep == (self.mathModel.N-1)):
                    YListNext = tf.stack([self.mathModel.g(self.mathModel.X)]*(self.mathModel.maxJumps+1), axis =1)
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
            Y0 = self.modelKeras.model(self.mathModel.getStatesU(0))[0][0]
            print(" Error",objError.numpy(),  " took %5.3f s" % rtime, "Y0 sofar ",Y0.numpy(), 'epoch', iout)
            self.listY0.append(Y0.numpy())
            self.lossList.append(objError)   
        return self.listY0 

# global as sum of local error due to regressions
# see algorithm 
class SolverGlobalSumLocalReg(SolverBase):
    def __init__(self, mathModel,   modelKeras, lRate):
        super().__init__(mathModel,   modelKeras, lRate)
      
    def train( self,  batchSize,  batchSizeVal, num_epoch, num_epochExt):
        @tf.function
        def regressOptim( nbSimul):
            # initialize
            self.mathModel.init(nbSimul)
            dN, listJumps, gaussJ  = self.mathModel.jumps()
            # Target
            error = 0.
            # get back Y
            YPrev, = self.modelKeras.model(self.mathModel.getStatesU(0))
            for istep in range(self.mathModel.N):
                # target
                toAdd = - self.mathModel.dt* self.mathModel.f(YPrev)
                # increment
                gaussian = tf.random.normal([nbSimul])
                dW =  np.sqrt(self.mathModel.dt)*gaussian 
                # jumps
                dN, listJumps, gaussJ  = self.mathModel.jumps()
                # Next step
                self.mathModel.oneStepFrom(dW, gaussJ, listJumps)
                # values
                if (istep == (self.mathModel.N-1)):
                    YNext = self.mathModel.g(self.mathModel.X)
                else:
                    YNext, = self.modelKeras(self.modelKeras.model(self.mathModel.getStatesU(0)))
                error = error +  tf.reduce_mean(tf.square(YPrev- YNext  + toAdd))
                YPrev = YNext
            return error
    
        # train to optimize control
        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= regressOptim( nbSimul)
            gradients= tape.gradient(objFunc, self.modelKeras.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKeras.model.trainable_variables))
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
            objError = regressOptim(batchSizeVal)
            self.mathModel.init(1)
            Y0 = self.modelKeras.model(self.mathModel.getStatesU(0))[0][0]
            print(" Error",objError.numpy(),  " took %5.3f s" % rtime, "Y0 sofar ",Y0.numpy(), 'epoch', iout)
            self.listY0.append(Y0.numpy())
            self.lossList.append(objError)   
        return self.listY0 
  

      
# global as multiStep regression  for hatY
# see algorithm 
# global as multiStep regression  for hatY
# see algorithm 
class SolverGlobalMultiStepReg(SolverBase):
    def __init__(self, mathModel,   modelKeras, lRate):
        super().__init__(mathModel,   modelKeras, lRate)

        
    def train( self,  batchSize,  batchSizeVal, num_epoch, num_epochExt):
        @tf.function
        def regressOptim( nbSimul):
            # initialize
            self.mathModel.init(nbSimul)
            # Target
            listOfForward = []
            for istep in range(self.mathModel.N): 
                # get back Y
                Y, = self.modelKeras.model(self.mathModel.getStatesU(0))
                # listforward
                listOfForward.append(Y)                 
                # to Add
                toAdd =- self.mathModel.dt* self.mathModel.f(Y)
                for i in range(len(listOfForward)):
                    listOfForward[i] = listOfForward[i] + toAdd
                # increment
                gaussian = tf.random.normal([nbSimul])
                dW =  np.sqrt(self.mathModel.dt)*gaussian
                # next t step
                dN, listJumps, gaussJ  = self.mathModel.jumps()
                self.mathModel.oneStepFrom(dW, gaussJ, listJumps)
            # final U
            Yfinal = self.mathModel.g(self.mathModel.X)
            listOfForward = tf.stack(listOfForward, axis=0) 
            return tf.reduce_sum(tf.reduce_mean(tf.reduce_mean(tf.square(listOfForward - tf.tile(tf.expand_dims(Yfinal,axis=0), [self.mathModel.N,1])), axis=-1), axis=-1))
                
        # train to optimize control
        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= regressOptim( nbSimul)
            gradients= tape.gradient(objFunc, self.modelKeras.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKeras.model.trainable_variables))
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
            objError = regressOptim(batchSizeVal)
            self.mathModel.init(1)
            Y0 = self.modelKeras.model(self.mathModel.getStatesU(0))[0][0]
            print(" Error",objError.numpy(),  " took %5.3f s" % rtime, "Y0 sofar ",Y0, 'epoch', iout)
            self.listY0.append(Y0)
            self.lossList.append(objError)   
        return self.listY0 
  
class SolverOsterleeFBSDE(SolverBase):
    def __init__(self, mathModel,   modelKeras, lRate, lamCoef):
        super().__init__(mathModel,   modelKeras, lRate)
        self.lamCoef = lamCoef

    def train(self,  batchSize,  batchSizeVal, num_epoch, num_epochExt):

        @tf.function
        def optimizeBSDE( nbSimul):
            #Compute Y0
            ####################################################################
            # initialize
            self.mathModel.init(nbSimul)   
            Y0 = 0                   
            for istep in range(self.mathModel.N):
                # increment
                gaussian = tf.random.normal([nbSimul])
                dW =  np.sqrt(self.mathModel.dt)*gaussian 
                # jump and compensation
                dN, listJumps, gaussJ  = self.mathModel.jumps()
                # get back U, Z, 
                Z = self.modelKeras.modelZ(self.mathModel.getStatesZ())
                Y = self.modelKeras.model(self.mathModel.getStatesU(0))
                # target
                Y0 += self.mathModel.dt* self.mathModel.f(Y)
                # next t step
                self.mathModel.oneStepFrom(dW, gaussJ, listJumps)
            Y0 += self.mathModel.g(self.mathModel.X)
            # initial value
            Y = tf.reduce_mean(Y0)
            #Compute error
            ####################################################################
            # initialize
            self.mathModel.init(nbSimul)  
            for istep in range(self.mathModel.N):
                # increment
                gaussian = tf.random.normal([nbSimul])
                dW =  np.sqrt(self.mathModel.dt)*gaussian 
                # jump and compensation
                dN, listJumps, gaussJ  = self.mathModel.jumps()
                # get back Y, Z, Gam
                Z = self.modelKeras.modelZ(self.mathModel.getStatesZ())
                YList = tf.stack([self.modelKeras.model(self.mathModel.getStatesU(l)) for l in range(self.mathModel.maxJumps + 1)], axis = 1)
                # adjoint variables
                Y = Y - self.mathModel.dt*self.mathModel.f(Y) + Z*dW + + tf.reduce_sum(YList[:,1:] - YList[:,:-1], axis = 1)\
                 - tf.reduce_mean(tf.reduce_sum(YList[:,1:] - YList[:,:-1], axis = 1))           
                # next t step
                self.mathModel.oneStepFrom(dW, gaussJ, listJumps)
            return  tf.reduce_mean(Y0) + self.lamCoef*(tf.reduce_mean(tf.square(Y - self.mathModel.g(self.mathModel.X)))), tf.reduce_mean(Y0)
                
        # train to optimize control
        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= optimizeBSDE(nbSimul)
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
            objError, Y0 = optimizeBSDE(batchSizeVal)
            print(" Error",objError.numpy(),  " took %5.3f s" % rtime, "Y0 sofar ",Y0.numpy(), 'epoch', iout)
            self.listY0.append(Y0.numpy())
            self.lossList.append(objError)   
        return self.listY0 