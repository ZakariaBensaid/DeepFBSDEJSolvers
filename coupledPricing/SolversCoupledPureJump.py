import numpy as np
import tensorflow as tf
import time
from tensorflow.keras import  optimizers

class SolverBase:
    # mathModel          Math model
    # modelKeras         Keras model 
    # lRate              Learning rate
    def __init__(self, mathModel, modelKerasU , modelKerasGam,  lRate):
        # to store les different networks
        self.mathModel = mathModel
        self.modelKerasU = modelKerasU
        self.modelKerasGam = modelKerasGam 
        self.lRate = lRate

class SolverGlobalFBSDE(SolverBase):
    def __init__(self, mathModel, modelKerasU , modelKerasGam,  lRate):
        super().__init__(mathModel, modelKerasU , modelKerasGam,  lRate)
        
    def train(self,  batchSize,  batchSizeVal, num_epoch, num_epochExt):
        @tf.function
        def optimizeBSDE( nbSimul):
            # initialize
            X = self.mathModel.init(nbSimul)
            # Target
            Y = self.modelKerasGam.Y0*tf.ones([nbSimul])
            for iStep in range(self.mathModel.N):
                # jumps
                gaussJ  = self.mathModel.jumps()
                # get  Gam
                Gam = self.modelKerasGam(tf.stack([iStep*tf.ones([nbSimul], dtype= tf.float32), X , gaussJ], axis=-1))[0]
                # target
                Y = Y - self.mathModel.dt*self.mathModel.f(Y) + Gam - tf.reduce_mean(Gam)
                # next t step
                X = self.mathModel.oneStepFrom(iStep, X, gaussJ, Y)
            return  tf.reduce_mean(tf.square(Y- self.mathModel.g(X)))

        # train to optimize control
        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc = optimizeBSDE( nbSimul)
            gradients= tape.gradient(objFunc, self.modelKerasGam.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKerasGam.trainable_variables))
            return objFunc

        optimizer = optimizers.Adam(learning_rate = self.lRate)

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
            Y0 = self.modelKerasGam.Y0
            print(" Error",objError.numpy(),  " elapsed time %5.3f s" % self.duration, "Y0 sofar ",Y0.numpy(), 'epoch', iout)
            self.listY0.append(Y0.numpy())
            self.lossList.append(objError)   
        return self.listY0 

class SolverMultiStepFBSDE1():
    def __init__(self, mathModel, modelKerasU , lRate):
        # to store les different networks
        self.mathModel = mathModel
        self.modelKerasU = modelKerasU
        self.lRate = lRate

    def train(self,  batchSize,  batchSizeVal, num_epoch, num_epochExt):

        @tf.function
        def optimizeBSDE( nbSimul):
            # initialize
            X = self.mathModel.init(nbSimul)
            # Target
            listOfForward = []
            for iStep in range(self.mathModel.N):
                # jumps
                gaussJ  = self.mathModel.jumps()
                # Adjoint variables 
                Y = self.modelKerasU(tf.stack([iStep* tf.ones([nbSimul], dtype= tf.float32) , X], axis=-1))[0]
                Gam =  self.modelKerasU(tf.stack([iStep* tf.ones([nbSimul], dtype= tf.float32), X + gaussJ], axis=-1))[0] \
                - self.modelKerasU(tf.stack([iStep* tf.ones([nbSimul], dtype= tf.float32), X], axis=-1))[0] 
                # target
                toAdd = - self.mathModel.dt*self.mathModel.f(Y) + Gam - tf.reduce_mean(Gam)
                #update list
                listOfForward.append(Y)
                #forward
                for i in range(len(listOfForward)):
                    listOfForward[i] = listOfForward[i] + toAdd
                # next t step
                X = self.mathModel.oneStepFrom(iStep, X, gaussJ, Y)
            #Final Y
            Yfinal = self.mathModel.g(X)
            listOfForward = tf.stack(listOfForward, axis=0)
            return tf.reduce_sum(tf.reduce_mean(tf.reduce_mean(tf.square(listOfForward - tf.tile(tf.expand_dims(Yfinal,axis=0), [self.mathModel.N,1])), axis=-1), axis=-1))
                
        # train to optimize control
        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= optimizeBSDE( nbSimul)
            gradients= tape.gradient(objFunc, self.modelKerasU.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKerasU.trainable_variables))
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
            X = self.mathModel.init(1)
            Y0 = self.modelKerasU( tf.stack([tf.zeros([1], dtype= tf.float32) , X], axis=-1))[0][0]
            print(" Error",objError.numpy(),  " took %5.3f s" % rtime, "Y0 sofar ",Y0.numpy(), 'epoch', iout)
            self.listY0.append(Y0.numpy())
            self.lossList.append(objError)   
        return self.listY0 

class SolverMultiStepFBSDE2(SolverBase):
    def __init__(self, mathModel, modelKerasU , modelKerasGam,  lRate):
        super().__init__(mathModel, modelKerasU , modelKerasGam,  lRate)

    def train(self,  batchSize,  batchSizeVal, num_epoch, num_epochExt):

        @tf.function
        def optimizeBSDE( nbSimul):
            # initialize
            X = self.mathModel.init(nbSimul)
            # Target
            listOfForward = []
            for iStep in range(self.mathModel.N):
                # Common and individual noises
                gaussJ  = self.mathModel.jumps()
                # Adjoint variables 
                Y = self.modelKerasU(tf.stack([iStep* tf.ones([nbSimul], dtype= tf.float32) , X], axis=-1))[0]
                Gam =  self.modelKerasGam(tf.stack([iStep* tf.ones([nbSimul], dtype= tf.float32),X , gaussJ], axis=-1) )[0]
                # target
                toAdd = - self.mathModel.dt*self.mathModel.f(Y) + Gam - tf.reduce_mean(Gam)
                #update list and error
                listOfForward.append(Y)
                #forward
                for i in range(len(listOfForward)):
                    listOfForward[i] = listOfForward[i] + toAdd
                # next t step
                X = self.mathModel.oneStepFrom(iStep, X, gaussJ, Y)
            #Final Y
            Yfinal = self.mathModel.g(X)
            listOfForward = tf.stack(listOfForward, axis=0)
            return tf.reduce_sum(tf.reduce_mean(tf.reduce_mean(tf.square(listOfForward - tf.tile(tf.expand_dims(Yfinal,axis=0), [self.mathModel.N,1])), axis=-1), axis=-1))
                
        # train to optimize control
        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= optimizeBSDE( nbSimul)
            gradients= tape.gradient(objFunc, self.modelKerasU.trainable_variables + self.modelKerasGam.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKerasU.trainable_variables + self.modelKerasGam.trainable_variables))
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
            X = self.mathModel.init(1)
            Y0 = self.modelKerasU( tf.stack([tf.zeros([1], dtype= tf.float32) , X], axis=-1))[0][0]
            print(" Error",objError.numpy(),  " took %5.3f s" % rtime, "Y0 sofar ",Y0.numpy(), 'epoch', iout)
            self.listY0.append(Y0.numpy())
            self.lossList.append(objError)   
        return self.listY0 
                
class SolverSumLocalFBSDE1():
    def __init__(self, mathModel, modelKerasU , lRate):
        # to store les different networks
        self.mathModel = mathModel
        self.modelKerasU = modelKerasU
        self.lRate = lRate

    def train(self,  batchSize,  batchSizeVal, num_epoch, num_epochExt):

        @tf.function
        def optimizeBSDE( nbSimul):
            # initialize
            X = self.mathModel.init(nbSimul)
            gaussJ  = self.mathModel.jumps()
            #error
            error = 0
            #init val
            YPrev = self.modelKerasU( tf.stack([tf.zeros([nbSimul], dtype= tf.float32) , X], axis=-1))[0]
            GamPrev = self.modelKerasU(tf.stack([tf.zeros([nbSimul], dtype= tf.float32), X + gaussJ], axis=-1))[0] \
            - self.modelKerasU(tf.stack([tf.zeros([nbSimul], dtype= tf.float32), X], axis=-1))[0]
            for iStep in range(self.mathModel.N):
                # target
                toAdd = self.mathModel.dt*self.mathModel.f(YPrev) - GamPrev + tf.reduce_mean(GamPrev)
                # next step
                X = self.mathModel.oneStepFrom(iStep, X, gaussJ, YPrev)
                # jumps
                gaussJ  = self.mathModel.jumps()  
                # update          
                if (iStep == (self.mathModel.N - 1)):
                    YNext = self.mathModel.g(X)
                else:
                    YNext = self.modelKerasU( tf.stack([iStep* tf.ones([nbSimul], dtype= tf.float32) , X], axis=-1))[0]
                    GamPrev = self.modelKerasU(tf.stack([tf.zeros([nbSimul], dtype= tf.float32), X + gaussJ], axis=-1))[0] \
                    - self.modelKerasU(tf.stack([tf.zeros([nbSimul], dtype= tf.float32), X], axis=-1))[0]
                error = error + tf.reduce_mean(tf.square(YNext - YPrev + toAdd))
                YPrev = YNext
            return error
                
        # train to optimize control
        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= optimizeBSDE( nbSimul)
            gradients= tape.gradient(objFunc, self.modelKerasU.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKerasU.trainable_variables))
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
            X = self.mathModel.init(1)
            Y0 = self.modelKerasU( tf.stack([tf.zeros([1], dtype= tf.float32) , X], axis=-1))[0][0]
            print(" Error",objError.numpy(),  " took %5.3f s" % rtime, "Y0 sofar ",Y0.numpy(), 'epoch', iout)
            self.listY0.append(Y0.numpy())
            self.lossList.append(objError)   
        return self.listY0

class SolverSumLocalFBSDE2(SolverBase):
    def __init__(self, mathModel, modelKerasU , modelKerasGam,  lRate):
        super().__init__(mathModel, modelKerasU , modelKerasGam,  lRate)

    def train(self,  batchSize,  batchSizeVal, num_epoch, num_epochExt):

        @tf.function
        def optimizeBSDE( nbSimul):
            # initialize
            X = self.mathModel.init(nbSimul)
            gaussJ  = self.mathModel.jumps()
            #error
            error = 0
            #init val
            YPrev = self.modelKerasU( tf.stack([tf.zeros([nbSimul], dtype= tf.float32) , X], axis=-1))[0]
            GamPrev = self.modelKerasGam(tf.stack([tf.zeros([nbSimul], dtype= tf.float32), X, gaussJ], axis=-1))[0]
            for iStep in range(self.mathModel.N):
                # target
                toAdd = self.mathModel.dt*self.mathModel.f(YPrev) - GamPrev + tf.reduce_mean(GamPrev)
                # next step
                X = self.mathModel.oneStepFrom(iStep, X, gaussJ, YPrev)
                # jumps
                gaussJ  = self.mathModel.jumps()   
    	        # update         
                if (iStep == (self.mathModel.N - 1)):
                    YNext = self.mathModel.g(X)
                else:
                    YNext = self.modelKerasU( tf.stack([iStep* tf.ones([nbSimul], dtype= tf.float32) , X], axis=-1))[0]
                    GamPrev = self.modelKerasGam(tf.stack([iStep* tf.ones([nbSimul], dtype= tf.float32), X, gaussJ], axis=-1) )[0]
                error = error + tf.reduce_mean(tf.square(YNext - YPrev + toAdd))
                YPrev = YNext
            return error
                
        # train to optimize control
        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= optimizeBSDE( nbSimul)
            gradients= tape.gradient(objFunc, self.modelKerasU.trainable_variables + self.modelKerasGam.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKerasU.trainable_variables + self.modelKerasGam.trainable_variables))
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
            X = self.mathModel.init(1)
            Y0 = self.modelKerasU( tf.stack([tf.zeros([1], dtype= tf.float32) , X], axis=-1))[0][0]
            print(" Error",objError.numpy(),  " took %5.3f s" % rtime, "Y0 sofar ",Y0.numpy(), 'epoch', iout)
            self.listY0.append(Y0.numpy())
            self.lossList.append(objError)   
        return self.listY0 

# global as sum of local error due to regressions
# see algorithm 
class SolverGlobalSumLocalReg(SolverBase):
    def __init__(self, mathModel, modelKerasU , modelKerasGam,  lRate):
        super().__init__(mathModel, modelKerasU , modelKerasGam,  lRate)
      
    def train( self,  batchSize,  batchSizeVal, num_epoch, num_epochExt):
        @tf.function
        def regressOptim( nbSimul):
            # initialize
            X = self.mathModel.init(nbSimul)
            # Target
            error = 0.
            # get back Y
            YPrev = self.modelKerasU(tf.stack([tf.zeros([nbSimul], dtype= tf.float32) , X], axis=-1))[0]
            for iStep in range(self.mathModel.N):
                # target
                toAdd = - self.mathModel.dt* self.mathModel.f(YPrev) 
                # jumps
                gaussJ  = self.mathModel.jumps()
                # Next step
                X = self.mathModel.oneStepFrom(iStep, X, gaussJ, YPrev)
                # values
                if (iStep == (self.mathModel.N - 1)):
                    YNext = self.mathModel.g(X)
                else:
                    YNext = self.modelKerasU(tf.stack([iStep*tf.ones([nbSimul], dtype= tf.float32) , X], axis=-1))[0]
                error = error +  tf.reduce_mean(tf.square(YPrev- YNext  + toAdd))
                YPrev = YNext
            return error
    
        # train to optimize control
        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= regressOptim( nbSimul)
            gradients= tape.gradient(objFunc, self.modelKerasU.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKerasU.trainable_variables))
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
            X = self.mathModel.init(1)
            Y0 = self.modelKerasU( tf.stack([tf.zeros([1], dtype= tf.float32) , X], axis=-1))[0][0]
            print(" Error",objError.numpy(),  " took %5.3f s" % rtime, "Y0 sofar ",Y0.numpy(), 'epoch', iout)
            self.listY0.append(Y0.numpy())
            self.lossList.append(objError)   
        return self.listY0 
  

      
# global as multiStep regression  for hatY
# see algorithm 
# global as multiStep regression  for hatY
# see algorithm 
class SolverGlobalMultiStepReg(SolverBase):
    def __init__(self, mathModel, modelKerasU , modelKerasGam,  lRate):
        super().__init__(mathModel, modelKerasU , modelKerasGam,  lRate)

        
    def train( self,  batchSize,  batchSizeVal, num_epoch, num_epochExt):
        @tf.function
        def regressOptim( nbSimul):
            # initialize
            X = self.mathModel.init(nbSimul)
            # Target
            listOfForward = []
            for iStep in range(self.mathModel.N): 
                # get back Y
                Y = self.modelKerasU(tf.stack([iStep*tf.ones([nbSimul], dtype= tf.float32) , X], axis=-1))[0]
                # listforward
                listOfForward.append(Y)                 
                # to Add
                toAdd =- self.mathModel.dt* self.mathModel.f(Y)
                for i in range(len(listOfForward)):
                    listOfForward[i] = listOfForward[i] + toAdd
                # jumps
                gaussJ  = self.mathModel.jumps()
                # next t step
                X = self.mathModel.oneStepFrom(iStep, X, gaussJ, Y)
            # final U
            Yfinal = self.mathModel.g(X)
            listOfForward = tf.stack(listOfForward, axis=0) 
            return tf.reduce_sum(tf.reduce_mean(tf.reduce_mean(tf.square(listOfForward - tf.tile(tf.expand_dims(Yfinal,axis=0), [self.mathModel.N,1])), axis=-1), axis=-1))
                
        # train to optimize control
        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= regressOptim( nbSimul)
            gradients= tape.gradient(objFunc, self.modelKerasU.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKerasU.trainable_variables))
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
            X = self.mathModel.init(1)
            Y0 = self.modelKerasU( tf.stack([tf.zeros([1], dtype= tf.float32) , X], axis=-1))[0][0]
            print(" Error",objError.numpy(),  " took %5.3f s" % rtime, "Y0 sofar ",Y0.numpy(), 'epoch', iout)
            self.listY0.append(Y0.numpy())
            self.lossList.append(objError)   
        return self.listY0 
  

class SolverOsterleeFBSDE(SolverBase):
    def __init__(self, mathModel, modelKerasU , modelKerasGam,  lRate, lamCoef):
        super().__init__(mathModel, modelKerasU , modelKerasGam,  lRate)
        self.lamCoef = lamCoef

    def train(self,  batchSize,  batchSizeVal, num_epoch, num_epochExt):

        @tf.function
        def optimizeBSDE( nbSimul):
            #Compute Y0
            ####################################################################
            # initialize
            X = self.mathModel.init(nbSimul)   
            Y0 = 0                 
            for iStep in range(self.mathModel.N):
                # jump 
                gaussJ  = self.mathModel.jumps()
                # get back Y, Z, Gam
                Y = self.modelKerasU(tf.stack([iStep* tf.ones([nbSimul], dtype= tf.float32), X], axis=-1))[0]
                Gam =  self.modelKerasGam(tf.stack([iStep* tf.ones([nbSimul], dtype= tf.float32), X, gaussJ], axis=-1))[0]
                # target
                Y0 += self.mathModel.dt* self.mathModel.f(Y) - Gam + tf.reduce_mean(Gam) 
                # next t step
                X = self.mathModel.oneStepFrom(iStep, X, gaussJ, Y)
            Y0 += self.mathModel.g(X)
            # initial value
            Y = tf.reduce_mean(Y0)
            #Compute error
            ####################################################################
            # initialize
            self.mathModel.init(nbSimul)  
            for iStep in range(self.mathModel.N): 
                # jump
                gaussJ  = self.mathModel.jumps()
                # get back Gam
                Gam = self.modelKerasGam(tf.stack([iStep* tf.ones([nbSimul], dtype= tf.float32),X, gaussJ], axis=-1) )[0]
                # adjoint variables
                Y = Y - self.mathModel.dt*self.mathModel.f(Y)+ Gam - tf.reduce_mean(Gam)            
                # next t step
                X = self.mathModel.oneStepFrom(iStep, X, gaussJ, Y)
            return  tf.reduce_mean(Y0) + self.lamCoef*(tf.reduce_mean(tf.square(Y - self.mathModel.g(X)))), tf.reduce_mean(Y0)
                
        # train to optimize control
        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= optimizeBSDE( nbSimul)
            gradients= tape.gradient(objFunc, self.modelKerasU.trainable_variables + self.modelKerasGam.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKerasU.trainable_variables + self.modelKerasGam.trainable_variables))
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