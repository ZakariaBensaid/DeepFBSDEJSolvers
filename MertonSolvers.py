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
    def __init__(self, mathModel,   modelKeras, lRate):
        super().__init__(mathModel,   modelKeras, lRate)

    def train(self,  batchSize,  batchSizeVal, num_epoch, num_epochExt):

        @tf.function
        def optimizeBSDE( nbSimul):
            # initialize
            self.mathModel.init(nbSimul)
            # Target
            Y = self.modelKeras.Y0
            # error compensator
            errorComp = 0
            for istep in range(self.mathModel.N):
                # increment
                gaussian = tf.random.normal([nbSimul])
                dW =  np.sqrt(self.mathModel.dt)*gaussian 
                # jump and compensation
                dN, gaussJ  = self.mathModel.jumps()
                # get back U, Z, 
                Z0, Gam, compens = self.modelKeras(self.mathModel.getStates(dN, gaussJ))
                # target
                YNext = Y - self.mathModel.dt* self.mathModel.f(Y) + Z0*dW + Gam - compens*self.mathModel.dt 
                # next t step
                self.mathModel.oneStepFrom(dW, gaussJ)
                Y = YNext
                # update error 
                errorComp += tf.reduce_mean(tf.square(Gam - self.mathModel.dt*compens))
            return  tf.reduce_mean(tf.square(Y- self.mathModel.g(self.mathModel.X))) + errorComp

        # train to optimize control
        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc = optimizeBSDE( nbSimul)
            gradients= tape.gradient(objFunc, self.modelKeras.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKeras.trainable_variables))
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
            Y0 = self.modelKeras.Y0
            print(" Error",objError.numpy(),  " elapsed time %5.3f s" % self.duration, "Y0 sofar ",Y0.numpy(), 'epoch', iout)
            self.listY0.append(Y0.numpy())
            self.lossList.append(objError)   
        return self.listY0 
    
    def simulateGlobalErr(self, nbSimul):
        # initialize
        self.mathModel.init(nbSimul)
        costFunc = 0.
        # Target
        Y = self.modelKeras.Y0
        for istep in range(self.mathModel.N):
            # get back U, Z, 
            Z0, Gam = self.modelKeras(self.mathModel.getStates())
            #costfunc
            costFunc += self.mathModel.dt* self.mathModel.f(Y)
            # increment
            gaussJ = tf.random.normal([nbSimul], self.mathModel.muJ, self.mathModel.sigJ)
            gaussian = tf.random.normal([nbSimul])
            dW =  np.sqrt(self.mathModel.dt)*gaussian
            # jump and compensation
            dN , compens  = self.mathModel.dN()
            # target
            YNext = Y - self.mathModel.dt* self.mathModel.f(Y) + Z0* dW + Gam*(dN*gaussJ-compens)
            #Go to next step                 
            self.mathModel.oneStepFrom(dW, dN, gaussJ)
            Y = YNext
        costFunc += self.mathModel.g(self.mathModel.X)
        return  tf.reduce_mean(costFunc), tf.reduce_mean(tf.square(Y- self.mathModel.g(self.mathModel.X)))

class SolverMultiStepFBSDE(SolverBase):
    def __init__(self, mathModel,   modelKeras, lRate):
        super().__init__(mathModel,   modelKeras, lRate)

    def train(self,  batchSize,  batchSizeVal, num_epoch, num_epochExt):

        @tf.function
        def optimizeBSDE( nbSimul):
            # initialize
            self.mathModel.init(nbSimul)
            # Target
            listOfForward = []
            # compensator error
            errorComp = 0.
            for istep in range(self.mathModel.N):
                # Common and individual noises
                gaussian = tf.random.normal([nbSimul])
                dW =  np.sqrt(self.mathModel.dt)*gaussian 
                # jump and compensation
                dN, gaussJ  = self.mathModel.jumps()
                # Adjoint variables 
                Y, Z0, Gam, compens = self.modelKeras(self.mathModel.getStates(dN, gaussJ))
                # target
                toAdd = - self.mathModel.dt*self.mathModel.f(Y) + Z0*dW + Gam - compens*self.mathModel.dt
                #update list and error
                errorComp += tf.reduce_mean(tf.square(Gam - compens*self.mathModel.dt))
                listOfForward.append(Y)
                #forward
                for i in range(len(listOfForward)):
                    listOfForward[i] = listOfForward[i] + toAdd
                # next t step
                self.mathModel.oneStepFrom(dW, gaussJ)
            #Final Y
            Yfinal = self.mathModel.g(self.mathModel.X)
            listOfForward = tf.stack(listOfForward, axis=0)
            return tf.reduce_sum(tf.reduce_mean(tf.reduce_mean(tf.square(listOfForward - tf.tile(tf.expand_dims(Yfinal,axis=0), [self.mathModel.N,1])), axis=-1), axis=-1)) + errorComp
                
        # train to optimize control
        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= optimizeBSDE( nbSimul)
            gradients= tape.gradient(objFunc, self.modelKeras.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKeras.trainable_variables))
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
            Y0 = self.modelKeras(self.mathModel.getStates(tf.zeros([1]), tf.zeros([1])))[0][0]
            print(" Error",objError.numpy(),  " took %5.3f s" % rtime, "Y0 sofar ",Y0.numpy(), 'epoch', iout)
            self.listY0.append(Y0.numpy())
            self.lossList.append(objError)   
        return self.listY0 
    
    def simulateGlobalErr(self, nbSimul):
        # initialize
        self.mathModel.init(nbSimul)
        costFunc = 0.
        for istep in range(self.mathModel.N):
            # get back U, Z, 
            Y, Z0, Gam = self.modelKeras(self.mathModel.getStates())
            #costfunc
            costFunc += self.mathModel.dt* self.mathModel.f(Y)
            # increment
            gaussJ = tf.random.normal([nbSimul], self.mathModel.muJ, self.mathModel.sigJ)
            gaussian = tf.random.normal([nbSimul])
            dW =  np.sqrt(self.mathModel.dt)*gaussian
            # jump and compensation
            dN   = self.mathModel.dN()
            #Go to next step                 
            self.mathModel.oneStepFrom(dW, dN, gaussJ)
        costFunc += self.mathModel.g(self.mathModel.X)
        return  tf.reduce_mean(costFunc), tf.reduce_mean(tf.square(Y- self.mathModel.g(self.mathModel.X)))
                
        
class SolverSumLocalFBSDE(SolverBase):
    def __init__(self, mathModel,   modelKeras, lRate):
        super().__init__(mathModel,   modelKeras, lRate)

    def train(self,  batchSize,  batchSizeVal, num_epoch, num_epochExt):

        @tf.function
        def optimizeBSDE( nbSimul):
            # initialize
            self.mathModel.init(nbSimul)
            dN, gaussJ  = self.mathModel.jumps()
            #error
            error = 0
            errorComp = 0.
            #init val
            YPrev, Z0Prev, GamPrev, compens = self.modelKeras(self.mathModel.getStates(dN, gaussJ))
            for istep in range(self.mathModel.N):
                # increment
                gaussian = tf.random.normal([nbSimul])
                dW =  np.sqrt(self.mathModel.dt)*gaussian 
                # target
                toAdd = self.mathModel.dt*self.mathModel.f(YPrev) - Z0Prev*dW - GamPrev + compens*self.mathModel.dt
                # next step
                errorComp = tf.reduce_mean(tf.square(GamPrev - compens*self.mathModel.dt))
                self.mathModel.oneStepFrom(dW, gaussJ)
                # jump and compensation
                dN, gaussJ  = self.mathModel.jumps()
                if (istep == (self.mathModel.N-1)):
                    YNext = self.mathModel.g(self.mathModel.X)
                else:
                    YNext, Z0Prev, GamPrev, compens = self.modelKeras(self.mathModel.getStates(dN, gaussJ))
                error = error + tf.reduce_mean(tf.square(YNext - YPrev + toAdd)) + errorComp
                YPrev = YNext
            return error
                
        # train to optimize control
        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= optimizeBSDE( nbSimul)
            gradients= tape.gradient(objFunc, self.modelKeras.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKeras.trainable_variables))
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
            dN, gaussJ  = self.mathModel.jumps()
            Y0 = self.modelKeras(self.mathModel.getStates(dN, gaussJ))[0][0]
            print(" Error",objError.numpy(),  " took %5.3f s" % rtime, "Y0 sofar ",Y0.numpy(), 'epoch', iout)
            self.listY0.append(Y0.numpy())
            self.lossList.append(objError)   
        return self.listY0 
    
    def simulateGlobalErr(self, nbSimul):
        # initialize
        self.mathModel.init(nbSimul)
        costFunc = 0.
        for istep in range(self.mathModel.N):
            # get back U, Z, 
            Y,_,_ = self.modelKeras(self.mathModel.getStates())
            #costfunc
            costFunc += self.mathModel.dt* self.mathModel.f(Y)
            # increment
            gaussJ = tf.random.normal([nbSimul], self.mathModel.muJ, self.mathModel.sigJ)
            gaussian = tf.random.normal([nbSimul])
            dW =  np.sqrt(self.mathModel.dt)*gaussian
            # jump and compensation
            dN , compens  = self.mathModel.dN()
            #Go to next step                 
            self.mathModel.oneStepFrom(dW, dN, gaussJ)
        costFunc += self.mathModel.g(self.mathModel.X)
        return  tf.reduce_mean(costFunc), tf.reduce_mean(tf.square(Y- self.mathModel.g(self.mathModel.X)))

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
            dN, gaussJ  = self.mathModel.jumps()
            # Target
            error = 0.
            # get back Y
            YPrev, = self.modelKeras(self.mathModel.getStates(dN, gaussJ))
            for istep in range(self.mathModel.N):
                # target
                toAdd = - self.mathModel.dt* self.mathModel.f(YPrev)
                # increment
                gaussian = tf.random.normal([nbSimul])
                dW =  np.sqrt(self.mathModel.dt)*gaussian 
                # Next step
                self.mathModel.oneStepFrom(dW, gaussJ)
                # jumps
                dN, gaussJ  = self.mathModel.jumps()
                # values
                if (istep == (self.mathModel.N-1)):
                    YNext = self.mathModel.g(self.mathModel.X)
                else:
                    YNext, = self.modelKeras(self.mathModel.getStates(dN, gaussJ))
                error = error +  tf.reduce_mean(tf.square(YPrev- YNext  + toAdd))
                YPrev = YNext
            return error
    
        # train to optimize control
        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= regressOptim( nbSimul)
            gradients= tape.gradient(objFunc, self.modelKeras.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKeras.trainable_variables))
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
            dN, gaussJ  = self.mathModel.jumps()
            Y0 = self.modelKeras(self.mathModel.getStates(dN, gaussJ))[0][0]
            print(" Error",objError.numpy(),  " took %5.3f s" % rtime, "Y0 sofar ",Y0.numpy(), 'epoch', iout)
            self.listY0.append(Y0.numpy())
            self.lossList.append(objError)   
        return self.listY0 
  
    def simulateGlobalErr(self, nbSimul):
        # initialize
        self.mathModel.init(nbSimul)
        costFunc = 0.
        for istep in range(self.mathModel.N):
            # get back U, Z, 
            Y, = self.modelKeras(self.mathModel.getStates())
            #cost function
            costFunc += self.mathModel.dt* self.mathModel.f(Y)
            # next t step
            # increment
            gaussJ = tf.random.normal([nbSimul], self.mathModel.muJ, self.mathModel.sigJ)
            gaussian = tf.random.normal([nbSimul])
            dW =  np.sqrt(self.mathModel.dt)*gaussian
            # jump and compensation
            dN , compens  = self.mathModel.dN()                 
            #Go to next step                 
            self.mathModel.oneStepFrom(dW, dN, gaussJ)
        costFunc += self.mathModel.g(self.mathModel.X)
        return   tf.reduce_mean(costFunc)
      
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
            dN, gaussJ  = self.mathModel.jumps()
            # Target
            listOfForward = []
            for istep in range(self.mathModel.N): 
                # get back Y
                Y, = self.modelKeras(self.mathModel.getStates(dN, gaussJ))
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
                self.mathModel.oneStepFrom(dW, gaussJ)
                dN, gaussJ  = self.mathModel.jumps()
            # final U
            Yfinal = self.mathModel.g(self.mathModel.X)
            listOfForward = tf.stack(listOfForward, axis=0) 
            return tf.reduce_sum(tf.reduce_mean(tf.reduce_mean(tf.square(listOfForward - tf.tile(tf.expand_dims(Yfinal,axis=0), [self.mathModel.N,1])), axis=-1), axis=-1))
                
        # train to optimize control
        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= regressOptim( nbSimul)
            gradients= tape.gradient(objFunc, self.modelKeras.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKeras.trainable_variables))
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
            Y0 = self.modelKeras(self.mathModel.getStates(tf.zeros([1]), tf.zeros([1])))[0][0].numpy()
            print(" Error",objError.numpy(),  " took %5.3f s" % rtime, "Y0 sofar ",Y0, 'epoch', iout)
            self.listY0.append(Y0)
            self.lossList.append(objError)   
        return self.listY0 
  
    def simulateGlobalErr(self, nbSimul):
        # initialize
        self.mathModel.init(nbSimul)
        costFunc = 0.
        for istep in range(self.mathModel.N):
            # get back U, Z, 
            Y, = self.modelKeras(self.mathModel.getStates())
            #cost function
            costFunc += self.mathModel.dt* self.mathModel.f(Y)
            # next t step
            # increment
            gaussJ = tf.random.normal([nbSimul], self.mathModel.muJ, self.mathModel.sigJ)
            gaussian = tf.random.normal([nbSimul])
            dW =  np.sqrt(self.mathModel.dt)*gaussian
            # jump and compensation
            dN = self.mathModel.dN()                 
            #Go to next step                 
            self.mathModel.oneStepFrom(dW, dN, gaussJ)
        costFunc += self.mathModel.g(self.mathModel.X)
        return   tf.reduce_mean(costFunc)

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
                dN, gaussJ  = self.mathModel.jumps()
                # get back Y, Z, Gam
                Y, Z0, Gam, compens = self.modelKeras(self.mathModel.getStates(dN, gaussJ))
                # target
                Y0 += self.mathModel.dt* self.mathModel.f(Y)
                # next t step
                self.mathModel.oneStepFrom(dW, gaussJ)
            Y0 += self.mathModel.g(self.mathModel.X)
            # initial value
            errorComp = 0.
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
                dN, gaussJ  = self.mathModel.jumps()
                # get back Y, Z, Gam
                Ynet, Z0, Gam, compens = self.modelKeras(self.mathModel.getStates(dN, gaussJ))
                # adjoint variables
                YNext = Y - self.mathModel.dt*self.mathModel.f(Y) + Z0*dW + Gam - compens*self.mathModel.dt             
                # next t step
                self.mathModel.oneStepFrom(dW, gaussJ)
                # Update 
                Y = YNext
                errorComp += tf.reduce_mean(tf.square(Gam - self.mathModel.dt*compens)) + tf.reduce_mean(tf.square(Ynet - Y))
            return  tf.reduce_mean(Y0) + self.lamCoef*(tf.reduce_mean(tf.square(Y - self.mathModel.g(self.mathModel.X))) + errorComp), tf.reduce_mean(Y0)
                
        # train to optimize control
        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= optimizeBSDE(nbSimul)
            gradients= tape.gradient(objFunc, self.modelKeras.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKeras.trainable_variables))
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