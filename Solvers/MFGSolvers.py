import numpy as np
import tensorflow as tf
import time
from tensorflow.keras import  optimizers

class SolverBase:
    # mathModel          Math model
    # modelKeras         Keras model 
    # lRate              Learning rate
    def __init__(self, mathModel,   modelKeras, lRate, couplage):
        # to store les different networks
        self.mathModel = mathModel
        self.modelKeras= modelKeras
        self.lRate= lRate
        self.couplage = couplage

class SolverGlobalFBSDE(SolverBase):
    def __init__(self, mathModel,   modelKeras, lRate, couplage):
        super().__init__(mathModel,   modelKeras, lRate, couplage)

    def train(self,  batchSize,  batchSizeVal, num_epoch, num_epochExt):

        @tf.function
        def optimizeBSDE( nbSimul):
            # initialize
            self.mathModel.init(nbSimul)
            # Target
            hY = self.modelKeras.model_hat.Y0_hat
            Y = self.modelKeras.model.Y0
            for istep in range(self.mathModel.N):
                # get back U, Z, 
                hZ0, hGam = self.modelKeras.model_hat(self.mathModel.getProjectedStates())
                Z0, Gam, Z = self.modelKeras.model(self.mathModel.getAllStates())
                # increment
                gaussian0, gaussian = tf.random.normal([nbSimul]), tf.random.normal([nbSimul])
                dW0, dW =  np.sqrt(self.mathModel.dt)*gaussian0, np.sqrt(self.mathModel.dt)*gaussian 
                # jump and compensation
                dN , compens  = self.mathModel.dN()
                # target
                hYNext = hY - self.mathModel.dt* self.mathModel.f(self.mathModel.hS) + hZ0* dW0 + hGam*(dN-compens)
                YNext = Y - self.mathModel.dt* self.mathModel.f(self.mathModel.S) + Z0* dW0 + Gam*(dN-compens) + Z*dW
                # next t step
                self.mathModel.oneStepFrom(dW0, dW, dN, hY, Y)
                hY, Y = hYNext, YNext
            if self.couplage == 'OFF':
              return tf.reduce_mean(tf.square(hY- self.mathModel.g(self.mathModel.hS))), tf.reduce_mean(tf.square(Y- self.mathModel.g(self.mathModel.S)))
            return tf.reduce_mean(tf.square(hY- self.mathModel.g(self.mathModel.hS))) + tf.reduce_mean(tf.square(Y- self.mathModel.g(self.mathModel.S)))
                
        # train to optimize control
        @tf.function
        def trainOpt_hat( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= optimizeBSDE( nbSimul)[0]
            gradients= tape.gradient(objFunc, self.modelKeras.model_hat.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKeras.model_hat.trainable_variables))
            return objFunc

        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= optimizeBSDE( nbSimul)[1]
            gradients= tape.gradient(objFunc, self.modelKeras.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKeras.model.trainable_variables))
            return objFunc

        @tf.function
        def trainOptCoupled( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= optimizeBSDE( nbSimul)
            CoupledVariables = self.modelKeras.model_hat.trainable_variables + self.modelKeras.model.trainable_variables
            gradients= tape.gradient(objFunc, CoupledVariables)
            optimizer.apply_gradients(zip(gradients, CoupledVariables))
            return objFunc

        optimizer= optimizers.Adam(learning_rate = self.lRate)

        self.listY0_hat = []
        self.listY0 = []
        if self.couplage == 'ON':
          for iout in range(num_epochExt):
              start_time = time.time()
              for epoch in range(num_epoch):
                  # un pas de gradient stochastique
                  trainOptCoupled(batchSize, optimizer)
              end_time = time.time()
              rtime = end_time-start_time 
              objError = optimizeBSDE(batchSizeVal)
              Y0_hat, Y0 =  self.modelKeras.model_hat.Y0_hat, self.modelKeras.model.Y0
              print("Error ",objError.numpy(),  " took %5.3f s" % rtime, "Y0_hat sofar ",Y0_hat.numpy(), 'Y0 sofar', Y0.numpy(), 'epoch', iout)
              self.listY0_hat.append(Y0_hat.numpy())
              self.listY0.append(Y0.numpy())      
        else :
            for iout in range(num_epochExt):
              start_time = time.time()
              for epoch in range(num_epoch):
                  # un pas de gradient stochastique
                  trainOpt_hat(batchSize, optimizer)
              end_time = time.time()
              rtime = end_time-start_time 
              objError = optimizeBSDE(batchSizeVal)[0]
              Y0_hat = self.modelKeras.model_hat.Y0_hat
              print("Error hat ",objError.numpy(),  " took %5.3f s" % rtime, "Y0_hat sofar ",Y0_hat.numpy(), 'epoch', iout)
              self.listY0_hat.append(Y0_hat.numpy()) 

            for iout in range(num_epochExt):
              start_time = time.time()
              for epoch in range(num_epoch):
                  # un pas de gradient stochastique
                  trainOpt(batchSize, optimizer)
              end_time = time.time()
              rtime = end_time-start_time 
              objError = optimizeBSDE(batchSizeVal)[1]
              Y0 = self.modelKeras.model.Y0
              print(" Error",objError.numpy(),  " took %5.3f s" % rtime, "Y0 sofar ",Y0.numpy(), 'epoch', iout)
              self.listY0.append(Y0.numpy())   
        return self.listY0_hat, self.listY0 
    
    def simulateGlobalErr(self, nbSimul):
        # initialize
        self.mathModel.init(nbSimul)
        costFunc_hat =0.
        costFunc = 0.
        # Target
        hY = self.modelKeras.model_hat.Y0_hat
        Y = self.modelKeras.model.Y0
        for istep in range(self.mathModel.N):
            # get back U, Z, 
            hZ0, hGam = self.modelKeras.model_hat(self.mathModel.getProjectedStates())
            Z0, Gam, Z = self.modelKeras.model(self.mathModel.getAllStates())
            #costfunc
            costFunc_hat += self.mathModel.dt* self.mathModel.f(self.mathModel.hS)
            costFunc += self.mathModel.dt* self.mathModel.f(self.mathModel.S)
            # increment
            gaussian0, gaussian = tf.random.normal([nbSimul]), tf.random.normal([nbSimul])
            dW0, dW =  np.sqrt(self.mathModel.dt)*gaussian0, np.sqrt(self.mathModel.dt)*gaussian
            # jump and compensation
            dN , compens  = self.mathModel.dN()
            # target
            hYNext = hY - self.mathModel.dt* self.mathModel.f(self.mathModel.hS) + hZ0* dW0 + hGam*(dN-compens)
            YNext = Y - self.mathModel.dt* self.mathModel.f(self.mathModel.S) + Z0* dW0 + Gam*(dN-compens) + Z*dW
            #Go to next step                 
            self.mathModel.oneStepFrom(dW0, dW, dN, hY, Y)
            hY, Y = hYNext, YNext
        costFunc_hat += self.mathModel.g(self.mathModel.hS)
        costFunc += self.mathModel.g(self.mathModel.S)
        return  tf.reduce_mean(costFunc_hat), tf.reduce_mean(costFunc), tf.reduce_mean(tf.square(hY- self.mathModel.g(self.mathModel.hS))) + tf.reduce_mean(tf.square(Y- self.mathModel.g(self.mathModel.S)))

    def followS(self, nbSimul):
        # initialize
        self.mathModel.init(nbSimul)
        averS_hat = [ tf.reduce_mean(self.mathModel.hS).numpy()]
        stdS_hat = [tf.math.reduce_std(self.mathModel.hS).numpy()]
        averS = [ tf.reduce_mean(self.mathModel.S).numpy()]
        stdS= [tf.math.reduce_std(self.mathModel.S).numpy()]
        # Target
        hY = self.modelKeras.model_hat.Y0_hat
        Y = self.modelKeras.model.Y0
        for istep in range(self.mathModel.N):
            # get back U, Z, 
            hZ0, hGam = self.modelKeras.model_hat(self.mathModel.getProjectedStates())
            Z0, Gam, Z = self.modelKeras.model(self.mathModel.getAllStates())
            # increment
            gaussian0, gaussian = tf.random.normal([nbSimul]), tf.random.normal([nbSimul])
            dW0, dW =  np.sqrt(self.mathModel.dt)*gaussian0, np.sqrt(self.mathModel.dt)*gaussian 
            # jump and compensation
            dN , compens  = self.mathModel.dN()
            # target
            hYNext = hY - self.mathModel.dt* self.mathModel.f(self.mathModel.hS) + hZ0* dW0 + hGam*(dN-compens)
            YNext = Y - self.mathModel.dt* self.mathModel.f(self.mathModel.S) + Z0* dW0 + Gam*(dN-compens) + Z*dW
            # next t step
            self.mathModel.oneStepFrom(dW0, dW, dN, hY, Y)
            hY, Y = hYNext, YNext
            # store aver, std 
            averS_hat.append(tf.reduce_mean(self.mathModel.hS).numpy())
            stdS_hat.append(tf.math.reduce_std(self.mathModel.hS).numpy())
            averS.append(tf.reduce_mean(self.mathModel.S).numpy())
            stdS.append(tf.math.reduce_std(self.mathModel.S).numpy())
        return averS_hat, stdS_hat, averS, stdS

class SolverMultiStepFBSDE(SolverBase):
    def __init__(self, mathModel,   modelKeras, lRate, couplage):
        super().__init__(mathModel,   modelKeras, lRate, couplage)

    def train(self,  batchSize,  batchSizeVal, num_epoch, num_epochExt):

        @tf.function
        def optimizeBSDE( nbSimul):
            # initialize
            self.mathModel.init(nbSimul)
            # Target
            listOfForward_hat = []
            listOfForward = []
            for istep in range(self.mathModel.N):
                # Adjoint variables 
                hY, hZ0, hGam = self.modelKeras.model_hat(self.mathModel.getProjectedStates())
                Y, Z0, Gam, Z = self.modelKeras.model(self.mathModel.getAllStates())
                # Common and individual noises
                gaussian0, gaussian = tf.random.normal([nbSimul]), tf.random.normal([nbSimul])
                dW0, dW =  np.sqrt(self.mathModel.dt)*gaussian0, np.sqrt(self.mathModel.dt)*gaussian 
                # jump and compensation
                dN , compens  = self.mathModel.dN()
                # target
                toAdd_hat = - self.mathModel.dt* self.mathModel.f(self.mathModel.hS) + hZ0* dW0 + hGam*(dN-compens)
                toAdd = - self.mathModel.dt* self.mathModel.f(self.mathModel.S) + Z0* dW0 + Gam*(dN-compens) + Z*dW
                #update list
                listOfForward_hat.append(hY)
                listOfForward.append(Y)
                #forward
                for i in range(len(listOfForward)):
                    listOfForward_hat[i] = listOfForward_hat[i] + toAdd_hat
                    listOfForward[i] = listOfForward[i] + toAdd
                # next t step
                self.mathModel.oneStepFrom(dW0, dW, dN, hY, Y)
            #Final Y
            Yfinal_hat =  self.mathModel.g(self.mathModel.hS)
            Yfinal = self.mathModel.g(self.mathModel.S)
            listOfForward_hat = tf.stack(listOfForward_hat, axis=0)
            listOfForward = tf.stack(listOfForward, axis=0)
            if self.couplage == 'OFF':
              return tf.reduce_sum(tf.reduce_mean(tf.reduce_mean(tf.square(listOfForward_hat - tf.tile(tf.expand_dims(Yfinal_hat,axis=0), [self.mathModel.N,1])), axis=-1), axis=-1))\
                , tf.reduce_sum(tf.reduce_mean(tf.reduce_mean(tf.square(listOfForward - tf.tile(tf.expand_dims(Yfinal,axis=0), [self.mathModel.N,1])), axis=-1), axis=-1))
            return tf.reduce_sum(tf.reduce_mean(tf.reduce_mean(tf.square(listOfForward_hat - tf.tile(tf.expand_dims(Yfinal_hat,axis=0), [self.mathModel.N,1])), axis=-1), axis=-1))\
                + tf.reduce_sum(tf.reduce_mean(tf.reduce_mean(tf.square(listOfForward - tf.tile(tf.expand_dims(Yfinal,axis=0), [self.mathModel.N,1])), axis=-1), axis=-1))
                
        # train to optimize control
        @tf.function
        def trainOpt_hat( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= optimizeBSDE( nbSimul)[0]
            gradients= tape.gradient(objFunc, self.modelKeras.model_hat.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKeras.model_hat.trainable_variables))
            return objFunc

        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= optimizeBSDE( nbSimul)[1]
            gradients= tape.gradient(objFunc, self.modelKeras.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKeras.model.trainable_variables))
            return objFunc

        @tf.function
        def trainOptCoupled( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= optimizeBSDE( nbSimul)
            CoupledVariables = self.modelKeras.model_hat.trainable_variables + self.modelKeras.model.trainable_variables
            gradients= tape.gradient(objFunc, CoupledVariables)
            optimizer.apply_gradients(zip(gradients, CoupledVariables))
            return objFunc

        optimizer= optimizers.Adam(learning_rate = self.lRate)

        self.listY0_hat = []
        self.listY0 = []
        if self.couplage == 'ON':
            for iout in range(num_epochExt):
                start_time = time.time()
                for epoch in range(num_epoch):
                    # un pas de gradient stochastique
                    trainOptCoupled(batchSize, optimizer)
                end_time = time.time()
                rtime = end_time-start_time 
                objError = optimizeBSDE(batchSizeVal)
                self.mathModel.init(1)
                Y0_hat, Y0 =  self.modelKeras.model_hat(self.mathModel.getProjectedStates())[0][0].numpy(), self.modelKeras.model(self.mathModel.getAllStates())[0][0].numpy()
                print("Error ",objError.numpy(),  " took %5.3f s" % rtime, "Y0_hat sofar ",Y0_hat, 'Y0 sofar', Y0, 'epoch', iout)
                self.listY0_hat.append(Y0_hat)
                self.listY0.append(Y0)      
        else :
            for iout in range(num_epochExt):
                start_time = time.time()
                for epoch in range(num_epoch):
                    # un pas de gradient stochastique
                    trainOpt_hat(batchSize, optimizer)
                end_time = time.time()
                rtime = end_time-start_time 
                objError = optimizeBSDE(batchSizeVal)[0]
                self.mathModel.init(1)
                Y0_hat = self.modelKeras.model_hat(self.mathModel.getProjectedStates())[0][0].numpy()
                print("Error hat ",objError.numpy(),  " took %5.3f s" % rtime, "Y0_hat sofar ",Y0_hat, 'epoch', iout)
                self.listY0_hat.append(Y0_hat) 

            for iout in range(num_epochExt):
              start_time = time.time()
              for epoch in range(num_epoch):
                  # un pas de gradient stochastique
                  trainOpt(batchSize, optimizer)
              end_time = time.time()
              rtime = end_time-start_time 
              objError = optimizeBSDE(batchSizeVal)[1]
              Y0 = self.modelKeras.model_hat(self.mathModel.getAllStates())[0][0].numpy()
              print("Error",objError.numpy(),  " took %5.3f s" % rtime, "Y0 sofar ",Y0)
              self.listY0.append(Y0)     
        return self.listY0_hat, self.listY0 
    
    def simulateGlobalErr(self, nbSimul):
        # initialize
        self.mathModel.init(nbSimul)
        costFunc_hat =0.
        costFunc = 0.
        for istep in range(self.mathModel.N):
            # get back U, Z, 
            hY, hZ0, hGam = self.modelKeras.model_hat(self.mathModel.getProjectedStates())
            Y, Z0, Gam, Z = self.modelKeras.model(self.mathModel.getAllStates())
            #print("AVEC ", tf.reduce_mean(dN)," COM ", tf.reduce_mean(dN-compens))
            costFunc_hat += self.mathModel.dt* self.mathModel.f(self.mathModel.hS)
            costFunc += self.mathModel.dt* self.mathModel.f(self.mathModel.S)
            # next t step
            # increment
            gaussian0, gaussian = tf.random.normal([nbSimul]), tf.random.normal([nbSimul])
            dW0, dW =  np.sqrt(self.mathModel.dt)*gaussian0, np.sqrt(self.mathModel.dt)*gaussian
            # jump and compensation
            dN , compens  = self.mathModel.dN()
            #Go to next step                 
            self.mathModel.oneStepFrom(dW0, dW, dN, hY, Y)
        costFunc_hat += self.mathModel.g(self.mathModel.hS)
        costFunc += self.mathModel.g(self.mathModel.S)
        return  tf.reduce_mean(costFunc_hat), tf.reduce_mean(costFunc), tf.reduce_mean(tf.square(hY- self.mathModel.g(self.mathModel.hS))) + tf.reduce_mean(tf.square(Y- self.mathModel.g(self.mathModel.S)))
                
        
class SolverSumLocalFBSDE(SolverBase):
    def __init__(self, mathModel,   modelKeras, lRate, couplage):
        super().__init__(mathModel,   modelKeras, lRate, couplage)

    def train(self,  batchSize,  batchSizeVal, num_epoch, num_epochExt):

        @tf.function
        def optimizeBSDE( nbSimul):
            # initialize
            self.mathModel.init(nbSimul)
            #error
            error_hat = 0
            error = 0
            errorCoupled = 0
            # initial values
            hYPrev, hZ0Prev, hGamPrev = self.modelKeras.model_hat(self.mathModel.getProjectedStates())
            YPrev, Z0Prev, GamPrev, ZPrev = self.modelKeras.model(self.mathModel.getAllStates())
            for istep in range(self.mathModel.N):
                # increment
                gaussian0, gaussian = tf.random.normal([nbSimul]), tf.random.normal([nbSimul])
                dW0, dW =  np.sqrt(self.mathModel.dt)*gaussian0, np.sqrt(self.mathModel.dt)*gaussian 
                # jump and compensation
                dN , compens  = self.mathModel.dN()
                # target
                toAdd_hat =  self.mathModel.dt* self.mathModel.f(self.mathModel.hS) - hZ0Prev* dW0 -  hGamPrev*(dN-compens)
                toAdd = self.mathModel.dt* self.mathModel.f(self.mathModel.S) - Z0Prev* dW0 - GamPrev*(dN-compens) - ZPrev*dW
                # next step
                self.mathModel.oneStepFrom(dW0, dW, dN, hYPrev, YPrev)
                if (istep == (self.mathModel.N-1)):
                    hYNext, YNext = self.mathModel.g(self.mathModel.hS), self.mathModel.g(self.mathModel.S)
                else :
                    hYNext, hZ0Prev, hGamPrev = self.modelKeras.model_hat(self.mathModel.getProjectedStates())
                    YNext, Z0Prev, GamPrev, ZPrev = self.modelKeras.model(self.mathModel.getAllStates())
                #loss functions
                error_hat = error_hat + tf.reduce_mean(tf.square(hYNext - hYPrev  +toAdd_hat ))
                error = error + tf.reduce_mean(tf.square(YNext - YPrev  +toAdd ))
                errorCoupled = errorCoupled + tf.reduce_mean(tf.square(hYNext - hYPrev  +toAdd_hat )) + tf.reduce_mean(tf.square(YNext - YPrev  +toAdd ))
                #update Y
                hYPrev = hYNext
                YPrev = YNext
            if self.couplage == 'OFF':
              return error_hat, error
            return errorCoupled
                
        # train to optimize control
        @tf.function
        def trainOpt_hat( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= optimizeBSDE( nbSimul)[0]
            gradients= tape.gradient(objFunc, self.modelKeras.model_hat.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKeras.model_hat.trainable_variables))
            return objFunc

        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= optimizeBSDE( nbSimul)[1]
            gradients= tape.gradient(objFunc, self.modelKeras.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKeras.model.trainable_variables))
            return objFunc

        @tf.function
        def trainOptCoupled( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= optimizeBSDE( nbSimul)
            CoupledVariables = self.modelKeras.model_hat.trainable_variables + self.modelKeras.model.trainable_variables
            gradients= tape.gradient(objFunc, CoupledVariables)
            optimizer.apply_gradients(zip(gradients, CoupledVariables))
            return objFunc

        optimizer= optimizers.Adam(learning_rate = self.lRate)

        self.listY0_hat = []
        self.listY0 = []
        if self.couplage == 'ON':
            for iout in range(num_epochExt):
                start_time = time.time()
                for epoch in range(num_epoch):
                    # un pas de gradient stochastique
                    trainOptCoupled(batchSize, optimizer)
                end_time = time.time()
                rtime = end_time-start_time 
                objError = optimizeBSDE(batchSizeVal)
                self.mathModel.init(1)
                Y0_hat, Y0 =  self.modelKeras.model_hat(self.mathModel.getProjectedStates())[0][0].numpy(), self.modelKeras.model(self.mathModel.getAllStates())[0][0].numpy()
                print("Error ",objError.numpy(),  " took %5.3f s" % rtime, "Y0_hat sofar ",Y0_hat, 'Y0 sofar', Y0, 'epoch', iout)
                self.listY0_hat.append(Y0_hat)
                self.listY0.append(Y0)      
        else :
            for iout in range(num_epochExt):
                start_time = time.time()
                for epoch in range(num_epoch):
                    # un pas de gradient stochastique
                    trainOpt_hat(batchSize, optimizer)
                end_time = time.time()
                rtime = end_time-start_time 
                objError = optimizeBSDE(batchSizeVal)[0]
                self.mathModel.init(1)
                Y0_hat = self.modelKeras.model_hat(self.mathModel.getProjectedStates())[0][0].numpy()
                print("Error hat ",objError.numpy(),  " took %5.3f s" % rtime, "Y0_hat sofar ",Y0_hat, 'epoch', iout)
                self.listY0_hat.append(Y0_hat) 

            for iout in range(num_epochExt):
              start_time = time.time()
              for epoch in range(num_epoch):
                  # un pas de gradient stochastique
                  trainOpt(batchSize, optimizer)
              end_time = time.time()
              rtime = end_time-start_time 
              objError = optimizeBSDE(batchSizeVal)[1]
              Y0 = self.modelKeras.model_hat(self.mathModel.getAllStates())[0][0].numpy()
              print("Error",objError.numpy(),  " took %5.3f s" % rtime, "Y0 sofar ",Y0)
              self.listY0.append(Y0)   
        return self.listY0_hat, self.listY0 
    
    def simulateGlobalErr(self, nbSimul):
        # initialize
        self.mathModel.init(nbSimul)
        costFunc_hat =0.
        costFunc = 0.
        for istep in range(self.mathModel.N):
            # get back U, Z, 
            hY,_,_  = self.modelKeras.model_hat(self.mathModel.getProjectedStates())
            Y,_,_,_ = self.modelKeras.model(self.mathModel.getAllStates())
            #print("AVEC ", tf.reduce_mean(dN)," COM ", tf.reduce_mean(dN-compens))
            costFunc_hat += self.mathModel.dt* self.mathModel.f(self.mathModel.hS)
            costFunc += self.mathModel.dt* self.mathModel.f(self.mathModel.S)
            # next t step
            # increment
            gaussian0, gaussian = tf.random.normal([nbSimul]), tf.random.normal([nbSimul])
            dW0, dW =  np.sqrt(self.mathModel.dt)*gaussian0, np.sqrt(self.mathModel.dt)*gaussian
            # jump and compensation
            dN , compens  = self.mathModel.dN()
            #Go to next step                 
            self.mathModel.oneStepFrom(dW0, dW, dN, hY, Y)
        #cost functions
        costFunc_hat += self.mathModel.g(self.mathModel.hS)
        costFunc += self.mathModel.g(self.mathModel.S)
        return  tf.reduce_mean(costFunc_hat), tf.reduce_mean(costFunc), tf.reduce_mean(tf.square(hY- self.mathModel.g(self.mathModel.hS))) + tf.reduce_mean(tf.square(Y- self.mathModel.g(self.mathModel.S)))

# global as sum of local error due to regressions
# see algorithm 
class SolverGlobalSumLocalReg(SolverBase):
    def __init__(self, mathModel,   modelKeras, lRate, couplage):
        super().__init__(mathModel,   modelKeras, lRate, couplage)
      
    def train( self,  batchSize,  batchSizeVal, num_epoch, num_epochExt):
        @tf.function
        def regressOptim( nbSimul):
            # initialize
            self.mathModel.init(nbSimul)
            # Target
            error_hat = 0
            error = 0
            errorCoupled = 0
            # get back Y
            hYPrev, = self.modelKeras.model_hat(self.mathModel.getProjectedStates())
            YPrev, = self.modelKeras.model(self.mathModel.getAllStates())
            for istep in range(self.mathModel.N):
                # increment
                gaussian0, gaussian = tf.random.normal([nbSimul]), tf.random.normal([nbSimul])
                dW0, dW =  np.sqrt(self.mathModel.dt)*gaussian0, np.sqrt(self.mathModel.dt)*gaussian 
                # jump and compensation
                dN , compens = self.mathModel.dN()
                # target
                toAdd_hat =  - self.mathModel.dt* self.mathModel.f(self.mathModel.hS)
                toAdd = - self.mathModel.dt* self.mathModel.f(self.mathModel.S)
                #Next step
                self.mathModel.oneStepFrom(dW0, dW, dN, hYPrev, YPrev)
                # values
                if (istep == (self.mathModel.N-1)):
                    hYNext = self.mathModel.g(self.mathModel.hS)
                    YNext = self.mathModel.g(self.mathModel.S)
                else:
                    hYNext, = self.modelKeras.model_hat(self.mathModel.getProjectedStates())
                    YNext, = self.modelKeras.model(self.mathModel.getAllStates())
                error_hat = error_hat +  tf.reduce_mean(tf.square(hYPrev- hYNext  + toAdd_hat))
                error = error +  tf.reduce_mean(tf.square(YPrev- YNext  + toAdd))
                errorCoupled = errorCoupled + tf.reduce_mean(tf.square(hYPrev- hYNext  + toAdd_hat)) + tf.reduce_mean(tf.square(YPrev- YNext  + toAdd))
                hYPrev = hYNext
                YPrev = YNext
            if self.couplage == 'OFF':
              return error_hat, error
            return errorCoupled
      
                
        # train to optimize control
        @tf.function
        def trainOpt_hat( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= regressOptim( nbSimul)[0]
            gradients= tape.gradient(objFunc, self.modelKeras.model_hat.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKeras.model_hat.trainable_variables))
            return objFunc

        # train to optimize control
        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= regressOptim( nbSimul)[1]
            gradients= tape.gradient(objFunc, self.modelKeras.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKeras.model.trainable_variables))
            return objFunc

        # train to optimize control
        @tf.function
        def trainOptCoupled( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= regressOptim( nbSimul)
            CoupledVariables = self.modelKeras.model_hat.trainable_variables + self.modelKeras.model.trainable_variables
            gradients= tape.gradient(objFunc, CoupledVariables)
            optimizer.apply_gradients(zip(gradients, CoupledVariables))
            return objFunc

        optimizer= optimizers.Adam(learning_rate = self.lRate)
            
        self.listY0_hat = []
        self.listY0 = []
        if self.couplage == 'ON':
            for iout in range(num_epochExt):
                start_time = time.time()
                for epoch in range(num_epoch):
                    # un pas de gradient stochastique
                    trainOptCoupled(batchSize, optimizer)
                end_time = time.time()
                rtime = end_time-start_time 
                objError = regressOptim(batchSizeVal)
                self.mathModel.init(1)
                Y0_hat, Y0 =  self.modelKeras.model_hat(self.mathModel.getProjectedStates())[0][0].numpy(), self.modelKeras.model(self.mathModel.getAllStates())[0][0].numpy()
                print("Error ",objError.numpy(),  " took %5.3f s" % rtime, "Y0_hat sofar ",Y0_hat, 'Y0 sofar', Y0, 'epoch', iout)
                self.listY0_hat.append(Y0_hat)
                self.listY0.append(Y0)      
        else :
            for iout in range(num_epochExt):
                start_time = time.time()
                for epoch in range(num_epoch):
                    # un pas de gradient stochastique
                    trainOpt_hat(batchSize, optimizer)
                end_time = time.time()
                rtime = end_time-start_time 
                objError = regressOptim(batchSizeVal)[0]
                self.mathModel.init(1)
                Y0_hat =  self.modelKeras.model_hat(self.mathModel.getProjectedStates())[0][0].numpy()
                print("Error hat ",objError.numpy(),  " took %5.3f s" % rtime, "Y0_hat sofar ",Y0_hat, 'epoch', iout)
                self.listY0_hat.append(Y0_hat) 

            for iout in range(num_epochExt):
                start_time = time.time()
                for epoch in range(num_epoch):
                    # un pas de gradient stochastique
                    trainOpt(batchSize, optimizer)
                end_time = time.time()
                rtime = end_time-start_time 
                objError = regressOptim(batchSizeVal)[1]
                self.mathModel.init(1)
                Y0 = self.modelKeras.model(self.mathModel.getAllStates())[0][0].numpy()
                print("Error",objError.numpy(),  " took %5.3f s" % rtime, "Y0 sofar ",Y0)
                self.listY0.append(Y0)   
        return self.listY0_hat, self.listY0 
  
    def simulateGlobalErr(self, nbSimul):
        # initialize
        self.mathModel.init(nbSimul)
        costFunc_hat =0.
        costFunc = 0.
        for istep in range(self.mathModel.N):
            # get back U, Z, 
            hY, = self.modelKeras.model_hat(self.mathModel.getProjectedStates())
            Y, = self.modelKeras.model(self.mathModel.getAllStates())
            #print("AVEC ", tf.reduce_mean(dN)," COM ", tf.reduce_mean(dN-compens))
            costFunc_hat += self.mathModel.dt* self.mathModel.f(self.mathModel.hS)
            costFunc += self.mathModel.dt* self.mathModel.f(self.mathModel.S)
            # next t step
            # increment
            gaussian0, gaussian = tf.random.normal([nbSimul]), tf.random.normal([nbSimul])
            dW0, dW =  np.sqrt(self.mathModel.dt)*gaussian0, np.sqrt(self.mathModel.dt)*gaussian
            # jump and compensation
            dN , compens  = self.mathModel.dN()                 
            self.mathModel.oneStepFrom(dW0, dW, dN, hY, Y)
        costFunc_hat += self.mathModel.g(self.mathModel.hS)
        costFunc += self.mathModel.g(self.mathModel.S)
        return   tf.reduce_mean(costFunc_hat), tf.reduce_mean(costFunc)
      
# global as multiStep regression  for hatY
# see algorithm 
# global as multiStep regression  for hatY
# see algorithm 
class SolverGlobalMultiStepReg(SolverBase):
    def __init__(self, mathModel,   modelKeras, lRate, couplage):
        super().__init__(mathModel,   modelKeras, lRate, couplage)

        
    def train( self,  batchSize,  batchSizeVal, num_epoch, num_epochExt):
        @tf.function
        def regressOptim( nbSimul):
            # initialize
            self.mathModel.init(nbSimul)
            # Target
            listOfForward_hat = []
            listOfForward = []
            for istep in range(self.mathModel.N): 
                # get back Y
                hY, = self.modelKeras.model_hat(self.mathModel.getProjectedStates())
                Y, = self.modelKeras.model(self.mathModel.getAllStates())
                # listforward
                listOfForward_hat.append(hY)
                listOfForward.append(Y)                 
                # to Add
                toAdd_hat =- self.mathModel.dt* self.mathModel.f(self.mathModel.hS)
                toAdd =- self.mathModel.dt* self.mathModel.f(self.mathModel.S)
                for i in range(len(listOfForward)):
                    listOfForward_hat[i]= listOfForward_hat[i]+ toAdd_hat
                    listOfForward[i] = listOfForward[i] + toAdd
                # increment
                gaussian0, gaussian = tf.random.normal([nbSimul]), tf.random.normal([nbSimul])
                dW0, dW =  np.sqrt(self.mathModel.dt)*gaussian0, np.sqrt(self.mathModel.dt)*gaussian
                # jump and compensation
                dN , compens  = self.mathModel.dN()  
                # next t step
                self.mathModel.oneStepFrom(dW0, dW, dN, hY, Y)
            # final U
            Yfinal_hat = self.mathModel.g(self.mathModel.hS)
            Yfinal = self.mathModel.g(self.mathModel.S)
            listOfForward_hat = tf.stack(listOfForward_hat, axis=0) 
            listOfForward = tf.stack(listOfForward, axis=0) 
            if self.couplage == 'OFF':
              return tf.reduce_sum(tf.reduce_mean(tf.reduce_mean(tf.square(listOfForward_hat - tf.tile(tf.expand_dims(Yfinal_hat,axis=0), [self.mathModel.N,1])), axis=-1), axis=-1))\
                , tf.reduce_sum(tf.reduce_mean(tf.reduce_mean(tf.square(listOfForward - tf.tile(tf.expand_dims(Yfinal,axis=0), [self.mathModel.N,1])), axis=-1), axis=-1))
            return tf.reduce_sum(tf.reduce_mean(tf.reduce_mean(tf.square(listOfForward_hat - tf.tile(tf.expand_dims(Yfinal_hat,axis=0), [self.mathModel.N,1])), axis=-1), axis=-1))\
                + tf.reduce_sum(tf.reduce_mean(tf.reduce_mean(tf.square(listOfForward - tf.tile(tf.expand_dims(Yfinal,axis=0), [self.mathModel.N,1])), axis=-1), axis=-1))
        
                
        # train to optimize control
        @tf.function
        def trainOpt_hat( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= regressOptim( nbSimul)[0]
            gradients= tape.gradient(objFunc, self.modelKeras.model_hat.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKeras.model_hat.trainable_variables))
            return objFunc

        # train to optimize control
        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= regressOptim( nbSimul)[1]
            gradients= tape.gradient(objFunc, self.modelKeras.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKeras.model.trainable_variables))
            return objFunc

        # train to optimize control
        @tf.function
        def trainOptCoupled( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= regressOptim( nbSimul)
            CoupledVariables = self.modelKeras.model_hat.trainable_variables + self.modelKeras.model.trainable_variables
            gradients= tape.gradient(objFunc, CoupledVariables)
            optimizer.apply_gradients(zip(gradients, CoupledVariables))
            return objFunc

        optimizer= optimizers.Adam(learning_rate = self.lRate)
            
        self.listY0_hat = []
        self.listY0 = []
        if self.couplage == 'ON':
            for iout in range(num_epochExt):
                start_time = time.time()
                for epoch in range(num_epoch):
                    # un pas de gradient stochastique
                    trainOptCoupled(batchSize, optimizer)
                end_time = time.time()
                rtime = end_time-start_time 
                objError = regressOptim(batchSizeVal)
                self.mathModel.init(1)
                Y0_hat, Y0 =  self.modelKeras.model_hat(self.mathModel.getProjectedStates())[0][0].numpy(), self.modelKeras.model(self.mathModel.getAllStates())[0][0].numpy()
                print("Error ",objError.numpy(),  " took %5.3f s" % rtime, "Y0_hat sofar ",Y0_hat, 'Y0 sofar', Y0, 'epoch', iout)
                self.listY0_hat.append(Y0_hat)
                self.listY0.append(Y0)      
        else :
            for iout in range(num_epochExt):
                start_time = time.time()
                for epoch in range(num_epoch):
                    # un pas de gradient stochastique
                    trainOpt_hat(batchSize, optimizer)
                end_time = time.time()
                rtime = end_time-start_time 
                objError = regressOptim(batchSizeVal)[0]
                self.mathModel.init(1)
                Y0_hat = self.modelKeras.model_hat(self.mathModel.getProjectedStates())[0][0].numpy()
                print("Error hat ",objError.numpy(),  " took %5.3f s" % rtime, "Y0_hat sofar ",Y0_hat, 'epoch', iout)
                self.listY0_hat.append(Y0_hat) 

            for iout in range(num_epochExt):
                start_time = time.time()
                for epoch in range(num_epoch):
                    # un pas de gradient stochastique
                    trainOpt(batchSize, optimizer)
                end_time = time.time()
                rtime = end_time-start_time 
                objError = regressOptim(batchSizeVal)[1]
                self.mathModel.init(1)
                Y0 = self.modelKeras.model(self.mathModel.getAllStates())[0][0].numpy()
                print("Error",objError.numpy(),  " took %5.3f s" % rtime, "Y0 sofar ",Y0)
                self.listY0.append(Y0)   
        return self.listY0_hat, self.listY0
 
    def simulateGlobalErr(self, nbSimul):
        # initialize
        self.mathModel.init(nbSimul)
        costFunc_hat =0.
        costFunc = 0.
        for istep in range(self.mathModel.N):
            # get back U, Z, 
            hY, = self.modelKeras.model_hat(self.mathModel.getProjectedStates())
            Y, = self.modelKeras.model(self.mathModel.getAllStates())
            #print("AVEC ", tf.reduce_mean(dN)," COM ", tf.reduce_mean(dN-compens))
            costFunc_hat += self.mathModel.dt* self.mathModel.f(self.mathModel.hS)
            costFunc += self.mathModel.dt* self.mathModel.f(self.mathModel.S)
            # next t step
            # increment
            gaussian0, gaussian = tf.random.normal([nbSimul]), tf.random.normal([nbSimul])
            dW0, dW =  np.sqrt(self.mathModel.dt)*gaussian0, np.sqrt(self.mathModel.dt)*gaussian
            # jump and compensation
            dN , compens  = self.mathModel.dN()                 
            self.mathModel.oneStepFrom(dW0, dW, dN, hY, Y)
        costFunc_hat += self.mathModel.g(self.mathModel.hS)
        costFunc += self.mathModel.g(self.mathModel.S)
        return  tf.reduce_mean(costFunc_hat), tf.reduce_mean(costFunc)

class SolverOsterleeFBSDE(SolverBase):
    def __init__(self, mathModel,   modelKeras, lRate, couplage, lamCoef):
        super().__init__(mathModel,   modelKeras, lRate, couplage)
        self.lamCoef = lamCoef

    def train(self,  batchSize,  batchSizeVal, num_epoch, num_epochExt):

        @tf.function
        def optimizeBSDE( nbSimul):
            #Compute initial hY and Y
            ####################################################################################################
            # initialize
            self.mathModel.init(nbSimul)   
            hY0 = 0
            Y0 = 0     
            for istep in range(self.mathModel.N):
                # get back U, Z, 
                hY, hZ0, hGam = self.modelKeras.model_hat(self.mathModel.getProjectedStates())
                Y, Z0, Gam, Z = self.modelKeras.model(self.mathModel.getAllStates())
                # increment
                gaussian0, gaussian = tf.random.normal([nbSimul]), tf.random.normal([nbSimul])
                dW0, dW =  np.sqrt(self.mathModel.dt)*gaussian0, np.sqrt(self.mathModel.dt)*gaussian 
                # jump and compensation
                dN , compens  = self.mathModel.dN()
                # target
                hY0 = hY0 + self.mathModel.dt* self.mathModel.f(self.mathModel.hS) - hZ0* dW0 - hGam*(dN-compens)
                Y0 = Y0 + self.mathModel.dt* self.mathModel.f(self.mathModel.S) - Z0* dW0 - Gam*(dN-compens) - Z*dW
                # next t step
                self.mathModel.oneStepFrom(dW0, dW, dN, hY, Y)
            hY0 += self.mathModel.g(self.mathModel.hS) 
            Y0 += self.mathModel.g(self.mathModel.S)
            # initial value for the next phase
            hY = tf.reduce_mean(hY0)
            Y = tf.reduce_mean(Y0)
            #Compute loss function
            #######################################################################################################
            # initialize
            self.mathModel.init(nbSimul)     
            for istep in range(self.mathModel.N):
                # get back U, Z, 
                _, hZ0, hGam = self.modelKeras.model_hat(self.mathModel.getProjectedStates())
                _, Z0, Gam, Z = self.modelKeras.model(self.mathModel.getAllStates())
                # increment
                gaussian0, gaussian = tf.random.normal([nbSimul]), tf.random.normal([nbSimul])
                dW0, dW =  np.sqrt(self.mathModel.dt)*gaussian0, np.sqrt(self.mathModel.dt)*gaussian 
                # jump and compensation
                dN , compens  = self.mathModel.dN()
                # target
                hYNext = hY - self.mathModel.dt* self.mathModel.f(self.mathModel.hS) + hZ0* dW0 + hGam*(dN-compens)
                YNext = Y - self.mathModel.dt* self.mathModel.f(self.mathModel.S) + Z0* dW0 + Gam*(dN-compens) + Z*dW  
                # next t step
                self.mathModel.oneStepFrom(dW0, dW, dN, hY, Y)
                # Update 
                hY, Y = hYNext, YNext
            if self.couplage == 'OFF':
              return tf.reduce_mean(hY0) + self.lamCoef*tf.reduce_mean(tf.square(hY - self.mathModel.g(self.mathModel.hS) ))\
              ,tf.reduce_mean(Y0)+ self.lamCoef*tf.reduce_mean(tf.square(Y - self.mathModel.g(self.mathModel.S) )), tf.reduce_mean(hY0), tf.reduce_mean(Y0)
            return tf.reduce_mean(hY0) + self.lamCoef*tf.reduce_mean(tf.square(hY - self.mathModel.g(self.mathModel.hS) )) \
            + tf.reduce_mean(Y0) + self.lamCoef*tf.reduce_mean(tf.square(Y - self.mathModel.g(self.mathModel.S) )), tf.reduce_mean(hY0), tf.reduce_mean(Y0)
                
        # train to optimize control
        @tf.function
        def trainOpt_hat( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= optimizeBSDE( 2*nbSimul)[0]
            gradients= tape.gradient(objFunc, self.modelKeras.model_hat.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKeras.model_hat.trainable_variables))
            return objFunc

        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= optimizeBSDE( 2*nbSimul)[1]
            gradients= tape.gradient(objFunc, self.modelKeras.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKeras.model.trainable_variables))
            return objFunc

        @tf.function
        def trainOptCoupled( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc,_,_ = optimizeBSDE( 2*nbSimul)
            CoupledVariables = self.modelKeras.model_hat.trainable_variables + self.modelKeras.model.trainable_variables
            gradients= tape.gradient(objFunc, CoupledVariables)
            optimizer.apply_gradients(zip(gradients, CoupledVariables))
            return objFunc

        optimizer= optimizers.Adam(learning_rate = self.lRate)

        self.listY0_hat = []
        self.listY0 = []
        if self.couplage == 'ON':
          for iout in range(num_epochExt):
              start_time = time.time()
              for epoch in range(num_epoch):
                  # un pas de gradient stochastique
                  trainOptCoupled(batchSize, optimizer)
              end_time = time.time()
              rtime = end_time-start_time 
              objError, Y0_hat, Y0 = optimizeBSDE(batchSizeVal)
              print("Error ",objError.numpy(),  " took %5.3f s" % rtime, "Y0_hat sofar ",Y0_hat.numpy(), 'Y0 sofar', Y0.numpy(), 'epoch', iout)
              self.listY0_hat.append(Y0_hat.numpy())
              self.listY0.append(Y0.numpy())      
        else :
            for iout in range(num_epochExt):
              start_time = time.time()
              for epoch in range(num_epoch):
                  # un pas de gradient stochastique
                  trainOpt_hat(batchSize, optimizer)
              end_time = time.time()
              rtime = end_time-start_time 
              objError = optimizeBSDE(batchSizeVal)[0]
              Y0_hat = optimizeBSDE(batchSizeVal)[2]
              print("Error hat ",objError.numpy(),  " took %5.3f s" % rtime, "Y0_hat sofar ",Y0_hat.numpy(), 'epoch', iout)
              self.listY0_hat.append(Y0_hat.numpy()) 

            for iout in range(num_epochExt):
              start_time = time.time()
              for epoch in range(num_epoch):
                  # un pas de gradient stochastique
                  trainOpt(batchSize, optimizer)
              end_time = time.time()
              rtime = end_time-start_time 
              objError = optimizeBSDE(batchSizeVal)[1]
              Y0 = optimizeBSDE(batchSizeVal)[3]
              print(" Error",objError.numpy(),  " took %5.3f s" % rtime, "Y0 sofar ",Y0.numpy(), 'epoch', iout)
              self.listY0.append(Y0.numpy())   
        return self.listY0_hat, self.listY0 
    


