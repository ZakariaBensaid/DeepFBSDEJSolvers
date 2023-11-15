import numpy as np
import tensorflow as tf
import time
from tensorflow.keras import  optimizers

class SolverBase:
    # mathModel          Math model
    # modelKeras         Keras model
    # lRate              Learning rate
    def __init__(self, mathModel, modelKerasUZ , modelKerasGam,  lRate):
        # to store les different networks
        self.mathModel = mathModel
        self.modelKerasUZ = modelKerasUZ
        self.modelKerasGam = modelKerasGam
        self.lRate = lRate

class SolverGlobalFBSDE(SolverBase):
    def __init__(self, mathModel, modelKerasUZ , modelKerasGam,  lRate):
        super().__init__(mathModel, modelKerasUZ , modelKerasGam,  lRate)

    def train(self,  batchSize,  batchSizeVal, num_epoch, num_epochExt):
        @tf.function
        def optimizeBSDE( nbSimul):
            # initialize
            X = self.mathModel.init(nbSimul)
            # Target
            Y = self.modelKerasUZ.Y0*tf.ones([nbSimul])
            for iStep in range(self.mathModel.N):
                # increment
                gaussian = tf.random.normal([nbSimul])
                dW =  np.sqrt(self.mathModel.dt)*gaussian
                # jump and compensation
                gaussJ  = self.mathModel.jumps(nbSimul)
                gaussJMC = self.mathModel.jumps(5000)
                # get  Z, Gam
                Z = self.modelKerasUZ(tf.stack([iStep*tf.ones([nbSimul], dtype= tf.float32) , X], axis=-1))[0]
                Gam = self.modelKerasGam(tf.stack([iStep*tf.ones([nbSimul], dtype= tf.float32), X , gaussJ], axis=-1))[0]
                GamCompensator = self.modelKerasGam(tf.stack([iStep*tf.ones([5000, nbSimul], dtype= tf.float32), tf.broadcast_to(X, [5000, nbSimul]) ,\
                                                              tf.ones([5000, nbSimul], dtype= tf.float32)*gaussJMC[:, tf.newaxis]], axis=-1))[0]
                # target
                Y = Y - self.mathModel.dt*self.mathModel.f(Y) + Z*dW + Gam - tf.reduce_mean(GamCompensator, axis = 0)
                # next t step
                X = self.mathModel.oneStepFrom(iStep, X, dW, gaussJ, Y)
            return  tf.reduce_mean(tf.square(Y- self.mathModel.g(X)))

        # train to optimize control
        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc = optimizeBSDE( nbSimul)
            gradients= tape.gradient(objFunc, self.modelKerasUZ.trainable_variables + self.modelKerasGam.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKerasUZ.trainable_variables+ self.modelKerasGam.trainable_variables))
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
            Y0 = self.modelKerasUZ.Y0
            print(" Error",objError.numpy(),  " elapsed time %5.3f s" % self.duration, "Y0 sofar ",Y0.numpy(), 'epoch', iout)
            self.listY0.append(Y0.numpy())
            self.lossList.append(objError)
        return self.listY0, self.duration

class SolverMultiStepFBSDE1():
    def __init__(self, mathModel, modelKerasUZ,  lRate):
        # to store les different networks
        self.mathModel = mathModel
        self.modelKerasUZ = modelKerasUZ
        self.lRate = lRate

    def train(self,  batchSize,  batchSizeVal, num_epoch, num_epochExt):

        @tf.function
        def optimizeBSDE( nbSimul):
            # initialize
            X = self.mathModel.init(nbSimul)
            # Target
            listOfForward = []
            Y0 = 0.
            for iStep in range(self.mathModel.N):
                # Common and individual noises
                gaussian = tf.random.normal([nbSimul])
                dW =  np.sqrt(self.mathModel.dt)*gaussian
                gaussJ  = self.mathModel.jumps(nbSimul)
                gaussJMC = self.mathModel.jumps(5000)
                # Adjoint variables
                Y, Z0 = self.modelKerasUZ(tf.stack([iStep* tf.ones([nbSimul], dtype= tf.float32) , X], axis=-1))
                Gam = self.modelKerasUZ(tf.stack([iStep* tf.ones([nbSimul], dtype= tf.float32), X*tf.math.exp(gaussJ)], axis=-1))[0]
                GamCompensator = self.modelKerasUZ(tf.stack([iStep* tf.ones([5000, nbSimul], dtype= tf.float32), tf.broadcast_to(X, [5000, nbSimul])*tf.math.exp(gaussJMC[:, tf.newaxis])], axis=-1))[0]
                # target
                toAdd = - self.mathModel.dt*self.mathModel.f(Y) + Z0*dW + Gam - tf.reduce_mean(GamCompensator, axis = 0)
                Y0 = Y0 - toAdd
                #update list and error
                listOfForward.append(Y)
                #forward
                for i in range(len(listOfForward)):
                    listOfForward[i] = listOfForward[i] + toAdd
                # next t step
                X = self.mathModel.oneStepFrom(iStep, X, dW, gaussJ, Y)
            #Final Y
            Yfinal = self.mathModel.g(X)
            Y0 += Yfinal
            listOfForward = tf.stack(listOfForward, axis=0)
            return tf.reduce_sum(tf.reduce_mean(tf.reduce_mean(tf.square(listOfForward - tf.tile(tf.expand_dims(Yfinal,axis=0), [self.mathModel.N,1])), axis=-1), axis=-1)), tf.reduce_mean(Y0)

        # train to optimize control
        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc,_ = optimizeBSDE( nbSimul)
            gradients= tape.gradient(objFunc, self.modelKerasUZ.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKerasUZ.trainable_variables))
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
            objError, _ = optimizeBSDE(batchSizeVal)
            X = self.mathModel.init(10**5)
            Y0 = tf.reduce_mean(self.modelKerasUZ( tf.stack([tf.zeros([10**5], dtype= tf.float32) , X], axis=-1))[0])
            #Y0 = []
            #for k in range(20):
              #Y0.append(optimizeBSDE(200)[1].numpy())
            #Y0 = np.mean(Y0)
            print(" Error",objError.numpy(),  " took %5.3f s" % self.duration, "Y0 sofar ",Y0.numpy(), 'epoch', iout)
            self.listY0.append(Y0.numpy())
            self.lossList.append(objError)
        return self.listY0, self.duration

class SolverMultiStepFBSDE2(SolverBase):
    def __init__(self, mathModel, modelKerasUZ , modelKerasGam,  lRate):
        super().__init__(mathModel, modelKerasUZ , modelKerasGam,  lRate)

    def train(self,  batchSize,  batchSizeVal, num_epoch, num_epochExt):

        @tf.function
        def optimizeBSDE( nbSimul):
            # initialize
            X = self.mathModel.init(nbSimul)
            # Target
            listOfForward = []
            Y0 = 0.
            for iStep in range(self.mathModel.N):
                # Common and individual noises
                gaussian = tf.random.normal([nbSimul])
                dW =  np.sqrt(self.mathModel.dt)*gaussian
                # jumps
                gaussJ  = self.mathModel.jumps(nbSimul)
                gaussJMC = self.mathModel.jumps(5000)
                # Adjoint variables
                Y, Z0 = self.modelKerasUZ(tf.stack([iStep* tf.ones([nbSimul], dtype= tf.float32) , X], axis=-1))
                Gam = self.modelKerasGam(tf.stack([iStep*tf.ones([nbSimul], dtype= tf.float32), X , tf.math.exp(gaussJ)], axis=-1))[0]
                GamCompensator = self.modelKerasGam(tf.stack([iStep*tf.ones([5000, nbSimul], dtype= tf.float32), tf.broadcast_to(X, [5000, nbSimul]) ,\
                                                              tf.ones([5000, nbSimul], dtype= tf.float32)*tf.math.exp(gaussJMC[:, tf.newaxis])], axis=-1))[0]
                # target
                toAdd = - self.mathModel.dt*self.mathModel.f(Y) + Z0*dW + Gam - tf.reduce_mean(GamCompensator, axis = 0)
                Y0 = Y0 - toAdd
                #update list and error
                listOfForward.append(Y)
                #forward
                for i in range(len(listOfForward)):
                    listOfForward[i] = listOfForward[i] + toAdd
                # next t step
                X = self.mathModel.oneStepFrom(iStep, X, dW, gaussJ, Y)
            #Final Y
            Yfinal = self.mathModel.g(X)
            Y0 += Yfinal
            listOfForward = tf.stack(listOfForward, axis=0)
            return tf.reduce_sum(tf.reduce_mean(tf.reduce_mean(tf.square(listOfForward - tf.tile(tf.expand_dims(Yfinal,axis=0), [self.mathModel.N,1])), axis=-1), axis=-1)), tf.reduce_mean(Y0)

        # train to optimize control
        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc, _ = optimizeBSDE( nbSimul)
            gradients= tape.gradient(objFunc, self.modelKerasUZ.trainable_variables + self.modelKerasGam.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKerasUZ.trainable_variables + self.modelKerasGam.trainable_variables))
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
            objError, _ = optimizeBSDE(batchSizeVal)
            X = self.mathModel.init(10**5)
            Y0 = tf.reduce_mean(self.modelKerasUZ( tf.stack([tf.zeros([10**5], dtype= tf.float32) , X], axis=-1))[0])
            #Y0 = []
            #for k in range(20):
              #Y0.append(optimizeBSDE(200)[1].numpy())
            #Y0 = np.mean(Y0)
            print(" Error",objError.numpy(),  " took %5.3f s" % self.duration, "Y0 sofar ",Y0.numpy(), 'epoch', iout)
            self.listY0.append(Y0.numpy())
            self.lossList.append(objError)
        return self.listY0, self.duration

class SolverSumLocalFBSDE1():
    def __init__(self, mathModel, modelKerasUZ , lRate):
        # to store les different networks
        self.mathModel = mathModel
        self.modelKerasUZ = modelKerasUZ
        self.lRate = lRate

    def train(self,  batchSize,  batchSizeVal, num_epoch, num_epochExt):

        @tf.function
        def optimizeBSDE(nbSimul):
            # initialize
            X = self.mathModel.init(nbSimul)
            gaussJ = self.mathModel.jumps(nbSimul)
            gaussJMC = self.mathModel.jumps(5000)
            #error
            error = 0
            #init val
            Y0 = 0.
            YPrev, Z0Prev = self.modelKerasUZ( tf.stack([tf.zeros([nbSimul], dtype= tf.float32) , X], axis=-1))
            GamPrev = self.modelKerasUZ(tf.stack([tf.zeros([nbSimul], dtype= tf.float32), X*tf.math.exp(gaussJ)], axis=-1))[0]
            GamCompensatorPrev = self.modelKerasUZ(tf.stack([tf.zeros([5000, nbSimul], dtype= tf.float32), tf.broadcast_to(X, [5000, nbSimul])*tf.math.exp(gaussJMC[:, tf.newaxis])], axis=-1))[0]
            for iStep in range(self.mathModel.N):
                # increment
                gaussian = tf.random.normal([nbSimul])
                dW =  np.sqrt(self.mathModel.dt)*gaussian
                # target
                toAdd = self.mathModel.dt*self.mathModel.f(YPrev) - Z0Prev*dW - GamPrev + tf.reduce_mean(GamCompensatorPrev, axis = 0)
                Y0 = Y0 + toAdd
                # next step
                X = self.mathModel.oneStepFrom(iStep, X, dW, gaussJ, YPrev)
                # jumps
                gaussJ = self.mathModel.jumps(nbSimul)
                gaussJMC = self.mathModel.jumps(5000)
                if (iStep == (self.mathModel.N - 1)):
                    YNext = self.mathModel.g(X)
                    Y0 += YNext
                else:
                    YNext, Z0Prev= self.modelKerasUZ( tf.stack([iStep* tf.ones([nbSimul], dtype= tf.float32) , X], axis=-1))
                    GamPrev = self.modelKerasUZ(tf.stack([iStep* tf.ones([nbSimul], dtype= tf.float32), X*tf.math.exp(gaussJ)], axis=-1))[0]
                    GamCompensatorPrev = self.modelKerasUZ(tf.stack([iStep* tf.ones([5000, nbSimul], dtype= tf.float32), tf.broadcast_to(X, [5000, nbSimul])*tf.math.exp(gaussJMC[:, tf.newaxis])], axis=-1))[0]
                error = error + tf.reduce_mean(tf.square(YNext - YPrev + toAdd))
                YPrev = YNext
            return error, tf.reduce_mean(Y0)

        # train to optimize control
        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc, _ = optimizeBSDE( nbSimul)
            gradients= tape.gradient(objFunc, self.modelKerasUZ.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKerasUZ.trainable_variables))
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
            objError, _ = optimizeBSDE(batchSizeVal)
            X = self.mathModel.init(10**5)
            Y0 = tf.reduce_mean(self.modelKerasUZ( tf.stack([tf.zeros([10**5], dtype= tf.float32) , X], axis=-1))[0])
            #Y0 = []
            #for k in range(20):
              #Y0.append(optimizeBSDE(200)[1].numpy())
            #Y0 = np.mean(Y0)
            print(" Error",objError.numpy(),  " took %5.3f s" % self.duration, "Y0 sofar ",Y0.numpy(), 'epoch', iout)
            self.listY0.append(Y0.numpy())
            self.lossList.append(objError)
        return self.listY0, self.duration

class SolverSumLocalFBSDE2(SolverBase):
    def __init__(self, mathModel, modelKerasUZ , modelKerasGam,  lRate):
        super().__init__(mathModel, modelKerasUZ , modelKerasGam,  lRate)

    def train(self,  batchSize,  batchSizeVal, num_epoch, num_epochExt):

        @tf.function
        def optimizeBSDE( nbSimul):
            # initialize
            X = self.mathModel.init(nbSimul)
            gaussJ = self.mathModel.jumps(nbSimul)
            gaussJMC = self.mathModel.jumps(5000)
            #error
            error = 0
            #init val
            Y0 = 0.
            YPrev, Z0Prev= self.modelKerasUZ( tf.stack([tf.zeros([nbSimul], dtype= tf.float32) , X], axis=-1))
            GamPrev = self.modelKerasGam(tf.stack([tf.zeros([nbSimul], dtype= tf.float32), X , tf.math.exp(gaussJ)], axis=-1))[0]
            GamCompensatorPrev = self.modelKerasGam(tf.stack([tf.zeros([5000, nbSimul], dtype= tf.float32), tf.broadcast_to(X, [5000, nbSimul]) ,\
                                                         tf.ones([5000, nbSimul], dtype= tf.float32)*tf.math.exp(gaussJMC[:, tf.newaxis])], axis=-1))[0]
            for iStep in range(self.mathModel.N):
                # increment
                gaussian = tf.random.normal([nbSimul])
                dW =  np.sqrt(self.mathModel.dt)*gaussian
                # target
                toAdd = self.mathModel.dt*self.mathModel.f(YPrev) - Z0Prev*dW - GamPrev + tf.reduce_mean(GamCompensatorPrev, axis = 0)
                Y0 = Y0 + toAdd
                # next step
                X = self.mathModel.oneStepFrom(iStep, X, dW, gaussJ, YPrev)
                # jumps
                gaussJ = self.mathModel.jumps(nbSimul)
                gaussJMC = self.mathModel.jumps(5000)
                if (iStep == (self.mathModel.N - 1)):
                    YNext = self.mathModel.g(X)
                    Y0 += YNext
                else:
                    YNext, Z0Prev= self.modelKerasUZ( tf.stack([iStep* tf.ones([nbSimul], dtype= tf.float32) , X], axis=-1))
                    GamPrev = self.modelKerasGam(tf.stack([iStep*tf.ones([nbSimul], dtype= tf.float32), X, tf.math.exp(gaussJ)], axis=-1))[0]
                    GamCompensatorPrev = self.modelKerasGam(tf.stack([iStep*tf.ones([5000, nbSimul], dtype= tf.float32), tf.broadcast_to(X, [5000, nbSimul]) ,\
                                                                  tf.ones([5000, nbSimul], dtype= tf.float32)*tf.math.exp(gaussJMC[:, tf.newaxis])], axis=-1))[0]
                error = error + tf.reduce_mean(tf.square(YNext - YPrev + toAdd))
                YPrev = YNext
            return error, tf.reduce_mean(Y0)

        # train to optimize control
        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc, _ = optimizeBSDE( nbSimul)
            gradients= tape.gradient(objFunc, self.modelKerasUZ.trainable_variables + self.modelKerasGam.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKerasUZ.trainable_variables + self.modelKerasGam.trainable_variables))
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
            objError, _ = optimizeBSDE(batchSizeVal)
            X = self.mathModel.init(10**5)
            Y0 = tf.reduce_mean(self.modelKerasUZ( tf.stack([tf.zeros([10**5], dtype= tf.float32) , X], axis=-1))[0])
            #Y0 = []
            #for k in range(20):
              #Y0.append(optimizeBSDE(200)[1].numpy())
            #Y0 = np.mean(Y0)
            print(" Error",objError.numpy(),  " took %5.3f s" % self.duration, "Y0 sofar ",Y0.numpy(), 'epoch', iout)
            self.listY0.append(Y0.numpy())
            self.lossList.append(objError)
        return self.listY0, self.duration

# global as sum of local error due to regressions
# see algorithm
class SolverGlobalSumLocalReg(SolverBase):
    def __init__(self, mathModel, modelKerasUZ , modelKerasGam,  lRate):
        super().__init__(mathModel, modelKerasUZ , modelKerasGam,  lRate)

    def train( self,  batchSize,  batchSizeVal, num_epoch, num_epochExt):
        @tf.function
        def regressOptim(nbSimul):
            # initialize
            X = self.mathModel.init(nbSimul)
            # Target
            error = 0.
            # get back Y
            YPrev = self.modelKerasUZ(tf.stack([tf.zeros([nbSimul], dtype= tf.float32) , X], axis=-1))[0]
            for iStep in range(self.mathModel.N):
                # target
                toAdd = self.mathModel.dt*self.mathModel.f(YPrev)
                # increment
                gaussian = tf.random.normal([nbSimul])
                dW =  np.sqrt(self.mathModel.dt)*gaussian
                # jumps
                gaussJ = self.mathModel.jumps(nbSimul)
                # Next step
                X = self.mathModel.oneStepFrom(iStep, X, dW, gaussJ, YPrev)
                # values
                if (iStep == (self.mathModel.N - 1)):
                    YNext = self.mathModel.g(X)
                else:
                    YNext = self.modelKerasUZ(tf.stack([iStep*tf.ones([nbSimul], dtype= tf.float32) , X], axis=-1))[0]
                error = error +  tf.reduce_mean(tf.square(YNext - YPrev + toAdd))
                YPrev = YNext
            return error

        # train to optimize control
        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= regressOptim(nbSimul)
            gradients= tape.gradient(objFunc, self.modelKerasUZ.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKerasUZ.trainable_variables))
            return objFunc

        optimizer= optimizers.Adam(learning_rate = self.lRate)

        self.listY0 = []
        self.lossList = []
        self.duration = 0
        for iout in range(num_epochExt):
            start_time = time.time()
            for epoch in range(num_epoch):
                # un pas de gradient stochastique
                trainOpt(1000*batchSize, optimizer)
            end_time = time.time()
            rtime = end_time-start_time
            self.duration += rtime
            objError = regressOptim(100*batchSizeVal)
            X = self.mathModel.init(10**5)
            Y0 = tf.reduce_mean(self.modelKerasUZ( tf.stack([tf.zeros([10**5], dtype= tf.float32) , X], axis=-1))[0])
            print(" Error",objError.numpy(),  " took %5.3f s" % self.duration, "Y0 sofar ",Y0.numpy(), 'epoch', iout)
            self.listY0.append(Y0.numpy())
            self.lossList.append(objError)
        return self.listY0, self.duration



# global as multiStep regression  for hatY
# see algorithm
# global as multiStep regression  for hatY
# see algorithm
class SolverGlobalMultiStepReg(SolverBase):
    def __init__(self, mathModel, modelKerasUZ , modelKerasGam,  lRate):
        super().__init__(mathModel, modelKerasUZ , modelKerasGam,  lRate)


    def train( self,  batchSize,  batchSizeVal, num_epoch, num_epochExt):
        @tf.function
        def regressOptim( nbSimul):
            # initialize
            X = self.mathModel.init(nbSimul)
            # Target
            listOfForward = []
            for iStep in range(self.mathModel.N):
                # get back Y
                Y = self.modelKerasUZ(tf.stack([iStep*tf.ones([nbSimul], dtype= tf.float32) , X], axis=-1))[0]
                # listforward
                listOfForward.append(Y)
                # to Add
                toAdd =- self.mathModel.dt* self.mathModel.f(Y)
                for i in range(len(listOfForward)):
                    listOfForward[i] = listOfForward[i] + toAdd
                # increment
                gaussian = tf.random.normal([nbSimul])
                dW =  np.sqrt(self.mathModel.dt)*gaussian
                gaussJ = self.mathModel.jumps(nbSimul)
                # next t step
                X = self.mathModel.oneStepFrom(iStep, X, dW, gaussJ, Y)
            # final U
            Yfinal = self.mathModel.g(X)
            listOfForward = tf.stack(listOfForward, axis=0)
            return tf.reduce_sum(tf.reduce_mean(tf.reduce_mean(tf.square(listOfForward - tf.tile(tf.expand_dims(Yfinal,axis=0), [self.mathModel.N,1])), axis=-1), axis=-1))

        # train to optimize control
        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= regressOptim( nbSimul)
            gradients= tape.gradient(objFunc, self.modelKerasUZ.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKerasUZ.trainable_variables))
            return objFunc

        optimizer= optimizers.Adam(learning_rate = self.lRate)

        self.listY0 = []
        self.lossList = []
        self.duration = 0
        for iout in range(num_epochExt):
            start_time = time.time()
            for epoch in range(num_epoch):
                # un pas de gradient stochastique
                trainOpt(1000*batchSize, optimizer)
            end_time = time.time()
            rtime = end_time-start_time
            self.duration += rtime
            objError = regressOptim(batchSizeVal)
            X = self.mathModel.init(10**5)
            Y0 = tf.reduce_mean(self.modelKerasUZ( tf.stack([tf.zeros([10**5], dtype= tf.float32) , X], axis=-1))[0])
            print(" Error",objError.numpy(),  " took %5.3f s" % self.duration, "Y0 sofar ",Y0.numpy(), 'epoch', iout)
            self.listY0.append(Y0.numpy())
            self.lossList.append(objError)
        return self.listY0, self.duration
