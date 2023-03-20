import numpy as np
import tensorflow as tf
import time
from tensorflow.keras import  optimizers

        
class SolverSumLocalFBSDE:
    def __init__(self, mathModel,   modelKerasUZ, lRate):
        self.mathModel = mathModel
        self.modelKerasUZ= modelKerasUZ
        self.lRate= lRate 

    def train(self,  batchSize,  batchSizeVal, num_epoch, num_epochExt):

        @tf.function
        def optimizeBSDE( nbSimul):
            # initialize
            X= self.mathModel.init(nbSimul)
            #error
            error = 0
            #init val
            YPrev, Z0Prev  = self.modelKerasUZ(tf.stack([tf.zeros([nbSimul], dtype= tf.float32) , X], axis=-1))
            for iStep in range(self.mathModel.N):
                # increment
                gaussian = tf.random.normal([nbSimul])
                dW =  np.sqrt(self.mathModel.dt)*gaussian 
                # jump and compensation
                dN, listJumps, gaussJ  = self.mathModel.jumps()
                #gam
                gam = self.modelKerasUZ(tf.stack([iStep* tf.ones([nbSimul], dtype= tf.float32) , X + gaussJ], axis=-1))[0] -  self.modelKerasUZ(tf.stack([iStep* tf.ones([nbSimul], dtype= tf.float32) , X], axis=-1))[0]
                # target
                toAdd = self.mathModel.dt*self.mathModel.f(YPrev) - Z0Prev*dW - gam + tf.reduce_mean(gam)
                # next step
                X= self.mathModel.oneStepFrom(iStep,X,dW, gaussJ, YPrev)
                YInput = YPrev
                if (iStep == (self.mathModel.N - 1)):
                    YNext = self.mathModel.g(X)
                else:
                    YNext, Z0Prev = self.modelKerasUZ(tf.stack([(iStep+1)* tf.ones([nbSimul], dtype= tf.float32) , X], axis=-1))
                error = error + tf.reduce_mean(tf.square(YNext - YPrev + toAdd))
                YPrev = YNext
            return error
                
        # train to optimize control
        @tf.function
        def trainOpt( nbSimul,optimizer):
            with tf.GradientTape() as tape:
                objFunc= optimizeBSDE( nbSimul)
            gradients= tape.gradient(objFunc, self.modelKerasUZ.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKerasUZ.trainable_variables))
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
            X  =self.mathModel.init(1)
            Y0 = self.modelKerasUZ(tf.stack([ tf.zeros([1], dtype= tf.float32) , X], axis=-1))[0][0]
            print(" Error",objError.numpy(),  " took %5.3f s" % rtime, "Y0 sofar ",Y0.numpy(), 'epoch', iout)
            self.listY0.append(Y0.numpy())
            self.lossList.append(objError)   
        return self.listY0
