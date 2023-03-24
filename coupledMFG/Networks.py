import sys
import tensorflow as tf
from tensorflow.keras import layers

#Neural Networks
class Net_hat( tf.keras.Model):
    def __init__( self,method, ndimOut, nbNeurons, activation= tf.nn.tanh):
        super().__init__()
        self.nbNeurons = nbNeurons
        self.name_ = "FeedForwardBSDEProjectedCase"
        self.ndimOut= ndimOut
        self.ListOfDense =  [layers.Dense( nbNeurons[i],activation= activation, kernel_initializer= tf.keras.initializers.GlorotNormal())  for i in range(len(nbNeurons)) ]+[layers.Dense(ndimOut, activation= None, kernel_initializer= tf.keras.initializers.GlorotNormal())]
        if method not in ['SumLocal', 'SumMultiStep','SumMultiStepReg', 'SumLocalReg', 'Osterlee']: 
            # not always used
            self.Y0_hat= tf.Variable(tf.keras.initializers.GlorotUniform()([]),  trainable = True, dtype=tf.float32)

    def call(self,inputs):
        x =tf.stack([inputs[0]*tf.ones_like(inputs[1])] + [ inputs[i] for i in range(1,len(inputs))], axis=-1)
        for layer in self.ListOfDense:
            x = layer(x)
        return [x[:,i] for i in range(self.ndimOut)]

class Net( tf.keras.Model):
    def __init__( self, method, ndimOut, nbNeurons, activation= tf.nn.tanh):
        super().__init__()
        self.nbNeurons = nbNeurons
        self.name_ = "FeedForwardBSDE"
        self.ndimOut= ndimOut
        self.ListOfDense =  [layers.Dense( nbNeurons[i],activation= activation, kernel_initializer= tf.keras.initializers.GlorotNormal())  for i in range(len(nbNeurons)) ]+[layers.Dense(ndimOut, activation= None, kernel_initializer= tf.keras.initializers.GlorotNormal())]
        if method not in ['SumLocal', 'SumMultiStep','SumMultiStepReg', 'SumLocalReg', 'Osterlee']: 
            # not always used
            self.Y0= tf.Variable(tf.keras.initializers.GlorotNormal()([]),  trainable = True, dtype=tf.float32)


    def call(self,inputs):
        x =tf.stack([inputs[0]*tf.ones_like(inputs[1])] + [ inputs[i] for i in range(1,len(inputs))], axis=-1)
        for layer in self.ListOfDense:
            x = layer(x)
        return [x[:,i] for i in range(self.ndimOut)]


class kerasModels:
    def __init__(self, Net_hat, Net, method, ndimOut_hat, ndimOut, nbNeurons_hat, nbNeurons, activation_hat, activation= tf.nn.tanh):
        super().__init__()
        self.model_hat = Net_hat(method, ndimOut_hat, nbNeurons_hat, activation_hat)
        self.model = Net(method, ndimOut, nbNeurons, activation)

