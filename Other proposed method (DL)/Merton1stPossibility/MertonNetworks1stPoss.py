import sys
import tensorflow as tf
from tensorflow.keras import layers

class Net( tf.keras.Model):
    def __init__( self, method, ndimOut, nbNeurons, activation= tf.nn.tanh):
        super().__init__()
        self.nbNeurons = nbNeurons
        self.name_ = "FeedForwardBSDE"
        self.ndimOut= ndimOut
        self.ListOfDense =  [tf.keras.layers.Dense( nbNeurons[i],activation= activation, kernel_initializer= tf.keras.initializers.GlorotUniform(), bias_initializer= tf.keras.initializers.RandomNormal(mean=0., stddev=1.))  for i in range(len(nbNeurons)) ]+[tf.keras.layers.Dense(ndimOut, activation= None, kernel_initializer= tf.keras.initializers.GlorotUniform(), bias_initializer= tf.keras.initializers.RandomNormal(mean=0., stddev=1.))]
        if method not in ['SumLocal', 'SumMultiStep','SumMultiStepReg', 'SumLocalReg', 'Osterlee']: 
            # not always used
            self.Y0= tf.Variable(tf.keras.initializers.GlorotUniform()([]),  trainable = True, dtype=tf.float32)
            
    def call(self,inputs):
        x = tf.stack([inputs[0]] + [ inputs[i] for i in range(1,len(inputs))], axis=-1)
        for layer in self.ListOfDense:
            x = layer(x)
        return [x[:,i] for i in range(self.ndimOut)]

class NetZ( tf.keras.Model):
    def __init__( self, method, ndimOut, nbNeurons, activation= tf.nn.tanh):
        super().__init__()
        self.nbNeurons = nbNeurons
        self.name_ = "FeedForwardBSDE"
        self.ndimOut= ndimOut
        self.ListOfDense =  [tf.keras.layers.Dense( nbNeurons[i],activation= activation, kernel_initializer= tf.keras.initializers.GlorotUniform(), bias_initializer= tf.keras.initializers.RandomNormal(mean=0., stddev=1.))  for i in range(len(nbNeurons)) ]+[tf.keras.layers.Dense(ndimOut, activation= None, kernel_initializer= tf.keras.initializers.GlorotUniform(), bias_initializer= tf.keras.initializers.RandomNormal(mean=0., stddev=1.))]

    def call(self,inputs):
        x = tf.stack([inputs[0]] + [ inputs[i] for i in range(1,len(inputs))], axis=-1)
        for layer in self.ListOfDense:
            x = layer(x)
        return [x[:,i] for i in range(self.ndimOut)]


class kerasModels:
    def __init__(self, Net, NetZ, method, ndimOut, nbNeurons, activation= tf.nn.tanh):
        super().__init__()
        self.modelZ = NetZ(method, ndimOut, nbNeurons, activation)
        self.model = Net(method, ndimOut, nbNeurons, activation)
