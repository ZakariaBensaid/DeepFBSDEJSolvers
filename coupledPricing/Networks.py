import sys
import tensorflow as tf
from tensorflow.keras import layers

#Neural Network
class Net( tf.keras.Model):
    def __init__( self, bY0, ndimOut, nbNeurons, activation= tf.nn.tanh):
        super().__init__()
        self.nbNeurons = nbNeurons
        self.name_ = "FeedForward"
        self.ndimOut = ndimOut
        self.ListOfDense =  [layers.Dense( nbNeurons[i],activation= activation, kernel_initializer= tf.keras.initializers.GlorotNormal())  for i in range(len(nbNeurons)) ]+[layers.Dense(self.ndimOut, activation= None, kernel_initializer= tf.keras.initializers.GlorotNormal())]
        if (bY0 ==1):
            self.Y0= tf.Variable(tf.keras.initializers.GlorotNormal()([]),  trainable = True, dtype=tf.float32)


    def call(self,inputs):
        x = inputs 
        for layer in self.ListOfDense:
            x = layer(x)
        return [x[:,i] for i in range(self.ndimOut)]