import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os
#from Networks import Next
from mathModelCoupled import MertonJumpModel
from SolversCoupled import SolverGlobalFBSDE #, SolverMultiStepFBSDE,SolverSumLocalFBSDE, SolverGlobalMultiStepReg, SolverGlobalSumLocalReg, SolverOsterleeFBSDE
from ClosedFormulaMerton import Option_param, Merton_process, Merton_pricer
import argparse
import sys 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

class Net( tf.keras.Model):
    def __init__( self, bY0, nbNeurons, activation= tf.nn.tanh):
        super().__init__()
        self.nbNeurons = nbNeurons
        self.name_ = "FeedForwardANd0"
        self.ListOfDense =  [layers.Dense( nbNeurons[i],activation= activation, kernel_initializer= tf.keras.initializers.GlorotNormal())  for i in range(len(nbNeurons)) ]+[layers.Dense(1, activation= None, kernel_initializer= tf.keras.initializers.GlorotNormal())]
        if (bY0 ==1):
            self.Y0= tf.Variable(tf.keras.initializers.GlorotNormal()([]),  trainable = True, dtype=tf.float32)


    def call(self,inputs):
        x = inputs #tf.stack(inputs, axis=-1)
        for layer in self.ListOfDense:
            x = layer(x)
        return x



    

os.environ['KMP_DUPLICATE_LIB_OK']='True'
parser = argparse.ArgumentParser()
parser.add_argument('--nbNeuron', type=int, default=20)
parser.add_argument('--nbLayer', type=int, default=2)
parser.add_argument('--nEpochExt', type=int, default=100)
parser.add_argument('--nEpoch', type=int, default=100)
parser.add_argument('--batchSize',  type=int, default=10**4)
parser.add_argument('--lRateY0',type=float, default =0.003)
parser.add_argument('--lRateLoc',type=float, default =0.0003)
parser.add_argument('--lRateReg',type=float, default =0.0003)
parser.add_argument('--activation',  type= str, default="tanh")
parser.add_argument('--coefOsterlee', type= float, default = 1000)
parser.add_argument('--nbSimul', type= int, default = 5)
parser.add_argument('--aLin', type= float, default = 0.1)
parser.add_argument('--lam', type= float, default = 1.)
parser.add_argument('--N',  type= int, default = 50)
    
args = parser.parse_args()
print("Args ", args)
nbNeuron = args.nbNeuron
print("Nb neurons ", nbNeuron)
nbLayer= args.nbLayer
print("Nb hidden layers ", nbLayer)
batchSize = args.batchSize
print("batchSize " , batchSize)
num_epochExt = args.nEpochExt
print("num_epochExt " , num_epochExt)
num_epoch =args.nEpoch
print("Number epoch ", num_epoch)
lRateY0 =args.lRateY0
print("Learning rate Global", lRateY0)
lRateLoc =args.lRateLoc
print("Learning rate Local", lRateLoc)
lRateReg =args.lRateReg
print("Learning rate Regression", lRateReg)
activation = args.activation
if activation not in ['tanh', 'relu']:
    print(activation, 'is invalid. Please choose tanh or relu.')
    sys.exit(0)
print('activation', activation)
coefOsterlee = args.coefOsterlee
print('Osterlee coefficient', coefOsterlee)
nbSimul = args.nbSimul
print('number of trajectories', nbSimul)
aLin = args.aLin
print("Linear coupling forward backward ", aLin)
lam  = args.lam
print("Jump intensity", lam)
N=  args.N
print(" N ", N)
# Layers
######################################
layerSize = nbNeuron*np.ones((nbLayer,), dtype=np.int32) 
# parameter models
######################################
# parameter models
dict_parameters = {'T':1 ,  'r':0.1, 'sig': 0.3,  'muJ': 0., 'sigJ': 0.2, 'K': 0.9, 'x0': 1}
T,  r, sig,  muJ, sigJ, K, x0 = dict_parameters.values()
maxJumps = np.amax(np.random.poisson(lam*T/N, size = 10**7)) + 1
print("maxJumps",maxJumps) 
print('Maximum number of Jumps:', maxJumps)
def func(x):
  return aLin*tf.math.abs(x)
# DL model
######################################
if activation == 'tanh':
    activ = tf.nn.tanh
elif activation == 'relu':
    activ = tf.nn.relu
# Closed formula Merton
######################################
opt_param = Option_param(x0, K, T, exercise="European", payoff="call" )
Merton_param = Merton_process(r, sig, lam, muJ, sigJ)
Merton = Merton_pricer(opt_param, Merton_param)
closedformula = Merton.closed_formula()
print('Merton real price:', closedformula)
# Train
#######################################
closedformula = Merton.closed_formula()
print(closedformula)
#sys.exit(0)
listLoss = []
listProcesses = []
# math model
##########################
mathModel = MertonJumpModel(T, N, r, muJ, sigJ, sig, lam, K, x0, maxJumps, func)
saveFig = "GLOBAL_BSize_"+str(batchSize)+"_lam_"+str(lam)+ "_N_"+str(N) +"_"
# DL model
##########################
if activation == 'tanh':
    activ = tf.nn.tanh
elif activation == 'relu':
    activ = tf.nn.relu
elif activation == 'sigmoid':
    activ = tf.math.sigmoid

kerasModelZ = Net(1, layerSize, activation)
kerasModelGam = Net(0, layerSize, activation)

# solver
#########################
solver = SolverGlobalFBSDE(mathModel,kerasModelZ,kerasModelGam, lRateY0)


# train and  get solution
Y0List=  solver.train(batchSize,batchSize*10, num_epoch,num_epochExt )
print('Y0',Y0List[-1])
# Store loss
listLoss.append(solver.lossList)  
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(Y0List, label = "Y0 DL ")
ax.plot(closedformula*np.ones(num_epochExt), label = 'Y0 closed formula', linestyle = 'dashed')
ax.grid()
plt.legend()
print("saveFig" , saveFig)
plt.savefig(saveFig+".png")
