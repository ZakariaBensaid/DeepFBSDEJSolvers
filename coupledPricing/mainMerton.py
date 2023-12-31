import numpy as np
import tensorflow as tf
import os
from Networks import Net
from pricingModels import MertonJumpModel
from SolversJumpDiff import SolverGlobalFBSDE, SolverMultiStepFBSDE1, SolverMultiStepFBSDE2, SolverSumLocalFBSDE1, SolverSumLocalFBSDE2, SolverGlobalMultiStepReg, SolverGlobalSumLocalReg
import argparse
import matplotlib.pyplot as plt
import sys 

os.environ['KMP_DUPLICATE_LIB_OK']='True'
parser = argparse.ArgumentParser()
parser.add_argument('--nbNeuron', type=int, default=21)
parser.add_argument('--nbLayer', type=int, default=2)
parser.add_argument('--nEpochExt', type=int, default=120)
parser.add_argument('--nEpoch', type=int, default=100)
parser.add_argument('--batchSize',  type=int, default=10)
parser.add_argument('--lRateY0',type=float, default =0.0004)
parser.add_argument('--lRateLoc',type=float, default =0.0003)
parser.add_argument('--lRateReg',type=float, default =0.0003)
parser.add_argument('--activation',  type= str, default="tanh")
parser.add_argument('--aLin', type= float, default = 0.1)
parser.add_argument('--limit', type= int, default = 30)
    
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
aLin = args.aLin
print('Linear coupling forward backward', aLin)
limit = args.limit
print('Limit Power Series', limit)
# Layers
######################################
layerSize = nbNeuron*np.ones((nbLayer,), dtype=np.int32) 
# parameter models
######################################
dict_parameters = {'T':1 , 'N':50, 'r':0.1, 'sig': 0.3, 'lam':3, 'muJ': 0., 'sigJ': 0.2, 'K': 0.9, 'x0': 1}
T, N, r, sig, lam, muJ, sigJ, K, x0 = dict_parameters.values()
#Couplage function
def func(x):
    return aLin*tf.math.abs(x)
# DL model
######################################
if activation == 'tanh':
    activ = tf.nn.tanh
elif activation == 'relu':
    activ = tf.nn.relu
# Real price
######################################
mathModel0 = MertonJumpModel(T, N, r, muJ, sigJ, sig, lam, K, x0, func, limit)
X = mathModel0.init(1)
Realprice = mathModel0.A(0, X).numpy()[0]
print('Merton real price:',Realprice )
# Train
#######################################
listLoss = []
listProcesses = []
fig, ax = plt.subplots(figsize=(10, 6))
for method in ['Global', 'SumMultiStep1', 'SumMultiStep2', 'SumLocal1', 'SumLocal2', 'SumLocalReg', 'SumMultiStepReg']:
#for method in ['SumMultiStep1']:
    # math model
    ##########################
    mathModel = MertonJumpModel(T, N, r, muJ, sigJ, sig, lam, K, x0, func, limit)
    # DL model
    ##########################
    if activation == 'tanh':
        activ = tf.nn.tanh
    elif activation == 'relu':
        activ = tf.nn.relu
    elif activation == 'sigmoid':
        activ = tf.math.sigmoid
    # Networks
    ############################  
    bY0 = 0
    ndimOut = 2
    if method == 'Global':
        bY0 = 1
        ndimOut = 1
    elif method in ['SumLocalReg', 'SumMultiStepReg']:
        ndimOut = 1
    kerasModelUZ =  Net(bY0, ndimOut, layerSize, activation)
    kerasModelGam = Net(0, 1, layerSize, activation)   
    # solver
    #########################
    if method == "Global":
        solver = SolverGlobalFBSDE(mathModel, kerasModelUZ , kerasModelGam, lRateY0)
    elif method == "SumMultiStep1":
        solver= SolverMultiStepFBSDE1(mathModel, kerasModelUZ, lRateLoc)
    elif method == "SumMultiStep2":
        solver= SolverMultiStepFBSDE2(mathModel, kerasModelUZ , kerasModelGam, lRateLoc)
    elif method == "SumLocal1":
        solver=  SolverSumLocalFBSDE1(mathModel, kerasModelUZ, lRateLoc)
    elif method == "SumLocal2":
        solver=  SolverSumLocalFBSDE2(mathModel, kerasModelUZ , kerasModelGam, lRateLoc)
    elif method == 'SumMultiStepReg':
        solver = SolverGlobalMultiStepReg(mathModel, kerasModelUZ , kerasModelGam, lRateReg)
    elif method == 'SumLocalReg':
        solver =  SolverGlobalSumLocalReg(mathModel, kerasModelUZ , kerasModelGam, lRateReg)
    # train and  get solution
    Y0List=  solver.train(batchSize,batchSize*10, num_epoch,num_epochExt )
    print('Y0',Y0List[-1])
    # Store loss
    listLoss.append(solver.lossList)   
    ax.plot(Y0List, label = f"Y0 DL {method}")
ax.plot(Realprice*np.ones(num_epochExt), label = 'Y0 closed formula', linestyle = 'dashed')
ax.grid()
plt.legend()
plt.show()
