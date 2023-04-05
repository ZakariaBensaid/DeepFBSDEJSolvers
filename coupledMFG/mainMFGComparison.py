import numpy as np
import tensorflow as tf
import os
from Networks import Net_hat, Net, kerasModels
from MFGModel import ModelCoupledFBSDE
from MFGSolvers import SolverGlobalFBSDE, SolverMultiStepFBSDE,SolverSumLocalFBSDE, SolverGlobalMultiStepReg, SolverGlobalSumLocalReg, SolverOsterleeFBSDE
import argparse
import matplotlib.pyplot as plt
import sys 


os.environ['KMP_DUPLICATE_LIB_OK']='True'
parser = argparse.ArgumentParser()
parser.add_argument('--nbNeuron_hat', type=int, default=20)
parser.add_argument('--nbNeuron', type=int, default=22)
parser.add_argument('--nbLayer_hat', type=int, default=2)
parser.add_argument('--nbLayer', type=int, default=2)
parser.add_argument('--nEpochExt', type=int, default=100)
parser.add_argument('--nEpoch', type=int, default=200)
parser.add_argument('--batchSize',  type=int, default=128)
parser.add_argument('--rafCoef',  type=int, default=1)
parser.add_argument('--jumpFac',type=float, default =2.16)
parser.add_argument('--nbDays',type=int, default =2)
parser.add_argument('--lRateY0',type=float, default =0.001)
parser.add_argument('--lRateLoc',type=float, default =0.00015)
parser.add_argument('--lRateReg',type=float, default =0.0001)
parser.add_argument('--couplage', type= str, default = 'ON')
parser.add_argument('--jumpModel', type= str, default = 'stochastic')
parser.add_argument('--activation_hat',  type= str, default="tanh")
parser.add_argument('--activation',  type= str, default="tanh")
parser.add_argument('--coefOsterlee', type= float, default = 100)
parser.add_argument('--nbSimulation', type= int, default = 10**5)
    
args = parser.parse_args()
print("Args ", args)
nbNeuron_hat = args.nbNeuron_hat
print("Nb neurons projected case", nbNeuron_hat)
nbNeuron = args.nbNeuron
print("Nb neurons ", nbNeuron)
nbLayer_hat = args.nbLayer_hat
print("Nb hidden layers projected case ", nbLayer_hat)
nbLayer= args.nbLayer
print("Nb hidden layers ", nbLayer)
batchSize = args.batchSize
print("batchSize " , batchSize)
num_epochExt = args.nEpochExt
print("num_epochExt " , num_epochExt)
num_epoch =args.nEpoch
print("Number epoch ", num_epoch)
rafCoef = args.rafCoef
print("refining coeff ",rafCoef)
jumpFactor=args.jumpFac
print("Jump factor ", jumpFactor)
nbDays= args.nbDays
print("Number of days", nbDays)
couplage = args.couplage
print('couplage Y_hat/Y', couplage)
jumpModel = args.jumpModel
print('jumpModel', jumpModel)
lRateY0 =args.lRateY0
print("Learning rate ", lRateY0)
lRateLoc =args.lRateLoc
print("Learning rate ", lRateLoc)
lRateReg =args.lRateReg
print("Learning rate ", lRateReg)
activation_hat = args.activation_hat
if activation_hat not in ['tanh', 'relu']:
    print(activation_hat, 'is invalid. Please choose tanh or relu.')
    sys.exit(0)
print('activation projected case', activation_hat)
activation = args.activation
if activation not in ['tanh', 'relu']:
    print(activation, 'is invalid. Please choose tanh or relu.')
    sys.exit(0)
print('activation', activation)
coefOsterlee = args.coefOsterlee
print('Osterlee coefficient', coefOsterlee)
nbSimul = args.nbSimulation
print('number of trajectories', nbSimul)
# Layers
###################################
layerSize_hat = nbNeuron_hat*np.ones((nbLayer_hat,), dtype=np.int32) 
layerSize = nbNeuron*np.ones((nbLayer_hat,), dtype=np.int32) 
# Parameters
###################################
QAverOneDay = np.array([0.26759617, 0.24771933, 0.23588383, 0.221369, 0.21174, 0.2047625, 0.20651067, 0.20098083, 0.20826067, 0.22095067,
  0.24346833, 0.27283267, 0.3382265, 0.42920433, 0.4875495, 0.50948433, 0.487712, 0.4537295, 0.40911717, 0.3728925,
  0.347346, 0.3419715, 0.32684, 0.320009, 0.32065767, 0.32586567, 0.31492483, 0.31607417, 0.30411783, 0.29950567,
 0.307519, 0.33259367, 0.375465, 0.45608333, 0.599178,0.70970583, 0.7364855, 0.736731, 0.70612667, 0.67284583,
  0.66692767, 0.64925583, 0.604485, 0.55684567, 0.515597, 0.45097333, 0.3822625, 0.31841833])
QAver = np.concatenate([QAverOneDay]*nbDays, axis=-1)
# tile
QAver = np.tile(np.expand_dims(QAver, axis=-1),[1,rafCoef]).flatten()
T  = float(nbDays)
dict_parameters = {'sigma': 0.56, 'sigma_0':0.31, 'theta':0.12, 'h0' : 0, 'h1' :0, 'h2':100, 'A' : 150, 'C' : 80, 'K' : 50, 'pi':0.5, 'p0' :6.159423723, 'p1': 87.4286117, 'f0':0, 'f1': 10**4, 
                    'R_0' : 2*0.12 , 's0':-0.5, 'alphaTarget':-0.2}
sig, sig0, theta, h0, h1, h2, A, C, K, pi, p0, p1, f0, f1, R0, S0, alphaTarget = dict_parameters.values()

# mathematical model for hY
###########################
mathModel = ModelCoupledFBSDE( T , QAver,   R0,  jumpFactor, A, K, pi, p0, p1, f0, f1, theta,C, S0, h1, h2,sig0, sig, alphaTarget, jumpModel, 1)

# DL model
###########################
if activation_hat == 'tanh':
    activ_hat = tf.nn.tanh
elif activation_hat == 'relu':
    activ_hat = tf.nn.relu

if activation == 'tanh':
    activ = tf.nn.tanh
elif activation == 'relu':
    activ = tf.nn.relu

# Storing all methods
############################ 
listKeras = []
listhY0List = []
listY0List = []
for method in ['Global', 'SumMultiStep', 'SumLocal', 'SumLocalReg', 'SumMultiStepReg', 'Osterlee']:    
    #method
    if method in ['SumMultiStepReg', 'SumLocalReg']:
        kerasModel = kerasModels(Net_hat, Net, method, 1, 1,layerSize_hat, layerSize, activation_hat, activation)
    elif method in ['SumMultiStep', 'SumLocal', 'Osterlee']:
        kerasModel = kerasModels(Net_hat, Net, method, 3, 4,layerSize_hat, layerSize, activation_hat, activation)
    else:
        kerasModel = kerasModels(Net_hat, Net, method, 2, 3,layerSize_hat, layerSize, activation_hat, activation)
    #solver
    if method == "Global":
        solver = SolverGlobalFBSDE(mathModel,kerasModel, lRateY0, couplage)
    elif method == "SumMultiStep":
        solver= SolverMultiStepFBSDE(mathModel,kerasModel, lRateLoc, couplage)
    elif method == "SumLocal":
        solver=  SolverSumLocalFBSDE(mathModel,kerasModel, lRateLoc, couplage)
    elif method == 'SumMultiStepReg':
        solver = SolverGlobalMultiStepReg(mathModel,kerasModel, lRateReg, couplage)
    elif method == 'SumLocalReg':
        solver =  SolverGlobalSumLocalReg(mathModel,kerasModel, lRateReg, couplage)
    elif method == 'Osterlee':
        solver = SolverOsterleeFBSDE(mathModel,kerasModel, lRateLoc, couplage, coefOsterlee) 
    # train and  get solution
    ############################
    hY0List, Y0List=  solver.train(batchSize,batchSize*10, num_epoch,num_epochExt )
    listhY0List.append(hY0List)
    listY0List.append(Y0List)
    # Storing the weights
    ###########################
    listKeras.append(kerasModel)
#Plots
#############################
fig, ax = plt.subplots()
for k, method in enumerate(['Global', 'SumMultiStep', 'SumLocal', 'SumLocalReg', 'SumMultiStepReg', 'Osterlee']):
    ax.plot(listhY0List[k], label = f'{method}')
plt.xlabel('epochs')
plt.ylabel(r'$\hat{Y}_{0}$')
plt.legend(prop={'size': 5})
plt.show()
############################
fig, ax = plt.subplots()
for k, method in enumerate(['Global', 'SumMultiStep', 'SumLocal', 'SumLocalReg', 'SumMultiStepReg', 'Osterlee']):
  ax.plot(listY0List[k], label = f'{method}')
plt.xlabel('epochs')
plt.ylabel(r'$Y_{0}$')
plt.legend(prop={'size': 5})
plt.show()