import numpy as np
import pandas as pd
import tensorflow as tf
import os
from Networks import Net_hat, Net, kerasModels
from Models import ModelCoupledFBSDE
from MFGSolvers import SolverGlobalFBSDE, SolverMultiStepFBSDE,SolverSumLocalFBSDE, SolverGlobalMultiStepReg, SolverGlobalSumLocalReg, SolverOsterleeFBSDE
from PlotsMFG import MFGSolutionsFixedTrajectory
import argparse
import time
import matplotlib.style as style 
import matplotlib.pyplot as plt
import sys 



os.environ['KMP_DUPLICATE_LIB_OK']='True'
parser = argparse.ArgumentParser()
parser.add_argument('--nbNeuron_hat', type=int, default=20)
parser.add_argument('--nbNeuron', type=int, default=25)
parser.add_argument('--nbLayer_hat', type=int, default=2)
parser.add_argument('--nbLayer', type=int, default=2)
parser.add_argument('--nEpochExt', type=int, default=100)
parser.add_argument('--nEpoch', type=int, default=200)
parser.add_argument('--batchSize',  type=int, default=128)
parser.add_argument('--rafCoef',  type=int, default=1)
parser.add_argument('--jumpFac',type=float, default =2.16)
parser.add_argument('--nbDays',type=int, default =1)
parser.add_argument('--lRateY0',type=float, default =0.001)
parser.add_argument('--lRateLoc',type=float, default =0.00015)
parser.add_argument('--lRateReg',type=float, default =0.0001)
parser.add_argument('--couplage', type= str, default = 'ON')
parser.add_argument('--jumpModel', type= str, default = 'stochastic')
parser.add_argument('--activation_hat',  type= str, default="tanh")
parser.add_argument('--activation',  type= str, default="tanh")
parser.add_argument('--method', type=str, default="Global")
parser.add_argument('--coefOsterlee', type= float, default = 1)
    
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
method=args.method
if method not in ['Global', "SumMultiStep", "SumLocal", 'SumMultiStepReg', 'SumLocalReg','Osterlee']:
    print(method, 'is invalid. Please choose Global or SumMultiStep or SumLocal or SumMultiStepReg or SumLocalReg or Osterlee')
    sys.exit(0)
print("METHOD", method)
coefOsterlee = args.coefOsterlee
if method == 'Osterlee':
    print('Osterlee coefficient', coefOsterlee)

#Number of simulations
nbSimul = 10**5
# Layers
layerSize_hat = nbNeuron_hat*np.ones((nbLayer_hat,), dtype=np.int32) 
layerSize = nbNeuron*np.ones((nbLayer_hat,), dtype=np.int32) 
# parameter models
QAverOneDay = np.array([0.26759617, 0.24771933, 0.23588383, 0.221369, 0.21174, 0.2047625, 0.20651067, 0.20098083, 0.20826067, 0.22095067,
  0.24346833, 0.27283267, 0.3382265, 0.42920433, 0.4875495, 0.50948433, 0.487712, 0.4537295, 0.40911717, 0.3728925,
  0.347346, 0.3419715, 0.32684, 0.320009, 0.32065767, 0.32586567, 0.31492483, 0.31607417, 0.30411783, 0.29950567,
 0.307519, 0.33259367, 0.375465, 0.45608333, 0.599178,0.70970583, 0.7364855, 0.736731, 0.70612667, 0.67284583,
  0.66692767, 0.64925583, 0.604485, 0.55684567, 0.515597, 0.45097333, 0.3822625, 0.31841833])


QAver = np.concatenate([QAverOneDay]*nbDays, axis=-1)

# tile
QAver = np.tile(np.expand_dims(QAver, axis=-1),[1,rafCoef]).flatten()

#Parameters
T  = float(nbDays)

dict_parameters = {'sigma': 0.56, 'sigma_0':0.31, 'theta':0.12, 'h0' : 0, 'h1' :0, 'h2':100, 'A' : 150, 'C' : 80, 'K' : 50, 'pi':0.5, 'p0' :6.159423723, 'p1': 87.4286117, 'f0':0, 'f1': 10**4, 
                    'R_0' : 2*0.12 , 's0':-0.5, 'alphaTarget':-0.2}
sig, sig0, theta, h0, h1, h2, A, C, K, pi, p0, p1, f0, f1, R0, S0, alphaTarget = dict_parameters.values()

# mathematical model for hY
###########################
mathModel0 = ModelCoupledFBSDE(T, QAver, R0, jumpFactor, A, K, pi, p0, p1, f0, f1, theta,C, S0, h1, h2,sig0, sig, alphaTarget, jumpModel, 1)

#Fixing the trajectory 
###########################
dW0_arr = np.sqrt(mathModel0.dt)*tf.random.normal([nbSimul, mathModel0.N+1])
dW_arrPlayer1 = np.sqrt(mathModel0.dt)*tf.random.normal([nbSimul, mathModel0.N+1])
dW_arrPlayer2 = np.sqrt(mathModel0.dt)*tf.random.normal([nbSimul, mathModel0.N+1])
Q = mathModel0.QAver*tf.exp(-0.5*mathModel0.sig0*mathModel0.sig0*np.arange(mathModel0.N+1)*mathModel0.dt + mathModel0.sig0*dW0_arr)
if jumpModel == 'stochastic':
    lam = jumpFactor*(Q)**2
else:
    lam = jumpFactor*tf.ones([nbSimul, mathModel0.N+1])
# number of jump in dt
dN = tf.random.poisson( [1], lam*mathModel0.dt)[0] 

# DL model
##########################
if activation_hat == 'tanh':
    activ_hat = tf.nn.tanh
elif activation_hat == 'relu':
    activ_hat = tf.nn.relu

if activation == 'tanh':
    activ = tf.nn.tanh
elif activation == 'relu':
    activ = tf.nn.relu


#style for plots
style.use('ggplot')
listPi = [0., 0.1, 0.5, 0.95]
#listPi = [0.5]
dictPoA = {}
for pi in listPi:
  #Mathematical models
  mathModelMFG = ModelCoupledFBSDE(T, QAver, R0,  jumpFactor, A, K, pi, p0, p1, f0, f1, theta,C, S0, h1, h2,sig0, sig, alphaTarget, jumpModel, 1)
  mathModelMFCagg = ModelCoupledFBSDE(T, QAver, R0,  jumpFactor, A, K, pi, p0, p1, f0, f1, theta,C, S0, h1, h2,sig0, sig, alphaTarget, jumpModel, 2)
  #Storing MFG and MFCagg
  listMathModel = [mathModelMFG, mathModelMFCagg]
  listSolutionsPlayer1 = []
  listSolutionsPlayer2 = []
  for k in range(len(listMathModel)): 
      #define the mathematical model
      mathModel = listMathModel[k]
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
      #train and  get solution
      hY0List, Y0List=  solver.train(batchSize,batchSize*10, num_epoch,num_epochExt)
      #Plotting
      solutionPlayer1 = MFGSolutionsFixedTrajectory(mathModel, kerasModel, method, dW0_arr, dW_arrPlayer1, dN, 'OFF')
      solutionPlayer2 = MFGSolutionsFixedTrajectory(mathModel, kerasModel, method, dW0_arr, dW_arrPlayer2, dN, 'OFF')
      #store solutions
      listSolutionsPlayer1.append(solutionPlayer1)
      listSolutionsPlayer2.append(solutionPlayer2)
  #Simulate processes
  #################################################
  #Player 1
  #MFG
  listSolutionsPlayer1[0].simulateAllProcesses(10)
  listSolutionsPlayer1[0].computeTarget(10)
  #MFCagg
  listSolutionsPlayer1[1].simulateAllProcesses(10)
  listSolutionsPlayer1[1].computeTarget(10)
  #Player2
  #MFG
  listSolutionsPlayer2[0].simulateAllProcesses(10)
  listSolutionsPlayer2[0].computeTarget(10)
  #MFCagg
  listSolutionsPlayer2[1].simulateAllProcesses(10)
  listSolutionsPlayer2[1].computeTarget(10)   
  #Plots
  ###############################################
  for j in range(10):
    #Consumption
    #############################################
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(listSolutionsPlayer1[0].t*listSolutionsPlayer1[0].dt*24, listSolutionsPlayer1[0].hQ[j], label = r'$\hat{Q}$', linewidth=2.2, color = 'dimgray')
    ax.plot(listSolutionsPlayer1[0].t*listSolutionsPlayer1[0].dt*24, listSolutionsPlayer1[0].Q[j], label = r'$Q^{1}$ player 1', color = 'lightsteelblue')
    ax.plot(listSolutionsPlayer2[0].t*listSolutionsPlayer2[0].dt*24, listSolutionsPlayer2[0].Q[j], label = r'$Q^{2}$ player 2', color = 'burlywood')
    plt.xlabel('time (hour)')
    plt.ylabel('consumption (kW)')
    plt.legend()
    plt.show()
    #intensity
    ############################################
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(listSolutionsPlayer1[0].t*listSolutionsPlayer1[0].dt*24, listSolutionsPlayer1[0].hQ[j], label = r'$\hat{Q}$', linewidth=2.2, color = 'dimgray')
    ax.set_xlabel('time (hour)')
    ax.set_ylabel('consumption (kW)')
    ax.legend(loc = 2)
    ax2 = ax.twinx()
    ax2.plot(listSolutionsPlayer1[0].t*listSolutionsPlayer1[0].dt*24, listSolutionsPlayer1[0].lam[j], label = r'$\lambda$: jump intensity', linestyle = 'dashed', color = 'tab:brown')
    ax2.set_ylabel('jump intensity')
    ax2.legend(loc = 1)
    plt.show()
    #Jumps
    #############################################
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(listSolutionsPlayer1[0].t*listSolutionsPlayer1[0].dt*24, listSolutionsPlayer1[0].R[j] < listSolutionsPlayer1[0].theta, label = 'jumps')
    plt.xlabel('time (hour)')
    plt.ylabel('jumps (kW)')
    plt.legend()
    plt.show()
    #Seasonalized consumption after equilibrium
    ###########################################
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(listSolutionsPlayer1[0].t*listSolutionsPlayer1[0].dt*24, 0, 1, where= listSolutionsPlayer1[0].R[j] < listSolutionsPlayer1[0].theta,
                color='green', alpha=0.2, transform=ax.get_xaxis_transform())
    ax.plot(listSolutionsPlayer1[0].t*listSolutionsPlayer1[0].dt*24, listSolutionsPlayer1[0].alphaTg[j] , color='tab:brown', label= r'$\alpha_{tg}$', linestyle = 'dashed')
    #MFG
    ax.plot(listSolutionsPlayer1[0].t*listSolutionsPlayer1[0].dt*24, listSolutionsPlayer1[0].hQ[j] + listSolutionsPlayer1[0].alpha_hat[j] \
            - listSolutionsPlayer1[0].QAver , label = 'MFG proj.', linewidth=2.2, color = 'dimgray')
    ax.plot(listSolutionsPlayer1[0].t*listSolutionsPlayer1[0].dt*24, listSolutionsPlayer1[0].Q[j] + listSolutionsPlayer1[0].alpha[j]\
            - listSolutionsPlayer1[0].QAver, label = 'MFG player1', color = 'lightsteelblue')
    ax.plot(listSolutionsPlayer2[0].t*listSolutionsPlayer2[0].dt*24, listSolutionsPlayer2[0].Q[j] + listSolutionsPlayer2[0].alpha[j]\
            - listSolutionsPlayer1[0].QAver, label = 'MFG player2', color = 'burlywood')
    #MFCagg
    ax.plot(listSolutionsPlayer1[1].t*listSolutionsPlayer1[1].dt*24, listSolutionsPlayer1[1].hQ[j] + listSolutionsPlayer1[1].alpha_hat[j]\
            - listSolutionsPlayer1[0].QAver, label = 'MFCagg proj.', linestyle='dotted', linewidth=2.2, color = 'dimgray')
    ax.plot(listSolutionsPlayer1[1].t*listSolutionsPlayer1[1].dt*24, listSolutionsPlayer1[1].Q[j] + listSolutionsPlayer1[1].alpha[j]\
            - listSolutionsPlayer1[0].QAver, label = 'MFCagg player1', linestyle='dotted', color = 'lightsteelblue')
    ax.plot(listSolutionsPlayer2[1].t*listSolutionsPlayer2[1].dt*24, listSolutionsPlayer2[1].Q[j] + listSolutionsPlayer2[1].alpha[j]\
            - listSolutionsPlayer1[0].QAver, label = 'MFCagg player2', linestyle='dotted', color = 'burlywood')
    ax.title.set_text(f'pi = {pi}')
    plt.xlabel('time (hour)')
    plt.ylabel('seasonalized consumption after eq. (kW)')
    plt.legend(prop={'size': 8})
    plt.show()
    #Consumption after equilibrium
    ############################################
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(listSolutionsPlayer1[0].t*listSolutionsPlayer1[0].dt*24, 0, 1, where= listSolutionsPlayer1[0].R[j] < listSolutionsPlayer1[0].theta,
                color='green', alpha=0.2, transform=ax.get_xaxis_transform())
    #MFG
    ax.plot(listSolutionsPlayer1[0].t*listSolutionsPlayer1[0].dt*24, listSolutionsPlayer1[0].hQ[j] + listSolutionsPlayer1[0].alpha_hat[j], label = f'MFG proj.', linewidth=2.2, color = 'dimgray')
    ax.plot(listSolutionsPlayer1[0].t*listSolutionsPlayer1[0].dt*24, listSolutionsPlayer1[0].Q[j] + listSolutionsPlayer1[0].alpha[j], label = f'MFG  player1', color = 'lightsteelblue')
    ax.plot(listSolutionsPlayer2[0].t*listSolutionsPlayer2[0].dt*24, listSolutionsPlayer2[0].Q[j] + listSolutionsPlayer2[0].alpha[j], label = f'MFG  player2', color = 'burlywood')
    #MFCagg
    ax.plot(listSolutionsPlayer1[1].t*listSolutionsPlayer1[1].dt*24, listSolutionsPlayer1[1].hQ[j] + listSolutionsPlayer1[1].alpha_hat[j], label = f'MFCagg proj. ', linestyle='dotted', linewidth=2.2, color = 'dimgray')
    ax.plot(listSolutionsPlayer1[1].t*listSolutionsPlayer1[1].dt*24, listSolutionsPlayer1[1].Q[j] + listSolutionsPlayer1[1].alpha[j], label = f'MFCagg player1', linestyle='dotted', color = 'lightsteelblue')
    ax.plot(listSolutionsPlayer2[1].t*listSolutionsPlayer2[1].dt*24, listSolutionsPlayer2[1].Q[j] + listSolutionsPlayer2[1].alpha[j], label = f'MFCagg player2', linestyle='dotted', color = 'burlywood')
    ax.title.set_text(f'pi = {pi}')
    plt.xlabel('time (hour)')
    plt.ylabel('consumption after eq. (kW)')
    plt.legend(prop={'size': 8})
    plt.show()
    #Cumulated deviation
    ##########################################
    fig, ax = plt.subplots(figsize=(14, 6))
    #MFG
    ax.plot(listSolutionsPlayer1[0].t*listSolutionsPlayer1[0].dt*24, listSolutionsPlayer1[0].hS[j], label = f'MFG hS', linewidth=2.2, color = 'dimgray')
    ax.plot(listSolutionsPlayer1[0].t*listSolutionsPlayer1[0].dt*24, listSolutionsPlayer1[0].S[j], label = f'MFG S player1', color = 'lightsteelblue')
    ax.plot(listSolutionsPlayer1[0].t*listSolutionsPlayer1[0].dt*24, listSolutionsPlayer2[0].S[j], label = f'MFG S player2', color = 'burlywood')
    #MFCagg
    ax.plot(listSolutionsPlayer1[1].t*listSolutionsPlayer1[1].dt*24, listSolutionsPlayer1[1].hS[j], label = f'MFCagg hS', linestyle='dotted', linewidth=2.2, color = 'dimgray')
    ax.plot(listSolutionsPlayer1[1].t*listSolutionsPlayer1[1].dt*24, listSolutionsPlayer1[1].S[j], label = f'MFCagg S player1', linestyle='dotted', color = 'lightsteelblue')
    ax.plot(listSolutionsPlayer1[1].t*listSolutionsPlayer1[1].dt*24, listSolutionsPlayer2[1].S[j], label = f'MFCagg S player2', linestyle='dotted', color = 'burlywood')
    ax.title.set_text(f'pi = {pi}')
    plt.xlabel('time (hour)')
    plt.ylabel(r'$S$')
    plt.legend(prop={'size': 8})
    plt.show()
    #Price
    #########################################
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(listSolutionsPlayer1[0].t*listSolutionsPlayer1[0].dt*24, 0, 1, where= listSolutionsPlayer1[0].R[j] < listSolutionsPlayer1[0].theta,
                color='green', alpha=0.2, transform=ax.get_xaxis_transform())
    #MFG
    ax.plot(listSolutionsPlayer1[0].t*listSolutionsPlayer1[0].dt*24, listSolutionsPlayer1[0].price(pi, 0)[j], label = f'MFG price without alpha')
    ax.plot(listSolutionsPlayer1[0].t*listSolutionsPlayer1[0].dt*24, listSolutionsPlayer1[0].price(pi, listSolutionsPlayer1[0].alpha_hat)[j],\
            label = f'MFG price')
    #MFCagg
    ax.plot(listSolutionsPlayer1[1].t*listSolutionsPlayer1[1].dt*24, listSolutionsPlayer1[1].price(pi, 0)[j], label = f'MFCagg price without alpha', linestyle='dotted')
    ax.plot(listSolutionsPlayer1[1].t*listSolutionsPlayer1[1].dt*24, listSolutionsPlayer1[1].price(pi, listSolutionsPlayer1[1].alpha_hat)[j],\
            label = f'MFCagg price', linestyle='dotted')
    ax.title.set_text(f'pi = {pi}')
    plt.xlabel('time (hour)')
    plt.ylabel('price (euros)')
    plt.legend(prop={'size': 8})
    plt.show()

  #Compute Price of Anarchy
  #MFG 0 in list
  listSolutionsPlayer1[0].simulateAllProcesses(nbSimul)
  listSolutionsPlayer1[0].computeTarget(nbSimul)
  MFGPlayer, stdMFGPlayer = listSolutionsPlayer1[0].objectiveFunction()
  #MFCagg 1 in list
  listSolutionsPlayer1[1].simulateAllProcesses(nbSimul)
  listSolutionsPlayer1[1].computeTarget(nbSimul)
  MFCaggPlayer, stdMFCaggPlayer = listSolutionsPlayer1[1].objectiveFunction()  
  #####################################
  PoA = MFGPlayer / MFCaggPlayer 
  dictPoA[pi] = ['{0:.3f}'.format(MFGPlayer)  + '(+/- {0:.3f}'.format(1.96*stdMFGPlayer/np.sqrt(nbSimul)) + ')',\
                 '{0:.3f}'.format(MFCaggPlayer) + '(+/- {0:.3f}'.format(1.96*stdMFCaggPlayer/np.sqrt(nbSimul)) + ')', PoA]
  #####################################
dfPoA = pd.DataFrame(dictPoA, index = ['MFG players', 'MFCagg players', 'PoA'])
print('The price of Anarchy for this model is: \n', dfPoA)