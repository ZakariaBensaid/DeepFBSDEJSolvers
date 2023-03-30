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
from matplotlib.backends.backend_pdf import PdfPages
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
QAver = QAver.astype('float32')

#Parameters
T  = float(nbDays)
dict_parameters = {'sigma': 0.3, 'sigma_0':0.1, 'theta':0.12, 'h0' : 0, 'h1' :0, 'h2': 600, 'A' : 150, 'C' : 80, 'K' : 50, 'R_0' : 2*0.12 , 's0':0, 'alphaTarget':-0.2, 'coeffOU': 7.}
sig, sig0, theta, h0, h1, h2, A, C, K, R0, S0, alphaTarget, coeffOU = dict_parameters.values()

# mathematical model for hY
###########################
mathModel0 = ModelCoupledFBSDE(T, QAver, R0, jumpFactor, coeffOU, A, K, 0.5, 0, 0, 0, 0, theta,C, S0, h1, h2,sig0, sig, alphaTarget, jumpModel, 1)

#Fixing the trajectory 
###########################
dW0_arr = np.sqrt(mathModel0.dt)*tf.random.normal([nbSimul, mathModel0.N+1])
dW_arrPlayer1 = np.sqrt(mathModel0.dt)*tf.random.normal([nbSimul, mathModel0.N+1])
dW_arrPlayer2 = np.sqrt(mathModel0.dt)*tf.random.normal([nbSimul, mathModel0.N+1])
dN = np.zeros((nbSimul, mathModel0.N + 1))
mathModel0.init(nbSimul)
dN[:,0] = mathModel0.dN()[0]
for istep in range(1, mathModel0.N + 1):
  mathModel0.oneStepFrom(dW0_arr[:,istep], 0,0,0,0)
  dN[:,istep] = mathModel0.dN()[0]

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


#Pretraining
#############################################
#method
NbSimulation = 5
if method in ['SumMultiStepReg', 'SumLocalReg']:
    kerasModel = kerasModels(Net_hat, Net, method, 1, 1,layerSize_hat, layerSize, activation_hat, activation)
elif method in ['SumMultiStep', 'SumLocal', 'Osterlee']:
    kerasModel = kerasModels(Net_hat, Net, method, 3, 4,layerSize_hat, layerSize, activation_hat, activation)
else:
    kerasModel = kerasModels(Net_hat, Net, method, 2, 3,layerSize_hat, layerSize, activation_hat, activation)
solutionPlayer1 = MFGSolutionsFixedTrajectory(mathModel0, kerasModel, method, dW0_arr, dW_arrPlayer1, dN, 'OFF')
solutionPlayer2 = MFGSolutionsFixedTrajectory(mathModel0, kerasModel, method, dW0_arr, dW_arrPlayer2, dN, 'OFF')
solutionPlayer1.simulateAllProcesses(NbSimulation)
solutionPlayer1.computeTarget(NbSimulation)
solutionPlayer2.simulateAllProcesses(NbSimulation)
solutionPlayer2.computeTarget(NbSimulation)
listFiguresPreTrain = []
for j in range(NbSimulation):
  #Consumption
  #############################################
  fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize=(12, 8))
  ax[0,0].plot(solutionPlayer1.t*solutionPlayer1.dt*24, solutionPlayer1.hQ[j], label = r'$\hat{Q}$', linewidth=2.2, color = 'dimgray')
  ax[0,0].plot(solutionPlayer1.t*solutionPlayer1.dt*24, solutionPlayer1.Q[j], label = r'$Q^{1}$ player 1', color = 'blue')
  ax[0,0].plot(solutionPlayer2.t*solutionPlayer2.dt*24, solutionPlayer2.Q[j], label = r'$Q^{2}$ player 2', color = 'red')
  ax[0,0].set_title('consumption (kW)')
  ax[0,0].legend()
  #intensity
  ############################################
  ax[0,1].plot(solutionPlayer1.t*solutionPlayer1.dt*24, solutionPlayer1.hQ[j], label = r'$\hat{Q}$', linewidth=2.2, color = 'dimgray')
  ax[0,1].set_title('intensity')
  #ax[0,1].legend(loc = 2)
  ax[0,1].set(ylabel = r'$\hat{Q}$')
  ax2 = ax[0,1].twinx()
  ax2.plot(solutionPlayer1.t*solutionPlayer1.dt*24, solutionPlayer1.lam[j], label = r'$\lambda$', linestyle = 'dashed', color = 'tab:brown')
  ax2.legend(loc = 1)
  #Jumps
  #############################################
  ax[1,1].plot(solutionPlayer1.t*solutionPlayer1.dt*24, solutionPlayer1.R[j] < solutionPlayer1.theta, label = 'jumps')
  ax[1,1].set_title('jumps')
  ax[1,0].axis('off')
  for ax in ax.flat:
      ax.set(xlabel='time (hours)')
  plt.legend()
  plt.show()
  listFiguresPreTrain.append(fig)

#style for plots
dict_cases = {'with jumps and with dynamic pricing': [ 6.159423723, 87.4286117, 0, 10**4], 'with jumps and without pricing': [0, 0, 0, 10**4], 'without jumps and with pricing' : [6.159423723, 87.4286117, 0, 0]}
dictFiguresPostTrain = {}
listPoAfigures = []
for string, [p0, p1, f0, f1] in dict_cases.items():
  listPi = [0., 0.1, 0.5, 0.95]
  dictPoA = {}
  for pi in listPi:
    #Mathematical models
    mathModelMFG = ModelCoupledFBSDE(T, QAver, R0,  jumpFactor, coeffOU, A, K, pi, p0, p1, f0, f1, theta,C, S0, h1, h2,sig0, sig, alphaTarget, jumpModel, 1)
    mathModelMFCagg = ModelCoupledFBSDE(T, QAver, R0,  jumpFactor, coeffOU, A, K, pi, p0, p1, f0, f1, theta,C, S0, h1, h2,sig0, sig, alphaTarget, jumpModel, 2)
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
    listSolutionsPlayer1[0].simulateAllProcesses(NbSimulation)
    listSolutionsPlayer1[0].computeTarget(NbSimulation)
    #MFCagg
    listSolutionsPlayer1[1].simulateAllProcesses(NbSimulation)
    listSolutionsPlayer1[1].computeTarget(NbSimulation)
    #Player2
    #MFG
    listSolutionsPlayer2[0].simulateAllProcesses(NbSimulation)
    listSolutionsPlayer2[0].computeTarget(NbSimulation)
    #MFCagg
    listSolutionsPlayer2[1].simulateAllProcesses(NbSimulation)
    listSolutionsPlayer2[1].computeTarget(NbSimulation)   
    #Plots
    ###############################################
    for j in range(NbSimulation):
      #Seasonalized consumption after equilibrium
      ###########################################
      fig, ax = plt.subplots(nrows = 2, ncols = 2 ,figsize=(12, 8))
      ax[0,0].fill_between(listSolutionsPlayer1[0].t*listSolutionsPlayer1[0].dt*24, 0, 1, where= listSolutionsPlayer1[0].R[j] < listSolutionsPlayer1[0].theta,
                  color='green', alpha=0.2, transform=ax[0,0].get_xaxis_transform())
      ax[0,0].plot(listSolutionsPlayer1[0].t*listSolutionsPlayer1[0].dt*24, listSolutionsPlayer1[0].alphaTg[j] , color='tab:brown', label= r'$\alpha_{tg}$', linestyle = 'dashed')
      #MFG
      ax[0,0].plot(listSolutionsPlayer1[0].t*listSolutionsPlayer1[0].dt*24, listSolutionsPlayer1[0].hQ[j] + listSolutionsPlayer1[0].alpha_hat[j] \
              - listSolutionsPlayer1[0].meanhQ , label = 'MFG proj.', linewidth=2.2, color = 'dimgray')
      ax[0,0].plot(listSolutionsPlayer1[0].t*listSolutionsPlayer1[0].dt*24, listSolutionsPlayer1[0].Q[j] + listSolutionsPlayer1[0].alpha[j]\
              - listSolutionsPlayer1[0].meanhQ, label = 'MFG player1', color = 'blue')
      ax[0,0].plot(listSolutionsPlayer2[0].t*listSolutionsPlayer2[0].dt*24, listSolutionsPlayer2[0].Q[j] + listSolutionsPlayer2[0].alpha[j]\
              - listSolutionsPlayer1[0].meanhQ, label = 'MFG player2', color = 'red')
      #MFCagg
      ax[0,0].plot(listSolutionsPlayer1[1].t*listSolutionsPlayer1[1].dt*24, listSolutionsPlayer1[1].hQ[j] + listSolutionsPlayer1[1].alpha_hat[j]\
              - listSolutionsPlayer1[0].meanhQ, label = 'MFCagg proj.', linestyle='dotted', linewidth=2.2, color = 'dimgray')
      ax[0,0].plot(listSolutionsPlayer1[1].t*listSolutionsPlayer1[1].dt*24, listSolutionsPlayer1[1].Q[j] + listSolutionsPlayer1[1].alpha[j]\
              - listSolutionsPlayer1[0].meanhQ, label = 'MFCagg player1', linestyle='dotted', color = 'blue')
      ax[0,0].plot(listSolutionsPlayer2[1].t*listSolutionsPlayer2[1].dt*24, listSolutionsPlayer2[1].Q[j] + listSolutionsPlayer2[1].alpha[j]\
              - listSolutionsPlayer1[0].meanhQ, label = 'MFCagg player2', linestyle='dotted', color = 'red')
      ax[0,0].title.set_text(r'$\tilde{Q} + \alpha$ (kW) / ' + string  + f' / pi = {pi}')
      ax[0,0].set_ylim(-0.5,0.5)
      ax[0,0].legend(prop={'size': 8})
      #Consumption after equilibrium
      ############################################
      ax[1,0].fill_between(listSolutionsPlayer1[0].t*listSolutionsPlayer1[0].dt*24, 0, 1, where= listSolutionsPlayer1[0].R[j] < listSolutionsPlayer1[0].theta,
                  color='green', alpha=0.2, transform=ax[1,0].get_xaxis_transform())
      #MFG
      ax[1,0].plot(listSolutionsPlayer1[0].t*listSolutionsPlayer1[0].dt*24, listSolutionsPlayer1[0].hQ[j] + listSolutionsPlayer1[0].alpha_hat[j], label = f'MFG proj.', linewidth=2.2, color = 'dimgray')
      ax[1,0].plot(listSolutionsPlayer1[0].t*listSolutionsPlayer1[0].dt*24, listSolutionsPlayer1[0].Q[j] + listSolutionsPlayer1[0].alpha[j], label = f'MFG  player1', color = 'blue')
      ax[1,0].plot(listSolutionsPlayer2[0].t*listSolutionsPlayer2[0].dt*24, listSolutionsPlayer2[0].Q[j] + listSolutionsPlayer2[0].alpha[j], label = f'MFG  player2', color = 'red')
      #MFCagg
      ax[1,0].plot(listSolutionsPlayer1[1].t*listSolutionsPlayer1[1].dt*24, listSolutionsPlayer1[1].hQ[j] + listSolutionsPlayer1[1].alpha_hat[j], label = f'MFCagg proj. ', linestyle='dotted', linewidth=2.2, color = 'dimgray')
      ax[1,0].plot(listSolutionsPlayer1[1].t*listSolutionsPlayer1[1].dt*24, listSolutionsPlayer1[1].Q[j] + listSolutionsPlayer1[1].alpha[j], label = f'MFCagg player1', linestyle='dotted', color = 'blue')
      ax[1,0].plot(listSolutionsPlayer2[1].t*listSolutionsPlayer2[1].dt*24, listSolutionsPlayer2[1].Q[j] + listSolutionsPlayer2[1].alpha[j], label = f'MFCagg player2', linestyle='dotted', color = 'red')
      ax[1,0].title.set_text(r'$Q + \alpha$ (kW) / ' + string + f' / pi = {pi}')
      ax[1,0].set_ylim(0.1,0.65)
      ax[1,0].legend(prop={'size': 8})
      #Cumulated deviation
      ##########################################
      #MFG
      ax[0,1].plot(listSolutionsPlayer1[0].t*listSolutionsPlayer1[0].dt*24, listSolutionsPlayer1[0].hS[j], label = f'MFG hS', linewidth=2.2, color = 'dimgray')
      ax[0,1].plot(listSolutionsPlayer1[0].t*listSolutionsPlayer1[0].dt*24, listSolutionsPlayer1[0].S[j], label = f'MFG S player1', color = 'blue')
      ax[0,1].plot(listSolutionsPlayer1[0].t*listSolutionsPlayer1[0].dt*24, listSolutionsPlayer2[0].S[j], label = f'MFG S player2', color = 'red')
      #MFCagg
      ax[0,1].plot(listSolutionsPlayer1[1].t*listSolutionsPlayer1[1].dt*24, listSolutionsPlayer1[1].hS[j], label = f'MFCagg hS', linestyle='dotted', linewidth=2.2, color = 'dimgray')
      ax[0,1].plot(listSolutionsPlayer1[1].t*listSolutionsPlayer1[1].dt*24, listSolutionsPlayer1[1].S[j], label = f'MFCagg S player1', linestyle='dotted', color = 'blue')
      ax[0,1].plot(listSolutionsPlayer1[1].t*listSolutionsPlayer1[1].dt*24, listSolutionsPlayer2[1].S[j], label = f'MFCagg S player2', linestyle='dotted', color = 'red')
      ax[0,1].title.set_text(r'$S$ / ' + string  + f' / pi = {pi}')
      ax[0,1].legend(prop={'size': 8})
      #Price
      #########################################
      ax[1,1].fill_between(listSolutionsPlayer1[0].t*listSolutionsPlayer1[0].dt*24, 0, 1, where= listSolutionsPlayer1[0].R[j] < listSolutionsPlayer1[0].theta,
                  color='green', alpha=0.2, transform=ax[1,1].get_xaxis_transform())
      #MFG
      ax[1,1].plot(listSolutionsPlayer1[0].t*listSolutionsPlayer1[0].dt*24, listSolutionsPlayer1[0].price(pi, 0)[j], label = f'MFG price without alpha')
      ax[1,1].plot(listSolutionsPlayer1[0].t*listSolutionsPlayer1[0].dt*24, listSolutionsPlayer1[0].price(pi, listSolutionsPlayer1[0].alpha_hat)[j],\
              label = f'MFG price')
      #MFCagg
      ax[1,1].plot(listSolutionsPlayer1[1].t*listSolutionsPlayer1[1].dt*24, listSolutionsPlayer1[1].price(pi, 0)[j], label = f'MFCagg price without alpha', linestyle='dotted')
      ax[1,1].plot(listSolutionsPlayer1[1].t*listSolutionsPlayer1[1].dt*24, listSolutionsPlayer1[1].price(pi, listSolutionsPlayer1[1].alpha_hat)[j],\
              label = f'MFCagg price', linestyle='dotted')
      ax[1,1].title.set_text('price (euros) / ' + string + f' / pi = {pi}')
      ax[1,1].legend(prop={'size': 8})
      for ax in ax.flat:
        ax.set(xlabel='time (hours)')
      plt.legend()
      plt.show()
      dictFiguresPostTrain[f'({j},{pi}, {string})'] = fig
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

  #Display dataframe
  fig4, ax4 = plt.subplots(figsize = (8, 2))
  ax4.table(cellText = dfPoA.values,
            rowLabels = dfPoA.index,
            colLabels =dfPoA.columns,
            loc = "center")
  ax4.set_title(string)
  ax4.axis('off')
  plt.show()
  listPoAfigures.append(fig4)
########################################################################################################################################
#save all figures:
# name your Pdf file
filename = "simulations_all_cases.pdf"  
pdf = PdfPages(filename)
for j in range(NbSimulation): 
  # and saving the files
  pdf.savefig(listFiguresPreTrain[j]) 
  for string in dict_cases.keys():
    for pi in listPi:
      # and saving the files
      pdf.savefig(dictFiguresPostTrain[f'({j},{pi}, {string})'])
for fig in listPoAfigures:
  # and saving the files
  pdf.savefig(fig)  
pdf.close()
##########################################################################################################################################
p = PdfPages('PoA.pdf')
for fig in listPoAfigures:
  # and saving the files
  p.savefig(fig) 
p.close()