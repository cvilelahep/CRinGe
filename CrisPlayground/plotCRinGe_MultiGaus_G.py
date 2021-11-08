import matplotlib
import tkinter
matplotlib.use('TkAgg')

import math
from math import atan2

import torch
import matplotlib.pyplot as plt
import sys

import numpy as np
import h5py
import scipy.stats as stats

N_GAUS = 3
StepSize = 0.01
maxIndex = 7787

if len(sys.argv) == 2 :
    N_GAUS = int(sys.argv[1])

print("PLOTTING NN from "+str(N_GAUS)+" GAUSSIANS in LOSS")

class CRinGeNet(torch.nn.Module) :
    def __init__(self) :
        super(CRinGeNet, self).__init__()

        self._mlp_pid = torch.nn.Sequential(
#            torch.nn.BatchNorm1d(3),
            torch.nn.Linear(3,512), torch.nn.ReLU(),
            torch.nn.Linear(512,512), torch.nn.ReLU()
        )

        self._mlp_pos = torch.nn.Sequential(
            torch.nn.BatchNorm1d(3),
            torch.nn.Linear(3,512), torch.nn.ReLU(),
            torch.nn.Linear(512,512), torch.nn.ReLU()
        )

        self._mlp_dir = torch.nn.Sequential(
            torch.nn.BatchNorm1d(3),
            torch.nn.Linear(3,512), torch.nn.ReLU(),
            torch.nn.Linear(512,512), torch.nn.ReLU()
        )

        self._mlp_E = torch.nn.Sequential(
            torch.nn.BatchNorm1d(1),
            torch.nn.Linear(1,512), torch.nn.ReLU(),
            torch.nn.Linear(512,512), torch.nn.ReLU()
        )

        self._mlp = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, 14784), torch.nn.ReLU()
        )


        self._upconvs = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 64, 4, 2),  torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3), torch.nn.ReLU(),

            torch.nn.ConvTranspose2d(64, 32, 4, 2), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3),  torch.nn.ReLU(),

            torch.nn.ConvTranspose2d(32, 32, 4, 2), torch.nn.ReLU(),
            #                    phit, gaussian, logvar, logmu, coefficient
            torch.nn.Conv2d(32, 1+N_GAUS*3, 3)
        )

        self._sigmoid = torch.nn.Sigmoid()

    def forward(self, x) :
        # Concatenate MLPs that treat PID, pos, dir and energy inputs separately
        net = torch.cat( (self._mlp_pid(x[:,0:3]),self._mlp_pos(x[:,3:6]),self._mlp_dir(x[:,6:9]),self._mlp_E(x[:,9].reshape(len(x[:,9]),1))), 1)

        # MegaMLP 
        net = self._mlp(net)
        
        # Reshape into 11 x 21 figure in 64 channels. Enough?!
        net = net.view(-1, 64, 11, 21)

        # Need to flatten? Maybe...
        net = self._upconvs(net).view(-1, 1+N_GAUS*3, 88*168)
        # return net
        # 0th channel is probability, pass through Sigmoid
        #hitprob = self._sigmoid(net[:,0]).view(-1, 1, 88*168)
        #hitprob = net[:,0].view(-1, 1, 88*168)

        # Last N_GAUS channels are coefficients, pass through softmax
        #coeffs = self._softmax(net[:,-N_GAUS:])

        #net = torch.cat( (hitprob, net[:,1:-N_GAUS], coeffs), dim=1)
        #net = torch.cat( (hitprob, net[:,1:]), dim=1)

        return net

def gaussian(nG, x, coeff, mu, var):
    gauss = [0]*(nG+1)
    #coefftot = 0
    tot = 0
    if nG != len(mu):
        print('Gaussian input dimensions do not agree!')
        return gauss
    for i in range(nG):
        tot += coeff[i]*(1./(2*np.pi*var[i])**0.5)*np.exp(-np.power(x - mu[i], 2.) / (2 * var[i]))
        gauss[i] = (coeff[i]*(1./(2*np.pi*var[i])**0.5)*np.exp(-np.power(x - mu[i], 2.) / (2 * var[i])))
    gauss[-1] = tot
    return  gauss

#def gauss(x, mu, var):
#    gauss = 1./(2*np.pi*var)**0.5*np.exp(-np.power(x - mu, 2.)) / (2 * var)
#    return gauss

def user_softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

net = CRinGeNet().cpu()
net.load_state_dict(torch.load("testCRinGe_MultiGaus_"+str(N_GAUS)+".cnn", map_location=lambda storage, loc: storage))

torch.set_grad_enabled(False)
net.eval()

nEvent = 1
#eventSel = [39680, 32640, 8197, 73502, 3958, 16250, 10122, 73510, 3813, 6041, 58898, 72889, 5, 2854, 3433, 7616, 8511, 10076]

#plt.figure()
#plt.imshow(plot_event[:,:,0])
#plt.colorbar()
#plt.show()
#plt.savefig("test_event.png")

muArr = [[] for i in range (N_GAUS)]
varArr = [[] for i in range (N_GAUS)]
muHitProbArr = [[] for i in range (N_GAUS)]
coeffArr = [[] for i in range (N_GAUS)]
hitProbArr = []

gammaArr = []
muonArr = []
eArr = []
energyArr = []
sinArr = []
cosArr = []

for data in [ [[1, 0, 0, 0., 0., 0., 0., 0., 1., 200]],
              [[0, 1, 0, 0., 0., 0., 0., 0., 1., 200]],
              [[0, 0, 1, 0., 0., 0., 0., 0., 1., 200]],
              
              [[1, 0, 0, 0., 0., 0., -0.6, 0., 0.8, 500]],
              [[0, 1, 0, 0., 0., 0., -0.6, 0., 0.8, 500]],
              [[0, 0, 1, 0., 0., 0., -0.6, 0., 0.8, 500]],
              
              [[1, 0, 0, 0., 0., 0., -0.9797958971132712, 0., 0.2, 700]],
              [[0, 1, 0, 0., 0., 0., -0.9797958971132712, 0., 0.2, 700]],
              [[0, 0, 1, 0., 0., 0., -0.9797958971132712, 0., 0.2, 700]] ] :

    gammaArr.append(data[0][0])
    eArr.append(data[0][1])
    muonArr.append(data[0][2])
    energyArr.append(data[0][-1])
    sinArr.append(data[0][8])
    cosArr.append(data[0][6])

    data = torch.as_tensor(data).cpu()
    eventHitProb = 1./(1+np.exp(net(data).cpu().detach().numpy()[0,0]))
    #eventHitProb = (1.-(net(data).cpu().detach().numpy()[0,0]))
    eventHitProb = eventHitProb.reshape((88,168))
    hitProbArr.append(eventHitProb.flatten())

    for iax in range(N_GAUS):            
        
        eventCoeff = net(data).cpu().detach().numpy()[0, -(N_GAUS - iax)]
        eventCoeff = eventCoeff.reshape((88, 168))
        coeffArr[iax].append(eventCoeff.flatten())

        eventMu = np.exp(net(data).cpu().detach().numpy()[0,2+iax*2])
        eventMu = eventMu.reshape((88,168))
        muArr[iax].append(eventMu.flatten())
    
        #    for i in range(N_GAUS):
        eventVariance = np.exp(net(data).cpu().detach().numpy()[0,1+iax*2])
        eventVariance = eventVariance.reshape((88,168))
        varArr[iax].append(eventVariance.flatten())
        
        eventMuHitProb = np.copy(eventMu[iax])*eventHitProb
        eventMuHitProb = eventMuHitProb.reshape((88,168))
        muHitProbArr[iax].append(eventMuHitProb.flatten())

    data = data.numpy()

        
figQDist = plt.figure(figsize=(24,12))
figTDist = plt.figure(figsize=(24,12))
#largeProb = hitProbArr < 0.05
#for mu, sig, coeff in (muArr[0], varArr[0], coeffArr[0]):

#print("Max HitProb", maxIndex)

for i in range(len(muArr[0])):
    thisAx = figQDist.add_subplot(3,3,i+1)
    x = np.arange(0, 10, StepSize)
    #vgauss = np.vectorize(asygauss)
    
    mutemp = []
    vartemp = []
    coefftemp = []
    
    for j in range(len(muArr)):
        #print(muArr[j][i][maxIndex])
        mutemp.append(muArr[j][i][maxIndex])
        vartemp.append(varArr[j][i][maxIndex])
        coefftemp.append(coeffArr[j][i][maxIndex])

    coefftemp = user_softmax(coefftemp)
    Gaussian = gaussian(N_GAUS, x, coefftemp, mutemp, vartemp)
    for k in range(N_GAUS+1):
        gmax = np.amax(Gaussian[k])
        gmaxid = np.where(Gaussian[k] == gmax)
        gupMask = (Gaussian[k] >= 0.5*gmax)
        gfwhm = np.array(gupMask, dtype=int)
        if k < N_GAUS:
            thisAx.plot(x, gaussian(N_GAUS, x, coefftemp, mutemp, vartemp)[k], linestyle = ':')
        else:
            thisAx.plot(x, gaussian(N_GAUS, x, coefftemp, mutemp, vartemp)[k], color = 'black', linewidth = 2)

        for text in thisAx.texts :
            txt.set_visible(False)
    thisAx.text(0.8, 0.9, "peak(FWHM) = %.2f(%.2f)" % (x[gmaxid], StepSize*np.sum(gfwhm)), horizontalalignment = 'center', verticalalignment = 'center', transform=thisAx.transAxes)
    thisAx.set_xlabel("Photoelectron")
        
    if gammaArr[i] == 1 :
        thisAx.set_title(r'$\gamma$ E='+str(energyArr[i])+' MeV $\phi$='+"{:.2f}".format(atan2(sinArr[i], cosArr[i])))
    elif eArr[i] == 1 :
        thisAx.set_title(r'$e$ E='+str(energyArr[i])+' MeV $\phi$='+"{:.2f}".format(atan2(sinArr[i], cosArr[i])))
    elif muonArr[i] == 1 :
        thisAx.set_title(r'$\mu$ E='+str(energyArr[i])+' MeV $\phi$='+"{:.2f}".format(atan2(sinArr[i], cosArr[i])))
    else :
        print("INVALID PID", data)
        
        
figQDist.tight_layout()
figQDist.savefig("PNGplots/CRinGe_MultiGaus_"+str(N_GAUS)+"_PMT_"+str(maxIndex)+"_GaussFit.png")

#plt.show()

