import matplotlib
matplotlib.use('Agg')

from math import atan2

import torch
import matplotlib.pyplot as plt
import sys

import numpy as np


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
            torch.nn.Conv2d(32, 3, 3)
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
        net = self._upconvs(net).view(-1, 3, 88*168)
        return net

net = CRinGeNet().cpu()
net.load_state_dict(torch.load("testCRinGe_Gaus_i_66266.0.cnn", map_location=lambda storage, loc: storage))

torch.set_grad_enabled(False)
net.eval()

figMu = plt.figure(figsize=(24,12))
figVariance = plt.figure(figsize=(24,12))
figHitProb = plt.figure(figsize=(24,12))
figMuHitProb = plt.figure(figsize=(24,12))

index = 1

muArr = []
varArr = []
hitProbArr = []

for data in [ [[1, 0, 0, 0., 0., 0., 0., 0., 1., 200]],
              [[0, 1, 0, 0., 0., 0., 0., 0., 1., 200]],
              [[0, 0, 1, 0., 0., 0., 0., 0., 1., 200]],
              
              [[1, 0, 0, 0., 0., 0., -0.6, 0., 0.8, 500]],
              [[0, 1, 0, 0., 0., 0., -0.6, 0., 0.8, 500]],
              [[0, 0, 1, 0., 0., 0., -0.6, 0., 0.8, 500]],
              
              [[1, 0, 0, 0., 0., 0., -0.9797958971132712, 0., 0.2, 700]],
              [[0, 1, 0, 0., 0., 0., -0.9797958971132712, 0., 0.2, 700]],
              [[0, 0, 1, 0., 0., 0., -0.9797958971132712, 0., 0.2, 700]] ] :
    


    data = torch.as_tensor(data).cpu()

    thisAxMu = figMu.add_subplot(3, 3, index)
    eventMu = np.exp(net(data).cpu().detach().numpy()[0, 1])
    eventMu = eventMu.reshape((88,168))
    imMu = thisAxMu.imshow(eventMu, vmin = 0, vmax = 5)
    figMu.colorbar(imMu)
    muArr.append(eventMu.flatten())
    
    thisAxVar = figVariance.add_subplot(3, 3, index)
    eventVariance = np.exp(net(data).cpu().detach().numpy()[0, 0])
    eventVariance = eventVariance.reshape((88,168))
    imVar = thisAxVar.imshow(eventVariance, vmin = 0, vmax = 25)
    figVariance.colorbar(imVar)
    varArr.append(eventVariance.flatten())

    thisAxHitProb = figHitProb.add_subplot(3, 3, index)
    eventHitProb = 1./(1+np.exp(net(data).cpu().detach().numpy()[0, 2]))
    eventHitProb = eventHitProb.reshape((88,168))
    imHitProb = thisAxHitProb.imshow(eventHitProb, vmin = 0, vmax = 1)
    figHitProb.colorbar(imHitProb)
    hitProbArr.append(eventHitProb.flatten())
    
    thisAxMuHitProb = figMuHitProb.add_subplot(3, 3, index)
    eventMuHitProb = np.copy(eventMu)*eventHitProb
    imMuHitProb = thisAxMuHitProb.imshow(eventMuHitProb, vmin = 0, vmax = 5)
    figMuHitProb.colorbar(imMuHitProb)
    
    data = data.numpy()
    
    for thisAx in [thisAxVar, thisAxMu, thisAxHitProb, thisAxMuHitProb] :
        
        if data[0][0] == 1 :
            thisAx.set_title(r'$\gamma$ E='+str(data[0][-1])+' MeV $\phi$='+"{:.2f}".format(atan2(data[0][8], data[0][6])))
        elif data[0][1] == 1 :
            thisAx.set_title(r'$e$ E='+str(data[0][-1])+' MeV $\phi$='+"{:.2f}".format(atan2(data[0][8], data[0][6])))
        elif data[0][2] == 1 :
            thisAx.set_title(r'$\mu$ E='+str(data[0][-1])+' MeV $\phi$='+"{:.2f}".format(atan2(data[0][8], data[0][6])))
        else :
            print("INVALID PID", data)
    index +=1


figMu.tight_layout()
figVariance.tight_layout()
figHitProb.tight_layout()
figMuHitProb.tight_layout()

figMu.savefig("CRinGe_Gaus_Mu.png")
figVariance.savefig("CRinGe_Gaus_Variance.png")
figHitProb.savefig("CRinGe_Gaus_HitProb.png")
figMuHitProb.savefig("CRinGe_Gaus_MuHitProb.png")

# figMuVar = plt.figure()
# largeProb = hitProbArr < 0.05
# plt.scatter(x = muArr[largeProb], y = varArr[largeProb])
# figMuVar.savefig("CRinGe_Gaus_MuVar.png")

plt.show()
