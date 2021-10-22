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
        )
        self._mlp_barrel = torch.nn.Sequential(
            torch.nn.Linear(1024, 8512), torch.nn.ReLU()   # 64 * 7 * 19
        )
        self._mlp_top = torch.nn.Sequential(
            torch.nn.Linear(1024, 2304), torch.nn.ReLU()   # 64 * 6 * 6
        )
        self._mlp_bottom = torch.nn.Sequential(
            torch.nn.Linear(1024, 2304), torch.nn.ReLU()
        )

        self._upconvs = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 64, 4, 2),  torch.nn.ReLU(),  # 16 x 40 
            torch.nn.Conv2d(64, 64, 3), torch.nn.ReLU(),               # 14 x 38 
                                                                                 
            torch.nn.ConvTranspose2d(64, 32, 4, 2), torch.nn.ReLU(),   # 30 x 78 
            torch.nn.Conv2d(32, 32, 3),  torch.nn.ReLU(),              # 28 x 76 
                                                                                 
            torch.nn.ConvTranspose2d(32, 32, 4, 2), torch.nn.ReLU(),   # 58 x 154
            torch.nn.Conv2d(32, 3, 3)                                  # 56 x 152
        )
        self._upconvs_top = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 64, 4, 2),  torch.nn.ReLU(),  # 14 x 14 
            torch.nn.Conv2d(64, 64, 3), torch.nn.ReLU(),               # 12 x 12 
                                                                                 
            torch.nn.ConvTranspose2d(64, 32, 4, 2), torch.nn.ReLU(),   # 26 x 26 
            torch.nn.Conv2d(32, 32, 3),  torch.nn.ReLU(),              # 24 x 24 
                                                                                 
            torch.nn.ConvTranspose2d(32, 32, 4, 2), torch.nn.ReLU(),   # 50 x 50
            torch.nn.Conv2d(32, 3, 3)                                  # 48 x 48
        )
        self._upconvs_bottom = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 64, 4, 2),  torch.nn.ReLU(),  # 14 x 14 
            torch.nn.Conv2d(64, 64, 3), torch.nn.ReLU(),               # 12 x 12 
                                                                                 
            torch.nn.ConvTranspose2d(64, 32, 4, 2), torch.nn.ReLU(),   # 26 x 26 
            torch.nn.Conv2d(32, 32, 3),  torch.nn.ReLU(),              # 24 x 24 
                                                                                 
            torch.nn.ConvTranspose2d(32, 32, 4, 2), torch.nn.ReLU(),   # 50 x 50
            torch.nn.Conv2d(32, 3, 3)                                  # 48 x 48
        )

        self._sigmoid = torch.nn.Sigmoid()

    def forward(self, x) :
        # Concatenate MLPs that treat PID, pos, dir and energy inputs separately
        net = torch.cat( (self._mlp_pid(x[:,0:3]),self._mlp_pos(x[:,3:6]),self._mlp_dir(x[:,6:9]),self._mlp_E(x[:,9].reshape(len(x[:,9]),1))), 1)

        # MegaMLP 
        net = self._mlp(net)
        net_barrel = self._mlp_barrel(net)
        net_top = self._mlp_top(net)
        net_bottom = self._mlp_bottom(net)
 
        # Reshape into 7 x 19 figure in 64 channels. Enough?!
        net_barrel = net_barrel.view(-1, 64, 7, 19)
        net_top = net_top.view(-1, 64, 6, 6)
        net_bottom = net_bottom.view(-1, 64, 6, 6)
        # Upconvs layers
        net_barrel = self._upconvs(net_barrel)[:,:,2:-3,1:-1]
        net_barrel = net_barrel.reshape(-1, 3, 51*150)
        net_top = self._upconvs_top(net_top).view(-1, 3, 48*48)
        net_bottom = self._upconvs_bottom(net_bottom).view(-1, 3, 48*48)

        return [net_barrel, net_bottom, net_top]

net = CRinGeNet().cpu()
net.load_state_dict(torch.load("testAllCRinGe_Gaus_SK_CapBar.cnn", map_location=lambda storage, loc: storage))

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
    eventMu = np.exp(net(data)[0].cpu().detach().numpy()[0,1])
    eventMu = eventMu.reshape((51,150))
#    eventMu = eventMu.reshape((48,48))
    imMu = thisAxMu.imshow(eventMu, vmin = 0, vmax = 5)
    figMu.colorbar(imMu)
    muArr.append(eventMu.flatten())
    
    thisAxVar = figVariance.add_subplot(3, 3, index)
    eventVariance = np.exp(net(data)[0].cpu().detach().numpy()[0,0])
    eventVariance = eventVariance.reshape((51,150))
#    eventVariance = eventVariance.reshape((48,48))
    imVar = thisAxVar.imshow(eventVariance, vmin = 0, vmax = 25)
    figVariance.colorbar(imVar)
    varArr.append(eventVariance.flatten())

    thisAxHitProb = figHitProb.add_subplot(3, 3, index)
    eventHitProb = 1./(1+np.exp(net(data)[0].cpu().detach().numpy()[0,2]))
    eventHitProb = eventHitProb.reshape((51,150))
 #   eventHitProb = eventHitProb.reshape((48,48))
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

figMu.savefig("CRinGe_Gaus_SK_barrel_Mu.png")
figVariance.savefig("CRinGe_Gaus_SK_barrel_Variance.png")
figHitProb.savefig("CRinGe_Gaus_SK_barrel_HitProb.png")
figMuHitProb.savefig("CRinGe_Gaus_SK_barrel_MuHitProb.png")

# figMuVar = plt.figure()
# largeProb = hitProbArr < 0.05
# plt.scatter(x = muArr[largeProb], y = varArr[largeProb])
# figMuVar.savefig("CRinGe_Gaus_MuVar.png")

plt.show()
