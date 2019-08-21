import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
import sys
import h5py
import scipy.stats as stats

# CRinGeNet
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
            torch.nn.ConvTranspose2d(64, 64, 4, 2),  torch.nn.ReLU(),  # 24 x 44 
            torch.nn.Conv2d(64, 64, 3), torch.nn.ReLU(),               # 22 x 42 
                                                                                 
            torch.nn.ConvTranspose2d(64, 32, 4, 2), torch.nn.ReLU(),   # 46 x 86 
            torch.nn.Conv2d(32, 32, 3),  torch.nn.ReLU(),              # 44 x 84 
                                                                                 
            torch.nn.ConvTranspose2d(32, 32, 4, 2), torch.nn.ReLU(),   # 90 x 170
            torch.nn.Conv2d(32, 3, 3)                                  # 88 x 168
        )

#        self._sigmoid = torch.nn.Sigmoid()

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
        # 3rd channel is probability, pass through Sigmoid
#        hitprob = self._sigmoid(net[:,2])
       
#        return torch.cat((net[:,0:2],hitprob.view(-1, 1, 88*168)), 1).view(-1, 3, 88*168)


net = CRinGeNet().cuda()

net.load_state_dict(torch.load("testCRinGe_Gaus_i_66266.0.cnn", map_location=lambda storage, loc: storage))

torch.set_grad_enabled(False)
net.eval()

N_Plot_Residual = 21
#N_Plot_Residual = 22

#fIn = h5py.File("/home/cvilela/HKML/varyAll/IWCDgrid_varyAll_e-_20-2000MeV_100k.h5")
fIn = h5py.File("/home/cvilela/HKML/varyAll/IWCDgrid_varyAll_mu-_20-2000MeV_100k.h5")
N_Events = len(fIn['event_data'])

print(fIn.keys())
print(N_Events)

directions = fIn['directions']
energies = fIn['energies']
event_data = fIn['event_data']
pids = fIn['pids']
positions = fIn['positions']

pulls = np.array([])
charges = np.array([])
mus = np.array([])
pHits = np.array([], dtype = bool)
hits = np.array([])

chargesThrown = np.array([])
pullsThrown = np.array([])

N_Events = 1000

for i in range(N_Events) :
    print("Reading event", i)

    data = event_data[i]
#    dim = data[0].shape
    event_q = data[:,:,0]
    
    if abs(pids[i][0]) == 11 :
        oneHotPID = [0., 1., 0]
    elif abs(pids[i][0]) == 13 :
        oneHotPID = [0., 0., 1.]
    elif abs(pids[i][0]) == 22 :
        oneHotPID = [1., 0., 0.]
    else :
        print("UNKOWN PID", pids[i][0])
        exit()

    params = [[np.concatenate((oneHotPID, positions[i][0], directions[i][0], energies[i]))]]
    params = torch.as_tensor(params).float().cuda().view(1,10)
    prediction = net(params).cpu().detach().numpy()[0]
    prediction = prediction.reshape((3,88,168))
    
    predMu = np.exp(prediction[1])
    predVar = np.exp(prediction[0])
    predHitProb = 1/(1+np.exp(-1*prediction[2]))

    thrownEvent = np.random.uniform(size = predHitProb.shape)
    thrownEvent = (thrownEvent > predHitProb).astype(float)
    thrownEvent *= np.multiply((predVar**0.5), np.random.normal(size = predHitProb.shape)) + predMu

    # Apply non-negative hit charge prior
    negatives = thrownEvent < 0
    while np.sum(negatives) != 0 :
        print(np.sum(negatives))
        thrownEvent[negatives] = 1.
        thrownEvent[negatives] *= np.multiply((predVar[negatives]**0.5), np.random.normal(size = predHitProb.shape)[negatives]) + predMu[negatives]
        negatives = thrownEvent < 0
    
    if i == N_Plot_Residual :
        figResidual = plt.figure()#figsize=(24,4))
        
        plt.subplot(2, 2, 1)
        plt.imshow(predMu)
        plt.title("NN Prediction")
        plt.colorbar()
        plt.subplot(2, 2, 3)
        plt.imshow(thrownEvent)
        plt.title("NN thrown event")
        plt.colorbar()
        plt.subplot(2, 2, 4)
        plt.imshow(event_q)
        plt.title("MC event")
        plt.colorbar()
        plt.subplot(2, 2, 2)
        plt.imshow(predMu-event_q)
        plt.title("Residual")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig("compare_event_prediction_"+str(i)+".png")

        plt.figure()
        thisQMask = (event_q > 1.5) & (event_q < 2.0)
        thisQMaskPos = thisQMask & ((event_q - predMu)/np.sqrt(predVar) > 0)
        thisQMaskNeg = thisQMask & ((event_q - predMu)/np.sqrt(predVar) < 0)
        debugEvent = np.zeros(thisQMask.shape)
        debugEvent[thisQMaskPos] = 1.
        debugEvent[thisQMaskNeg] = -1.
        plt.imshow(debugEvent)
        plt.colorbar()
        
        
    hitMask = event_q > 0
    thesePulls = np.hstack((event_q[hitMask] - predMu[hitMask])/np.sqrt(predVar[hitMask]))
    charges = np.append(charges, np.hstack(event_q[hitMask]))
    mus = np.append(mus, np.hstack(predMu[hitMask]))
    pulls = np.append(pulls, thesePulls)
    pHits = np.append(pHits, np.hstack(predHitProb))
    hits = np.append(hits, np.hstack(hitMask)).astype(bool)

    thrownHitMask = thrownEvent > 0
    theseThrownPulls = np.hstack((thrownEvent[thrownHitMask] - predMu[thrownHitMask])/np.sqrt(predVar[thrownHitMask]))
    chargesThrown = np.append(chargesThrown, np.hstack(thrownEvent[thrownHitMask]))
    pullsThrown = np.append(pullsThrown, theseThrownPulls)

plt.figure()
nPhitsAll, bins, patches = plt.hist(pHits, bins = 100, range = (0, 1))
nPhitsHits, bins, patches = plt.hist(pHits[hits], bins = 100, range = (0,1))
print(len(nPhitsHits/nPhitsAll))
print(len(bins))
print(len((bins[:-1] + bins[1:])/2.))

plt.figure()
plt.plot([0,1],[1,0], linestyle = ':', color = '0.75')
plt.plot((bins[:-1] + bins[1:])/2.,  nPhitsHits/nPhitsAll)
plt.ylabel("Fraction of hit PMTs")
plt.xlabel(r"$P_{unhit}$")

plt.figure()
plt.hist(pulls, bins = 200, range = (-10, 10), density = True, histtype = 'step', label = r'$\frac{q_{MC}-\mu}{\sigma}$')
plt.hist(pullsThrown, bins = 200, range = (-10, 10), density = True, histtype = 'step', label = r'$\frac{q_{Throw}-\mu}{\sigma}$')
x = np.linspace(-10, 10, 200)
plt.plot(x, stats.norm.pdf(x, 0., 1.), label = r'$\mathcal{N}(0,1)$')
plt.legend()


plt.figure()
for i in np.arange(0.5, 4.0, 0.5) :
    thisQ = (charges > (i-0.5)) & (charges < i)
    plt.hist(pulls[thisQ], bins = 150, range = (-3, 3), density = True, histtype = 'step', label = r'$\frac{q_{MC}-\mu}{\sigma}, q \in ['+str(i-0.5)+','+str(i)+']$')

#hiQ = charges > 10
#plt.hist(pulls[hiQ], bins = 200, range = (-10, 10), density = True, histtype = 'step', label = r'$\frac{q-\mu}{\sigma}$')
x = np.linspace(-3, 3, 150)
plt.plot(x, stats.norm.pdf(x, 0., 1.), label = r'$\mathcal{N}(0,1)$')
plt.legend()

plt.figure()
for i in np.arange(0.5, 4.0, 0.5) :
    thisQ = (chargesThrown > (i-0.5)) & (chargesThrown < i)
    plt.hist(pullsThrown[thisQ], bins = 150, range = (-3, 3), density = True, histtype = 'step', label = r'$\frac{q_{Throw}-\mu}{\sigma}, q \in ['+str(i-0.5)+','+str(i)+']$')

#hiQ = charges > 10
#plt.hist(pulls[hiQ], bins = 200, range = (-10, 10), density = True, histtype = 'step', label = r'$\frac{q-\mu}{\sigma}$')
x = np.linspace(-3, 3, 150)
plt.plot(x, stats.norm.pdf(x, 0., 1.), label = r'$\mathcal{N}(0,1)$')
plt.legend()



plt.figure()
#plt.hist2d(x = mus, y = charges, bins = (100, 100))
#plt.hist2d(x = mus, y = charges, bins = (100, 100), range = ((0, 100), (0, 100)))
plt.scatter(x = mus, y = charges, alpha = 0.1)

plt.figure()
plt.hist(mus, histtype='step', bins = 100, range = (0, 500), label = r'$\mu$')
plt.hist(charges, histtype='step', bins = 100, range = (0, 500), label = r'$q$')
plt.yscale('log')
plt.legend()

plt.show()
