import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
import sys

sqrtpi = np.sqrt(np.pi)
sqrt2 = np.sqrt(2.)

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
            torch.nn.Conv2d(32, 4, 3)                                  # 88 x 168
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
        net = self._upconvs(net).view(-1, 4, 88*168)
        return net
        # 3rd channel is probability, pass through Sigmoid
#        hitprob = self._sigmoid(net[:,2])
       
#        return torch.cat((net[:,0:2],hitprob.view(-1, 1, 88*168)), 1).view(-1, 3, 88*168)


# blobbedy blob blob
class BLOB :
    pass
blob = BLOB()

blob.net = CRinGeNet().cuda()
#blob.bceloss = torch.nn.BCELoss(reduction = 'sum')
blob.bceloss = torch.nn.BCEWithLogitsLoss()
#blob.criterion = torch.nn.SmoothL1Loss()
# Clip gradient norm to avoid exploding gradients:
torch.nn.utils.clip_grad.clip_grad_norm_(blob.net.parameters(), 1.0)
blob.optimizer = torch.optim.Adam(blob.net.parameters(), lr = 0.0002)
blob.data = None
blob.label = None

# Forward path
def forward(blob, train=True) :
    with torch.set_grad_enabled(train) :
        data = torch.as_tensor(blob.data).cuda()
        prediction = blob.net(data)

        # Training
        loss, acc = -1, -1
        if blob.label is not None :
            label = torch.as_tensor(blob.label).type(torch.FloatTensor).cuda()
#            loss = blob.criterion(prediction, label)

            logvar = prediction[:,0]
            logmu = prediction[:,1]
            loglambda = prediction[:,2]
            punhit = prediction[:,3]
            
            var = torch.exp(logvar)
            mu = torch.exp(logmu)
            lambd = torch.exp(loglambda)

            unhitMask = (label == 0)

            unhitTarget = torch.as_tensor(unhitMask).type(torch.FloatTensor).cuda()
            fracUnhit = unhitTarget.sum()/unhitTarget.numel()
            
            loss = fracUnhit*blob.bceloss(punhit, unhitTarget)

#            print("bceloss", loss.mean())
            
            loss -= (1-fracUnhit)*torch.log(lambd[~unhitMask]).mean()/2.
#            print ("add lambda over 2", loss.mean())
            loss -= (1-fracUnhit)*(lambd[~unhitMask]/2*(2*mu[~unhitMask] + lambd[~unhitMask]*var[~unhitMask] - 2*label[~unhitMask])).mean()
#            print("add exponential factor", loss.mean())
            
            x = (mu[~unhitMask] + lambd[~unhitMask] * var[~unhitMask] - label[~unhitMask])/(sqrt2*var[~unhitMask]**0.5)
            negX = x < 0
            # xAbs = torch.abs(x)
            # erfc approximation from Karagiannidis & Lioumpas (2007). Discontinuity around x = 0 !!!
            # erfc = ( 1 - torch.exp(-1.98*xAbs))*torch.exp(-xAbs**2)/(1.135*sqrtpi*xAbs)

            # erfc approximation from Tsai et al (2012)
            erfcPos = torch.exp(-1.09500814703333*x[~negX] - 0.75651138383854*x[~negX]**2)
#            print("positive erfc", erfcPos.mean())
            erfcNeg = 2 - torch.exp(-1.09500814703333*(-1*x[negX]) - 0.75651138383854*(-1*x[negX])**2)
#            print("negative erfc", erfcNeg.mean())
#            print("posx", x[~negX])
#            print("negx", x[negX])
#            print("negative erfc entries posx", (erfcPos< 0).sum())
#            print("negative erfc entries negx", (erfcNeg< 0).sum())
            
#            torch.set_printoptions(profile="full")
#            print(torch.cat((erfcPos, erfcNeg)))
#            torch.set_printoptions(profile="default")
#            print(torch.cat((erfcPos, erfcNeg)).size())
#            print(torch.cat((erfcPos, erfcNeg)).mean())
            loss -= (1-fracUnhit)*torch.log(torch.cat((erfcPos, erfcNeg))+1e-10).mean()
#            print("Add erfc", loss.mean())
            
            
            if loss != loss :
                with open("fDebug.p", "wb") as f:
                    print("################################################################################")
                    print("NAN LOSS")
                    print("################################################################################")
                    print("LOSS")
                    print(loss.cpu().detach().numpy())
                    pickle.dump(loss.cpu().detach().numpy(), f)
                    print("################################################################################")
                    print("DATA")
                    print(data.cpu().numpy())
                    pickle.dump(data.cpu().numpy(), f)
                    print("--------------------------------------------------------------------------------")
                    print("LABEL")
                    print(label.cpu().numpy())
                    pickle.dump(label.cpu().numpy(), f)
                    print("--------------------------------------------------------------------------------")
                    print("punhit")
                    print(punhit.cpu().detach().numpy())
                    pickle.dump(punhit.cpu().detach().numpy(), f)
                    print("--------------------------------------------------------------------------------")
                    print("MU")
                    print(mu.cpu().detach().numpy())
                    pickle.dump(mu.cpu().detach().numpy(), f)
                    print("--------------------------------------------------------------------------------")
                    print("VAR")
                    pickle.dump(var.cpu().detach().numpy(), f)
                    print(var.cpu().detach().numpy())
                    print("--------------------------------------------------------------------------------")
                    print("LOGMU")
                    print(logmu.cpu().detach().numpy())
                    pickle.dump(logmu.cpu().detach().numpy(), f)
                    print("--------------------------------------------------------------------------------")
                    print("LOGVAR")
                    pickle.dump(logvar.cpu().detach().numpy(), f)
                    print(logvar.cpu().detach().numpy())
                    sys.stdout.flush()
                exit()

            blob.loss = loss
        return {'prediction' : prediction.cpu().detach().numpy(),
                'loss' : loss.cpu().detach().item()}
# Backward path
def backward(blob) :
    blob.optimizer.zero_grad()
    blob.loss.backward()
    blob.optimizer.step()


# Data loaders
from iotools import loader_factory
DATA_DIRS=['/home/cvilela/HKML/varyAll/']
#DATA_DIRS=['/storage/shared/cvilela/HKML/varyAll']
train_loader=loader_factory('H5Dataset', batch_size=200, shuffle=True, num_workers=8, data_dirs=DATA_DIRS, flavour='1M.h5', start_fraction=0.0, use_fraction=0.75, read_keys= ["positions","directions", "energies"])
test_loader=loader_factory('H5Dataset', batch_size=200, shuffle=True, num_workers=2, data_dirs=DATA_DIRS, flavour='1M.h5', start_fraction=0.75, use_fraction=0.25, read_keys= [ "positions","directions", "energies"])

# Useful function
def fillLabel (blob, data) :
    # Label is vector of charges. Mind = Blown
    dim = data[0].shape

    blob.label = data[0][:,:,:,0].reshape(-1,dim[1]*dim[2])

def fillData (blob,data) :
    # Data is particle state

    oneHotGamma = np.array(data[1] == 0)
    oneHotE = np.array(data[1] == 1)
    oneHotMu = np.array(data[1] == 2)
    
    blob.data =  np.hstack((oneHotGamma.reshape(len(oneHotGamma),1), oneHotE.reshape(len(oneHotE),1), oneHotMu.reshape(len(oneHotMu),1), # One-hot PID
                            data[2][:,0,:], # Positions
                            data[3][:,0,:], # Directions
                            data[4][:,0].reshape(len(data[4][:,0]),1) ) ) # Energy
                    

# Training loop
TRAIN_EPOCH = 10.
blob.net.train()
epoch = 0.
iteration = 0.

#torch.autograd.set_detect_anomaly(True)

while epoch < TRAIN_EPOCH :
    print('Epoch', epoch, int(epoch+0.5), 'Starting @',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for i,data in enumerate(train_loader) :
        fillLabel(blob,data)
        fillData(blob,data)

        res = forward(blob, True)
        backward(blob)
        epoch += 1./len(train_loader)
        iteration += 1

        # Report progress
        if i == 0 or (i+1)%10 == 0 :
            print('TRAINING', 'Iteration', iteration, 'Epoch', epoch, 'Loss', res['loss'])
            
        if (i+1)%100 == 0 :
            with torch.no_grad() :
                blob.net.eval()
                test_data = next(iter(test_loader))
                fillLabel(blob,test_data)
                fillData(blob,test_data)
                res = forward(blob, False)
                print('VALIDATION', 'Iteration', iteration, 'Epoch', epoch, 'Loss', res['loss'])

        if (iteration+1)%7363 == 0 :
            torch.save(blob.net.state_dict(), "testCRinGe_EMG_i_"+str(iteration)+".cnn")
        if epoch >= TRAIN_EPOCH :
            break

torch.save(blob.net.state_dict(), "testCRinGe_EMG.cnn")
