import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
import sys

N_GAUS=1

if len(sys.argv) == 2 :
    N_GAUS = int(sys.argv[1])

print("RUNNING WITH "+str(N_GAUS)+" GAUSSIANS and Timing Peak")

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
            # phit, gaussian logvar, logmu for both time and charge, coefficient
            torch.nn.Conv2d(32, 1+N_GAUS*6, 3)                                  # 88 x 168
        )

        self._sigmoid = torch.nn.Sigmoid()
#        self._softmax = torch.nn.Softmax(dim=1)

    def forward(self, x) :
        # Concatenate MLPs that treat PID, pos, dir and energy inputs separately
        net = torch.cat( (self._mlp_pid(x[:,0:3]),self._mlp_pos(x[:,3:6]),self._mlp_dir(x[:,6:9]),self._mlp_E(x[:,9].reshape(len(x[:,9]),1))), 1)

        # MegaMLP 
        net = self._mlp(net)
        
        # Reshape into 11 x 21 figure in 64 channels. Enough?!
        net = net.view(-1, 64, 11, 21)
        # Need to flatten? Maybe...
        net = self._upconvs(net).view(-1, 1+N_GAUS*6, 88*168)
        # better way?
        for i in range(N_GAUS):
            a11 = torch.exp(net[:, 1+i*5, :])
            a22 = torch.exp(net[:, 1+i*5+2, :])
            a12 = net[:,1+i*5+4,:]
            
            net[:,1+i*5+4,:] = (a11+a22)*self._sigmoid(a12) - 0.5*(a11+a22)
            

        return net


# blobbedy blob blob
class BLOB :
    pass
blob = BLOB()

blob.net = CRinGeNet().cuda()
#blob.bceloss = torch.nn.BCELoss(reduction = 'mean')
blob.bceloss = torch.nn.BCEWithLogitsLoss(reduction = 'mean')
#blob.criterion = torch.nn.SmoothL1Loss()
# Clip gradient norm to avoid exploding gradients:
torch.nn.utils.clip_grad.clip_grad_norm_(blob.net.parameters(), 1.0)
blob.optimizer = torch.optim.Adam(blob.net.parameters(), lr = 0.0002)
blob.data = None
blob.label = None
blob.pmtx = None
blob.dist = None
blob.time = None


# define pmt coordinates
#def fillPMTx(blob):
    # Label is vector of charges. Mind = Blown
#    dim = [200, 88, 168]
    #print(dim[1],dim[2])
#    blob.pmtx = torch.ones([dim[0], dim[1]*dim[2], 3], dtype=torch.float).cuda()
    #print(blob.pmtx.shape)
#    for i in range(dim[1]):
#        for j in range(dim[2]):
#            pmtpos = [1336*np.cos((j/dim[2]-0.5)/np.pi), 1336*np.sin((j/dim[2]-0.5)/np.pi), (i/dim[1]-0.5)*4400]
#            blob.pmtx[:, i*j] = torch.as_tensor(pmtpos).type(torch.FloatTensor).cuda() 
#            print(blob.pmtx[0, i, j, :])

# Forward path
def forward(blob, train=True) :
    with torch.set_grad_enabled(train) :
        data = torch.as_tensor(blob.data).cuda()
        prediction = blob.net(data)
            
        # Training
        loss, acc = -1, -1
        if blob.label is not None :
            label = torch.as_tensor(blob.label).type(torch.FloatTensor).cuda()
            time = torch.as_tensor(blob.time).type(torch.FloatTensor).cuda()
#            loss = blob.criterion(prediction, label)
            label_n = torch.stack( [ label for i in range(N_GAUS) ] ) # better way of doing this?
            time_n = torch.stack( [ time for i in range(N_GAUS) ], dim = 0)
#            ones = torch.tensor([1]).expand_as(label_n).type(torch.FloatTensor).cuda()

            punhit = prediction[:,0]
            coeff = torch.nn.functional.softmax(prediction[:, -N_GAUS:], dim=1)
            coefficients = torch.stack( [ coeff[:,i]  for i in range(N_GAUS) ], dim = 0 )

            loga12 = torch.stack( [ prediction[:,1+i*5+4] for i in range(N_GAUS) ], dim = 0 )
            a12 = torch.exp(loga12)
            
            loga11 = torch.stack( [ prediction[:, 1+i*5] for i in range(N_GAUS) ], dim = 0 )
            a11 = torch.exp(loga11)
            tmu = torch.stack( [ prediction[:, 1+i*5+1] for i in range(N_GAUS) ], dim = 0 )
            tmu_sub_time = time_n - tmu
#            tVAR = tvar + var_share
#            logtVAR = torch.log(tVAR)
            
            loga22 = torch.stack( [ prediction[:,1+i*5+2] for i in range(N_GAUS) ], dim = 0 )
            a22 = torch.exp(loga22)
            logmu = torch.stack( [ prediction[:,1+i*5+3] for i in range(N_GAUS) ], dim = 0 )
            mu = torch.exp(logmu)            
            mu_sub_label = label_n - mu
#            VAR = var + var_share
#            logVAR = torch.log(VAR)


#            covariance = var_share
#            corr = covariance/((VAR*tVAR)**0.5)
#            corrterm = ones - corr**2            
            unhitMask = (label == 0)


            unhitTarget = torch.as_tensor(unhitMask).type(torch.FloatTensor).cuda()
            fracUnhit = unhitTarget.sum()/unhitTarget.numel()

            loss = blob.bceloss(punhit, unhitTarget)
            CTloss = (1-fracUnhit)*np.log(2*np.pi)
            CTloss += -(1-fracUnhit)*torch.logsumexp(torch.log(coefficients[:,~unhitMask]) + loga11[:,~unhitMask]+loga22[:,~unhitMask] - 1/2.*((tmu_sub_time[:,~unhitMask]*a11[:,~unhitMask])**2 + mu_sub_label[:,~unhitMask]**2*(a22[:,~unhitMask]**2 + a12[:, ~unhitMask]**2) + 2*tmu_sub_time[:,~unhitMask]*mu_sub_label[:,~unhitMask]*a11[:,~unhitMask]*a12[:,~unhitMask]), dim=0).mean()
      


            loss += CTloss

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
                    print("A22")
                    pickle.dump(a22.cpu().detach().numpy(), f)
                    print(a22.cpu().detach().numpy())
                    print("--------------------------------------------------------------------------------")
                    print("LOGMU")
                    print(logmu.cpu().detach().numpy())
                    pickle.dump(logmu.cpu().detach().numpy(), f)
                    print("--------------------------------------------------------------------------------")
                    print("LOGA22")
                    pickle.dump(loga22.cpu().detach().numpy(), f)
                    print(loga22.cpu().detach().numpy())
                    sys.stdout.flush()
                exit()
            blob.loss = loss
        return {'prediction' : prediction.cpu().detach().numpy(),
                'loss' : loss.cpu().detach().item(),
                'CTloss' : CTloss.cpu().detach().item()}
# Backward path
def backward(blob) :
    blob.optimizer.zero_grad()
    blob.loss.backward()
    blob.optimizer.step()


# Data loaders
from iotools import loader_factory
#DATA_DIRS=['/home/cvilela/HKML/varyAll/']
#DATA_DIRS=['/storage/shared/cvilela/HKML/varyAll']
DATA_DIRS=['/home/junjiex/projects/def-pdeperio/junjiex/HKML/varyAll']
train_loader=loader_factory('H5Dataset', batch_size=200, shuffle=True, num_workers=4, data_dirs=DATA_DIRS, flavour='1M.h5', start_fraction=0.0, use_fraction=0.75, read_keys= ["positions","directions", "energies"])
test_loader=loader_factory('H5Dataset', batch_size=200, shuffle=True, num_workers=1, data_dirs=DATA_DIRS, flavour='1M.h5', start_fraction=0.75, use_fraction=0.24, read_keys= [ "positions","directions", "energies"])

# Useful function
def fillLabel (blob, data) :
    # Label is vector of charges. Mind = Blown
    dim = data[0].shape
    blob.label = data[0][:,:,:,0].reshape(-1,dim[1]*dim[2])

def fillTime (blob, data) :
    dim = data[0].shape
    blob.time = 0.1*(data[0][:,:,:,1]).reshape(-1, dim[1]*dim[2])

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
#sys.stdout = open("N2GausTrain.log","w")
fillPMTx(blob)

while epoch < TRAIN_EPOCH :
    print('Epoch', epoch, int(epoch+0.5), 'Starting @',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for i,data in enumerate(train_loader) :
        fillLabel(blob,data)
        fillData(blob,data)
        fillTime(blob,data)

        res = forward(blob, True)
        backward(blob)
        epoch += 1./len(train_loader)
        iteration += 1

        # Report progress
        if i == 0 or (i+1)%10 == 0 :
            print('TRAINING', 'Iteration', iteration, 'Epoch', epoch, 'HitProb Loss', res['loss'] - res['CTloss'], 'CT Loss', res['CTloss'])
            
        if (i+1)%100 == 0 :
            with torch.no_grad() :
                blob.net.eval()
                test_data = next(iter(test_loader))
                fillLabel(blob,test_data)
                fillData(blob,test_data)
                fillTime(blob,test_data)
                res = forward(blob, False)
                print('VALIDATION', 'Iteration', iteration, 'Epoch', epoch, 'HitProb Loss', res['loss'] - res['CTloss'], 'CT Loss', res['CTloss'])

        if (iteration+1)%7363 == 0 :
            torch.save(blob.net.state_dict(), "testCRinGe_MultiGausTimeCorr_"+str(N_GAUS)+"_i_"+str(iteration)+".cnn")
        if epoch >= TRAIN_EPOCH :
            break

torch.save(blob.net.state_dict(), "testCRinGe_MultiGausTimeCorr_"+str(N_GAUS)+".cnn")
#sys.stdout.close
