import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
import sys

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
blob.label_top = None
blob.label_bottom = None
blob.top_mask = None
blob.bottom_mask = None

# Forward path
def forward(blob, train=True) :
    with torch.set_grad_enabled(train) :
        data = torch.as_tensor(blob.data).cuda()
        prediction = blob.net(data)
        prediction_top = prediction[2]
        prediction_bottom = prediction[1]
        prediction = prediction[0]

        # Training
        loss, acc = -1, -1
        if blob.label is not None :
            label = torch.as_tensor(blob.label).type(torch.FloatTensor).cuda()
#            loss = blob.criterion(prediction, label)

            logvar = prediction[:,0]
            logmu = prediction[:,1]
            punhit = prediction[:,2]
            
            var = torch.exp(logvar)
            mu = torch.exp(logmu)

            #print (logvar.size())
            #print(logmu.size())

            unhitMask = (label == 0)

            unhitTarget = torch.as_tensor(unhitMask).type(torch.FloatTensor).cuda()
            fracUnhit = unhitTarget.sum()/unhitTarget.numel()
            
#            loss = fracUnhit*blob.bceloss(punhit, unhitTarget)
            loss = blob.bceloss(punhit, unhitTarget)

            loss += (1-fracUnhit)*(1/2.)*(logvar[~unhitMask] + (label[~unhitMask]-mu[~unhitMask])**2/var[~unhitMask]).mean()
            loss += (1-fracUnhit)*(1/2.)*np.log(2*np.pi)
            # Calculate the loss function on top and bottom caps
            if blob.label_top is not None:
                label_top = torch.as_tensor(blob.label_top).type(torch.FloatTensor).cuda()
                mask_top = torch.as_tensor(blob.top_mask).type(torch.FloatTensor).cuda()
                
                logvar_top = prediction_top[:,0] * mask_top
                logmu_top = prediction_top[:,1] * mask_top
                punhit_top = prediction_top[:,2] * mask_top
#                logvar_top = prediction_top[:,0] 
#                logmu_top = prediction_top[:,1] 
#                punhit_top = prediction_top[:,2] 

                var_top = torch.exp(logvar_top)
                mu_top = torch.exp(logmu_top)
               
                unhitMask_top = (label_top == 0)
                unhitTarget_top = torch.as_tensor(unhitMask_top).type(torch.FloatTensor).cuda()
                fracUnhit_top = unhitTarget_top.sum()/unhitTarget_top.numel() 

                loss += blob.bceloss(punhit_top, unhitTarget_top)
                loss += (1-fracUnhit_top)*(1/2.)*(logvar_top[~unhitMask_top] +(label_top[~unhitMask_top]-mu_top[~unhitMask_top])**2/var_top[~unhitMask_top]).mean()
                loss += (1-fracUnhit_top)*(1/2.)*np.log(2*np.pi)
 
            else:
                print("No label on top cap!")
                raise ValueError

            if blob.label_bottom is not None:
                label_bottom = torch.as_tensor(blob.label_bottom).type(torch.FloatTensor).cuda()
                mask_bottom = torch.as_tensor(blob.bottom_mask).type(torch.FloatTensor).cuda()
                
                logvar_bottom = prediction_bottom[:,0] * mask_bottom
                logmu_bottom = prediction_bottom[:,1] * mask_bottom
                punhit_bottom = prediction_bottom[:,2] * mask_bottom
#                logvar_bottom = prediction_bottom[:,0] 
#                logmu_bottom = prediction_bottom[:,1] 
#                punhit_bottom = prediction_bottom[:,2] 

                var_bottom = torch.exp(logvar_bottom)
                mu_bottom = torch.exp(logmu_bottom)
               
                unhitMask_bottom = (label_bottom == 0)
                unhitTarget_bottom = torch.as_tensor(unhitMask_bottom).type(torch.FloatTensor).cuda()
                fracUnhit_bottom = unhitTarget_bottom.sum()/unhitTarget_bottom.numel() 

                loss += blob.bceloss(punhit_bottom, unhitTarget_bottom)
                loss += (1-fracUnhit_bottom)*(1/2.)*(logvar_bottom[~unhitMask_bottom] +(label_bottom[~unhitMask_bottom]-mu_bottom[~unhitMask_bottom])**2/var_bottom[~unhitMask_bottom]).mean()
                loss += (1-fracUnhit_bottom)*(1/2.)*np.log(2*np.pi)
 
            else:
                print("No label on bottom cap!")
                raise ValueError

            if loss != loss :
                f_debug = open("input_debug.txt","wb")
                data_debug = data.cpu().numpy()
                np.savetxt(f_debug, data_debug, fmt='%s')
                f_debug.close()
       
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
                    #            loss = (torch.log(2*np.pi*prediction) + ((label-prediction)**2/prediction)).mean()
            blob.loss = loss
        return {'prediction' : prediction.cpu().detach().numpy(),
                'prediction_top' : prediction_top.cpu().detach().numpy(),
                'prediction_bottom' : prediction_bottom.cpu().detach().numpy(),
                'loss' : loss.cpu().detach().item()}
# Backward path
def backward(blob) :
    blob.optimizer.zero_grad()
    blob.loss.backward()
    blob.optimizer.step()

#reading in mask array
import h5py
file_handle = h5py.File("/storage/shared/mojia/trainingSamples/WCSim_mu-_npztoh5_test.h5", mode='r')
dim_cap = file_handle['mask'][0].shape
blob.top_mask = file_handle['mask'][0].reshape(-1, dim_cap[0]*dim_cap[1])
blob.bottom_mask = file_handle['mask'][1].reshape(-1, dim_cap[0]*dim_cap[1])
file_handle.close()

print(blob.top_mask)
print("mask array!")
# Data loaders
from iotools import loader_factory
DATA_DIRS=['/storage/shared/mojia/trainingSamples']
#DATA_DIRS=['/storage/shared/cvilela/HKML/varyAll']
train_loader=loader_factory('H5Dataset', batch_size=200, shuffle=True, num_workers=8, data_dirs=DATA_DIRS, flavour='test.h5', start_fraction=0.0, use_fraction=0.75, read_keys= ["positions","directions", "energies", "event_data_top", "event_data_bottom"])
test_loader=loader_factory('H5Dataset', batch_size=200, shuffle=True, num_workers=2, data_dirs=DATA_DIRS, flavour='test.h5', start_fraction=0.75, use_fraction=0.25, read_keys= [ "positions","directions", "energies", "event_data_top", "event_data_bottom"])

# Useful function
def fillLabel (blob, data) :
    # Label is vector of charges. Mind = Blown
    dim = data[0].shape

    blob.label = data[0][:,:,:,0].reshape(-1,dim[1]*dim[2])

    dim_cap = data[5].shape
    blob.label_top = data[5][:,:,:,0].reshape(-1, dim_cap[1]*dim_cap[2])
    blob.label_bottom = data[6][:,:,:,0].reshape(-1, dim_cap[1]*dim_cap[2])

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
TRAIN_EPOCH = 10.  #default 10
blob.net.train()
epoch = 0.
iteration = 0.

f = open("Training_data.txt","wb")
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
#        print('TRAINING', 'Iteration', iteration, 'Epoch', epoch, 'Loss', res['loss'])
        if i == 0 or (i+1)%10 == 0 :
            print('TRAINING', 'Iteration', iteration, 'Epoch', epoch, 'Loss', res['loss'])
            data = np.column_stack(['TRAINING', 'Iteration', iteration, 'Epoch',epoch,'Loss', res['loss']])
            np.savetxt(f, data, fmt= '%s')
       
        if (i+1)%100 == 0 :
            with torch.no_grad() :
                blob.net.eval()
                test_data = next(iter(test_loader))
                fillLabel(blob,test_data)
                fillData(blob,test_data)
                res = forward(blob, False)
                print('VALIDATION', 'Iteration', iteration, 'Epoch', epoch, 'Loss', res['loss'])
                data = np.column_stack(["VALIDATION", 'Iteration', iteration, 'Epoch', epoch, 'Loss', res['loss']])
                np.savetxt(f, data, fmt = "%s")

        if (iteration+1)%7363 == 0 :
            torch.save(blob.net.state_dict(), "testAllCRinGe_Gaus_SK_CapBar_i_"+str(iteration)+".cnn")
        if epoch >= TRAIN_EPOCH :
            break

f.close()
torch.save(blob.net.state_dict(), "testAllCRinGe_Gaus_SK_CapBar.cnn")
