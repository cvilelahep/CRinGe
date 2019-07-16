import torch
import numpy as np
import time
import matplotlib.pyplot as plt

BATCH_SIZE=200

# CRinGeNet
class CRinGeNet(torch.nn.Module) :
    def __init__(self) :
        super(CRinGeNet, self).__init__()

#        self.mu = torch.nn.Parameter(torch.zeros(5))
#        self.logvar = torch.nn.Parameter(torch.ones(5))

#        self.varPars = torch.nn.Parameter(torch.zeros(5))
        
        self._mlp_pid = torch.nn.Sequential(
            torch.nn.Linear(3,512), torch.nn.ReLU(),
            torch.nn.Linear(512,512), torch.nn.ReLU()
        )

        self._mlp_pos = torch.nn.Sequential(
            torch.nn.Linear(3,512), torch.nn.ReLU(),
            torch.nn.Linear(512,512), torch.nn.ReLU()
        )

        self._mlp_dir = torch.nn.Sequential(
            torch.nn.Linear(3,512), torch.nn.ReLU(),
            torch.nn.Linear(512,512), torch.nn.ReLU()
        )

        self._mlp_E = torch.nn.Sequential(
            torch.nn.Linear(1,512), torch.nn.ReLU(),
            torch.nn.Linear(512,512), torch.nn.ReLU()
        )

        self._mlp_varPars = torch.nn.Sequential(
            torch.nn.Linear(5, 512), torch.nn.ReLU(),
            torch.nn.Linear(512, 512), torch.nn.ReLU()
        )
        
        self._mlp = torch.nn.Sequential(
            torch.nn.Linear(2560, 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, 14784), torch.nn.ReLU()
        )


        self._upconvs = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 64, 4, 2), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3), torch.nn.ReLU(),

            torch.nn.ConvTranspose2d(64, 32, 4, 2), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3), torch.nn.ReLU(),

            torch.nn.ConvTranspose2d(32, 32, 4, 2), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 1, 3)
        )

    def forward(self, x) :
        # Concatenate MLPs that treat PID, pos, dir and energy inputs separately

        varPars = torch.autograd.Variable(x[:,10:15], requires_grad = True)
        
        net = torch.cat( (self._mlp_pid(x[:,0:3]),self._mlp_pos(x[:,3:6]),self._mlp_dir(x[:,6:9]),self._mlp_E(x[:,9].reshape(len(x[:,9]),1)), self._mlp_varPars(varPars)), 1)

        # MegaMLP 
        net = self._mlp(net)
        
        # Reshape into 11 x 21 figure in 64 channels. Enough?!
        net = net.view(-1, 64, 11, 21)

        # Need to flatten? Maybe...
        return self._upconvs(net).view(-1, 88*168), varPars

   
# blobbedy blob blob
class BLOB :
    pass
blob = BLOB()

blob.net = CRinGeNet().cuda()
blob.criterion = torch.nn.SmoothL1Loss()
blob.latentLoss = torch.nn.MSELoss()#reduce=True, reduction='mean')
blob.optimizer = torch.optim.Adam(blob.net.parameters())
blob.data = None
blob.label = None

# Forward path
def forward(blob, train=True) :
    with torch.set_grad_enabled(train) :
#        print('forward called', torch.cuda.memory_allocated())
        data = torch.as_tensor(blob.data).cuda()
#        print(data[0])
#        print('send data to gpu', torch.cuda.memory_allocated())
        prediction, varPars = blob.net(data)
#        print('first prediction', torch.cuda.memory_allocated())
#        print("VAR PARS")
#        print(varPars)
        # Training
        loss, acc = -1, -1
        if blob.label is not None :

            label = torch.as_tensor(blob.label).type(torch.FloatTensor).cuda()
#            print('Send label to gpu', torch.cuda.memory_allocated())
           
            # UPDATE VARPARS using the backward path:
            thisLoss = blob.criterion(prediction, label)
#            print('Get Loss', torch.cuda.memory_allocated())
#            thisLoss.backward()
#            print('Go backward', torch.cuda.memory_allocated())
#            print ("GRAD")
#            dVarPars = varPars.grad
            dVarPars = torch.autograd.grad(thisLoss, varPars, retain_graph = True)[0]
#            print(dVarPars)
#            dOtherPars = torch.autograd.grad(thisLoss, blob.net.parameters())
#            print(dOtherPars)
#            print(dVarPars.size())
#            print(dOtherPars.size())
#            exit()
 
            varParsUpdated = varPars.clone()
            varParsUpdated.add_(-1, dVarPars)
#            print("UPDATED VAR PARS")
#            print(varParsUpdated)
#            print('update vars', torch.cuda.memory_allocated())
            
            data[:,10:15] = varParsUpdated.data
#            print('update data', torch.cuda.memory_allocated())
            # Let's go again, this time it counts
            prediction, varPars = blob.net(data)
#            print('second prediction', torch.cuda.memory_allocated())
            
            lossIMAGE = blob.criterion(prediction, label)
            lossLATENT = blob.latentLoss(varPars, torch.zeros_like(varPars))
#            print('Get losses', torch.cuda.memory_allocated())
            loss = lossIMAGE + lossLATENT
        blob.loss = loss

        return {'prediction' : prediction.cpu().detach().numpy(),
                'loss' : loss.cpu().detach().item(),
                'lossIMAGE' : lossIMAGE.cpu().detach().item(),
                'lossLATENT' : lossLATENT.cpu().detach().item()}

# Backward path
def backward(blob) :
#    print('backward called', torch.cuda.memory_allocated())
    blob.optimizer.zero_grad()
#    print('optimizer zero grad', torch.cuda.memory_allocated())
    blob.loss.backward()
#    print('go backward', torch.cuda.memory_allocated())
    blob.optimizer.step()
#    print('optimizer step', torch.cuda.memory_allocated())


# Data loaders
from iotools import loader_factory
DATA_DIRS=['/storage/shared/cvilela/HKML/varyAll']
train_loader=loader_factory('H5Dataset', batch_size=BATCH_SIZE, shuffle=True, num_workers=8, data_dirs=DATA_DIRS, flavour='1M.h5', start_fraction=0.0, use_fraction=0.75, read_keys= ["positions","directions", "energies"])
test_loader=loader_factory('H5Dataset', batch_size=BATCH_SIZE, shuffle=True, num_workers=2, data_dirs=DATA_DIRS, flavour='1M.h5', start_fraction=0.75, use_fraction=0.25, read_keys= [ "positions","directions", "energies"])

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

    varPars = torch.randn(len(oneHotGamma), 5)
    
    blob.data =  np.hstack((oneHotGamma.reshape(len(oneHotGamma),1), oneHotE.reshape(len(oneHotE),1), oneHotMu.reshape(len(oneHotMu),1), # One-hot PID
                            data[2][:,0,:], # Positions
                            data[3][:,0,:], # Directions
                            data[4][:,0].reshape(len(data[4][:,0]),1), # Energy
                            varPars)) # Free Parameters
                    

# Training loop
TRAIN_EPOCH = 0.5
blob.net.train()
epoch = 0.
iteration = 0.

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
            print('TRAINING', 'Iteration', iteration, 'Epoch', epoch, 'Loss', res['loss'], 'LossIMAGE', res['lossIMAGE'], 'lossLATENT', res['lossLATENT'])
            print(torch.cuda.memory_allocated())

            
        if (i+1)%100 == 0 :
            test_data = next(iter(test_loader))
#            print('got test data', torch.cuda.memory_allocated())
            fillLabel(blob,test_data)
#            print('Filled label', torch.cuda.memory_allocated())
            fillData(blob,test_data)
#            print('Filled data', torch.cuda.memory_allocated())
            res = forward(blob, True)
#            print('VALIDATION', 'Iteration', iteration, 'Epoch', epoch, 'Loss', res['loss'], 'LossIMAGE', res['lossIMAGE'], 'lossLATENT', res['lossLATENT'])
            del blob.loss
#            print('deleted loss', torch.cuda.memory_allocated())

        if epoch >= TRAIN_EPOCH :
            break

torch.save(blob.net.state_dict(), "testCRinGe_FIP2.cnn")
