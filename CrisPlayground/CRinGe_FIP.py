import torch
import numpy as np
import time
import matplotlib.pyplot as plt

# CRinGeNet
class CRinGeNet(torch.nn.Module) :
    def __init__(self) :
        super(CRinGeNet, self).__init__()

        self.mu = torch.nn.Parameter(torch.zeros(5))
        self.logvar = torch.nn.Parameter(torch.ones(5))

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

        if x.size()[1] == 10 :
        
            std = torch.exp(0.5*self.logvar)
            #        eps = torch.randn_like((x.size()[0],std.size()))
            eps = torch.randn(x.size()[0], std.size()[0]).cuda()
            
            varPars = self.mu + eps*std
        else :
            varPars = x[:,9:13]
        
        net = torch.cat( (self._mlp_pid(x[:,0:3]),self._mlp_pos(x[:,3:6]),self._mlp_dir(x[:,6:9]),self._mlp_E(x[:,9].reshape(len(x[:,9]),1)), self._mlp_varPars(varPars)), 1)

        # MegaMLP 
        net = self._mlp(net)
        
        # Reshape into 11 x 21 figure in 64 channels. Enough?!
        net = net.view(-1, 64, 11, 21)

        # Need to flatten? Maybe...
        return self._upconvs(net).view(-1, 88*168), self.mu, self.logvar

# blobbedy blob blob
class BLOB :
    pass
blob = BLOB()

blob.net = CRinGeNet().cuda()
blob.criterion = torch.nn.SmoothL1Loss()
blob.optimizer = torch.optim.Adam(blob.net.parameters())
blob.data = None
blob.label = None

# Forward path
def forward(blob, train=True) :
    with torch.set_grad_enabled(train) :
        data = torch.as_tensor(blob.data).cuda()
        prediction, mu, logvar = blob.net(data)

        # Training
        loss, acc = -1, -1
        if blob.label is not None :
            label = torch.as_tensor(blob.label).type(torch.FloatTensor).cuda()
            lossIMAGE = blob.criterion(prediction, label)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = lossIMAGE + KLD
        blob.loss = loss

        return {'prediction' : prediction.cpu().detach().numpy(),
                'loss' : loss.cpu().detach().item(),
                'lossIMAGE' : lossIMAGE.cpu().detach().item(),
                'KLD' : KLD.cpu().detach().item()}

# Backward path
def backward(blob) :
    blob.optimizer.zero_grad()
    blob.loss.backward()
    blob.optimizer.step()


# Data loaders
from iotools import loader_factory
DATA_DIRS=['/storage/shared/cvilela/HKML/varyAll']
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
TRAIN_EPOCH = 1.
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
            print('TRAINING', 'Iteration', iteration, 'Epoch', epoch, 'Loss', res['loss'], 'LossIMAGE', res['lossIMAGE'], 'KLD', res['KLD'])
            
        if (i+1)%100 == 0 :
            with torch.no_grad() :
                blob.net.eval()
                test_data = next(iter(test_loader))
                fillLabel(blob,test_data)
                fillData(blob,test_data)
                res = forward(blob, False)
                print('VALIDATION', 'Iteration', iteration, 'Epoch', epoch, 'Loss', res['loss'], 'LossIMAGE', res['lossIMAGE'], 'KLD', res['KLD'])

        if epoch >= TRAIN_EPOCH :
            break

torch.save(blob.net.state_dict(), "testCRinGe_FIP.cnn")
