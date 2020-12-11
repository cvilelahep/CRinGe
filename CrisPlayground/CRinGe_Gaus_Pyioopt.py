import torch
from torch.utils.data import DataLoader, BatchSampler
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
            torch.nn.Linear(1024, 8512), torch.nn.ReLU()
        )


        self._upconvs = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 64, 4, 2),  torch.nn.ReLU(),  # 16 x 40 
            torch.nn.Conv2d(64, 64, 3), torch.nn.ReLU(),               # 14 x 38 
                                                                                 
            torch.nn.ConvTranspose2d(64, 32, 4, 2), torch.nn.ReLU(),   # 30 x 78 
            torch.nn.Conv2d(32, 32, 3),  torch.nn.ReLU(),              # 28 x 76 
                                                                                 
            torch.nn.ConvTranspose2d(32, 32, 4, 2), torch.nn.ReLU(),   # 58 x 154
            torch.nn.Conv2d(32, 3, 3)                                  # 56 x 152

        )

        self._sigmoid = torch.nn.Sigmoid()

    def forward(self, x) :
        # Concatenate MLPs that treat PID, pos, dir and energy inputs separately
        net = torch.cat( (self._mlp_pid(x[:,0:3]),self._mlp_pos(x[:,3:6]),self._mlp_dir(x[:,6:9]),self._mlp_E(x[:,9].reshape(len(x[:,9]),1))), 1)

        # MegaMLP 
        net = self._mlp(net)
        
        # Reshape into 7 x 19 figure in 64 channels. Enough?!

        net = net.view(-1, 64, 7, 19)


        # Get rid of extra rows and columns: 2 on the long side, 5 on the short side

        net = self._upconvs(net)[:,:,2:-3,1:-1]

        # Need to flatten? Maybe...
#        net = net.view(-1, 3, 51*150)

        return net


# blobbedy blob blob
class BLOB :
    pass
blob = BLOB()

blob.net = CRinGeNet().cuda()
#blob.net = CRinGeNet().cpu()
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
#        data = torch.as_tensor(blob.data).cpu()
        data = torch.as_tensor(blob.data).cuda()
        prediction = blob.net(data)

        # Training
        loss, acc = -1, -1
        if blob.label is not None :
#            label = torch.as_tensor(blob.label).type(torch.FloatTensor).cpu()
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
#            unhitTarget = torch.as_tensor(unhitMask).type(torch.FloatTensor).cpu()
            fracUnhit = unhitTarget.sum()/unhitTarget.numel()
            
#            loss = fracUnhit*blob.bceloss(punhit, unhitTarget)
            loss = blob.bceloss(punhit, unhitTarget)

            loss += (1-fracUnhit)*(1/2.)*(logvar[~unhitMask] + (label[~unhitMask]-mu[~unhitMask])**2/var[~unhitMask]).mean()
            loss += (1-fracUnhit)*(1/2.)*np.log(2*np.pi)

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
                    #            loss = (torch.log(2*np.pi*prediction) + ((label-prediction)**2/prediction)).mean()
            blob.loss = loss
        return {'prediction' : prediction.cpu().detach().numpy(),
                'loss' : loss.cpu().detach().item()}
# Backward path
def backward(blob) :
    blob.optimizer.zero_grad()
    blob.loss.backward()
    blob.optimizer.step()


# Data loaders
from iotools.pyioopt_dataset import PyiooptDataset, SemiRandomSampler
#DATA_FILES = "/disk/cvilela/WCML/TrainingSampleWCSim/*/*/WCSim/out/WCSim_TrainingSample_*.root"
DATA_FILES = "/disk/cvilela/WCML/TrainingSampleWCSim_SmallFiles/*/*/WCSim/out/*.root"
from pyioopt.wcsim import wcsim_reader

event_reader = wcsim_reader.Reader()
event_reader.addFile(DATA_FILES)
mask = event_reader.mask

train_dataset = PyiooptDataset(reader = event_reader,
                               transform = None,
                               start_fraction = 0.,
                               use_fraction = 0.75)
train_loader = DataLoader( train_dataset,
                           num_workers = 4,
                           pin_memory = True,
                           persistent_workers = True,
                           batch_sampler = BatchSampler(SemiRandomSampler(data_source = train_dataset, sequence_length = 11), batch_size = 200, drop_last = False))
                           
test_dataset = PyiooptDataset(reader = event_reader,
                               transform = None,
                               start_fraction = 0.75,
                               use_fraction = 0.25)
test_loader = DataLoader(test_dataset,
                         num_workers = 4,
                         pin_memory = True,
                         persistent_workers = True,
                         batch_sampler = BatchSampler(SemiRandomSampler(data_source = test_dataset, sequence_length = 11), batch_size = 200, drop_last = False))
                         

print("Finished data loader init")

# Useful function
def fillLabel (blob, data) :
    blob.label = data["q_barrel"]
#    print(blob.label)
#    print(blob.label.size())
    
def fillData (blob, data) :
    blob.data = torch.hstack((data["pid"], data["pos"], data["dir"], data["Ekin"]))
#    print(blob.data)
#    print(blob.data.size())
    

# Training loop
TRAIN_EPOCH = 10.
blob.net.train()
epoch = 0.
iteration = 0.


while epoch < TRAIN_EPOCH :
    print('Epoch', epoch, int(epoch+0.5), 'Starting @',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    for i, data in enumerate(train_loader) :
        fillLabel(blob, data)
        fillData(blob, data)

        res = forward(blob, True)
        backward(blob)
        epoch += 1./len(train_loader)
        iteration += 1

        if iteration >= 200 :
            exit()
        
        # Report progress
#        if i == 0 or (i+1)%10 == 0 :
        print('TRAINING', 'Iteration', iteration, 'Epoch', epoch, 'Loss', res['loss'])
            
        if (i+1)%100 == 0 :
            with torch.no_grad() :
                blob.net.eval()
                test_data = next(iter(test_loader))
                fillLabel(blob, test_data)
                fillData(blob, test_data)
                res = forward(blob, False)
                print('VALIDATION', 'Iteration', iteration, 'Epoch', epoch, 'Loss', res['loss'])

        if (iteration+1)%7363 == 0 :
            torch.save(blob.net.state_dict(), "testCRinGe_Gaus_i_"+str(iteration)+".cnn")
        if epoch >= TRAIN_EPOCH :
            break

torch.save(blob.net.state_dict(), "testCRinGe_Gaus.cnn")
