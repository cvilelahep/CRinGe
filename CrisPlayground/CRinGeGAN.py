import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from math import pi

# CRinGeNet
class CRinGeGen(torch.nn.Module) :
    def __init__(self) :
        super(CRinGeGen, self).__init__()

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

        self._mlp_var = torch.nn.Sequential(
            torch.nn.Linear(5,512), torch.nn.ReLU(),
            torch.nn.Linear(512,512), torch.nn.ReLU()
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
        net = torch.cat( (self._mlp_pid(x[:,0:3]) ,self._mlp_pos(x[:,3:6]), self._mlp_dir(x[:,6:9]), self._mlp_E(x[:,9].reshape(len(x[:,9]),1) ), self._mlp_var(x[:,9:14])), 1)

        # MegaMLP 
        net = self._mlp(net)
        
        # Reshape into 11 x 21 figure in 64 channels. Enough?!
        net = net.view(-1, 64, 11, 21)

        # Need to flatten? Maybe...
        return self._upconvs(net).view(-1, 88*168)
    
class CRinGeDisc(torch.nn.Module) :
    def __init__(self) :
        super(CRinGeDisc, self).__init__()

        self._convs = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3), torch.nn.ReLU(), torch.nn.MaxPool2d(2,2), # 88*168 -> 86*166 -> 43*83
            torch.nn.Conv2d(32, 32, 4), torch.nn.ReLU(), torch.nn.MaxPool2d(2,2), # 43*83 -> 40*80 -> 20*40
            torch.nn.Conv2d(32, 64, 3), torch.nn.ReLU(), torch.nn.MaxPool2d(2,2), # 20*40 -> 18*38 ->  9*19
            torch.nn.Conv2d(64, 64, 4), torch.nn.ReLU(), torch.nn.MaxPool2d(2,2), # 9*19 -> 6*16 ->  3*8
        )
        
        self._mlp = torch.nn.Sequential(
            torch.nn.Linear(64*3*8+10, 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, 1)
        )
        
        
    def forward(self, x) :

        # First 88*168 elements are event
        net = x[:,:88*168].view(-1, 1, 88, 168)

        # Put this through the convolutional layers:
        net = self._convs(net)

        # Now should be 64 channels * 9 * 14, add generator imput to this and put through final MLP
#        print(net.size())
#        print(x[:, 88*168:].size())
        return self._mlp(torch.cat((net.view(-1, 64*3*8), x[:, 88*168:]),1))
    
# blobbedy blob blob
class BLOB :
    pass
blob = BLOB()

blob.gen = CRinGeGen().cuda()
blob.disc = CRinGeDisc().cuda()
blob.criterion = torch.nn.SmoothL1Loss()
blob.optimizerGen = torch.optim.Adam(blob.gen.parameters())
blob.optimizerDisc = torch.optim.Adam(blob.disc.parameters())
blob.data = None
blob.label = None
blob.RandomNoise = None
blob.RandomData = None

# Forward path
#def forward(blob, train=True) :
#    with torch.set_grad_enabled(train) :
#        data = torch.as_tensor(blob.data).cuda()
#        prediction = blob.net(data)
#
#        # Training
#        loss, acc = -1, -1
#        if blob.label is not None :
#            label = torch.as_tensor(blob.label).type(torch.FloatTensor).cuda()
#            loss = blob.criterion(prediction, label)
#        blob.loss = loss
#
#        return {'prediction' : prediction.cpu().detach().numpy(),
#                'loss' : loss.cpu().detach().item()}
## Backward path
#def backward(blob) :
#    blob.optimizer.zero_grad()
#    blob.loss.backward()
#    blob.optimizer.step()


# Data loaders
from iotools import loader_factory
DATA_DIRS=['/storage/shared/cvilela/HKML/varyAll']
train_loader=loader_factory('H5Dataset', batch_size=200, shuffle=True, num_workers=8, data_dirs=DATA_DIRS, flavour='1M.h5', start_fraction=0.0, use_fraction=0.75, read_keys= ["positions","directions", "energies"])
test_loader=loader_factory('H5Dataset', batch_size=200, shuffle=True, num_workers=2, data_dirs=DATA_DIRS, flavour='1M.h5', start_fraction=0.75, use_fraction=0.25, read_keys= ["positions","directions", "energies"])

# Useful function
def fillLabel (blob, data) :
    dim = data[0].shape    
    blob.label = data[0][:,:,:,0].reshape(-1,dim[1]*dim[2])
                         
def fillDataRandom (blob, data) :
    dim = data[0].shape

    # PID
    randomPID = torch.LongTensor(dim[0]).random_(0,3).data

    oneHotGamma = randomPID == 0
    oneHotE = randomPID == 1
    oneHotMu = randomPID == 2

    phi = torch.rand(dim[0],1) * 2 * pi
    r = 250 * torch.rand(dim[0],1)**0.5
    x = r*torch.cos(phi,)
    z = r*torch.sin(phi)
    y = torch.rand(dim[0],1)*1100-550

    dx = torch.randn(dim[0],1)
    dy = torch.randn(dim[0],1)
    dz = torch.randn(dim[0],1)

    dx /= ((dx**2 + dy**2 + dz**2)**0.5)
    dy /= ((dx**2 + dy**2 + dz**2)**0.5)
    dz /= ((dx**2 + dy**2 + dz**2)**0.5)
    
    energy = (torch.rand(dim[0])*1980+20).reshape(dim[0], 1)

    var0 = torch.rand(dim[0],1)
    var1 = torch.rand(dim[0],1)
    var2 = torch.rand(dim[0],1)
    var3 = torch.rand(dim[0],1)
    var4 = torch.rand(dim[0],1)
    
    blob.RandomData = np.hstack( (oneHotGamma.reshape(len(oneHotGamma),1), oneHotE.reshape(len(oneHotE),1), oneHotMu.reshape(len(oneHotMu),1),
                            x, y, z,
                            dx, dy, dz,
                            energy ) )
    blob.RandomNoise = np.hstack((var0, var1, var2, var3, var4) )
    
def fillData (blob,data) :
    # Data is particle state
                         
    oneHotGamma = np.array(data[1] == 0)
    oneHotE = np.array(data[1] == 1)
    oneHotMu = np.array(data[1] == 2)
    
    blob.data = np.hstack((oneHotGamma.reshape(len(oneHotGamma),1), oneHotE.reshape(len(oneHotE),1), oneHotMu.reshape(len(oneHotMu),1), # One-hot PID
                           data[2][:,0,:], # Positions
                           data[3][:,0,:], # Directions
                           data[4][:,0].reshape(len(data[4][:,0]),1) ) ) # Energy
                    

# Training loop
TRAIN_EPOCH = 0.01
blob.disc.train()
blob.gen.train()
epoch = 0.
iteration = 0.

while epoch < TRAIN_EPOCH :
    print('Epoch', epoch, int(epoch+0.5), 'Starting @',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for i ,data in enumerate(train_loader) :
        epoch += 1./len(train_loader)
        iteration += 1
        fillLabel(blob,data)
        fillData(blob,data)
        fillDataRandom(blob,data)
        
        batch_size = data[0].shape[0]
        
        real = torch.autograd.Variable(torch.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False).cuda()
        fake = torch.autograd.Variable(torch.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False).cuda()

        blob.optimizerGen.zero_grad()

        genRings = blob.gen(torch.cat((torch.FloatTensor(blob.RandomData), torch.FloatTensor(blob.RandomNoise)), 1).cuda())
#        genRings = blob.gen(torch.cat((torch.FloatTensor(blob.data), torch.FloatTensor(blob.RandomNoise)), 1).cuda())
        
        # Update generator
        validity = blob.disc(torch.cat((genRings, torch.FloatTensor(blob.RandomData).cuda()), 1))
#        validity = blob.disc(torch.cat((genRings, torch.FloatTensor(blob.data).cuda()), 1))
        gen_loss = blob.criterion(validity, real)
        gen_loss.backward()
        blob.optimizerGen.step()
                             
        # Update discriminator
        blob.optimizerDisc.zero_grad()
        validity_real = blob.disc(torch.cat((torch.FloatTensor(blob.label).cuda(), torch.FloatTensor(blob.data).cuda()), 1))
        disc_real_loss = blob.criterion(validity_real, real)
        
        validity_fake = blob.disc(torch.cat((genRings.detach(), torch.FloatTensor(blob.RandomData).cuda()),1))

        disc_fake_loss = blob.criterion(validity_fake, fake)
        
        disc_loss = (disc_real_loss+disc_fake_loss)/2.

        disc_loss.backward()

        blob.optimizerDisc.step()
        torch.cuda.empty_cache() 
        
        # Report progress
        if i == 0 or (i+1)%10 == 0 :
            print('TRAINING', 'Iteration', iteration, 'Epoch', epoch, 'Generator loss', gen_loss.item(), "Discriminator loss", disc_loss.item())
            
        if (i+1)%100 == 0 :
            with torch.no_grad() :
                blob.gen.eval()
                blob.disc.eval()
                test_data = next(iter(test_loader))
                fillLabel(blob,data)
                fillData(blob,data)
                fillDataRandom(blob,data)
                
                batch_size = (data[0].shape)[0]
                
                real = torch.autograd.Variable(torch.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False).cuda()
                fake = torch.autograd.Variable(torch.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False).cuda()
                
                genRings = blob.gen(torch.cat((torch.FloatTensor(blob.RandomData).cuda(), torch.FloatTensor(blob.RandomNoise).cuda()), 1).cuda())
#                genRings = blob.gen(torch.cat((torch.FloatTensor(blob.data).cuda(), torch.FloatTensor(blob.RandomNoise).cuda()), 1).cuda())

                validity = blob.disc(torch.cat((genRings, torch.FloatTensor(blob.RandomData).cuda()), 1))
#                validity = blob.disc(torch.cat((genRings, torch.FloatTensor(blob.data).cuda()), 1))
                gen_loss = blob.criterion(validity, real)
        
                validity_real = blob.disc(torch.cat((torch.FloatTensor(blob.label).cuda(), torch.FloatTensor(blob.data).cuda()),1))
                disc_real_loss = blob.criterion(validity_real, real)
                validity_fake = blob.disc(torch.cat((genRings, torch.FloatTensor(blob.RandomData).cuda()),1))
#                validity_fake = blob.disc(torch.cat((genRings, torch.FloatTensor(blob.data).cuda()),1))
                disc_fake_loss = blob.criterion(validity_fake, fake)
                
                disc_loss = (disc_real_loss+disc_fake_loss)/2.
                torch.cuda.empty_cache() 
                print('TEST', 'Iteration', iteration, 'Epoch', epoch, 'Generator loss', gen_loss.item(), "Discriminator loss", disc_loss.item())


        if epoch >= TRAIN_EPOCH :
            break

torch.save(blob.gen.state_dict(), "testCRinGeGAN_generator.cnn")
torch.save(blob.disc.state_dict(), "testCRinGeGAN_discriminator.cnn")
